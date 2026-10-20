import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.workbench import dispatch, execute_job
from nixtla_scaffold.workbench_confirmation import _radii
from nixtla_scaffold.workbench_store import WorkbenchStore, WorkbenchError
from nixtla_scaffold.schema import ForecastSpec, forecast_spec_from_dict


def test_calibration_is_per_horizon_and_uses_finite_sample_order_statistics():
    calibration = pd.DataFrame({"unique_id": ["x"] * 40, "horizon_step": [1] * 20 + [2] * 20,
                                "y": [0.] * 40, "Blend": list(range(1, 21)) + list(range(101, 121))})
    radii = _radii(calibration, [80, 95])
    assert radii[radii["level"] == 95]["radius"].tolist() == [20., 120.]
    assert radii[radii["level"] == 80]["radius"].tolist() == [17., 117.]
    assert _radii(calibration.head(3), [95]).empty


@pytest.mark.parametrize("method,aggregate,policy", [
    ("native", False, "legacy"), ("split_conformal", False, "legacy"),
    ("native", True, "legacy"), ("native", False, "greedy"),
])
def test_frozen_confirmation_calibrates_then_adopts_without_changing_parent(tmp_path, monkeypatch, method, aggregate, policy):
    if method != "native":
        pytest.importorskip("conformalforecast")
    dispatch(tmp_path, "initialize", {})
    store = WorkbenchStore(tmp_path)
    def profile(periods, reserved):
        path = tmp_path / f"history-{periods}.csv"
        pd.DataFrame({"unique_id": ["x"] * periods, "ds": pd.date_range("2020-01-01", periods=periods, freq="D" if aggregate else "MS"),
                      "y": 100 + np.arange(periods) + np.sin(np.arange(periods))}).to_csv(path, index=False)
        return dispatch(tmp_path, "profile", {"path": str(path), "horizon": 3, "holdoutWindows": reserved})["sourceId"]
    sid = profile(48, 1)
    job = dispatch(tmp_path, "start_job", {"kind": "forecast", "sourceId": sid, "requestKey": "original",
                                         "improvementPolicy": policy,
                                         "spec": {"horizon": 3, "model_policy": "baseline",
                                                  "interval_calibration": {"method": method, "aggregate_totals": aggregate}}})
    original = execute_job(store, job["jobId"])["result"]["runId"]
    blends = dispatch(tmp_path, "view", {"runId": original})["blends"]
    blend = next(item for item in blends if item["recipe"]["policy"] == "greedy") if policy == "greedy" else blends[0]
    protected = profile(120, 21)
    from nixtla_scaffold import workbench_confirmation
    native_forecast = workbench_confirmation.run_forecast
    fitted_history = []
    def observed_forecast(frame, spec):
        fitted_history.append(frame[["unique_id", "ds", "y"]].copy())
        return native_forecast(frame, spec)
    monkeypatch.setattr(workbench_confirmation, "run_forecast", observed_forecast)
    request = {"kind": "confirm_blend", "sourceId": protected, "runId": original, "blendId": blend["id"],
               "expectedReceipt": blend["receiptHash"], "referenceModel": "Naive", "requestKey": "confirmation"}
    job = dispatch(tmp_path, "start_job", request)
    confirmed = execute_job(store, job["jobId"])["result"]
    receipt = confirmed["receipt"]
    assert receipt["confirmation_status"] == "independently_reserved_within_workspace"
    assert receipt["calibration"]["windows"] == 20
    assert len(receipt["calibration"]["radii"]) == 6
    assert all(row["samples"] == 20 for row in receipt["calibration"]["radii"])
    assert len(receipt["confirmation_coverage"]) == 6
    assert [len(frame) for frame in fitted_history] == [*range(57, 118, 3), 120]
    assert all(frame["ds"].max() < pd.Timestamp("2029-10-01") for frame in fitted_history[:-1])
    partition = pd.read_csv(store.owned(f"jobs/{job['jobId']}/partitions.csv"))
    assert not partition.duplicated(["unique_id", "ds"]).any()
    with pytest.raises(WorkbenchError, match="prior job"):
        dispatch(tmp_path, "start_job", {**request, "requestKey": "reuse"})
    selected = dispatch(tmp_path, "view", {"runId": confirmed["runId"]})["blends"][0]
    adopted_job = dispatch(tmp_path, "start_job", {"kind": "adopt", "runId": confirmed["runId"], "blendId": selected["id"],
                                                   "expectedReceipt": selected["receiptHash"], "approved": True, "requestKey": "adopt"})
    adopted = execute_job(store, adopted_job["jobId"])["result"]
    view = dispatch(tmp_path, "view", {"runId": adopted["runId"]})
    assert {"yhat_lo_80", "yhat_hi_95"}.issubset(view["columns"])
    assert all(row["cv_windows"] == 21 for row in view["rows"])
    assert {row["interval_method"] for row in view["rows"]} == {
        "blend_residual_order_statistic" if method == "native" else "conformalforecast_split_conformal"}
    diagnostics = json.loads(store.read_artifact(adopted["runId"], "manifest.json"))
    assert diagnostics["spec"]["interval_calibration"]["aggregate_totals"] == aggregate
    if aggregate:
        support = dispatch(tmp_path, "evidence", {"runId": adopted["runId"], "name": "appendix/aggregate_intervals.csv"})
        assert support["rows"] and all(row["lower"] in (None, "") for row in support["rows"])
    evidence_path = diagnostics["outputs"]["candidate_evidence"]
    evidence = json.loads(store.read_artifact(adopted["runId"], evidence_path))
    assert all(fit["intervals"]["calibration"]["windows"] == (21 if method != "native" and fit["stage"] == "future" else 20)
               for fit in evidence["fits"])
    assert all(fit["intervals"]["confirmation_windows"] == 1 for fit in evidence["fits"])
    assert all(row["interval_status"] != "calibrated" for row in view["rows"])
    if method != "native":
        support = dispatch(tmp_path, "evidence", {"runId": adopted["runId"], "name": "appendix/interval_support.csv"})
        assert support["rows"]
        assert diagnostics["spec"]["interval_calibration"]["method"] == method
        fresh = profile(123, 1)
        refresh_job = dispatch(tmp_path, "start_job", {"kind": "refresh", "runId": adopted["runId"],
            "sourceId": fresh, "approved": True, "requestKey": "refresh-experimental"})
        refreshed = execute_job(store, refresh_job["jobId"])["result"]
        assert refreshed["confirmation"] == "new_snapshot_requires_new_confirmation"
        refreshed_manifest = json.loads(store.read_artifact(refreshed["runId"], "manifest.json"))
        assert refreshed_manifest["spec"]["interval_calibration"]["method"] == method
        recalibrated = json.loads(store.read_artifact(refreshed["runId"], "ensemble_adoption.json"))
        assert recalibrated["interval_calibration_source"] == "retrospective_frozen_replay"
        assert recalibrated.get("confirmation_status") != "independently_reserved_within_workspace"
        new_support = dispatch(tmp_path, "evidence", {
            "runId": refreshed["runId"], "name": "appendix/interval_support.csv"})
        assert new_support["rows"]
    assert store.run(original)


def test_paired_validation_defaults_on_for_new_specs_and_preserves_legacy():
    assert ForecastSpec().paired_validation
    assert not forecast_spec_from_dict({}).paired_validation
    assert forecast_spec_from_dict(ForecastSpec(paired_validation=True).to_dict()).paired_validation
