from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.intervals import interval_provenance, interval_support, minimum_rank_samples
from nixtla_scaffold.outputs import build_interval_diagnostics, build_trust_summary
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec


def _run(alias, levels=(80, 95)):
    t = np.arange(40)
    data = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2020-01-01", periods=40, freq="MS"),
                         "y": 100 + 0.3 * t + 3 * np.sin(t * np.pi / 2)})
    return run_forecast(data, ForecastSpec(
        freq="MS", season_length=4, horizon=3, model_allowlist=(alias,), levels=levels,
        evaluation=EvaluationPolicy(max_windows=2),
    ))


@pytest.mark.parametrize("alias,method", [
    ("AutoTBATS", "statsforecast_native_tbats"), ("AutoETS", "statsforecast_conformal_distribution"),
    ("MSTL", "statsforecast_native"), ("Ridge", "mlforecast_conformal_distribution"),
])
def test_actual_method_and_calibration_are_recorded_at_each_fit(alias, method):
    run = _run(alias)
    for stage in ("future", "backtest"):
        provenance = interval_provenance(run, "A", alias, stage=stage)
        assert provenance["method"] == method
        assert all(row["levels"] == [80, 95] for row in provenance["receipts"])
        for row in provenance["receipts"]:
            calibration = row["calibration"]
            if "conformal" in method:
                assert 3 <= calibration["windows"] <= run.spec.evaluation.max_calibration_windows
                assert len(calibration["score_sha256"]) == 64
                for period in calibration["periods"]:
                    assert all(pd.Timestamp(target) > pd.Timestamp(period["cutoff"]) for target in period["targets"])
            else:
                assert calibration is None
    assert build_trust_summary(run).interval_status.eq("insufficient_observations").all()
    diagnostics = build_interval_diagnostics(run)
    assert diagnostics.interval_method.eq(method).all()
    assert diagnostics.mean_interval_score.ge(0).all()
    assert diagnostics.disjoint_windows.le(2).all()


def test_api_minimum_cannot_certify_95_percent_coverage():
    run = _run("AutoETS")
    assert minimum_rank_samples(95) == 20
    assert minimum_rank_samples(80) == 5
    assert not interval_support(run, "A", "AutoETS", 95, stage="future")
    evidence = run.model_policy_resolution["candidate_evidence"]
    for row in evidence["fits"]:
        row["intervals"]["calibration"]["windows"] = 20
    assert interval_support(run, "A", "AutoETS", 95, stage="future")
    # Even adequate within-fit samples cannot replace later empirical evidence.
    assert build_interval_diagnostics(run).interval_status.eq("insufficient_observations").all()


def test_more_calibration_windows_do_not_remove_ml_lags_to_manufacture_support():
    from nixtla_scaffold.models import _mlforecast_interval_plan
    plan = _mlforecast_interval_plan(min_obs=63, horizon=3, season_length=4,
                                    lags=list(range(1, 14)), levels=(95,),
                                    max_windows=20, allow_lag_cap=False)
    assert plan.lags == tuple(range(1, 14))
    assert plan.n_windows == 16


def test_ml_calibration_reserves_real_training_rows_not_only_one_after_the_lags():
    from nixtla_scaffold.models import _mlforecast_interval_plan
    plan = _mlforecast_interval_plan(min_obs=96, horizon=3, season_length=12,
                                    lags=list(range(1, 14)), levels=(95,), max_windows=20,
                                    allow_lag_cap=False, minimum_effective_rows=24)
    assert plan.n_windows == 19
    assert 96 - plan.n_windows * 3 - max(plan.lags) >= 24


def test_missing_execution_evidence_is_not_inferred_from_model_family():
    run = _run("AutoTBATS")
    legacy = replace(run, model_policy_resolution={})
    assert interval_provenance(legacy, "A", "AutoTBATS")["method"] == "unverified_legacy"
    assert not interval_support(legacy, "A", "AutoTBATS", 95, stage="future")


def test_point_only_run_never_inherits_conformal_label():
    run = _run("AutoETS", levels=())
    assert interval_provenance(run, "A", "AutoETS")["method"] == "point_forecast_only"
