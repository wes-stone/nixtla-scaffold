from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import (
    DriverSnapshotGrant, DriverSourceContract, EvaluationPolicy, ForecastSpec,
    KnownFutureRegressor, run_forecast,
)
from nixtla_scaffold.drivers import parse_known_future_regressors, prepare_mlforecast_regressor_features
from nixtla_scaffold.feature_sources import DriverSourceError, source_features
from nixtla_scaffold.schema import forecast_spec_from_dict


def fixture(tmp_path, *, n=48, availability="historical_only", ids=("001",)):
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    rows = []
    targets = []
    for index, uid in enumerate(ids):
        values = np.random.default_rng(18 + index).normal(100, 10, n)
        rows.append(pd.DataFrame({
            "unique_id": uid, "ds": dates, "value": values, "available_at": dates,
            "period_end": dates, "vintage": "original", "units": "seats",
        }))
        targets.append(pd.DataFrame({"unique_id": uid, "ds": dates, "y": np.roll(values, 3) * 2}))
    snapshot = pd.concat(rows, ignore_index=True)
    path = tmp_path / "source.csv"
    snapshot.to_csv(path, index=False)
    grant = DriverSnapshotGrant("seats-source", str(path), sha256(path.read_bytes()).hexdigest(), ids)
    contract = DriverSourceContract("seats-source", "MS", "seats", "stock", "mock-query:v1",
                                    "Seats precede demand by three monthly periods")
    driver = KnownFutureRegressor("seats", value_col="seats", mode="model_candidate",
                                   availability=availability, source=contract,
                                   historical_lags=(3,) if availability == "historical_only" else ())
    spec = ForecastSpec(horizon=3, freq="MS", season_length=1, levels=(),
                        model_allowlist=("Ridge",), weighted_ensemble=False,
                        evaluation=EvaluationPolicy(max_windows=3),
                        regressors=(driver,), driver_snapshots=(grant,),
                        train_known_future_regressors=True)
    return pd.concat(targets, ignore_index=True), snapshot, spec


def publish(snapshot, spec):
    from pathlib import Path
    path = Path(spec.driver_snapshots[0].path)
    snapshot.to_csv(path, index=False)
    return replace(spec, driver_snapshots=(replace(spec.driver_snapshots[0],
                                                 sha256=sha256(path.read_bytes()).hexdigest()),))


def test_contract_and_grants_roundtrip_through_saved_specs_and_cli_declarations(tmp_path):
    _, _, spec = fixture(tmp_path)
    assert forecast_spec_from_dict(spec.to_dict()) == spec
    assert parse_known_future_regressors([json.dumps(spec.regressors[0].to_dict())]) == spec.regressors
    from nixtla_scaffold.cli import _parse_driver_snapshots
    assert _parse_driver_snapshots([json.dumps(spec.driver_snapshots[0].to_dict())]) == spec.driver_snapshots


def test_unapproved_source_is_rejected_before_any_file_read(tmp_path, monkeypatch):
    from nixtla_scaffold import feature_sources
    history, _, spec = fixture(tmp_path)
    monkeypatch.setattr(feature_sources, "read_granted_snapshot",
                        lambda _: pytest.fail("unapproved file was read"))
    with pytest.raises(DriverSourceError, match="no explicit snapshot grant"):
        source_features(history, history.iloc[:0], replace(spec, driver_snapshots=()),
                        spec.regressors[0], freq="MS", origin=history.ds.max())


@pytest.mark.parametrize("defect,message", [
    ("duplicate", "duplicate"), ("ambiguous_revision", "publication time"),
    ("grain", "grain differs"), ("entity", "entity scope"),
    ("missing_entity", "missing required columns"), ("units", "units"),
    ("date", "invalid ds"), ("value", "nonfinite"), ("missing_vintage", "vintage"),
    ("observation_end", "ends before"),
])
def test_bad_join_grain_units_vintages_and_values_fail_closed(tmp_path, defect, message):
    history, snapshot, spec = fixture(tmp_path)
    if defect == "duplicate":
        snapshot = pd.concat([snapshot, snapshot.iloc[[0]]], ignore_index=True)
    elif defect == "ambiguous_revision":
        snapshot = pd.concat([snapshot, snapshot.iloc[[0]].assign(vintage="revision")], ignore_index=True)
    elif defect == "grain":
        spec = replace(spec, regressors=(replace(spec.regressors[0],
                           source=replace(spec.regressors[0].source, grain="D")),))
    elif defect == "entity":
        snapshot.loc[0, "unique_id"] = "another-target"
    elif defect == "missing_entity":
        snapshot = snapshot.drop(columns="unique_id")
    elif defect == "units":
        snapshot.loc[0, "units"] = "USD"
    elif defect == "date":
        snapshot["ds"] = snapshot["ds"].astype(str)
        snapshot.loc[0, "ds"] = "invalid"
    elif defect == "value":
        snapshot.loc[0, "value"] = np.inf
    elif defect == "missing_vintage":
        snapshot.loc[0, "vintage"] = ""
    elif defect == "observation_end":
        snapshot.loc[0, "period_end"] = pd.Timestamp("1900-01-01")
    spec = publish(snapshot, spec)
    with pytest.raises(DriverSourceError, match=message):
        source_features(history, history.iloc[:0], spec, spec.regressors[0],
                        freq="MS", origin=history.ds.max())


def test_changed_snapshot_is_not_silently_refreshed(tmp_path):
    history, snapshot, spec = fixture(tmp_path)
    publish(snapshot.assign(value=1), spec)
    with pytest.raises(DriverSourceError, match="snapshot changed"):
        source_features(history, history.iloc[:0], spec, spec.regressors[0],
                        freq="MS", origin=history.ds.max())


def test_revisions_and_partial_periods_are_withheld_per_origin(tmp_path):
    history, snapshot, spec = fixture(tmp_path)
    origin = history.ds.iloc[30]
    revised = snapshot.iloc[[10]].assign(value=1e9, vintage="revised", available_at=origin + pd.offsets.MonthBegin())
    snapshot.loc[30, "period_end"] = origin + pd.offsets.MonthEnd()
    spec = publish(pd.concat([snapshot, revised], ignore_index=True), spec)
    before, _, receipt = source_features(history.iloc[:31], history.iloc[:0], spec, spec.regressors[0],
                                         freq="MS", origin=origin)
    assert before.loc[10, "seats"] == snapshot.loc[10, "value"]
    assert pd.isna(before.loc[30, "seats"])
    assert receipt["selected_vintages"] == ["original"]
    after, _, receipt = source_features(history.iloc[:32], history.iloc[:0], spec, spec.regressors[0],
                                        freq="MS", origin=history.ds.iloc[31])
    assert after.loc[10, "seats"] == 1e9
    assert "revised" in receipt["selected_vintages"]


def test_text_ids_and_ragged_series_do_not_broadcast_or_borrow_later_information(tmp_path):
    history, snapshot, spec = fixture(tmp_path, ids=("001", "NA"))
    history = history.loc[~(history.unique_id.eq("001") & history.ds.gt(history.ds.iloc[29]))]
    snapshot.loc[snapshot.unique_id.eq("001"), "available_at"] = history.ds.max()
    spec = publish(snapshot, spec)
    joined, _, receipt = source_features(history, history.iloc[:0], spec, spec.regressors[0],
                                         freq="MS", origin=history.ds.max())
    assert set(joined.unique_id) == {"001", "NA"}
    assert joined.loc[joined.unique_id.eq("001"), "seats"].isna().all()
    assert joined.loc[joined.unique_id.eq("NA"), "seats"].notna().all()
    assert receipt["series_origins"]["001"] != receipt["series_origins"]["NA"]


def test_safe_lag_can_use_released_values_despite_unavailable_recent_source_rows(tmp_path):
    history, snapshot, spec = fixture(tmp_path)
    spec = replace(spec, regressors=(replace(spec.regressors[0], historical_lags=(5,)),))
    snapshot.loc[46:, "available_at"] = pd.Timestamp("2099-01-01")
    spec = publish(snapshot, spec)
    bundle = prepare_mlforecast_regressor_features(
        history, spec, freq="MS", horizon=3, season_length=1,
        min_training_rows=12, forecast_origin=history.ds.max(),
    )
    assert bundle.feature_columns == ("seats_lag_5",)
    np.testing.assert_allclose(bundle.future["seats_lag_5"], snapshot.value.iloc[-5:-2])
    assert bundle.source_receipts[0]["rows_withheld_by_time"] == 2


def test_short_lag_does_not_substitute_realized_future_values(tmp_path):
    history, _, spec = fixture(tmp_path)
    spec = replace(spec, regressors=(replace(spec.regressors[0], historical_lags=(1,)),))
    bundle = prepare_mlforecast_regressor_features(
        history, spec, freq="MS", horizon=3, season_length=1,
        min_training_rows=12, forecast_origin=history.ds.max(),
    )
    assert not bundle.feature_columns
    assert "shorter than" in bundle.audit.iloc[0]["reason"]


def test_future_driver_forecasts_are_not_treated_as_realized_known_inputs(tmp_path):
    history, _, spec = fixture(tmp_path, availability="forecasted")
    with pytest.raises(DriverSourceError, match="paired upstream/downstream"):
        source_features(history, history.iloc[:0], spec, spec.regressors[0],
                        freq="MS", origin=history.ds.max())


def test_actual_ml_fits_reconstruct_snapshot_and_preserve_origin_receipts(tmp_path):
    history, _, spec = fixture(tmp_path)
    run = run_forecast(history, spec)
    assert "Ridge" in run.all_models
    assert run.backtest_predictions.Ridge.notna().all()
    evidence = run.model_policy_resolution["candidate_evidence"]
    recipes = evidence["recipes"]
    fits = [row for row in evidence["fits"] if row["model"] == "Ridge" and row["status"] == "succeeded"]
    assert len(fits) == 4
    for fit in fits:
        execution = recipes[fit["recipe_id"]]["execution"]
        assert execution["external_feature_columns"] == ["seats_lag_3"]
        receipt = execution["driver_source_receipts"][0]
        assert receipt["origin"] == fit["cutoff"]
        assert receipt["snapshot_sha256"] == spec.driver_snapshots[0].sha256


def test_unsorted_panel_features_are_joined_by_keys_not_positions(tmp_path):
    history, _, spec = fixture(tmp_path, ids=("001", "NA"))
    options = dict(freq="MS", horizon=3, season_length=1, min_training_rows=12, forecast_origin=history.ds.max())
    ordered = prepare_mlforecast_regressor_features(history, spec, **options)
    shuffled = prepare_mlforecast_regressor_features(history.sample(frac=1, random_state=13), spec, **options)
    for role in ("historical", "future"):
        pd.testing.assert_frame_equal(
            getattr(ordered, role).sort_values(["unique_id", "ds"]).reset_index(drop=True),
            getattr(shuffled, role).sort_values(["unique_id", "ds"]).reset_index(drop=True),
        )


def test_saved_snapshot_survives_original_change_and_workbench_relocation(tmp_path):
    from pathlib import Path
    from nixtla_scaffold.feature_sources import bind_saved_driver_snapshots
    from nixtla_scaffold.workbench_store import WorkbenchStore
    history, _, spec = fixture(tmp_path)
    run = run_forecast(history, spec)
    directory = run.to_directory(tmp_path / "run")
    manifest = json.loads((directory / "manifest.json").read_text())
    saved = manifest["driver_snapshot_artifacts"]["seats-source"]
    assert saved["status"] == "frozen"
    original = Path(spec.driver_snapshots[0].path)
    original.write_text("changed original\n")
    rebound = bind_saved_driver_snapshots(spec, directory, manifest)
    assert rebound.driver_snapshots[0].sha256 == spec.driver_snapshots[0].sha256
    (tmp_path / "workbench").mkdir()
    store = WorkbenchStore(tmp_path / "workbench")
    store.initialize()
    imported = store.import_run(directory)
    owned = store.run(imported["id"])[0]
    relocated = bind_saved_driver_snapshots(spec, owned, manifest)
    assert Path(relocated.driver_snapshots[0].path).is_relative_to(owned)
    assert relocated.driver_snapshots[0].sha256 == rebound.driver_snapshots[0].sha256
    # Windows workbench storage uses local extended paths, not network shares.
    from nixtla_scaffold.feature_sources import read_granted_snapshot
    assert sha256(read_granted_snapshot(relocated.driver_snapshots[0])).hexdigest() == saved["sha256"]
    with pytest.raises(DriverSourceError, match="changed"):
        read_granted_snapshot(spec.driver_snapshots[0])
    changed_scope = json.loads(json.dumps(manifest))
    changed_scope["driver_snapshot_artifacts"]["seats-source"]["series_ids"] = ["another"]
    with pytest.raises(DriverSourceError, match="scope differs"):
        bind_saved_driver_snapshots(spec, owned, changed_scope)


def test_saved_snapshot_cannot_escape_its_run_or_silently_regrant_missing_data(tmp_path):
    from nixtla_scaffold.feature_sources import bind_saved_driver_snapshots, persist_driver_snapshots
    _, _, spec = fixture(tmp_path)
    directory = tmp_path / "copy"
    receipts = persist_driver_snapshots(spec, directory)
    receipts["seats-source"]["artifact"] = "../source.csv"
    with pytest.raises(DriverSourceError, match="escapes"):
        bind_saved_driver_snapshots(spec, directory, {"driver_snapshot_artifacts": receipts})
    with pytest.raises(DriverSourceError, match="unavailable"):
        bind_saved_driver_snapshots(spec, directory, {"driver_snapshot_artifacts": {}})


def test_network_snapshot_is_rejected_before_path_resolution_or_file_read(monkeypatch):
    from pathlib import Path
    from nixtla_scaffold.feature_sources import read_granted_snapshot
    grant = DriverSnapshotGrant("network", r"\\unapproved-host\share\source.csv", "0" * 64, ("A",))
    monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("network path was resolved"))
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("network file was opened"))
    with pytest.raises(DriverSourceError, match="not network"):
        read_granted_snapshot(grant)


def test_query_identity_is_metadata_not_executable_code(tmp_path, monkeypatch):
    import subprocess
    history, _, spec = fixture(tmp_path)
    spec = replace(spec, regressors=(replace(spec.regressors[0], source=replace(
        spec.regressors[0].source, query_ref="query:label; not a script invocation")),))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("source metadata executed a process"))
    _, _, receipt = source_features(history, history.iloc[:0], spec, spec.regressors[0],
                                    freq="MS", origin=history.ds.max())
    assert receipt["query_ref"] == spec.regressors[0].source.query_ref
