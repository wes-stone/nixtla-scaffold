from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.finance import finance_scorecard, aggregate_intervals
from nixtla_scaffold.schema import ForecastSpec, SelectionPolicy, EvaluationPolicy, IntervalCalibrationSpec, forecast_spec_from_dict
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.models import select_champions


def protocol_fixture():
    history = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2025-07-01", "2026-10-11"), "y": 100.})
    origins = pd.date_range("2026-07-12", "2026-10-04", freq="7D")
    cv = pd.concat([history.loc[history.ds.gt(c) & history.ds.le(c + pd.Timedelta(days=81))]
                    .assign(cutoff=c, Model=110., Blend=100.) for c in origins], ignore_index=True)
    return history, cv


def test_exact_finn_window_contract_and_numeric_oracle():
    history, cv = protocol_fixture()
    scores, windows = finance_scorecard(history, cv, ["Model", "Blend"], horizon=81)
    assert set(scores.landing_windows) == {20}
    assert set(scores.next28_windows) == {10}
    monthly = windows.loc[windows.model.eq("Model") & windows.kind.eq("month")]
    assert monthly.offset.value_counts().to_dict() == {0: 12, 1: 8}
    expected_errors = 10 * monthly.forecast_days / monthly.period_days
    result = scores.set_index("model").loc["Model"]
    assert result.landing_mape_pct == pytest.approx(expected_errors.mean())
    assert result.landing_bias_pct == pytest.approx(expected_errors.mean())
    assert result.landing_worst_pct == pytest.approx(expected_errors.max())
    assert result.next28_mape_pct == pytest.approx(10)
    assert result.daily_wmape_pct == pytest.approx(10)
    assert result.daily_rmse == pytest.approx(10)
    assert scores.set_index("model").loc["Blend", "landing_mape_pct"] == 0


def test_clip_before_date_average_and_immutable_inputs():
    history, cv = protocol_fixture()
    cv["Model"] = np.where(cv.cutoff.eq(cv.cutoff.min()), -100., 100.)
    before = cv.copy(deep=True)
    scores, _ = finance_scorecard(history, cv, ["Model"], horizon=81, clip_negative=True)
    points = cv.assign(Model=cv.Model.clip(lower=0)).groupby("ds")[["Model", "y"]].mean()
    expected = 100 * (points.Model - points.y).abs().sum() / points.y.abs().sum()
    assert scores.iloc[0].daily_wmape_pct == pytest.approx(expected)
    pd.testing.assert_frame_equal(before, cv)


def test_clipping_preserves_release_metadata_and_actuals():
    from nixtla_scaffold.finance import clip_model_result
    from nixtla_scaffold.models import ModelResult
    frame = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-01-02")],
                          "available_at": [pd.Timestamp("2026-01-03")], "y": [-10.], "Model": [-5.]})
    result = ModelResult(forecast=frame, backtest_predictions=frame, backtest_metrics=pd.DataFrame(),
                         engine="fixture", model_weights=pd.DataFrame())
    clipped = clip_model_result(result)
    assert clipped.forecast.Model.item() == 0 and clipped.forecast.y.item() == -10
    pd.testing.assert_series_equal(clipped.forecast.available_at, frame.available_at)


def test_missing_candidate_window_cannot_improve_score():
    history, cv = protocol_fixture()
    cv.loc[cv.index[0], "Model"] = np.nan
    scores, windows = finance_scorecard(history, cv, ["Model", "Blend"], horizon=81)
    bad = scores.set_index("model").loc["Model"]
    assert np.isnan(bad.landing_mape_pct) and bad.landing_windows == 20
    assert scores.set_index("model").loc["Blend", "landing_mape_pct"] == 0
    assert windows.status.str.contains("incomplete").any()
    with pytest.raises(ValueError, match="unique"):
        finance_scorecard(history, pd.concat([cv, cv.iloc[:1]]), ["Model"], horizon=81)


def test_zero_total_is_undefined_not_an_omitted_easy_window():
    history, cv = protocol_fixture()
    history["y"] = 0.
    cv["y"] = 0.
    scores, _ = finance_scorecard(history, cv, ["Model"], horizon=81)
    assert np.isnan(scores.iloc[0].landing_mape_pct)
    assert np.isnan(scores.iloc[0].daily_wmape_pct)


def test_nonfinite_totals_fail_instead_of_persisting_infinite_scores():
    history, cv = protocol_fixture()
    history["y"], cv["y"] = 1e308, 1e308
    with pytest.raises(ValueError, match="overflows"):
        finance_scorecard(history, cv, ["Model"], horizon=81)


def test_closed_origin_period_is_not_a_forecast_and_late_actuals_are_ineligible():
    history, cv = protocol_fixture()
    origin = pd.Timestamp("2026-07-31")
    cv = history.loc[history.ds.gt(origin) & history.ds.le(origin + pd.Timedelta(days=81))].assign(
        cutoff=origin, Model=100.)
    _, windows = finance_scorecard(history, cv, ["Model"], horizon=81)
    assert windows.loc[windows.kind.eq("month"), "offset"].tolist() == [1]
    history["available_at"] = history.ds + pd.Timedelta(days=1)
    origin = pd.Timestamp("2026-08-01")
    cv = cv.loc[cv.ds.gt(origin)].assign(cutoff=origin)
    scores, windows = finance_scorecard(history, cv, ["Model"], horizon=81)
    assert "actuals_not_available_at_origin" in windows.status.to_list()
    assert np.isnan(scores.iloc[0].landing_mape_pct)


def test_monthly_selection_honors_primary_and_ties():
    future = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-11-01")], "A": [100.], "B": [100.]})
    metrics = pd.DataFrame({"unique_id": ["A", "A"], "model": ["A", "B"],
                            "landing_mape_pct": [2., 1.], "next28_mape_pct": [1., 50.],
                            "daily_wmape_pct": [1., 50.], "rmse": [1., 100.], "mae": [1., 100.],
                            "abs_bias": [0., 0.]})
    assert select_champions(future, metrics, policy=SelectionPolicy(objective="monthly_landing")).selected_model.item() == "B"
    metrics["landing_mape_pct"] = 1.
    assert select_champions(future, metrics, policy=SelectionPolicy(objective="monthly_landing")).selected_model.item() == "A"


def test_native_scorecard_outputs_roundtrip_and_legacy_default(tmp_path):
    history, _ = protocol_fixture()
    spec = ForecastSpec(horizon=60, freq="D", model_policy="baseline", weighted_ensemble=False,
                        levels=(80, 95), evaluation=EvaluationPolicy(max_windows=3, step_size=7),
                        selection_policy=SelectionPolicy(objective="monthly_landing", clip_negative=True),
                        interval_calibration=IntervalCalibrationSpec(aggregate_totals=True))
    assert forecast_spec_from_dict(spec.to_dict()) == spec
    assert forecast_spec_from_dict({}).selection_policy.objective == "daily_rmse"
    run = run_forecast(history, spec)
    assert "landing_mape_pct" in run.model_selection
    assert run.model_policy_resolution["finance_scorecard"]["scores"]
    assert all(row["status"] != "experimental" for row in run.model_policy_resolution["aggregate_intervals"])
    run.to_directory(tmp_path / "run")
    assert (tmp_path / "run" / "appendix" / "finance_scorecard.csv").is_file()
    assert (tmp_path / "run" / "appendix" / "aggregate_intervals.csv").is_file()
    from nixtla_scaffold.research import _run_evidence, _select_primary_metric, _paired_tuning_evidence
    assert _select_primary_metric(_run_evidence(run, tmp_path / "run")) == "avg_landing_mape_pct"
    paired = _paired_tuning_evidence(tmp_path / "run", tmp_path / "run", primary_metric="avg_landing_mape_pct")
    assert paired["exact_match"]
    from nixtla_scaffold.finance import enrich_frozen_run
    replay = enrich_frozen_run(run)
    pd.testing.assert_frame_equal(run.forecast, replay.forecast)
    assert replay.model_selection.landing_mape_pct.tolist() == run.model_selection.landing_mape_pct.tolist()
    from nixtla_scaffold.driver_research import assess_driver_ablation
    from nixtla_scaffold.schema import PromotionPolicy
    ablation = assess_driver_ablation(tmp_path / "run", tmp_path / "run", "Naive",
                                     primary="avg_landing_mape_pct", secondary="avg_next28_mape_pct",
                                     policy=PromotionPolicy())
    assert ablation["control"]["avg_landing_mape_pct"] == 0
    assert ablation["feature"]["avg_next28_mape_pct"] == 0
    from nixtla_scaffold.workbench import dispatch, _evaluate
    from nixtla_scaffold.workbench_store import WorkbenchStore
    from nixtla_scaffold.blend import BlendRecipe
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dispatch(workspace, "initialize", {})
    imported = dispatch(workspace, "import_run", {"path": str(tmp_path / "run")})
    receipt = _evaluate(WorkbenchStore(workspace), imported["id"],
                        BlendRecipe(members=("Naive", "HistoricAverage"), policy="equal"))
    assert receipt["receipt"]["finance_scorecard"]["cohort"] == "exact_paired_blend_replay"
    assert all(row["landing_mape_pct"] == 0 for row in receipt["receipt"]["finance_scorecard"]["scores"])


def test_aggregate_oracle_disjoint_complete_total_errors():
    history = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2000-01-01", "2024-01-10"), "y": 10.})
    origins = [pd.Timestamp(year=year, month=1, day=10) for year in range(2000, 2024)]
    cv = pd.concat([history.loc[history.ds.gt(c) & history.ds.le(c + pd.Timedelta(days=21))]
                    .assign(cutoff=c, Model=11.) for c in origins], ignore_index=True)
    future = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2024-01-11", periods=81), "Model": 11.})
    rows = aggregate_intervals(history, cv, future, ["Model"], horizon=81, levels=(80, 95))
    month = next(r for r in rows if r["kind"] == "month" and r["offset"] == 0 and r["level"] == 95)
    assert month["status"] == "experimental"
    assert month["point_total"] == 331
    assert month["lower"] == 310 and month["upper"] == 352
    assert month["calibration_windows"] == 24 and month["order_statistic"] == 24
    assert all(r["lower"] is None for r in rows if r["kind"] == "quarter")
    incomplete = aggregate_intervals(history, cv, future.iloc[:2], ["Model"], horizon=81, levels=(95,))
    assert all(r["lower"] is None for r in incomplete)


def test_finance_flags_match_all_cli_routes(monkeypatch):
    from test_runtime_contracts import _cli_spec
    for command in ("forecast", "compare-models", "experiment", "optimize", "ingest"):
        spec = _cli_spec(monkeypatch, command, "--selection-objective", "monthly_landing",
                         "--clip-negative-points", "--aggregate-intervals")
        assert spec.selection_policy.objective == "monthly_landing"
        assert spec.selection_policy.clip_negative and spec.interval_calibration.aggregate_totals


def test_finance_confirmation_uses_same_closed_windows():
    from nixtla_scaffold.research import _confirmation_metrics
    history, cv = protocol_fixture()
    origin = cv.cutoff.min()
    scored = cv.loc[cv.cutoff.eq(origin), ["unique_id", "ds", "y", "Model"]].rename(
        columns={"y": "y_actual", "Model": "yhat"})
    spec = ForecastSpec(horizon=81, selection_policy=SelectionPolicy(objective="monthly_landing"))
    result = _confirmation_metrics(scored, history.loc[history.ds.le(origin)], season_length=7,
                                    variant="candidate", window="confirmation", spec=spec)
    expected, _ = finance_scorecard(history, cv.loc[cv.cutoff.eq(origin)], ["Model"], horizon=81)
    assert result[0]["landing_mape_pct"] == pytest.approx(expected.iloc[0].landing_mape_pct)
    assert result[0]["next28_mape_pct"] == pytest.approx(expected.iloc[0].next28_mape_pct)


def test_recorded_intent_and_research_recipes_do_not_broaden_models():
    from nixtla_scaffold.experiments import _variant_catalog
    spec = ForecastSpec(model_policy="mlforecast", model_allowlist=("Ridge",))
    for name in ("rolling_features", "calendar_dummies", "direct_horizons"):
        recipe = _variant_catalog()[name]
        changed = recipe.build_spec(spec)
        assert changed.model_policy == spec.model_policy and changed.model_allowlist == spec.model_allowlist
        assert not recipe.applicable(None, replace(spec, model_policy="baseline"))[0]
