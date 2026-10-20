import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import EvaluationPolicy, ForecastSpec, run_forecast
from nixtla_scaffold.evaluation import cv_parameters, plan_origins, training_slice
from nixtla_scaffold.models import ModelResult, _metrics_from_cv, paired_comparison_result, select_champions
from nixtla_scaffold.profile import profile_dataset


def history(n=38):
    return pd.DataFrame({"unique_id": ["A"] * n,
                         "ds": pd.date_range("2023-07-01", periods=n, freq="MS"),
                         "y": np.arange(n, dtype=float) + 100})


def test_monthly_schedule_uses_nine_full_origins_without_reading_target_values():
    frame = history()
    spec = ForecastSpec(horizon=6, freq="MS", season_length=12)
    plan = plan_origins(frame, spec, 12)
    assert len(plan) == 9
    assert [origin.training_rows for origin in plan] == list(range(24, 33))
    assert plan[0].cutoff == pd.Timestamp("2025-06-01")
    assert plan[-1].cutoff == pd.Timestamp("2026-02-01")
    assert all(len(origin.targets) == 6 and min(origin.targets) > origin.cutoff for origin in plan)
    assert plan_origins(frame.assign(y=np.nan), spec, 12) == plan


@pytest.mark.parametrize("settings", [{"max_windows": 0}, {"step_size": 0}, {"step_size": True},
                                     {"training_window": 2}, {"minimum_training_observations": 1}])
def test_invalid_evaluation_contract_is_rejected(settings):
    with pytest.raises(ValueError):
        EvaluationPolicy(**settings)


def test_evaluation_policy_survives_serialization():
    from nixtla_scaffold.schema import forecast_spec_from_dict
    spec = ForecastSpec(evaluation=EvaluationPolicy(training_window=24, max_windows=9, step_size=2))
    assert forecast_spec_from_dict(spec.to_dict()) == spec
    legacy = forecast_spec_from_dict({})
    assert legacy.evaluation.strategy == "legacy"
    assert cv_parameters(38, 6, 12, spec.evaluation, strict=True) == (6, 5, 2)


def test_rolling_rule_is_used_at_each_origin_and_final_refit(monkeypatch):
    from nixtla_scaffold import forecast as module
    original = module.forecast_with_policy
    fits = []

    def record(frame, profile, spec):
        fits.append((frame.copy(), spec))
        return original(frame, profile, spec)

    monkeypatch.setattr(module, "forecast_with_policy", record)
    frame = history()
    spec = ForecastSpec(horizon=3, freq="MS", season_length=1, model_policy="baseline",
                        weighted_ensemble=False, evaluation=EvaluationPolicy(training_window=10, max_windows=3))
    run = run_forecast(frame, spec)
    assert len(fits) == 4
    assert all(len(training) == 10 and inner.evaluation.strategy == "none" for training, inner in fits)
    assert fits[0][0]["ds"].max() == frame["ds"].max()
    for training, _ in fits[1:]:
        cutoff = training["ds"].max()
        prediction = run.backtest_predictions.loc[run.backtest_predictions.cutoff.eq(cutoff)]
        assert len(prediction) == 3
        assert prediction["ds"].min() > cutoff
        assert prediction["RandomWalkWithDrift"].iloc[0] == pytest.approx(training["y"].iloc[-1] + 1)
    assert run.model_policy_resolution["evaluation"]["policy"]["training_window"] == 10


def test_future_target_cannot_leak_through_interpolation_or_error_scaling():
    frame = history(20)
    frame.loc[12, "y"] = np.nan
    spec = ForecastSpec(horizon=3, freq="MS", season_length=1, model_policy="baseline",
                        weighted_ensemble=False, fill_method="interpolate",
                        evaluation=EvaluationPolicy(max_windows=3, step_size=2))
    first = run_forecast(frame, spec)
    changed = frame.copy()
    changed.loc[13:, "y"] += 1_000_000
    second = run_forecast(changed, spec)
    cutoff = first.backtest_predictions.cutoff.min()
    cols = ["Naive", "RandomWalkWithDrift", "HistoricAverage"]
    pd.testing.assert_frame_equal(first.backtest_predictions.loc[first.backtest_predictions.cutoff.eq(cutoff), cols],
                                  second.backtest_predictions.loc[second.backtest_predictions.cutoff.eq(cutoff), cols])
    from nixtla_scaffold.outputs import build_backtest_long
    pd.testing.assert_series_equal(build_backtest_long(first)["rmsse_scale"],
                                   build_backtest_long(second)["rmsse_scale"])
    assert first.observed_history.loc[first.observed_history.ds.eq(frame.loc[12, "ds"]), "y"].isna().all()


def test_imputed_targets_are_not_scored_as_observed_actuals():
    frame = history(20)
    frame.loc[18, "y"] = np.nan
    spec = ForecastSpec(horizon=3, freq="MS", season_length=1, model_policy="baseline",
                        weighted_ensemble=False, evaluation=EvaluationPolicy(max_windows=2))
    run = run_forecast(frame, spec)
    assert run.backtest_predictions.loc[run.backtest_predictions.ds.eq(frame.loc[18, "ds"]), "y"].isna().all()
    assert not run.backtest_metrics.selection_eligible.any()
    assert "backtest not available" in run.model_selection.iloc[0].selection_reason


def test_scale_reference_does_not_compress_missing_calendar_positions():
    from nixtla_scaffold.external_scoring import _cutoff_scale
    frame = history(5).assign(y=[0.0, 1.0, np.nan, 10.0, 11.0])
    scales = _cutoff_scale(frame, frame.ds.max(), season_length=1)
    assert scales["mase_scale"] == 1.0
    assert scales["rmsse_scale"] == 1.0


def test_missing_hard_period_disqualifies_candidate_instead_of_shrinking_reference():
    frame = history(12)
    spec = ForecastSpec(horizon=3, freq="MS")
    cv = pd.DataFrame({"unique_id": ["A"] * 3, "cutoff": pd.to_datetime(["2024-03-01"] * 3),
                       "ds": pd.date_range("2024-04-01", periods=3, freq="MS"), "y": [10., 1000., 20.],
                       "Naive": [11., 1100., 21.], "Partial": [10., np.nan, 20.]})
    future = pd.DataFrame({"unique_id": ["A"] * 3, "ds": pd.date_range("2024-07-01", periods=3, freq="MS"),
                           "Naive": [10.] * 3, "Partial": [20.] * 3})
    result = ModelResult(future, _metrics_from_cv(cv), cv, "test", pd.DataFrame())
    paired = paired_comparison_result(result, frame, profile_dataset(frame, spec), spec)
    scores = paired.backtest_metrics.set_index("model")
    assert scores.loc["Naive", "observations"] == 3
    assert scores.loc["Naive", "wape"] == pytest.approx(102 / 1030)
    assert not scores.loc["Partial", "selection_eligible"]
    assert select_champions(future, paired.backtest_metrics).iloc[0].selected_model == "Naive"


def test_ensemble_warmup_preserves_the_component_tournament():
    spec = ForecastSpec(horizon=6, freq="MS", model_policy="baseline")
    run = run_forecast(history(), spec)
    metrics = run.backtest_metrics.set_index("model")
    assert metrics.loc["Naive", "cv_windows"] == 9
    assert metrics.loc["Naive", "observations"] == 54
    assert metrics.loc["Naive", "cv_disjoint_windows"] == 2
    assert metrics.loc["WeightedEnsemble", "cv_windows"] < 9
    assert metrics.loc["WeightedEnsemble", "comparison_support"] == "ensemble_later_advisory"


def test_training_window_is_applied_independently_within_each_series():
    data = pd.concat([history(12), history(8).assign(unique_id="B")], ignore_index=True)
    sliced = training_slice(data, pd.Timestamp("2024-01-01"), 4)
    assert sliced.groupby("unique_id").size().to_dict() == {"A": 4, "B": 4}
    assert sliced.ds.max() == pd.Timestamp("2024-01-01")


def test_late_and_unknown_publication_values_are_withheld_not_dropped():
    frame = history(8)
    frame["available_at"] = frame["ds"]
    frame.loc[3, "available_at"] = pd.Timestamp("2030-01-01")
    frame.loc[4, "available_at"] = pd.NaT
    selected = training_slice(frame, frame.ds.iloc[5], None)
    assert len(selected) == 6
    assert selected.loc[[3, 4], "y"].isna().all()
    assert selected.attrs["withheld_at_origin"] == 2


def test_late_revisions_cannot_alter_earlier_fits():
    frame = history(20)
    frame["available_at"] = frame["ds"]
    frame.loc[11, "available_at"] = pd.Timestamp("2030-01-01")
    spec = ForecastSpec(horizon=3, freq="MS", season_length=1, model_policy="baseline",
                        weighted_ensemble=False, evaluation=EvaluationPolicy(max_windows=3, step_size=2))
    first = run_forecast(frame, spec)
    changed = frame.copy()
    changed.loc[11, "y"] += 1_000_000
    second = run_forecast(changed, spec)
    cutoff = first.backtest_predictions.cutoff.min()
    cols = ["Naive", "RandomWalkWithDrift", "HistoricAverage"]
    pd.testing.assert_frame_equal(first.backtest_predictions.loc[first.backtest_predictions.cutoff.eq(cutoff), cols],
                                  second.backtest_predictions.loc[second.backtest_predictions.cutoff.eq(cutoff), cols])
    from nixtla_scaffold.outputs import build_backtest_long
    pd.testing.assert_series_equal(build_backtest_long(first)["rmsse_scale"], build_backtest_long(second)["rmsse_scale"])


@pytest.mark.parametrize("model", ["AutoETS", "Ridge", "AutoMFLES"])
def test_actual_native_and_ml_fits_obey_rolling_window(monkeypatch, model):
    from nixtla_scaffold import forecast as module
    original = module.forecast_with_policy
    sizes = []

    def record(frame, profile, spec):
        sizes.append(len(frame))
        return original(frame, profile, spec)

    monkeypatch.setattr(module, "forecast_with_policy", record)
    spec = ForecastSpec(horizon=3, freq="MS", season_length=4, levels=(),
                        model_allowlist=(model,), weighted_ensemble=False,
                        evaluation=EvaluationPolicy(training_window=36, max_windows=2))
    run = run_forecast(history(48), spec)
    assert sizes == [36, 36, 36]
    assert model in run.all_models
    assert run.backtest_predictions[model].notna().all()
    assert run.backtest_metrics.set_index("model").loc[model, "observations"] == 6


@pytest.mark.parametrize("remaining", [0, -1])
def test_frozen_plan_rejects_deleted_or_duplicated_hard_rows(remaining):
    from dataclasses import replace
    spec = ForecastSpec(horizon=3, freq="MS", model_policy="baseline", weighted_ensemble=False)
    run = run_forecast(history(), spec)
    result = ModelResult(run.all_models, run.backtest_metrics, run.backtest_predictions, run.engine, run.model_weights,
                         model_policy_resolution=run.model_policy_resolution)
    with pytest.raises(ValueError, match="frozen origin plan"):
        paired_comparison_result(replace(result, backtest_predictions=result.backtest_predictions.iloc[:remaining]),
                                 run.observed_history, run.profile, spec)
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison_result(replace(result, backtest_predictions=pd.concat([
            result.backtest_predictions, result.backtest_predictions.iloc[:1]
        ])), run.observed_history, run.profile, spec)


def test_failed_origin_retains_expected_rows_and_removes_candidate_eligibility(monkeypatch):
    from nixtla_scaffold import forecast as module
    original = module.forecast_with_policy
    calls = 0

    def fail_one(frame, profile, spec):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional origin failure")
        return original(frame, profile, spec)

    monkeypatch.setattr(module, "forecast_with_policy", fail_one)
    run = run_forecast(history(), ForecastSpec(
        horizon=3, freq="MS", model_policy="baseline", weighted_ensemble=False,
        evaluation=EvaluationPolicy(max_windows=3)))
    assert len(run.backtest_predictions) == 9
    assert run.model_policy_resolution["evaluation"]["executions"][0]["status"] == "failed"
    assert not run.backtest_metrics.selection_eligible.any()
    assert (run.backtest_metrics.selection_horizon == 0).all()


def test_overlapping_windows_do_not_pass_disjoint_horizon_gate():
    from nixtla_scaffold.outputs import build_forecast_long, build_trust_summary
    run = run_forecast(history(), ForecastSpec(
        horizon=6, freq="MS", model_policy="baseline", weighted_ensemble=False,
        evaluation=EvaluationPolicy(max_windows=5)))
    trust = build_trust_summary(run).iloc[0]
    assert trust.cv_windows == 5
    assert trust.cv_disjoint_windows == 1
    assert not trust.full_horizon_claim_allowed
    assert trust.horizon_gate_result == "warning_insufficient_disjoint_windows"
    assert not build_forecast_long(run).planning_eligible.any()


def test_shorter_horizon_diagnostic_records_real_spacing_and_no_full_horizon_claim():
    run = run_forecast(history(8), ForecastSpec(
        horizon=6, freq="MS", season_length=1, strict_cv_horizon=False,
        model_policy="baseline", weighted_ensemble=False, evaluation=EvaluationPolicy(step_size=None)))
    metrics = run.backtest_metrics
    assert set(metrics.cv_step_size) == {2}
    assert set(metrics.selection_horizon) == {2}
    assert set(metrics.cv_disjoint_windows) == {0}


def test_observed_history_export_preserves_missing_actuals(tmp_path):
    frame = history(20)
    frame.loc[18, "y"] = np.nan
    run = run_forecast(frame, ForecastSpec(
        horizon=3, freq="MS", season_length=1, model_policy="baseline", weighted_ensemble=False))
    run.to_directory(tmp_path)
    observed = pd.read_csv(tmp_path / "appendix" / "observed_history.csv")
    assert observed.loc[18, "y"] != observed.loc[18, "y"]
    assert pd.notna(run.history.loc[18, "y"])


def test_research_keeps_undefined_series_in_paired_metric_cohort(tmp_path):
    from nixtla_scaffold.research import _paired_tuning_evidence
    frames = []
    for uid, scale in [("001", 1.), ("NA", np.nan)]:
        frames.append(pd.DataFrame({
            "unique_id": [uid] * 3, "cutoff": ["2024-03-01"] * 3,
            "ds": pd.date_range("2024-04-01", periods=3, freq="MS"),
            "horizon_step": [1, 2, 3], "y_actual": [1., 2., 3.], "yhat": [0., 1., 2.],
            "mase_scale": [scale] * 3, "rmsse_scale": [scale] * 3, "is_selected_model": [True] * 3}))
    rows = pd.concat(frames)
    for name in ["baseline", "candidate"]:
        directory = tmp_path / name / "appendix"
        directory.mkdir(parents=True)
        rows.to_csv(directory / "backtest_long.csv", index=False)
    evidence = _paired_tuning_evidence(tmp_path / "baseline", tmp_path / "candidate", primary_metric="avg_rmsse")
    assert evidence["window_count"] == 1
    assert not evidence["metrics_complete"]
    assert not evidence["exact_match"]
    assert evidence["stable_window_fraction"] == 0
    assert evidence["frame"]["baseline"].isna().all()


def test_research_rejects_duplicate_selected_predictions(tmp_path):
    from nixtla_scaffold.research import _selected_backtest_rows
    path = tmp_path / "appendix"
    path.mkdir()
    pd.DataFrame({"unique_id": ["001"] * 2, "cutoff": ["2024-03-01"] * 2,
                  "ds": ["2024-04-01"] * 2, "horizon_step": [1] * 2,
                  "is_selected_model": [True] * 2}).to_csv(path / "backtest_long.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        _selected_backtest_rows(tmp_path)


def test_confirmation_training_applies_publication_cutoff():
    from nixtla_scaffold.research import _training_before_window
    frame = history(20)
    frame["available_at"] = frame["ds"]
    frame.loc[11, "available_at"] = pd.Timestamp("2030-01-01")
    training = _training_before_window(frame, frame.tail(3))
    assert len(training) == 17
    assert training.loc[training.ds.eq(frame.loc[11, "ds"]), "y"].isna().all()
    assert training.ds.max() < frame.tail(3).ds.min()


def test_disabling_outer_cv_preserves_automfles_internal_tuning():
    from dataclasses import replace
    from nixtla_scaffold.models import _statsforecast_model_factories
    frame = history(48)
    spec = ForecastSpec(horizon=3, freq="MS", season_length=4, model_policy="statsforecast",
                        evaluation=EvaluationPolicy(max_windows=5, step_size=2))
    profile = profile_dataset(frame, spec)
    expected = dict(_statsforecast_model_factories(frame, profile, spec))["AutoMFLES"]()
    inner = replace(spec, evaluation=replace(spec.evaluation, strategy="none"))
    actual = dict(_statsforecast_model_factories(frame, profile, inner))["AutoMFLES"]()
    assert actual.test_size == expected.test_size
    assert actual.n_windows == expected.n_windows == 2
    assert actual.step_size == expected.step_size == 2
