import pandas as pd
import pytest

from nixtla_scaffold.ensemble import observable_backtest_rows
from nixtla_scaffold.models import _metrics_from_cv, ModelResult, paired_comparison_result


def test_overlapping_origins_only_use_landed_outcomes():
    frame = pd.DataFrame({
        "cutoff": ["2025-01-31"] * 3,
        "ds": ["2025-02-28", "2025-03-31", "2025-04-30"],
        "y": [1, 2, 3],
    })
    rows = observable_backtest_rows(frame, pd.Timestamp("2025-02-28"))
    assert rows["y"].tolist() == [1]


def test_publication_lag_and_unknown_vintage_are_not_observable():
    frame = pd.DataFrame({
        "cutoff": ["2025-01-31"] * 3,
        "ds": ["2025-02-28"] * 3,
        "available_at": ["2025-02-28", "2025-03-10", None],
        "y": [1, 2, 3],
    })
    assert observable_backtest_rows(frame, pd.Timestamp("2025-02-28"))["y"].tolist() == [1]


def test_metric_denominator_uses_same_finite_rows_as_errors():
    frame = pd.DataFrame({
        "unique_id": ["A"] * 3,
        "ds": pd.date_range("2025-01-01", periods=3, freq="MS"),
        "cutoff": [pd.Timestamp("2024-12-01")] * 3,
        "y": [10.0, 1000.0, 20.0],
        "Model": [12.0, float("nan"), float("inf")],
    })
    metrics = _metrics_from_cv(frame).iloc[0]
    assert metrics["wape"] == pytest.approx(0.2)
    assert metrics["rmse"] == pytest.approx(2)
    assert metrics["observations"] == 1


def test_native_paired_tournament_cannot_reward_missing_hard_periods():
    from nixtla_scaffold.schema import ForecastSpec
    from nixtla_scaffold.profile import profile_dataset
    history = pd.DataFrame({"unique_id": ["x"] * 12, "ds": pd.date_range("2024-01-01", periods=12, freq="MS"), "y": range(12)})
    cv = pd.DataFrame({"unique_id": ["x"] * 3, "cutoff": pd.to_datetime(["2024-12-01"] * 3),
                       "ds": pd.date_range("2025-01-01", periods=3, freq="MS"), "y": [10., 1000., 20.],
                       "Naive": [11., 1100., 21.], "Partial": [12., float("nan"), 22.]})
    future = pd.DataFrame({"unique_id": ["x"], "ds": [pd.Timestamp("2025-04-01")], "Naive": [10.], "Partial": [20.]})
    result = ModelResult(future, _metrics_from_cv(cv), cv, "test", pd.DataFrame())
    spec = ForecastSpec(horizon=3, freq="MS", paired_validation=True, run_mode="legacy", policy_version=0)
    paired = paired_comparison_result(result, history, profile_dataset(history, spec), spec)
    assert paired.backtest_metrics["observations"].tolist() == [2, 2]
    assert paired.backtest_metrics["wape"].tolist() == pytest.approx([2 / 30, 4 / 30])
    assert paired.backtest_metrics["selection_horizon"].tolist() == [1, 1]
    assert paired.model_policy_resolution["comparison"]["cohorts"][0]["excluded_rows"] == 1
    assert paired.model_policy_resolution["comparison"]["cohorts"][0]["missing_rows_by_model"] == {"Partial": 1}
    assert any("retains 2/3 rows" in warning for warning in paired.warnings)


def test_learned_ensemble_warmup_discloses_loss_of_an_entire_origin():
    from nixtla_scaffold.schema import ForecastSpec
    from nixtla_scaffold.profile import profile_dataset
    history = pd.DataFrame({"unique_id": ["x"] * 12, "ds": pd.date_range("2024-01-01", periods=12, freq="MS"), "y": range(12)})
    cv = pd.DataFrame({
        "unique_id": ["x"] * 4, "cutoff": pd.to_datetime(["2024-08-01"] * 2 + ["2024-10-01"] * 2),
        "ds": pd.date_range("2024-09-01", periods=4, freq="MS"), "y": [8., 9., 10., 11.],
        "Naive": [7., 7., 9., 9.], "WeightedEnsemble": [float("nan"), float("nan"), 10., 10.],
    })
    future = pd.DataFrame({"unique_id": ["x"], "ds": [pd.Timestamp("2025-01-01")], "Naive": [11.], "WeightedEnsemble": [12.]})
    result = ModelResult(future, _metrics_from_cv(cv), cv, "test", pd.DataFrame())
    spec = ForecastSpec(horizon=2, freq="MS", paired_validation=True, run_mode="legacy", policy_version=0)
    paired = paired_comparison_result(result, history, profile_dataset(history, spec), spec)
    receipt = paired.model_policy_resolution["comparison"]["cohorts"][0]
    assert receipt["missing_rows_by_model"] == {"WeightedEnsemble": 2}
    assert receipt["available_windows"] == 2 and receipt["paired_windows"] == 1
    assert (paired.backtest_metrics.cv_windows == 1).all()
    assert any("1/2 origins" in warning for warning in paired.warnings)


def test_blend_frozen_reference_uses_identical_later_keys_and_rejects_missing_hard_rows():
    import numpy as np
    from nixtla_scaffold.blend import BlendRecipe, evaluate_blend

    cv = pd.DataFrame({
        "unique_id": ["A"] * 8,
        "cutoff": pd.to_datetime(["2024-01-01"] * 2 + ["2024-03-01"] * 2 + ["2024-05-01"] * 2 + ["2024-07-01"] * 2),
        "ds": pd.date_range("2024-02-01", periods=8, freq="MS"),
        "y": [10.] * 8, "Naive": [12.] * 8, "First": [11.] * 8, "Second": [9.] * 8,
    })
    future = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-10-01")],
                           "First": [11.], "Second": [9.]})
    recipe = BlendRecipe(policy="inverse_error", members=("First", "Second"))
    paired, _, scores, receipt = evaluate_blend(cv, future, recipe)
    assert receipt["warmup_rows"] == 2 and receipt["planned_later_rows"] == 6
    assert receipt["comparison_eligible"]
    assert receipt["reference_models"] == {"A": "Naive"}
    assert scores["observations"].eq(len(paired)).all()
    assert scores["cutoffs"].eq(3).all()
    changed = cv.copy()
    changed.loc[7, "Second"] = np.nan
    partial, _, _, incomplete = evaluate_blend(changed, future, recipe)
    assert len(partial) < incomplete["planned_later_rows"]
    assert not incomplete["comparison_eligible"]
    pd.testing.assert_frame_equal(paired.iloc[:4], partial.iloc[:4])


def test_native_blend_reference_is_rescored_not_borrowed_from_full_tournament():
    from nixtla_scaffold.ensemble import selected_ensemble_comparisons
    cv = pd.DataFrame({
        "unique_id": ["A"] * 4, "cutoff": pd.to_datetime(["2024-01-01"] * 2 + ["2024-03-01"] * 2),
        "ds": pd.date_range("2024-02-01", periods=4, freq="MS"), "y": [10., 10., 10., 10.],
        "Naive": [1010., 1010., 12., 12.], "WeightedEnsemble": [float("nan"), float("nan"), 11., 11.],
    })
    selection = pd.DataFrame({"unique_id": ["A"], "selected_model": ["Naive"]})
    row = selected_ensemble_comparisons(cv, selection)[0]
    assert row["metrics"][0]["rmse"] == 2
    assert row["metrics"][1]["rmse"] == 1
    assert row["planned_later_rows"] == row["paired_rows"] == 2
    assert row["warmup_rows"] == 2 and row["advisory_only"]
    assert not row["promotion_recommended"]


def test_separate_cohort_cannot_win_pareto_benchmark_or_exported_rank():
    from dataclasses import replace
    from nixtla_scaffold.forecast import run_forecast
    from nixtla_scaffold.schema import ForecastSpec, EvaluationPolicy
    from nixtla_scaffold.experiments import build_model_leaderboard
    from nixtla_scaffold.outputs import build_model_pareto_frontier, build_model_win_rates
    history = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2020-01-01", periods=20, freq="MS"), "y": range(20)})
    run = run_forecast(history, ForecastSpec(horizon=2, freq="MS", model_policy="baseline",
                                            evaluation=EvaluationPolicy(max_windows=2)))
    partial = run.backtest_metrics.iloc[[0]].copy()
    partial["model"], partial["rmse"], partial["mae"] = "Partial", 0., 0.
    partial["selection_eligible"], partial["comparison_support"] = False, "incomplete_diagnostic"
    changed = replace(run, backtest_metrics=pd.concat([run.backtest_metrics, partial], ignore_index=True))
    pareto = build_model_pareto_frontier(changed).set_index("model")
    assert pareto.loc["Partial", "pareto_status"] == "separate_cohort"
    assert not pareto.loc["Partial", "is_pareto_optimal"]
    assert "Partial" not in build_model_win_rates(changed)["model"].tolist()
    assert pd.isna(build_model_leaderboard(changed).set_index("model").loc["Partial", "leaderboard_rank"])


def _monthly_policy_fixture():
    import numpy as np
    from types import SimpleNamespace
    from nixtla_scaffold.schema import EnsembleSpec, ForecastSpec, SelectionPolicy

    history = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2025-01-01", "2025-07-31"), "y": 100.})
    origins = pd.to_datetime(["2025-01-31", "2025-03-31", "2025-05-31"])
    cv = pd.concat([
        history.loc[history.ds.gt(cutoff) & history.ds.le(cutoff + pd.Timedelta(days=31))]
        .assign(cutoff=cutoff, Naive=100 + np.r_[np.tile([-10., 10.], 15), 0.], Ridge=101.)
        for cutoff in origins
    ], ignore_index=True)
    spec = ForecastSpec(horizon=31, freq="D", season_length=7, levels=(),
                        selection_policy=SelectionPolicy(objective="monthly_landing"),
                        ensemble=EnsembleSpec(policies=("top_k_average", "family_diverse_average"), max_models=1))
    future = pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2025-08-01", periods=31),
                           "Naive": 100., "Ridge": 101.})
    return SimpleNamespace(spec=spec, history=history, observed_history=history,
                           backtest_predictions=cv, all_models=future)


@pytest.mark.parametrize("policy", ["top_k_average", "family_diverse_average"])
def test_advisory_members_honor_monthly_objective_not_daily_rmse(policy):
    from dataclasses import replace
    from nixtla_scaffold.ensemble import _select_policy_models, _build_ensemble_forecast
    from nixtla_scaffold.schema import SelectionPolicy

    run = _monthly_policy_fixture()
    kwargs = {"history": run.history, "spec": run.spec}
    assert _select_policy_models(policy, run.backtest_predictions, ["Naive", "Ridge"], 1, **kwargs) == ["Naive"]
    daily = replace(run.spec, selection_policy=SelectionPolicy(objective="daily_rmse"))
    assert _select_policy_models(policy, run.backtest_predictions, ["Naive", "Ridge"], 1,
                                 history=run.history, spec=daily) == ["Ridge"]
    deployed = _build_ensemble_forecast(run, (policy,), ["Naive", "Ridge"])
    assert deployed.base_models.eq("Naive").all()


def test_monthly_advisory_policy_winner_and_receipt_honor_objective():
    from nixtla_scaffold.ensemble import _build_ensemble_selection, _policy_receipts

    run = _monthly_policy_fixture()
    cv = run.backtest_predictions
    paths = pd.concat([
        cv[["unique_id", "cutoff", "ds", "y"]].assign(
            policy=policy, yhat=cv[model], error=cv[model] - cv.y,
        )
        for policy, model in (("top_k_average", "Naive"), ("family_diverse_average", "Ridge"))
    ], ignore_index=True)
    selected = _build_ensemble_selection(run, paths).set_index("policy")
    assert selected.loc["top_k_average", "selected_advisory_policy"]
    assert not selected.loc["family_diverse_average", "selected_advisory_policy"]
    assert selected.loc["top_k_average", "rmse"] > selected.loc["family_diverse_average", "rmse"]
    assert selected.loc["top_k_average", "landing_mape_pct"] == pytest.approx(0.)
    assert selected.objective.eq("monthly_landing").all()
    receipt = _policy_receipts(run, ("top_k_average",), len(cv), 31)
    assert receipt.objective.eq("monthly_landing").all()


def test_monthly_member_selection_never_falls_back_to_rmse_without_landed_origins():
    from nixtla_scaffold.ensemble import _select_policy_models

    run = _monthly_policy_fixture()
    partial = run.backtest_predictions.groupby("cutoff").head(20)
    assert _select_policy_models("top_k_average", partial, ["Naive", "Ridge"], 1,
                                 spec=run.spec, history=run.history) == []
    with pytest.raises(ValueError, match="observed source history"):
        _select_policy_models("top_k_average", partial, ["Naive", "Ridge"], 1, spec=run.spec)


def test_monthly_members_ignore_unlanded_and_unreleased_outcomes():
    from nixtla_scaffold.ensemble import _build_ensemble_backtest

    run = _monthly_policy_fixture()
    first = run.backtest_predictions.loc[run.backtest_predictions.cutoff.eq(pd.Timestamp("2025-01-31"))]
    overlapping = first.assign(cutoff=pd.Timestamp("2025-02-07"), ds=first.ds + pd.Timedelta(days=7))
    run.backtest_predictions = pd.concat([first, overlapping, run.backtest_predictions.iloc[31:]], ignore_index=True)
    before = _build_ensemble_backtest(run, ("top_k_average",), ["Naive", "Ridge"])
    assert pd.Timestamp("2025-02-07") not in set(before.cutoff)
    assert before.base_models.eq("Naive").all()
    changed = run.backtest_predictions.copy()
    changed.loc[changed.ds.gt(pd.Timestamp("2025-03-31")), "y"] = 10000.
    run.history = run.history.copy()
    run.history.loc[run.history.ds.gt(pd.Timestamp("2025-03-31")), "y"] = 10000.
    run.observed_history = run.history
    run.backtest_predictions = changed
    after = _build_ensemble_backtest(run, ("top_k_average",), ["Naive", "Ridge"])
    assert after.loc[after.cutoff.eq(pd.Timestamp("2025-03-31")), "base_models"].eq("Naive").all()
    changed["available_at"] = pd.Timestamp("2026-01-01")
    run.backtest_predictions = changed
    assert _build_ensemble_backtest(run, ("top_k_average",), ["Naive", "Ridge"]).empty
