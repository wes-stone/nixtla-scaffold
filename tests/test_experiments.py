from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nixtla_scaffold import (
    ChallengerSpec,
    ExperimentHypothesis,
    ForecastContext,
    ForecastSpec,
    PromotionPolicy,
    ResearchBudget,
    compare_models,
    run_experiment,
    run_optimizer,
)
from nixtla_scaffold.experiments import (
    _apply_candidate_set_compatibility,
    _detect_candidate_drivers,
    _lagged_correlation_evidence,
    _select_challenger_configuration,
)
from nixtla_scaffold.research import (
    _assess_tuning_candidate,
    _build_chronological_split,
    _paired_tuning_evidence,
    _review_iteration,
    _run_confirmation,
)
from nixtla_scaffold.models import _with_resolved_candidate_identity


def _demo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
            "y": [100, 103, 107, 111, 116, 120, 124, 129, 133, 138, 144, 149],
        }
    )


def _driver_candidate_frame() -> pd.DataFrame:
    minutes = [100, 105, 111, 118, 126, 137, 149, 162, 176, 191, 207, 224, 242, 261]
    rolling = pd.Series(minutes).rolling(3, min_periods=1).mean().shift(1).bfill()
    return pd.DataFrame(
        {
            "unique_id": ["Product minutes"] * len(minutes),
            "usage_month": pd.date_range("2025-01-31", periods=len(minutes), freq="ME"),
            "minutes": minutes,
            "rolling_minutes": rolling,
            "active_flag": [True] * len(minutes),
        }
    )


def test_compare_models_writes_leaderboard_and_preserves_normal_run_outputs(tmp_path) -> None:
    output_dir = tmp_path / "compare"

    leaderboard = compare_models(_demo_frame(), ForecastSpec(horizon=2, model_policy="baseline"), output_dir=output_dir)

    assert not leaderboard.empty
    assert {"unique_id", "leaderboard_rank", "model", "is_selected_model"}.issubset(leaderboard.columns)
    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "compare_models_leaderboard.csv").exists()
    manifest = json.loads((output_dir / "compare_models_manifest.json").read_text(encoding="utf-8"))
    assert "advisory leaderboard" in manifest["purpose"].lower()


def test_run_experiment_records_skips_and_autoresearch_next_step(tmp_path) -> None:
    output_dir = tmp_path / "experiment"

    result = run_experiment(
        _demo_frame(),
        ForecastSpec(horizon=2, model_policy="baseline"),
        output_dir=output_dir,
        variants=("baseline", "events", "known_future_regressors"),
        max_variants=3,
    )

    summary = result.summary.set_index("variant")
    assert summary.loc["baseline", "status"] == "success"
    assert summary.loc["events", "status"] == "not_applicable"
    assert summary.loc["known_future_regressors", "status"] == "not_applicable"
    assert (output_dir / "variants" / "baseline" / "forecast.csv").exists()
    assert (output_dir / "experiment_summary.csv").exists()
    assert (output_dir / "experiment_manifest.json").exists()
    assert (output_dir / "experiment_llm_context.json").exists()
    recommendation = (output_dir / "experiment_recommendation.md").read_text(encoding="utf-8")
    assert "Autoresearch next iteration" in recommendation
    assert "Keep rule" in recommendation
    next_iteration = result.llm_context["recommendation"]["autoresearch_next_iteration"]
    assert next_iteration["executor"] == "uv run nixtla-scaffold experiment"
    assert "avg_rmse primary" in next_iteration["metric"]
    assert "avg_mae" in next_iteration["metric"]
    assert "avg_wape" not in next_iteration["metric"]
    assert "avg_wape" not in recommendation


def test_explicit_experiment_adds_matched_control_and_candidate_fingerprint(tmp_path) -> None:
    result = run_experiment(
        _demo_frame(),
        ForecastSpec(horizon=2, model_policy="baseline"),
        output_dir=tmp_path / "matched",
        variants=("log1p_transform",),
        max_variants=1,
    )

    rows = result.summary.set_index("variant")
    assert list(result.summary["variant"]) == ["log1p_transform", "baseline"]
    assert result.manifest["matched_control"] == {
        "enabled": True,
        "auto_added": True,
        "variant": "baseline",
    }
    assert len(rows.loc["baseline", "resolved_candidate_fingerprint"]) == 64
    assert (
        rows.loc["log1p_transform", "resolved_candidate_fingerprint"]
        == rows.loc["baseline", "resolved_candidate_fingerprint"]
    )
    assert bool(rows.loc["log1p_transform", "candidate_set_compatible"]) is True


def test_unexpected_candidate_fingerprint_drift_blocks_ranking() -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "status": "success",
                "resolved_candidate_fingerprint": "a" * 64,
                "promotion_evidence_eligible": True,
                "evidence_class": "native_chronological_backtest",
            },
            {
                "variant": "rolling_features",
                "status": "success",
                "resolved_candidate_fingerprint": "b" * 64,
                "promotion_evidence_eligible": True,
                "evidence_class": "native_chronological_backtest",
            },
        ]
    )

    assessed = _apply_candidate_set_compatibility(summary).set_index("variant")

    assert bool(assessed.loc["rolling_features", "candidate_set_changed"]) is True
    assert bool(assessed.loc["rolling_features", "candidate_set_compatible"]) is False
    assert bool(assessed.loc["rolling_features", "promotion_evidence_eligible"]) is False
    assert assessed.loc["rolling_features", "evidence_class"] == "candidate_set_incompatible"


def test_optional_runtime_version_changes_candidate_fingerprint(monkeypatch) -> None:
    versions = {"value": "1.0"}
    monkeypatch.setattr(
        "nixtla_scaffold.models._installed_package_version",
        lambda package: versions["value"] if package == "scikit-learn" else None,
    )
    resolution = {
        "model_policy": "baseline",
        "model_allowlist": [],
        "families": [
            {
                "family": "baseline",
                "contributed_models": ["Naive"],
            }
        ],
    }

    first = _with_resolved_candidate_identity(
        resolution,
        resolved_candidates=["Naive"],
    )
    versions["value"] = "2.0"
    second = _with_resolved_candidate_identity(
        resolution,
        resolved_candidates=["Naive"],
    )

    assert (
        first["resolved_candidate_fingerprint"]
        != second["resolved_candidate_fingerprint"]
    )


def test_run_optimizer_writes_iteration_receipts(tmp_path) -> None:
    output_dir = tmp_path / "optimizer"

    result = run_optimizer(
        _demo_frame(),
        ForecastSpec(horizon=2, model_policy="baseline"),
        output_dir=output_dir,
        variants=("baseline", "events", "rolling_features"),
        max_iterations=1,
        max_variants=2,
    )

    assert result.manifest["schema_version"] == "nixtla_scaffold.optimizer.v2"
    assert result.manifest["executed_iterations"] == 1
    assert (output_dir / "iteration_001" / "experiment_manifest.json").exists()
    assert (output_dir / "iteration_001" / "hypothesis.json").exists()
    assert (output_dir / "iteration_001" / "reviews.json").exists()
    assert (output_dir / "research_plan.json").exists()
    assert (output_dir / "chronological_split.csv").exists()
    assert (output_dir / "iteration_ledger.csv").exists()
    assert (output_dir / "knowledge_ledger.jsonl").exists()
    assert (output_dir / "promotion_decision.json").exists()
    assert (output_dir / "stop_receipt.json").exists()
    assert (output_dir / "iteration_summary.csv").exists()
    assert (output_dir / "iteration_decisions.jsonl").exists()
    assert (output_dir / "next_iteration_questions.md").exists()
    assert result.decisions
    assert result.decisions[0]["research_decision"] == "failed"
    assert result.manifest["promotion_recommended"] is False
    assert result.manifest["stopped_reason"] == "budget_exhausted"
    stop_receipt = json.loads((output_dir / "stop_receipt.json").read_text(encoding="utf-8"))
    assert stop_receipt["remaining_feasible_hypotheses"] == 1
    assert not (output_dir / "iteration_002").exists()


def test_chronological_split_keeps_confirmation_rows_out_of_tuning() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 20,
            "ds": pd.date_range("2024-01-31", periods=20, freq="ME"),
            "y": range(20),
        }
    )

    split = _build_chronological_split(frame, horizon=2, confirmation_cutoffs=2)

    assert split.all_series_eligible is True
    assert len(split.confirmation_windows) == 2
    assert all(len(window) == 2 for _, window in split.confirmation_windows)
    confirmation_dates = pd.concat([window for _, window in split.confirmation_windows])["ds"]
    assert split.tuning_data["ds"].max() < confirmation_dates.min()
    assert set(split.receipt["role"]) == {"tuning", "confirmation"}


def test_untouched_confirmation_scores_only_selected_candidate(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 16,
            "ds": pd.date_range("2024-01-31", periods=16, freq="ME"),
            "y": [float(value) for value in range(100, 116)],
        }
    )
    split = _build_chronological_split(frame, horizon=2, confirmation_cutoffs=1)
    actual_lookup = frame.set_index("ds")["y"].to_dict()
    training_ends: list[pd.Timestamp] = []

    class FakeRun:
        def __init__(self, forecast: pd.DataFrame):
            self.forecast = forecast
            self.profile = SimpleNamespace(season_length=1)

        def to_directory(self, output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path

    def fake_run_forecast(model_input, run_spec):
        observed = model_input[model_input["y"].notna()]
        training_end = pd.to_datetime(observed["ds"]).max()
        training_ends.append(training_end)
        future_dates = [
            date
            for date in frame["ds"]
            if date > training_end
        ][: run_spec.horizon]
        error = 1.0 if run_spec.transform.target == "log1p" else 5.0
        return FakeRun(
            pd.DataFrame(
                {
                    "unique_id": ["Revenue"] * len(future_dates),
                    "ds": future_dates,
                    "yhat": [actual_lookup[date] + error for date in future_dates],
                }
            )
        )

    monkeypatch.setattr("nixtla_scaffold.research.run_forecast", fake_run_forecast)
    monkeypatch.setattr(
        "nixtla_scaffold.research._new_gate_failure",
        lambda *_: True,
    )
    hypothesis = ExperimentHypothesis(
        hypothesis_id="h001-log1p",
        statement="Log1p should improve multiplicative growth.",
        changed_dimension="target transform",
        expected_mechanism="stabilize growth",
        predicted_effect="lower error",
        required_data=("target history",),
        falsifying_outcome="no improvement",
        leakage_risk="low",
        horizon_risk="full horizon required",
        estimated_cost={"compute_units": 1},
        variant="log1p_transform",
    )

    receipt = _run_confirmation(
        frame,
        ForecastSpec(horizon=2, model_policy="baseline"),
        split,
        {"hypothesis": hypothesis, "assessment": {}},
        primary_metric="avg_rmsse",
        secondary_metric="avg_mase",
        policy=PromotionPolicy(
            minimum_primary_metric_improvement=0.1,
            maximum_secondary_metric_regression=0.1,
            require_no_new_gate_failures=False,
        ),
        output_dir=tmp_path / "confirmation",
    )

    confirmation_start = pd.concat([window for _, window in split.confirmation_windows])["ds"].min()
    assert all(training_end < confirmation_start for training_end in training_ends)
    assert receipt["passed"] is True
    assert receipt["exact_coverage"] == 1.0
    assert receipt["extra_rows"] == 0
    assert receipt["no_new_gate_failures"] is False
    assert receipt["gate_policy_pass"] is True


def test_confirmation_coverage_uses_exact_candidate_baseline_pairs(
    monkeypatch,
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 16,
            "ds": pd.date_range("2024-01-31", periods=16, freq="ME"),
            "y": [float(value) for value in range(100, 116)],
        }
    )
    split = _build_chronological_split(frame, horizon=2, confirmation_cutoffs=1)
    actual_lookup = frame.set_index("ds")["y"].to_dict()

    class FakeRun:
        def __init__(self, forecast):
            self.forecast = forecast
            self.profile = SimpleNamespace(season_length=1)

        def to_directory(self, output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path

    def fake_run_forecast(model_input, run_spec):
        training_end = pd.to_datetime(
            model_input.loc[model_input["y"].notna(), "ds"]
        ).max()
        future_dates = [
            date for date in frame["ds"] if date > training_end
        ][: run_spec.horizon]
        error = 1.0 if run_spec.transform.target == "log1p" else 5.0
        if run_spec.transform.target == "log1p":
            future_dates = future_dates[:1]
        return FakeRun(
            pd.DataFrame(
                {
                    "unique_id": ["Revenue"] * len(future_dates),
                    "ds": future_dates,
                    "yhat": [
                        actual_lookup[date] + error for date in future_dates
                    ],
                }
            )
        )

    monkeypatch.setattr("nixtla_scaffold.research.run_forecast", fake_run_forecast)
    hypothesis = ExperimentHypothesis(
        hypothesis_id="h001-log1p",
        statement="Log1p should improve multiplicative growth.",
        changed_dimension="target transform",
        expected_mechanism="stabilize growth",
        predicted_effect="lower error",
        required_data=("target history",),
        falsifying_outcome="no improvement",
        leakage_risk="low",
        horizon_risk="full horizon required",
        estimated_cost={"compute_units": 1},
        variant="log1p_transform",
    )

    receipt = _run_confirmation(
        frame,
        ForecastSpec(horizon=2, model_policy="baseline"),
        split,
        {"hypothesis": hypothesis, "assessment": {}},
        primary_metric="avg_rmsse",
        secondary_metric="avg_mase",
        policy=PromotionPolicy(
            exact_cutoff_coverage=0.75,
            minimum_primary_metric_improvement=0.0,
            maximum_secondary_metric_regression=1.0,
        ),
        output_dir=tmp_path / "partial_confirmation",
    )

    assert receipt["matched_rows"] == 1
    assert receipt["expected_rows"] == 2
    assert receipt["exact_coverage"] == 0.5
    assert receipt["coverage_pass"] is False
    assert receipt["passed"] is False


def test_run_experiment_detects_candidate_drivers_for_autoresearch_context(tmp_path) -> None:
    output_dir = tmp_path / "experiment_context"

    result = run_experiment(
        _driver_candidate_frame(),
        ForecastSpec(horizon=2, time_col="usage_month", target_col="minutes", model_policy="baseline"),
        output_dir=output_dir,
        variants=("baseline",),
        max_variants=1,
    )

    candidates = result.llm_context["candidate_drivers"]
    assert candidates
    assert candidates[0]["value_col"] == "rolling_minutes"
    assert candidates[0]["regressor_json"]["availability"] == "historical_only"
    assert candidates[0]["regressor_json"]["mode"] == "model_candidate"
    assert "same_period_correlation_abs" in candidates[0]
    assert "best_lag" in candidates[0]
    assert "relationship_timing" in candidates[0]
    assert "active_flag" not in {candidate["value_col"] for candidate in candidates}
    assert result.llm_context["manifest"]["base_spec"]["regressors"] == []
    assert result.llm_context["autoresearch_hypotheses"][0]["command_seed"].count("--train-known-future-regressors") == 1
    assert "avg_mae" in result.llm_context["autoresearch_hypotheses"][0]["metric"]
    assert "avg_wape" not in result.llm_context["autoresearch_hypotheses"][0]["metric"]

    recommendation = (output_dir / "experiment_recommendation.md").read_text(encoding="utf-8")
    assert "Context to collect from human" in recommendation
    assert "Automatically detected candidate drivers" in recommendation
    assert "rolling_minutes" in recommendation
    assert '"availability":"historical_only"' in recommendation
    assert "Detected candidates are not trained automatically" in recommendation


def test_candidate_driver_lag_metadata_uses_positive_lag_for_driver_leading_target() -> None:
    signal = [0, 10, 2, 8, 4, 6, 1, 9, 3, 7, 5, 11]
    y = [None, None, *signal[:-2]]
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * len(signal),
            "ds": pd.date_range("2025-01-31", periods=len(signal), freq="ME"),
            "y": y,
            "usage_signal": signal,
        }
    )

    candidates = _detect_candidate_drivers(frame, ForecastSpec(horizon=2), limit=3)
    candidate = next(item for item in candidates if item["value_col"] == "usage_signal")

    assert candidate["best_lag"] == 2
    assert candidate["relationship_timing"] == "driver_leads_target"
    assert "leads `y` by 2 period" in candidate["lag_interpretation"]


def test_lagged_driver_screen_does_not_shift_across_panel_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6,
            "ds": list(pd.date_range("2025-01-31", periods=6, freq="ME")) * 2,
            "y": [1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100],
            "usage_signal": [10, 10, 10, 10, 10, 10, -10, -10, -10, -10, -10, -10],
        }
    )

    evidence = _lagged_correlation_evidence(frame, "usage_signal", max_lag=3)

    assert evidence["relationship_timing"] == "insufficient_lag_evidence"
    assert evidence["best_lag_paired_observations"] == 0


def test_run_experiment_caps_requested_variants(tmp_path) -> None:
    result = run_experiment(
        _demo_frame(),
        ForecastSpec(horizon=2, model_policy="baseline"),
        output_dir=tmp_path / "experiment_cap",
        variants=("baseline", "all_models", "rolling_features"),
        max_variants=1,
    )

    capped = result.summary[result.summary["status"].eq("max_variants_cap")]
    assert set(capped["variant"]) == {"all_models", "rolling_features"}


def test_commands_reject_enabled_challenger_when_finn_is_not_scheduled(tmp_path) -> None:
    spec = ForecastSpec(
        horizon=2,
        model_policy="baseline",
        challengers=(ChallengerSpec(engine="finn", enabled=True),),
    )

    with pytest.raises(ValueError, match="refusing to persist inactive challenger settings"):
        compare_models(_demo_frame(), spec)
    with pytest.raises(ValueError, match="no `finn` experiment variant"):
        run_experiment(
            _demo_frame(),
            spec,
            output_dir=tmp_path / "experiment",
            variants=("baseline",),
        )
    with pytest.raises(ValueError, match="no `finn` hypothesis"):
        run_optimizer(
            _demo_frame(),
            spec,
            output_dir=tmp_path / "optimizer",
            variants=("baseline",),
        )


def test_finn_variant_executes_and_partial_evidence_cannot_rank(monkeypatch, tmp_path) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_run_challengers(run_dir, challengers):
        calls.append(tuple(challengers))
        native = pd.read_csv(run_dir / "appendix" / "backtest_long.csv")
        external = native[native["is_selected_model"].map(bool)].copy()
        external["model"] = "finnts_mock"
        external["source_id"] = "finn"
        external["scenario_name"] = "Base"
        external["scoring_status"] = "scored"
        external["yhat"] = external["y_actual"]
        alternate = external.copy()
        alternate["scenario_name"] = "Bull"
        alternate["yhat"] = alternate["y_actual"] + 999.0
        external = pd.concat([external, alternate], ignore_index=True)
        finn_dir = run_dir / "finn"
        finn_dir.mkdir(parents=True, exist_ok=True)
        external_path = finn_dir / "external_backtest_long.csv"
        external.to_csv(external_path, index=False)
        leaderboard = pd.DataFrame(
            [
                {
                    "unique_id": "Revenue",
                    "model": "finnts_mock",
                    "lane": "challenger",
                    "source_id": "finn",
                    "scenario_name": "Base",
                    "comparable": False,
                    "cutoff_coverage": 0.5,
                    "rmse": 1.0,
                    "mae": 1.0,
                    "mase": 0.1,
                    "rmsse": 0.1,
                    "bias": 0.0,
                }
            ]
        )
        appendix = run_dir / "appendix"
        appendix.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(appendix / "challenger_leaderboard.csv", index=False)
        return {
            "challengers": [
                {
                    "status": "completed",
                    "source_id": "finn",
                    "outputs": {
                        "external_backtest_long": str(external_path),
                    },
                }
            ],
            "completed": 1,
            "skipped": 0,
        }

    monkeypatch.setattr("nixtla_scaffold.challengers.run_challengers", fake_run_challengers)
    spec = ForecastSpec(
        horizon=2,
        model_policy="baseline",
        challengers=(ChallengerSpec(engine="finn", enabled=True),),
    )

    result = run_experiment(
        _demo_frame(),
        spec,
        output_dir=tmp_path / "finn_experiment",
        variants=("baseline", "finn"),
        max_variants=1,
    )

    assert calls
    row = result.summary.iloc[0]
    assert row["status"] == "success"
    assert row["evidence_class"] == "directional_external_evidence"
    assert bool(row["promotion_evidence_eligible"]) is False
    assert pd.isna(row["advisory_rank"])
    selected_external = Path(row["backtest_path"])
    assert selected_external.exists()
    selected_external_frame = pd.read_csv(selected_external)
    assert set(selected_external_frame["scenario_name"]) == {"Base"}
    paired = _paired_tuning_evidence(
        result.output_dir / "variants" / "finn",
        result.output_dir / "variants" / "finn",
        primary_metric="avg_rmsse",
        candidate_backtest_path=selected_external,
    )
    assert paired["exact_coverage"] == 1.0
    assert paired["frame"]["candidate"].eq(0.0).all()


def test_paired_tuning_rejects_and_excludes_candidate_only_rows(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    (baseline_dir / "appendix").mkdir(parents=True)
    candidate_dir.mkdir()
    baseline = pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "cutoff": ["2025-01-31"],
            "ds": ["2025-02-28"],
            "horizon_step": [1],
            "y_actual": [100.0],
            "yhat": [90.0],
            "mase_scale": [10.0],
            "rmsse_scale": [10.0],
            "is_selected_model": [True],
        }
    )
    baseline.to_csv(baseline_dir / "appendix" / "backtest_long.csv", index=False)
    candidate = pd.concat(
        [
            baseline.assign(yhat=100.0),
            baseline.assign(
                ds="2025-03-31",
                horizon_step=2,
                y_actual=110.0,
                yhat=60.0,
            ),
        ],
        ignore_index=True,
    ).drop(columns=["is_selected_model"])
    candidate_path = candidate_dir / "external_selected_backtest_long.csv"
    candidate.to_csv(candidate_path, index=False)

    paired = _paired_tuning_evidence(
        baseline_dir,
        candidate_dir,
        primary_metric="avg_rmsse",
        candidate_backtest_path=candidate_path,
        require_explicit_candidate_backtest=True,
    )

    assert paired["exact_coverage"] == 1.0
    assert paired["extra_rows"] == 1
    assert paired["exact_match"] is False
    assert paired["frame"]["candidate"].eq(0.0).all()


def test_finn_pairing_requires_an_explicit_external_backtest(tmp_path) -> None:
    run_dir = tmp_path / "run"
    appendix = run_dir / "appendix"
    appendix.mkdir(parents=True)
    pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "cutoff": ["2025-01-31"],
            "ds": ["2025-02-28"],
            "horizon_step": [1],
            "y_actual": [100.0],
            "yhat": [99.0],
            "mase_scale": [1.0],
            "rmsse_scale": [1.0],
            "is_selected_model": [True],
        }
    ).to_csv(appendix / "backtest_long.csv", index=False)

    paired = _paired_tuning_evidence(
        run_dir,
        run_dir,
        primary_metric="avg_rmsse",
        require_explicit_candidate_backtest=True,
    )

    assert paired["exact_coverage"] == 0.0
    assert paired["reason"] == "paired evidence unavailable"


def test_promotion_policy_can_allow_new_accuracy_gate_failures() -> None:
    hypothesis = SimpleNamespace(
        hypothesis_id="h001-log1p",
        variant="log1p_transform",
    )
    baseline = {
        "status": "success",
        "avg_rmsse": 1.0,
        "avg_mase": 1.0,
        "gate_status": "planning_ready",
    }
    candidate = {
        "status": "success",
        "avg_rmsse": 0.8,
        "avg_mase": 0.8,
        "gate_status": "blocked",
        "promotion_evidence_eligible": True,
    }
    paired = {
        "exact_coverage": 1.0,
        "window_count": 2,
        "stable_window_fraction": 1.0,
    }

    assessment = _assess_tuning_candidate(
        baseline,
        candidate,
        paired,
        hypothesis=hypothesis,
        primary_metric="avg_rmsse",
        secondary_metric="avg_mase",
        policy=PromotionPolicy(require_no_new_gate_failures=False),
    )

    assert assessment["decision"] == "candidate_for_confirmation"
    assert "candidate introduced a stronger accuracy-gate failure" not in assessment["blockers"]


def test_forecast_reviewer_blocks_on_tuning_rejection_reason() -> None:
    blocker = "candidate tied or underperformed; retain the simpler baseline"

    reviews = _review_iteration(
        ForecastSpec(horizon=2, model_policy="baseline"),
        SimpleNamespace(all_series_eligible=True),
        SimpleNamespace(variant="log1p_transform"),
        {"status": "success"},
        {"exact_coverage": 1.0},
        {"decision": "reject", "blockers": [blocker]},
    )

    review = next(item for item in reviews if item["reviewer"] == "forecast_skeptic")
    assert review["verdict"] == "block"
    assert review["blocking_gaps"] == [blocker]


def test_required_finn_runs_before_patience_can_stop_research(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_run_challengers(run_dir, challengers):
        calls.append("finn")
        leaderboard = pd.DataFrame(
            [
                {
                    "unique_id": "Revenue",
                    "model": "finnts_mock",
                    "lane": "challenger",
                    "source_id": "finn",
                    "comparable": False,
                    "cutoff_coverage": 0.0,
                    "rmse": 10.0,
                    "mae": 10.0,
                    "mase": 10.0,
                    "rmsse": 10.0,
                    "bias": 0.0,
                }
            ]
        )
        appendix = run_dir / "appendix"
        appendix.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(appendix / "challenger_leaderboard.csv", index=False)
        return {
            "challengers": [{"status": "completed", "source_id": "finn"}],
            "completed": 1,
            "skipped": 0,
        }

    monkeypatch.setattr(
        "nixtla_scaffold.challengers.run_challengers",
        fake_run_challengers,
    )
    spec = ForecastSpec(
        horizon=2,
        model_policy="baseline",
        challengers=(ChallengerSpec(engine="finn", enabled=True),),
        context=ForecastContext(
            research_budget=ResearchBudget(
                profile="custom",
                max_iterations=2,
                max_compute_units=3,
            )
        ),
    )

    result = run_optimizer(
        _demo_frame(),
        spec,
        output_dir=tmp_path / "required_finn",
        variants=("rolling_features", "finn"),
        max_iterations=2,
        patience=1,
    )

    assert calls == ["finn"]
    assert "finn" in set(result.iteration_summary["variant"])


def test_required_finn_rejects_an_insufficient_compute_budget(tmp_path) -> None:
    spec = ForecastSpec(
        horizon=2,
        model_policy="baseline",
        challengers=(ChallengerSpec(engine="finn", enabled=True),),
        context=ForecastContext(
            research_budget=ResearchBudget(
                profile="custom",
                max_iterations=2,
                max_compute_units=2,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="cannot fund the 3-unit baseline and required challenger hypotheses",
    ):
        run_optimizer(
            _demo_frame(),
            spec,
            output_dir=tmp_path / "underfunded_finn",
            variants=("rolling_features", "finn"),
            max_iterations=2,
        )


def test_external_research_selects_one_panel_configuration() -> None:
    external = pd.DataFrame(
        [
            {
                "unique_id": "A",
                "source_id": "finn",
                "scenario_name": "Base",
                "model": "ets",
                "rmsse": 1.0,
                "rmse": 1.0,
                "comparable": True,
                "cutoff_coverage": 1.0,
            },
            {
                "unique_id": "B",
                "source_id": "finn",
                "scenario_name": "Base",
                "model": "ets",
                "rmsse": 4.0,
                "rmse": 4.0,
                "comparable": True,
                "cutoff_coverage": 1.0,
            },
            {
                "unique_id": "A",
                "source_id": "finn",
                "scenario_name": "Bull",
                "model": "arima",
                "rmsse": 5.0,
                "rmse": 5.0,
                "comparable": True,
                "cutoff_coverage": 1.0,
            },
            {
                "unique_id": "B",
                "source_id": "finn",
                "scenario_name": "Bull",
                "model": "arima",
                "rmsse": 1.0,
                "rmse": 1.0,
                "comparable": True,
                "cutoff_coverage": 1.0,
            },
        ]
    )

    selected, receipt = _select_challenger_configuration(
        external,
        expected_series={"A", "B"},
    )

    assert receipt["full_series_coverage"] is True
    assert receipt["scenario_name"] == "Base"
    assert receipt["model"] == "ets"
    assert set(selected["scenario_name"]) == {"Base"}
    assert set(selected["model"]) == {"ets"}
