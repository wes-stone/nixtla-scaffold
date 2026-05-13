from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold import ForecastSpec, compare_models, run_experiment


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

