from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold import (
    AccuracyPolicy,
    CandidateDriver,
    ChallengerSpec,
    CleaningSpec,
    ContextSource,
    EnsembleSpec,
    FeatureRecipeSpec,
    ForecastContext,
    ForecastSpec,
    ParallelSpec,
    PromotionPolicy,
    ResearchBudget,
    forecast_spec_preset,
    preset_catalog,
)
from nixtla_scaffold.cli import main
from nixtla_scaffold.schema import forecast_spec_from_dict


def _small_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
            "y": [100, 104, 107, 111, 118, 121, 126, 130, 135, 141, 148, 154],
        }
    )


def test_forecast_spec_presets_are_named_and_overridable() -> None:
    quick = forecast_spec_preset("quick", horizon=3, freq="ME")
    strict = forecast_spec_preset("strict")
    hierarchy = forecast_spec_preset("hierarchy", model_policy="baseline")
    accuracy_first = forecast_spec_preset("accuracy-first")

    assert isinstance(quick, ForecastSpec)
    assert quick.horizon == 3
    assert quick.freq == "ME"
    assert quick.model_policy == "baseline"
    assert strict.require_backtest is True
    assert strict.strict_cv_horizon is True
    assert hierarchy.hierarchy_reconciliation == "bottom_up"
    assert hierarchy.model_policy == "baseline"
    assert accuracy_first.model_policy == "standard"
    assert accuracy_first.strict_cv_horizon is True
    assert accuracy_first.require_backtest is False
    assert accuracy_first.context is not None
    assert ForecastSpec().context is None
    catalog = {row["name"]: row for row in preset_catalog()}
    assert catalog["standard"]["model_policy"] == "standard"
    assert catalog["standard"]["aliases"] == ["finance"]
    assert catalog["quick"]["verbose"] is False
    assert catalog["quick"]["accuracy_first"] is False
    assert catalog["accuracy-first"]["accuracy_first"] is True


def test_legacy_finance_and_auto_aliases_canonicalize_to_standard_light() -> None:
    assert forecast_spec_preset("finance").model_policy == "standard"
    assert ForecastSpec(model_policy="auto").model_policy == "light"
    assert forecast_spec_from_dict({"model_policy": "auto"}).model_policy == "light"


def test_nested_finn_inspired_specs_round_trip() -> None:
    spec = ForecastSpec(
        horizon=4,
        feature_recipe=FeatureRecipeSpec(
            fiscal_year_start=7,
            fourier_periods=(12,),
            lag_periods=(1, 3),
            rolling_window_periods=(3, 6),
            recipes_to_run=("finance_calendar",),
            pca=False,
            feature_selection=True,
            weekly_to_daily=False,
        ),
        cleaning=CleaningSpec(clean_missing_values=True, clean_outliers=True, negative_forecast=False, combo_cleanup_date="2025-06-30"),
        ensemble=EnsembleSpec(policies=("legacy_weighted", "top_k_average", "family_diverse_average"), max_models=2),
        parallel=ParallelSpec(processing="local_machine", inner_parallel=True, num_cores=4),
    )

    payload = spec.to_dict()
    restored = forecast_spec_from_dict(payload)

    assert payload["feature_recipe"]["lag_periods"] == [1, 3]
    assert payload["ensemble"]["policies"] == ["legacy_weighted", "top_k_average", "family_diverse_average"]
    assert restored.feature_recipe == spec.feature_recipe
    assert restored.cleaning == spec.cleaning
    assert restored.ensemble == spec.ensemble
    assert restored.parallel == spec.parallel


def test_accuracy_context_round_trips_and_custom_budget_requires_a_bound() -> None:
    context = ForecastContext(
        decision="Set capacity plan",
        audience="Finance",
        target_semantics="Monthly hosted compute cost",
        units="USD",
        grain="monthly by product",
        requested_horizon=4,
        refresh_cadence="monthly",
        source_discovery_enabled=True,
        sources=(
            ContextSource(
                source_id="usage",
                kind="kusto",
                status="available",
                query_ref="queries/usage.kql",
                query_count=2,
                row_count=48,
            ),
        ),
        candidate_drivers=(
            CandidateDriver(
                name="Hosted minutes",
                source_id="usage",
                status="eligible_for_experiment",
                future_availability="planned",
                leakage_verdict="pass",
                business_rationale="Compute minutes drive cost.",
            ),
        ),
        research_budget=ResearchBudget(profile="custom", max_iterations=3, max_source_queries=6),
        accuracy_policy=AccuracyPolicy(minimum_trust_score=75),
        promotion_policy=PromotionPolicy(minimum_primary_metric_improvement=0.03),
    )
    spec = ForecastSpec(horizon=4, context=context)

    restored = forecast_spec_from_dict(spec.to_dict())

    assert restored.context == context
    assert restored.context is not None
    assert restored.context.sources[0].query_count == 2
    assert restored.context.candidate_drivers[0].status == "eligible_for_experiment"

    try:
        ResearchBudget(profile="custom")
    except ValueError as error:
        assert "hard bound" in str(error)
    else:
        raise AssertionError("expected an unbounded custom research budget to fail")


def test_challenger_spec_round_trips_and_validates() -> None:
    challenger = ChallengerSpec(
        engine="FINN",
        enabled=True,
        on_error="skip",
        models=("ets", "arima", "ets"),
        back_test_scenarios=6,
        back_test_spacing=1,
        run_ensemble_models=True,
        timeout_seconds=900,
        extra=(("hist_end_date", "2025-12-31"),),
    )
    spec = ForecastSpec(horizon=4, challengers=(challenger,))

    payload = spec.to_dict()
    restored = forecast_spec_from_dict(payload)

    assert challenger.engine == "finn"
    assert challenger.source_id == "finn"
    assert challenger.model_name == "FINN"
    assert challenger.models == ("ets", "arima")
    assert payload["challengers"][0]["extra"] == {"hist_end_date": "2025-12-31"}
    assert restored.challengers == spec.challengers
    assert "challengers" not in ForecastSpec(horizon=4).to_dict()

    try:
        ChallengerSpec(on_error="explode")
    except ValueError as error:
        assert "on_error" in str(error)
    else:
        raise AssertionError("expected invalid on_error to raise")

    try:
        ForecastSpec(challengers=(ChallengerSpec(), ChallengerSpec(model_name="FINN2")))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("expected duplicate challenger engines to raise")


def test_forecast_cli_preset_applies_defaults_and_allows_overrides(tmp_path) -> None:
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "quick"
    _small_monthly_frame().to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--preset",
            "quick",
            "--horizon",
            "2",
            "--freq",
            "ME",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["spec"]["horizon"] == 2
    assert diagnostics["spec"]["model_policy"] == "baseline"
    assert diagnostics["spec"]["verbose"] is False
    assert "context" not in diagnostics["spec"]
    assert not (output_dir / "appendix" / "accuracy_gate.json").exists()


def test_accuracy_first_short_history_writes_directional_claim_receipts(tmp_path, capsys) -> None:
    input_path = tmp_path / "short.csv"
    output_dir = tmp_path / "accuracy_first"
    context_path = tmp_path / "forecast_context.json"
    _small_monthly_frame().head(6).to_csv(input_path, index=False)
    context = ForecastContext(
        decision="Set the next-quarter plan",
        audience="Finance",
        target_semantics="Monthly revenue",
        units="USD",
        grain="monthly",
        requested_horizon=3,
        refresh_cadence="monthly",
        source_discovery_enabled=False,
        sources=(ContextSource(source_id="target_csv", kind="csv", status="opted_out"),),
        research_budget=ResearchBudget(profile="time-boxed"),
    )
    context_path.write_text(json.dumps(context.to_dict(), indent=2), encoding="utf-8")

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--preset",
            "accuracy-first",
            "--context-file",
            str(context_path),
            "--model-policy",
            "baseline",
            "--horizon",
            "3",
            "--freq",
            "ME",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    appendix = output_dir / "appendix"
    gate = json.loads((appendix / "accuracy_gate.json").read_text(encoding="utf-8"))
    receipt = json.loads((appendix / "context_receipt.json").read_text(encoding="utf-8"))
    budget = json.loads((appendix / "research_budget.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert gate["status"] == "directional_only"
    assert gate["forecast_produced"] is True
    assert gate["planning_ready_claim_allowed"] is False
    assert receipt["context_complete"] is True
    assert receipt["source_discovery"]["status"] == "opted_out"
    assert budget["profile"] == "time-boxed"
    assert budget["remaining"]["iterations"] == 2
    assert manifest["outputs"]["accuracy_gate"] == "appendix/accuracy_gate.json"
    assert "Accuracy claim status: directional_only" in capsys.readouterr().out


def test_forecast_cli_preset_can_be_overridden_explicitly(tmp_path) -> None:
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "override"
    _small_monthly_frame().to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--preset",
            "quick",
            "--model-policy",
            "statsforecast",
            "--verbose",
            "--horizon",
            "2",
            "--freq",
            "ME",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["spec"]["model_policy"] == "statsforecast"
    assert diagnostics["spec"]["verbose"] is True


def test_forecast_cli_records_ensemble_and_feature_recipe_artifacts(tmp_path) -> None:
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "ensemble_recipe"
    _small_monthly_frame().to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--model-policy",
            "baseline",
            "--horizon",
            "2",
            "--freq",
            "ME",
            "--ensemble-policy",
            "top_k_average",
            "--ensemble-max-models",
            "2",
            "--fiscal-year-start",
            "7",
            "--lag-period",
            "1",
            "--rolling-window-period",
            "3",
            "--feature-selection",
            "--clean-outliers",
            "--no-negative-forecast",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["spec"]["ensemble"]["policies"] == ["top_k_average"]
    assert diagnostics["spec"]["ensemble"]["max_models"] == 2
    assert diagnostics["spec"]["feature_recipe"]["fiscal_year_start"] == 7
    assert diagnostics["spec"]["feature_recipe"]["lag_periods"] == [1]
    assert diagnostics["spec"]["feature_recipe"]["rolling_window_periods"] == [3]
    assert diagnostics["spec"]["cleaning"]["clean_outliers"] is True
    assert diagnostics["spec"]["cleaning"]["negative_forecast"] is False
    assert (output_dir / "appendix" / "ensemble_policy_receipts.csv").exists()
    assert (output_dir / "appendix" / "ensemble_backtest.csv").exists()
    assert (output_dir / "appendix" / "ensemble_selection.csv").exists()
    assert (output_dir / "appendix" / "ensemble_forecast.csv").exists()


def test_guide_presets_prints_catalog(capsys) -> None:
    exit_code = main(["guide", "presets"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in payload} == {"quick", "accuracy-first", "standard", "strict", "hierarchy"}
    standard = next(row for row in payload if row["name"] == "standard")
    assert standard["aliases"] == ["finance"]
