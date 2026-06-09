from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold import ChallengerSpec, CleaningSpec, EnsembleSpec, FeatureRecipeSpec, ForecastSpec, ParallelSpec, forecast_spec_preset, preset_catalog
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

    assert isinstance(quick, ForecastSpec)
    assert quick.horizon == 3
    assert quick.freq == "ME"
    assert quick.model_policy == "baseline"
    assert strict.require_backtest is True
    assert strict.strict_cv_horizon is True
    assert hierarchy.hierarchy_reconciliation == "bottom_up"
    assert hierarchy.model_policy == "baseline"
    catalog = {row["name"]: row for row in preset_catalog()}
    assert catalog["standard"]["model_policy"] == "standard"
    assert catalog["standard"]["aliases"] == ["finance"]
    assert catalog["quick"]["verbose"] is False


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
    assert {row["name"] for row in payload} == {"quick", "standard", "strict", "hierarchy"}
    standard = next(row for row in payload if row["name"] == "standard")
    assert standard["aliases"] == ["finance"]
