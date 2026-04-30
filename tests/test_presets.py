from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold import ForecastSpec, forecast_spec_preset, preset_catalog
from nixtla_scaffold.cli import main


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
    assert catalog["finance"]["model_policy"] == "auto"
    assert catalog["quick"]["verbose"] is False


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


def test_guide_presets_prints_catalog(capsys) -> None:
    exit_code = main(["guide", "presets"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in payload} == {"quick", "finance", "strict", "hierarchy"}
