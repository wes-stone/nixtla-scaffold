from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nixtla_scaffold import (
    DriverEvent,
    ForecastSpec,
    TransformSpec,
    add_fiscal_calendar,
    label_anomalies,
    normalize_by_factor,
    run_forecast,
)
from nixtla_scaffold.cli import main
from nixtla_scaffold.transformations import apply_event_adjustments


def test_add_fiscal_calendar_uses_february_year_start() -> None:
    frame = pd.DataFrame({"ds": ["2025-01-31", "2025-02-28"], "y": [10, 20]})

    out = add_fiscal_calendar(frame, fiscal_year_start_month=2)

    assert out["fiscal_year"].tolist() == [2025, 2026]
    assert out["fiscal_month"].tolist() == [12, 1]
    assert out["fiscal_quarter"].tolist() == [4, 1]


def test_normalize_by_factor_creates_adjusted_target() -> None:
    frame = pd.DataFrame({"y": [110, 120], "price_factor": [1.1, 1.2]})

    out = normalize_by_factor(frame, factor_col="price_factor")

    assert out["y_adjusted"].round(2).tolist() == [100.0, 100.0]


def test_run_forecast_applies_log1p_transform_and_reports_original_scale(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 18,
            "ds": pd.date_range("2024-01-01", periods=18, freq="MS"),
            "y": [100 + i * 8 for i in range(18)],
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(horizon=3, freq="MS", model_policy="baseline", transform=TransformSpec(target="log1p")),
    )
    output_dir = run.to_directory(tmp_path / "log1p_run")

    assert run.history["y"].round(6).tolist() == df["y"].astype(float).round(6).tolist()
    assert run.forecast["yhat"].min() > 100
    assert run.backtest_predictions["y"].min() > 100
    assert not run.transformation_audit.empty
    assert run.transformation_audit["target_transform"].unique().tolist() == ["log1p"]
    assert (run.transformation_audit["y_modeled"].round(8) == pd.Series(np.log1p(df["y"])).round(8)).all()
    assert any("inverse-transformed" in warning for warning in run.warnings)
    receipts = {item["id"]: item for item in run.best_practice_receipts()}
    assert receipts["target_transform_audited"]["status"] == "passed"

    assert (output_dir / "audit" / "target_transform_audit.csv").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec"]["transform"]["target"] == "log1p"
    assert "Target Transform Audit" in pd.ExcelFile(output_dir / "forecast.xlsx").sheet_names
    assert "Target transformation audit" in (output_dir / "report.html").read_text(encoding="utf-8")
    assert 'read_csv("target_transform_audit.csv")' in (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Target transform: `log1p`" in (output_dir / "diagnostics.md").read_text(encoding="utf-8")


def test_run_forecast_applies_factor_normalization_for_modeling() -> None:
    raw_values = [110.0, 121.0, 132.0, 143.0, 154.0, 165.0, 176.0, 187.0]
    factors = [1.10, 1.10, 1.20, 1.20, 1.30, 1.30, 1.40, 1.40]
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * len(raw_values),
            "ds": pd.date_range("2025-01-01", periods=len(raw_values), freq="MS"),
            "y": raw_values,
            "price_factor": factors,
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=2,
            freq="MS",
            model_policy="baseline",
            transform=TransformSpec(normalization_factor_col="price_factor", normalization_label="FY26 pricing"),
        ),
    )

    expected_adjusted = pd.Series(raw_values) / pd.Series(factors)
    assert run.history["y"].round(6).tolist() == expected_adjusted.round(6).tolist()
    assert run.transformation_audit["y_adjusted"].round(6).tolist() == expected_adjusted.round(6).tolist()
    assert run.transformation_audit["output_scale"].unique().tolist() == ["normalized_units"]
    assert any("target normalization applied" in warning for warning in run.warnings)


def test_cli_target_transform_and_normalization_flags_are_audited(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "run"
    raw_values = [110.0, 121.0, 132.0, 143.0, 154.0, 165.0, 176.0, 187.0, 198.0, 209.0, 220.0, 231.0]
    factors = [1.10, 1.10, 1.20, 1.20, 1.30, 1.30, 1.40, 1.40, 1.50, 1.50, 1.60, 1.60]
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * len(raw_values),
            "ds": pd.date_range("2025-01-01", periods=len(raw_values), freq="MS"),
            "y": raw_values,
            "price_factor": factors,
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "2",
            "--freq",
            "MS",
            "--model-policy",
            "baseline",
            "--target-transform",
            "log1p",
            "--normalization-factor-col",
            "price_factor",
            "--normalization-label",
            "FY26 pricing",
            "--unit-label",
            "$",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    audit = pd.read_csv(output_dir / "audit" / "target_transform_audit.csv")
    assert audit["target_transform"].unique().tolist() == ["log1p"]
    assert audit["output_scale"].unique().tolist() == ["normalized_units"]
    assert audit["normalization_factor_col"].unique().tolist() == ["price_factor"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec"]["transform"]["target"] == "log1p"
    assert manifest["spec"]["transform"]["normalization_factor_col"] == "price_factor"
    assert manifest["spec"]["unit_label"] == "$"


def test_label_anomalies_flags_large_outlier() -> None:
    frame = pd.DataFrame({"unique_id": ["A"] * 7, "y": [10, 11, 10, 12, 11, 10, 100]})

    out = label_anomalies(frame, threshold=3.5)

    assert bool(out["anomaly_label"].iloc[-1])
    assert not bool(out["anomaly_label"].iloc[0])


def test_apply_event_adjustments_adds_scenario_columns() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2026-01-31", "2026-02-28"]),
            "yhat": [100.0, 100.0],
        }
    )
    event = DriverEvent(name="Launch", start="2026-02-28", effect="multiplicative", magnitude=0.2)

    out = apply_event_adjustments(forecast, [event])

    assert out["yhat_scenario"].tolist() == [100.0, 120.0]
    assert out["event_names"].tolist() == ["", "Launch"]


def test_run_forecast_applies_driver_event_scenario() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=2,
            model_policy="baseline",
            events=(DriverEvent(name="Launch", start="2025-10-31", magnitude=0.1),),
        ),
    )

    assert "yhat_scenario" in run.forecast.columns
    assert run.forecast.loc[run.forecast["event_names"] == "Launch", "yhat_scenario"].iloc[0] > run.forecast.loc[
        run.forecast["event_names"] == "Launch", "yhat"
    ].iloc[0]
    assert any("driver/event" in warning for warning in run.warnings)


def test_cli_event_json_is_applied(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "run"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
            "--event",
            '{"name":"Launch","start":"2025-10-31","magnitude":0.1}',
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    forecast = pd.read_csv(output_dir / "forecast.csv")
    assert "yhat_scenario" in forecast.columns


def test_cli_event_null_affected_ids_is_treated_as_all_series(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "run"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
            "--event",
            '{"name":"Launch","start":"2025-10-31","magnitude":0.1,"affected_unique_ids":null}',
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
