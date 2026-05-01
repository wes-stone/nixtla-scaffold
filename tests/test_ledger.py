from __future__ import annotations

import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from nixtla_scaffold import (
    ForecastSpec,
    compare_versions,
    ingest_actuals,
    ingest_adjustments,
    init_ledger,
    lock_version,
    register_run,
    run_forecast,
)
from nixtla_scaffold.cli import main


def _history(multiplier: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["ARR"] * 14,
            "ds": pd.date_range("2025-01-31", periods=14, freq="ME"),
            "y": [round((100 + i * 4) * multiplier, 2) for i in range(14)],
        }
    )


def _write_run(tmp_path, name: str, multiplier: float = 1.0):
    run = run_forecast(_history(multiplier), ForecastSpec(horizon=3, freq="ME", model_policy="baseline", verbose=False))
    run_dir = tmp_path / name
    run.to_directory(run_dir)
    return run, run_dir


def test_ledger_register_lock_actuals_adjustments_compare_and_export(tmp_path) -> None:
    ledger = tmp_path / "forecast_ledger"
    march_run, march_dir = _write_run(tmp_path, "march_run", multiplier=1.0)
    april_run, april_dir = _write_run(tmp_path, "april_run", multiplier=1.08)

    initialized = init_ledger(ledger).to_dict()
    assert initialized["schema_version"] == "1.0"
    assert (ledger / "ledger.sqlite").exists()

    march = register_run(ledger, march_dir, forecast_key="Actions ARR", version_label="March refresh").to_dict()
    april = register_run(ledger, april_dir, forecast_key="Actions ARR", version_label="April refresh").to_dict()
    assert march["forecast_version_id"] != april["forecast_version_id"]
    assert json.loads((march_dir / "ledger_context.json").read_text(encoding="utf-8"))["forecast_key"] == "Actions ARR"

    lock = lock_version(
        ledger,
        version_id=march["forecast_version_id"],
        lock_label="March lock",
        audience="leadership",
        planning_cycle="FY26",
        reason="Submitted to leadership",
    ).to_dict()
    assert lock["status"] == "locked"

    actuals = march_run.forecast[["unique_id", "ds", "yhat"]].head(2).rename(columns={"yhat": "y"}).copy()
    actuals["y"] = actuals["y"] * 1.05
    actuals_path = tmp_path / "actuals.csv"
    actuals.to_csv(actuals_path, index=False)
    actual_result = ingest_actuals(
        ledger,
        actuals_path,
        forecast_key="Actions ARR",
        source_kind="kusto",
        source_id="unit-test-refresh",
        revision_label="actuals refresh 1",
        known_as_of="2026-05-01",
    ).to_dict()
    assert actual_result["rows"] == 2

    adjustments = pd.DataFrame(
        {
            "unique_id": ["ARR", "ARR"],
            "start_ds": [actuals["ds"].iloc[0], actuals["ds"].iloc[1]],
            "end_ds": [actuals["ds"].iloc[0], actuals["ds"].iloc[1]],
            "adjustment_type": ["replacement", "regime_change"],
            "adjusted_y": [float(actuals["y"].iloc[0]) * 1.1, None],
            "metric_mapping": [None, "seat_to_token"],
            "conversion_factor": [None, 1.25],
            "expected_elasticity": [None, 0.8],
            "confidence": [None, 0.7],
            "reason": ["Late-arriving usage was understated", "Seat pricing moves to token pricing"],
            "known_as_of": ["2026-05-01", "2026-05-01"],
            "approval_status": ["approved", "approved"],
        }
    )
    adjustments_path = tmp_path / "adjustments.csv"
    adjustments.to_csv(adjustments_path, index=False)
    adjustment_result = ingest_adjustments(ledger, adjustments_path, forecast_key="Actions ARR").to_dict()
    assert adjustment_result["rows"] == 2
    assert adjustment_result["regime_changes"] == 1

    comparison = compare_versions(
        ledger,
        forecast_key="Actions ARR",
        against_lock="March lock",
        latest_version_id=april["forecast_version_id"],
        call_up_pct=0.02,
        call_down_pct=0.02,
    ).to_dict()
    assert comparison["rows"] == 3

    exports = ledger / "exports"
    versions = pd.read_csv(exports / "forecast_versions.csv")
    locks = pd.read_csv(exports / "official_forecast_locks.csv")
    forecast_actuals = pd.read_csv(exports / "forecast_actuals.csv")
    performance = pd.read_csv(exports / "forecast_performance.csv")
    corrected_actuals = pd.read_csv(exports / "corrected_actuals.csv")
    regime_changes = pd.read_csv(exports / "regime_changes.csv")
    deltas = pd.read_csv(exports / "forecast_version_deltas.csv")

    assert set(versions["version_label"]) == {"March refresh", "April refresh"}
    assert locks["lock_label"].tolist() == ["March lock"]
    assert not forecast_actuals.empty
    assert not performance.empty
    assert corrected_actuals["applied_adjustment_ids"].astype(str).str.len().max() > 0
    assert regime_changes["metric_mapping"].tolist() == ["seat_to_token"]
    assert deltas["comparison_id"].notna().all()
    assert deltas["forecast_key"].eq("Actions ARR").all()
    assert deltas["base_version_id"].eq(march["forecast_version_id"]).all()
    assert deltas["comparison_version_id"].eq(april["forecast_version_id"]).all()
    assert deltas["base_lock_label"].eq("March lock").all()
    assert set(deltas["status_label"]) <= {"call_up", "call_down", "on_track", "watch", "threshold_not_configured"}


def test_ledger_cli_registers_forecast_and_official_lock(tmp_path) -> None:
    data = _history()
    input_path = tmp_path / "history.csv"
    data.to_csv(input_path, index=False)
    output_dir = tmp_path / "run"
    ledger = tmp_path / "ledger"

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "2",
            "--freq",
            "ME",
            "--model-policy",
            "baseline",
            "--output",
            str(output_dir),
            "--ledger",
            str(ledger),
            "--forecast-key",
            "Actions ARR",
            "--version-label",
            "March refresh",
            "--lock-official",
            "--lock-label",
            "March lock",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "ledger_context.json").exists()
    assert (ledger / "exports" / "forecast_snapshot.csv").exists()
    locks = pd.read_csv(ledger / "exports" / "official_forecast_locks.csv")
    assert locks["lock_label"].tolist() == ["March lock"]
    report_html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Forecast ledger" in report_html
    assert "Static ledger preview embedded from" in report_html
    assert "March lock" in report_html
    app_source = (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Forecasts as they moved over time" in app_source
    assert "Latest actuals" in app_source
    assert "lighter lines" in app_source
    assert "Latest historicals used in ledger visuals" in app_source
    assert "ledger_forecast_evolution" in app_source
    assert "Forecast operating loop" in app_source
    assert "Audit tables" in app_source

    app = AppTest.from_file(str(output_dir / "streamlit_app.py"), default_timeout=120)
    app.session_state["active_workbench_section"] = "Forecast ledger"
    app.run()
    assert not app.exception
