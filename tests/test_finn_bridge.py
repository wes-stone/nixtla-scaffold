from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nixtla_scaffold.cli import main
from nixtla_scaffold.finn_bridge import (
    FINN_BRIDGE_SCHEMA_VERSION,
    compare_finn_forecasts,
    canonicalize_finn_forecasts,
    check_finn_environment,
    run_finn_bridge,
    score_finn_forecasts,
)
from nixtla_scaffold.reports import build_streamlit_app


def _finn_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "ds": ["2026-01-31", "2026-02-28"],
            "model": ["FINN_Auto", "FINN_Auto"],
            "yhat": [110.0, 118.0],
        }
    )


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 6,
            "ds": pd.date_range("2025-01-31", periods=6, freq="ME"),
            "y": [100.0, 102.0, 105.0, 107.0, 109.0, 112.0],
        }
    )


def _run_dir(tmp_path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    forecast = pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "ds": ["2026-01-31", "2026-02-28"],
            "yhat": [100.0, 110.0],
            "model": ["ScaffoldChampion", "ScaffoldChampion"],
        }
    )
    forecast.to_csv(run_dir / "forecast.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({"outputs": {"forecast": "forecast.csv"}}) + "\n", encoding="utf-8")
    (run_dir / "llm_context.json").write_text(json.dumps({"artifact_index": {"forecast": "forecast.csv"}}) + "\n", encoding="utf-8")
    (run_dir / "control_pane_state.json").write_text(
        json.dumps({"feature_map": [], "artifacts": {"forecast": "forecast.csv"}}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _finn_backtest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "ds": ["2025-05-31", "2025-06-30"],
            "model": ["FINN_Auto", "FINN_Auto"],
            "yhat": [108.0, 113.0],
            "cutoff": ["2025-04-30", "2025-05-31"],
        }
    )


def test_canonicalize_finn_forecasts_marks_advisory_external_contract(tmp_path) -> None:
    source = tmp_path / "finn_raw.csv"
    _finn_forecast_frame().to_csv(source, index=False)

    result = canonicalize_finn_forecasts(source, output_dir=tmp_path / "out")

    assert result.manifest["schema_version"] == FINN_BRIDGE_SCHEMA_VERSION
    assert result.manifest["operation"] == "canonicalize"
    assert set(result.forecasts["source_system"]) == {"FINN"}
    assert result.forecasts["advisory_only"].eq(True).all()
    assert (tmp_path / "out" / "finn_forecasts.csv").exists()
    assert (tmp_path / "out" / "finn_manifest.json").exists()


def test_check_finn_environment_missing_rscript_has_install_hint() -> None:
    result = check_finn_environment(rscript="definitely_missing_Rscript_for_test")

    assert result.manifest["available"] is False
    assert result.manifest["rscript_available"] is False
    assert "Direct FINN execution requires R/Rscript plus the finnts R package" in result.manifest["install_hint"]


def test_finn_cli_ingest_writes_manifest(tmp_path) -> None:
    source = tmp_path / "finn_raw.csv"
    output = tmp_path / "finn_ingest"
    _finn_forecast_frame().to_csv(source, index=False)

    exit_code = main(["finn", "ingest", "--input", str(source), "--output", str(output)])

    assert exit_code == 0
    manifest = json.loads((output / "finn_manifest.json").read_text(encoding="utf-8"))
    assert manifest["operation"] == "canonicalize"
    assert manifest["metadata"]["models"] == ["FINN_Auto"]


def test_finn_run_without_runner_writes_template_only_manifest(tmp_path) -> None:
    input_path = tmp_path / "history.csv"
    output = tmp_path / "finn_run"
    _history_frame().to_csv(input_path, index=False)

    result = run_finn_bridge(input_path, output)

    assert result.manifest["status"] == "template_only"
    assert (output / "finn_runner_template.R").exists()
    assert (output / "finn_manifest.json").exists()


def test_score_finn_forecasts_manifest_points_to_scoring_outputs(tmp_path) -> None:
    forecasts = tmp_path / "finn_backtest.csv"
    actuals = tmp_path / "history.csv"
    output = tmp_path / "finn_score"
    _finn_backtest_frame().to_csv(forecasts, index=False)
    _history_frame().to_csv(actuals, index=False)

    result = score_finn_forecasts(forecasts, actuals, output, requested_horizon=1)

    assert result.manifest["operation"] == "score"
    assert result.manifest["outputs"]["external_backtest_long"] == "external_backtest_long.csv"
    assert result.manifest["outputs"]["external_model_metrics"] == "external_model_metrics.csv"
    assert result.manifest["outputs"]["external_scoring_manifest"] == "external_scoring_manifest.json"
    assert (output / "external_backtest_long.csv").exists()
    assert (output / "external_model_metrics.csv").exists()


def test_finn_compare_defaults_to_run_finn_and_registers_agent_artifacts(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    source = tmp_path / "finn_future.csv"
    _finn_forecast_frame().to_csv(source, index=False)

    result = compare_finn_forecasts(run_dir, source)

    assert result.manifest["operation"] == "compare"
    assert (run_dir / "finn" / "finn_forecasts.csv").exists()
    assert (run_dir / "finn" / "forecast_comparison.csv").exists()
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["outputs"]["finn_forecasts"] == "finn/finn_forecasts.csv"
    assert run_manifest["outputs"]["finn_forecast_comparison"] == "finn/forecast_comparison.csv"
    assert run_manifest["outputs"]["finn_forecast_comparison_manifest"] == "finn/comparison_manifest.json"
    llm_context = json.loads((run_dir / "llm_context.json").read_text(encoding="utf-8"))
    assert llm_context["artifact_index"]["finn_manifest"] == "finn/finn_manifest.json"
    control = json.loads((run_dir / "control_pane_state.json").read_text(encoding="utf-8"))
    assert any(row["mechanism"] == "FINN advisory bridge" and row["status"] == "available" for row in control["feature_map"])


def test_finn_score_can_attach_to_run_finn_for_reporting(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    forecasts = tmp_path / "finn_backtest.csv"
    actuals = tmp_path / "history.csv"
    _finn_backtest_frame().to_csv(forecasts, index=False)
    _history_frame().to_csv(actuals, index=False)

    result = score_finn_forecasts(forecasts, actuals, run_dir=run_dir, requested_horizon=1)

    assert result.manifest["operation"] == "score"
    assert (run_dir / "finn" / "external_backtest_long.csv").exists()
    assert (run_dir / "finn" / "external_model_metrics.csv").exists()
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["outputs"]["finn_external_model_metrics"] == "finn/external_model_metrics.csv"
    assert run_manifest["outputs"]["finn_external_scoring_manifest"] == "finn/external_scoring_manifest.json"


def test_streamlit_app_includes_finn_advisory_section() -> None:
    app = build_streamlit_app()

    assert '"FINN advisory"' in app
    assert '"Pipeline map"' in app
    assert "FINN now sits beside native Nixtla" in app
    assert "external challenger lane and shared cutoff/actual scoring spine" in app
    assert "Historical actuals" in app
    assert "Scaffold vs FINN with recent actuals" in app
    assert 'read_scoped_csv("finn", "external_model_metrics.csv")' in app
    assert "finn/finn_manifest.json / finn/external_model_metrics.csv" in app
