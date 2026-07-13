from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold import ContextSource, ForecastContext, ForecastSpec, run_forecast
from nixtla_scaffold.ops import (
    CommandResult,
    build_doctor_payload,
    build_drift_payload,
    build_status_payload,
    build_validation_receipt,
    run_operating_loop,
)
from nixtla_scaffold.refresh import write_refresh_artifacts


def _demo_frame(values: list[int] | None = None) -> pd.DataFrame:
    series = values or [100, 104, 109, 115, 122, 130, 139, 149]
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * len(series),
            "ds": pd.date_range("2025-01-31", periods=len(series), freq="ME"),
            "y": series,
        }
    )


def test_run_outputs_include_operational_receipts(tmp_path) -> None:
    run = run_forecast(_demo_frame(), ForecastSpec(horizon=2, model_policy="baseline"))
    output_dir = run.to_directory(tmp_path / "run")

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["run_receipt"] == "appendix/run_receipt.json"
    assert manifest["outputs"]["validation_receipt"] == "appendix/validation_receipt.csv"
    assert (output_dir / "appendix" / "run_receipt.json").exists()
    assert (output_dir / "appendix" / "run_receipt.md").exists()
    assert (output_dir / "appendix" / "validation_receipt.csv").exists()

    validation = build_validation_receipt(output_dir)
    assert set(validation["check_id"]) >= {"canonical_history_columns", "forecast_contract_columns", "target_numeric"}
    assert "fail" not in set(validation["status"])


def test_status_and_doctor_summarize_local_runs(tmp_path) -> None:
    run = run_forecast(_demo_frame(), ForecastSpec(horizon=2, model_policy="baseline"))
    output_dir = run.to_directory(tmp_path / "runs" / "monthly")

    status = build_status_payload(runs=tmp_path / "runs")
    assert status["run_count"] == 1
    assert status["runs"][0]["run_dir"] == str(output_dir)
    assert status["runs"][0]["missing_artifacts"] == 0

    doctor = build_doctor_payload(output_dir)
    assert doctor["overall_status"] in {"pass", "warn"}
    checks = {row["check_id"]: row["status"] for row in doctor["checks"]}
    assert checks["manifest_exists"] == "pass"
    assert checks["run_receipt_exists"] == "pass"


def test_doctor_surfaces_authoritative_accuracy_gate(tmp_path) -> None:
    context = ForecastContext(
        source_discovery_enabled=False,
        sources=(ContextSource(source_id="target", kind="csv", status="opted_out"),),
    )
    output_dir = run_forecast(
        _demo_frame(),
        ForecastSpec(horizon=2, model_policy="baseline", context=context),
    ).to_directory(tmp_path / "accuracy_run")
    expected_gate = json.loads(
        (output_dir / "appendix" / "accuracy_gate.json").read_text(encoding="utf-8")
    )

    doctor = build_doctor_payload(output_dir)

    assert doctor["accuracy_gate"]["available"] is True
    assert doctor["accuracy_gate"]["status"] == expected_gate["status"]
    assert doctor["checks"][1]["check_id"] == "accuracy_gate_review"
    assert doctor["overall_status"] in {"pass", "warn"}


def test_drift_rollup_reads_refresh_delta(tmp_path) -> None:
    previous = run_forecast(_demo_frame(), ForecastSpec(horizon=2, model_policy="baseline")).to_directory(tmp_path / "previous")
    current = run_forecast(_demo_frame([101, 106, 112, 119, 127, 136, 146, 157]), ForecastSpec(horizon=2, model_policy="baseline")).to_directory(
        tmp_path / "current"
    )
    write_refresh_artifacts(previous, current)

    payload = build_drift_payload(previous_run=previous, refreshed_run=current)

    signals = {row["signal"] for row in payload["signals"]}
    assert "forecast_yhat_change" in signals
    assert "run_pair" in signals


def test_operating_loop_is_linear_and_stops_on_required_failure(tmp_path) -> None:
    config = tmp_path / "operate.yaml"
    config.write_text(
        """
steps:
  - name: first
    args: ["status", "--runs", "runs"]
  - name: second
    args: ["doctor", "--run", "missing"]
  - name: third
    args: ["status", "--runs", "never"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def executor(command: list[str], cwd, timeout_seconds) -> CommandResult:
        if "doctor" in command:
            return CommandResult(exit_code=2, stderr="missing")
        return CommandResult(exit_code=0, stdout="ok")

    payload = run_operating_loop(config, tmp_path / "operate", executor=executor)

    assert payload["status"] == "failed"
    assert [step["name"] for step in payload["steps"]] == ["first", "second"]
    assert (tmp_path / "operate" / "operate_manifest.json").exists()
