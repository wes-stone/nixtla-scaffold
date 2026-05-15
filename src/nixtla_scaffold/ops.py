from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nixtla_scaffold.mcp_contracts import CANONICAL_COLUMNS

RUN_RECEIPT_SCHEMA_VERSION = "nixtla_scaffold.run_receipt.v1"
VALIDATION_RECEIPT_SCHEMA_VERSION = "nixtla_scaffold.validation_receipt.v1"
RUN_STATUS_SCHEMA_VERSION = "nixtla_scaffold.run_status.v1"
DOCTOR_SCHEMA_VERSION = "nixtla_scaffold.doctor.v1"
DRIFT_SCHEMA_VERSION = "nixtla_scaffold.drift.v1"
OPERATE_SCHEMA_VERSION = "nixtla_scaffold.operate.v1"

VALIDATION_RECEIPT_COLUMNS = [
    "check_id",
    "scope",
    "status",
    "severity",
    "evidence",
    "next_action",
]
DOCTOR_COLUMNS = VALIDATION_RECEIPT_COLUMNS
RUN_STATUS_COLUMNS = [
    "run_dir",
    "status",
    "generated_at_utc",
    "modified_at_utc",
    "forecast_origin",
    "horizon",
    "freq",
    "model_policy",
    "selected_models",
    "trust_high",
    "trust_medium",
    "trust_low",
    "missing_artifacts",
    "has_refresh_delta",
    "ledger_forecast_key",
    "ledger_version_id",
    "next_action",
]
DRIFT_COLUMNS = [
    "signal",
    "scope",
    "status",
    "current_value",
    "baseline_value",
    "delta",
    "note",
]


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


def write_operational_receipts(run_dir: str | Path) -> dict[str, str]:
    """Write local-run receipts from artifacts that already exist in a run folder."""

    run = Path(run_dir)
    appendix = run / "appendix"
    appendix.mkdir(parents=True, exist_ok=True)
    receipt_json = appendix / "run_receipt.json"
    receipt_md = appendix / "run_receipt.md"
    validation_csv = appendix / "validation_receipt.csv"
    validation_json = appendix / "validation_receipt.json"

    for path in (receipt_json, receipt_md, validation_csv, validation_json):
        path.touch(exist_ok=True)

    validation = build_validation_receipt(run)
    validation.to_csv(validation_csv, index=False)
    validation_json.write_text(
        json.dumps(
            {
                "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
                "generated_at_utc": _utc_now(),
                "run_dir": str(run),
                "summary": _status_counts(validation),
                "checks": validation.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = build_run_receipt(run)
    receipt_json.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    receipt_md.write_text(format_run_receipt_markdown(receipt), encoding="utf-8")
    return {
        "run_receipt": str(receipt_json),
        "run_receipt_markdown": str(receipt_md),
        "validation_receipt": str(validation_csv),
        "validation_receipt_json": str(validation_json),
    }


def build_run_receipt(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = _read_json(run / "manifest.json")
    diagnostics = _read_json(run / "diagnostics.json")
    profile = manifest.get("profile", {}) if isinstance(manifest, dict) else {}
    spec = manifest.get("spec", {}) if isinstance(manifest, dict) else {}
    reproducibility = manifest.get("reproducibility", {}) if isinstance(manifest, dict) else {}
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    artifact_rows = []
    for name, rel_path in sorted(outputs.items()):
        path = run / str(rel_path)
        state = _file_state(path)
        artifact_rows.append({"name": name, "path": str(path), **state})

    return {
        "schema_version": RUN_RECEIPT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_dir": str(run),
        "forecast_origin": reproducibility.get("forecast_origin") or profile.get("end"),
        "freq": spec.get("freq") or reproducibility.get("frequency") or profile.get("freq"),
        "horizon": spec.get("horizon"),
        "season_length": spec.get("season_length") or reproducibility.get("season_length"),
        "model_policy": spec.get("model_policy"),
        "model_allowlist": spec.get("model_allowlist"),
        "data_hash_sha256": reproducibility.get("data_hash_sha256"),
        "git_sha": reproducibility.get("git_sha"),
        "python_version": reproducibility.get("python_version"),
        "package_versions": reproducibility.get("package_versions", {}),
        "diagnostics_status": diagnostics.get("status"),
        "warning_count": len(manifest.get("warnings", [])) if isinstance(manifest, dict) else 0,
        "next_actions": diagnostics.get("next_diagnostic_steps", []),
        "artifacts": artifact_rows,
    }


def format_run_receipt_markdown(receipt: dict[str, Any]) -> str:
    missing = [item for item in receipt.get("artifacts", []) if not item.get("exists")]
    lines = [
        "# Run receipt",
        "",
        f"- Schema: `{receipt.get('schema_version')}`",
        f"- Generated: `{receipt.get('generated_at_utc')}`",
        f"- Run: `{receipt.get('run_dir')}`",
        f"- Forecast origin: `{receipt.get('forecast_origin')}`",
        f"- Frequency: `{receipt.get('freq')}`",
        f"- Horizon: `{receipt.get('horizon')}`",
        f"- Model policy: `{receipt.get('model_policy')}`",
        f"- Data hash: `{receipt.get('data_hash_sha256')}`",
        f"- Git SHA: `{receipt.get('git_sha')}`",
        "",
        "## Artifact health",
        "",
        f"- Artifact count: `{len(receipt.get('artifacts', []))}`",
        f"- Missing artifacts: `{len(missing)}`",
    ]
    next_actions = receipt.get("next_actions") or []
    if next_actions:
        lines.extend(["", "## Next actions", ""])
        lines.extend(f"- {action}" for action in next_actions[:8])
    return "\n".join(lines) + "\n"


def build_validation_receipt(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    rows: list[dict[str, str]] = []
    manifest = _read_json(run / "manifest.json")
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}

    _append_check(rows, "manifest_exists", "run", (run / "manifest.json").exists(), "manifest.json is present.", "Run forecast again or restore manifest.json.")
    _append_check(rows, "diagnostics_exists", "run", (run / "diagnostics.json").exists(), "diagnostics.json is present.", "Run forecast again or restore diagnostics.json.")
    _append_check(rows, "forecast_exists", "run", (run / "forecast.csv").exists(), "forecast.csv is present.", "Run forecast again or restore forecast.csv.")

    history = _read_csv(run / "appendix" / "history.csv")
    history_required = set(CANONICAL_COLUMNS)
    history_has_required = not history.empty and history_required.issubset(history.columns)
    _append_check(
        rows,
        "canonical_history_columns",
        "data_contract",
        history_has_required,
        f"History columns: {', '.join(history.columns.astype(str)) if not history.empty else 'none'}.",
        "Ensure the run writes appendix\\history.csv with unique_id, ds, and y.",
    )
    if history_has_required:
        duplicates = int(history.duplicated(["unique_id", "ds"]).sum())
        _append_check(
            rows,
            "history_unique_series_dates",
            "data_contract",
            duplicates == 0,
            f"Duplicate unique_id/ds rows: {duplicates}.",
            "Deduplicate the input at the forecast grain before rerunning.",
            fail_severity="warn",
        )
        numeric_y = pd.to_numeric(history["y"], errors="coerce")
        _append_check(
            rows,
            "target_numeric",
            "data_contract",
            int(numeric_y.notna().sum()) == len(history),
            f"Numeric target rows: {int(numeric_y.notna().sum())} of {len(history)}.",
            "Fix nonnumeric target values before rerunning.",
        )

    forecast = _read_csv(run / "forecast.csv")
    forecast_required = {"unique_id", "ds", "yhat"}
    _append_check(
        rows,
        "forecast_contract_columns",
        "output_contract",
        not forecast.empty and forecast_required.issubset(forecast.columns),
        f"Forecast columns: {', '.join(forecast.columns.astype(str)) if not forecast.empty else 'none'}.",
        "Regenerate the forecast output.",
    )

    profile = manifest.get("profile", {}) if isinstance(manifest, dict) else {}
    _append_check(
        rows,
        "frequency_recorded",
        "run_contract",
        bool(profile.get("freq") or manifest.get("spec", {}).get("freq") if isinstance(manifest, dict) else False),
        f"Frequency: {profile.get('freq') or manifest.get('spec', {}).get('freq') if isinstance(manifest, dict) else ''}.",
        "Pass --freq or verify frequency inference in profile.json.",
        fail_severity="warn",
    )

    missing_outputs = _missing_manifest_outputs(run, outputs)
    _append_check(
        rows,
        "manifest_outputs_exist",
        "artifact_contract",
        not missing_outputs,
        f"Missing outputs: {len(missing_outputs)}.",
        "Open doctor output for missing paths and regenerate the run/report if needed.",
        fail_severity="warn",
    )

    if "hierarchy_depth" in forecast.columns:
        _append_check(
            rows,
            "hierarchy_artifacts_exist",
            "hierarchy_contract",
            (run / "appendix" / "hierarchy_coherence.csv").exists() or (run / "audit" / "hierarchy_coherence_post.csv").exists(),
            "Hierarchy forecast detected.",
            "Regenerate hierarchy outputs or rerun with hierarchy reconciliation if parent/child coherence matters.",
            fail_severity="warn",
        )

    driver_audit = _read_csv(run / "appendix" / "driver_availability_audit.csv")
    if not driver_audit.empty:
        status_columns = [column for column in ("audit_status", "status", "modeling_decision") if column in driver_audit.columns]
        failed = 0
        if status_columns:
            text = driver_audit[status_columns].astype(str).agg(" ".join, axis=1).str.lower()
            failed = int(text.str.contains("fail|blocked|rejected|leak").sum())
        _append_check(
            rows,
            "known_future_regressor_audit",
            "driver_contract",
            failed == 0,
            f"Driver audit rows: {len(driver_audit)}; flagged rows: {failed}.",
            "Review appendix\\driver_availability_audit.csv before using drivers as model regressors.",
            fail_severity="warn",
        )

    return pd.DataFrame(rows, columns=VALIDATION_RECEIPT_COLUMNS)


def build_status_payload(*, run: str | Path | None = None, runs: str | Path | None = None) -> dict[str, Any]:
    if run is not None:
        run_rows = [summarize_run_status(run)]
        root = str(Path(run).parent)
    else:
        root_path = Path(runs or "runs")
        run_rows = discover_run_status(root_path)
        root = str(root_path)
    return {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "root": root,
        "run_count": len(run_rows),
        "runs": run_rows,
    }


def discover_run_status(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    manifest_paths = sorted(root_path.rglob("manifest.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    rows = []
    for manifest_path in manifest_paths:
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict) or "spec" not in manifest or "outputs" not in manifest:
            continue
        rows.append(summarize_run_status(run_dir))
    return rows


def summarize_run_status(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = _read_json(run / "manifest.json")
    diagnostics = _read_json(run / "diagnostics.json")
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    spec = manifest.get("spec", {}) if isinstance(manifest, dict) else {}
    profile = manifest.get("profile", {}) if isinstance(manifest, dict) else {}
    reproducibility = manifest.get("reproducibility", {}) if isinstance(manifest, dict) else {}
    trust = _read_csv(run / "appendix" / "trust_summary.csv")
    model_selection = _read_csv(run / "audit" / "model_selection.csv")
    ledger_context = _read_json(run / "ledger_context.json")
    trust_counts = _trust_counts(trust)
    missing = _missing_manifest_outputs(run, outputs)
    return {
        "run_dir": str(run),
        "status": diagnostics.get("status", "unknown"),
        "generated_at_utc": diagnostics.get("generated_at_utc") or diagnostics.get("created_at"),
        "modified_at_utc": _mtime_utc(run / "manifest.json"),
        "forecast_origin": reproducibility.get("forecast_origin") or profile.get("end"),
        "horizon": spec.get("horizon"),
        "freq": spec.get("freq") or profile.get("freq"),
        "model_policy": spec.get("model_policy"),
        "selected_models": _selected_models(model_selection),
        "trust_high": trust_counts.get("high", 0),
        "trust_medium": trust_counts.get("medium", 0),
        "trust_low": trust_counts.get("low", 0),
        "missing_artifacts": len(missing),
        "has_refresh_delta": (run / "appendix" / "refresh_delta.csv").exists(),
        "ledger_forecast_key": ledger_context.get("forecast_key", ""),
        "ledger_version_id": ledger_context.get("forecast_version_id", ""),
        "next_action": _first_next_action(diagnostics),
    }


def write_status_outputs(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "run_status_summary.csv"
    json_path = out / "run_status_summary.json"
    pd.DataFrame(payload.get("runs", []), columns=RUN_STATUS_COLUMNS).to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {"summary": str(summary_path), "json": str(json_path)}


def build_doctor_payload(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    rows: list[dict[str, str]] = []
    _append_check(rows, "run_directory_exists", "run", run.exists(), f"Run directory: {run}.", "Point --run to a forecast output folder.")
    manifest = _read_json(run / "manifest.json")
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    _append_check(rows, "manifest_exists", "run", (run / "manifest.json").exists(), "manifest.json is present.", "Regenerate the run.")
    _append_check(rows, "diagnostics_exists", "run", (run / "diagnostics.json").exists(), "diagnostics.json is present.", "Regenerate the run.")
    _append_check(rows, "forecast_exists", "run", (run / "forecast.csv").exists(), "forecast.csv is present.", "Regenerate the run.")

    missing_outputs = _missing_manifest_outputs(run, outputs)
    _append_check(
        rows,
        "manifest_output_paths_exist",
        "artifact_contract",
        not missing_outputs,
        "; ".join(missing_outputs[:10]) if missing_outputs else "All declared manifest outputs exist.",
        "Regenerate the run or report artifacts for missing files.",
        fail_severity="warn",
    )
    _append_check(
        rows,
        "run_receipt_exists",
        "ops_receipts",
        (run / "appendix" / "run_receipt.json").exists(),
        "appendix\\run_receipt.json records local reproducibility evidence.",
        "Rerun the forecast with the current package to generate the receipt.",
        fail_severity="warn",
    )
    _append_check(
        rows,
        "validation_receipt_exists",
        "ops_receipts",
        (run / "appendix" / "validation_receipt.csv").exists(),
        "appendix\\validation_receipt.csv indexes local data/readiness checks.",
        "Rerun the forecast with the current package to generate the receipt.",
        fail_severity="warn",
    )

    policy_resolution = manifest.get("model_policy_resolution", {}) if isinstance(manifest, dict) else {}
    smooth_rows = [
        row
        for row in policy_resolution.get("families", [])
        if isinstance(row, dict) and str(row.get("family", "")).lower() == "smooth"
    ]
    if smooth_rows:
        smooth = smooth_rows[0]
        smooth_requested = bool(smooth.get("requested"))
        smooth_ran = bool(smooth.get("ran"))
        smooth_reason = str(smooth.get("reason_if_not_ran") or "")
        _append_check(
            rows,
            "smooth_optional_family_status",
            "optional_models",
            (not smooth_requested) or smooth_ran,
            "Smooth ADAM ran." if smooth_ran else f"Smooth ADAM did not run: {smooth_reason or 'not requested'}.",
            "Install with `uv sync --extra dev --extra smooth` if this run should include smooth candidates.",
            fail_severity="warn",
        )

    trust = _read_csv(run / "appendix" / "trust_summary.csv")
    if not trust.empty:
        low_count = _trust_counts(trust).get("low", 0)
        _append_check(
            rows,
            "trust_summary_review",
            "forecast_readiness",
            low_count == 0,
            f"Low-trust series: {low_count}.",
            "Review appendix\\trust_summary.csv before stakeholder use.",
            fail_severity="warn",
        )
        if "full_horizon_claim_allowed" in trust.columns:
            no_claim = int(trust["full_horizon_claim_allowed"].astype(str).str.lower().isin({"false", "0", "no"}).sum())
            _append_check(
                rows,
                "full_horizon_claim_review",
                "forecast_readiness",
                no_claim == 0,
                f"Series without full-horizon claim: {no_claim}.",
                "Shorten horizon, add history, or disclose far-horizon rows as directional.",
                fail_severity="warn",
            )

    refresh_manifest = run / "refresh_manifest.json"
    if refresh_manifest.exists():
        refresh_delta = _read_csv(run / "appendix" / "refresh_delta.csv")
        _append_check(
            rows,
            "refresh_delta_review",
            "refresh",
            not refresh_delta.empty,
            f"Refresh delta rows: {len(refresh_delta)}.",
            "Review appendix\\refresh_delta.csv for spec, model, trust, and forecast changes.",
            fail_severity="warn",
        )

    ledger_context = _read_json(run / "ledger_context.json")
    if ledger_context:
        exports = Path(str(ledger_context.get("exports_path", "")))
        _append_check(
            rows,
            "ledger_exports_exist",
            "ledger",
            exports.exists(),
            f"Ledger exports path: {exports}.",
            "Run ledger export or re-register the run.",
            fail_severity="warn",
        )
    else:
        _append_check(
            rows,
            "ledger_context_optional",
            "ledger",
            True,
            "No ledger_context.json found; this run is not registered in a forecast ledger.",
            "Register the run if it is part of a recurring operating loop.",
        )

    checks = pd.DataFrame(rows, columns=DOCTOR_COLUMNS)
    return {
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_dir": str(run),
        "overall_status": _overall_status(checks),
        "summary": _status_counts(checks),
        "checks": checks.to_dict(orient="records"),
    }


def write_doctor_outputs(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks_path = out / "doctor_checks.csv"
    json_path = out / "doctor_summary.json"
    pd.DataFrame(payload.get("checks", []), columns=DOCTOR_COLUMNS).to_csv(checks_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {"checks": str(checks_path), "json": str(json_path)}


def build_drift_payload(
    *,
    ledger: str | Path | None = None,
    previous_run: str | Path | None = None,
    refreshed_run: str | Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if refreshed_run is not None:
        rows.extend(_refresh_drift_rows(previous_run=previous_run, refreshed_run=refreshed_run))
    if ledger is not None:
        rows.extend(_ledger_drift_rows(ledger))
    if not rows:
        rows.append(
            _drift_row(
                "evidence_available",
                "drift",
                "warn",
                "0",
                "",
                "",
                "No ledger exports or refresh delta artifacts were available for drift review.",
            )
        )
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "ledger": str(ledger) if ledger is not None else "",
        "previous_run": str(previous_run) if previous_run is not None else "",
        "refreshed_run": str(refreshed_run) if refreshed_run is not None else "",
        "summary": _drift_status_counts(rows),
        "signals": rows,
    }


def write_drift_report(
    output_dir: str | Path,
    *,
    ledger: str | Path | None = None,
    previous_run: str | Path | None = None,
    refreshed_run: str | Path | None = None,
) -> dict[str, str]:
    payload = build_drift_payload(ledger=ledger, previous_run=previous_run, refreshed_run=refreshed_run)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "drift_summary.csv"
    json_path = out / "drift_summary.json"
    md_path = out / "drift_report.md"
    pd.DataFrame(payload["signals"], columns=DRIFT_COLUMNS).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(format_drift_markdown(payload), encoding="utf-8")
    return {"summary": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def format_drift_markdown(payload: dict[str, Any]) -> str:
    rows = payload.get("signals", [])
    counts = payload.get("summary", {})
    lines = [
        "# Drift report",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Signals: `{len(rows)}`",
        f"- Warnings: `{counts.get('warn', 0)}`",
        f"- Failures: `{counts.get('fail', 0)}`",
        "",
        "| Signal | Scope | Status | Current | Baseline | Delta | Note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('signal', '')} | {row.get('scope', '')} | {row.get('status', '')} | "
            f"{row.get('current_value', '')} | {row.get('baseline_value', '')} | {row.get('delta', '')} | {row.get('note', '')} |"
        )
    return "\n".join(lines) + "\n"


def run_operating_loop(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    executor: Callable[[list[str], Path | None, int | None], CommandResult] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    steps = config.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("operate config requires a non-empty 'steps' list")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    executor = executor or _execute_command
    results: list[dict[str, Any]] = []
    status = "passed"
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError("each operate step must be an object")
        name = str(step.get("name") or f"step_{index}")
        required = bool(step.get("required", True))
        timeout = step.get("timeout_seconds")
        timeout_int = int(timeout) if timeout is not None else None
        cwd = Path(str(step["cwd"])) if step.get("cwd") else None
        command = _step_command(step)
        result = executor(command, cwd, timeout_int)
        step_status = "passed" if result.exit_code == 0 else ("failed" if required else "warn")
        record = {
            "index": index,
            "name": name,
            "required": required,
            "status": step_status,
            "exit_code": result.exit_code,
            "command": command,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        results.append(record)
        if required and result.exit_code != 0:
            status = "failed"
            break
        if result.exit_code != 0 and status != "failed":
            status = "warn"
    payload = {
        "schema_version": OPERATE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "config_path": str(config_file),
        "status": status,
        "steps": results,
    }
    manifest_path = out / "operate_manifest.json"
    steps_path = out / "operate_steps.csv"
    manifest_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(results).to_csv(steps_path, index=False)
    payload["paths"] = {"manifest": str(manifest_path), "steps": str(steps_path)}
    return payload


def _step_command(step: dict[str, Any]) -> list[str]:
    if "args" in step:
        args = step["args"]
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError("operate step args must be a list of strings")
        return _normalize_cli_args(args)
    command = step.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("operate step requires either args or command")
    return _normalize_cli_args(shlex.split(command, posix=False))


def _normalize_cli_args(tokens: Sequence[str]) -> list[str]:
    items = list(tokens)
    if not items:
        raise ValueError("empty operate command")
    if items[0] == "uv":
        return items
    if items[0] == "nixtla-scaffold":
        return [sys.executable, "-m", "nixtla_scaffold.cli", *items[1:]]
    if items[0] in {"forecast", "refresh", "status", "doctor", "drift", "ledger", "report", "compare-models", "experiment"}:
        return [sys.executable, "-m", "nixtla_scaffold.cli", *items]
    return items


def _execute_command(command: list[str], cwd: Path | None, timeout_seconds: int | None) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return CommandResult(exit_code=int(completed.returncode), stdout=completed.stdout, stderr=completed.stderr)


def _refresh_drift_rows(*, previous_run: str | Path | None, refreshed_run: str | Path) -> list[dict[str, Any]]:
    current = Path(refreshed_run)
    delta = _read_csv(current / "appendix" / "refresh_delta.csv")
    if delta.empty:
        return [
            _drift_row(
                "refresh_delta_available",
                str(current),
                "warn",
                "0",
                "",
                "",
                "No appendix\\refresh_delta.csv rows were available; run refresh or inspect refresh_manifest.json.",
            )
        ]
    rows: list[dict[str, Any]] = []
    for delta_type, group in delta.groupby("delta_type", dropna=False):
        status = "warn" if str(delta_type) in {"model_selection_change", "trust_change", "spec_change"} else "pass"
        rows.append(
            _drift_row(
                str(delta_type),
                str(current),
                status,
                str(len(group)),
                "",
                "",
                "Refresh delta rows grouped by type; review changed rows before stakeholder use.",
            )
        )
    if previous_run is not None:
        rows.append(_drift_row("run_pair", "refresh", "pass", str(current), str(previous_run), "", "Previous and refreshed run paths recorded."))
    return rows


def _ledger_drift_rows(ledger: str | Path) -> list[dict[str, Any]]:
    ledger_dir = Path(ledger)
    exports = ledger_dir / "exports"
    versions = _read_csv(exports / "forecast_versions.csv")
    performance = _read_csv(exports / "forecast_performance.csv")
    locks = _read_csv(exports / "official_forecast_locks.csv")
    rows = [
        _drift_row("ledger_versions", str(ledger_dir), "pass" if not versions.empty else "warn", str(len(versions)), "", "", "Registered forecast versions."),
        _drift_row("official_locks", str(ledger_dir), "pass" if not locks.empty else "warn", str(len(locks)), "", "", "Official submitted forecast locks."),
    ]
    if performance.empty:
        rows.append(_drift_row("forecast_performance", str(ledger_dir), "warn", "0", "", "", "No landed actuals scored yet."))
        return rows
    perf = performance.copy()
    perf["rmse"] = pd.to_numeric(perf.get("rmse"), errors="coerce")
    perf["mae"] = pd.to_numeric(perf.get("mae"), errors="coerce")
    perf["bias"] = pd.to_numeric(perf.get("bias"), errors="coerce")
    for forecast_key, group in perf.groupby("forecast_key", dropna=False):
        rows.append(
            _drift_row(
                "forecast_performance_rmse",
                str(forecast_key),
                "pass",
                _format_number(group["rmse"].mean()),
                "",
                "",
                f"Mean RMSE over {len(group)} scored series/version rows.",
            )
        )
        rows.append(
            _drift_row(
                "forecast_performance_bias",
                str(forecast_key),
                "warn" if abs(float(group["bias"].mean(skipna=True) or 0.0)) > 0 else "pass",
                _format_number(group["bias"].mean()),
                "",
                "",
                "Mean bias across scored rows; directional only until enough actuals land.",
            )
        )
    return rows


def _append_check(
    rows: list[dict[str, str]],
    check_id: str,
    scope: str,
    passed: bool,
    evidence: str,
    next_action: str,
    *,
    fail_severity: str = "fail",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "scope": scope,
            "status": "pass" if passed else fail_severity,
            "severity": "info" if passed else fail_severity,
            "evidence": evidence,
            "next_action": "" if passed else next_action,
        }
    )


def _missing_manifest_outputs(run: Path, outputs: dict[str, Any]) -> list[str]:
    missing = []
    for name, rel_path in sorted(outputs.items()):
        path = run / str(rel_path)
        if not path.exists():
            missing.append(f"{name}: {rel_path}")
    return missing


def _trust_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "trust_level" not in frame.columns:
        return {"high": 0, "medium": 0, "low": 0}
    values = frame["trust_level"].astype(str).str.lower()
    return {
        "high": int(values.eq("high").sum()),
        "medium": int(values.eq("medium").sum()),
        "low": int(values.eq("low").sum()),
    }


def _selected_models(frame: pd.DataFrame) -> str:
    if frame.empty or "selected_model" not in frame.columns:
        return ""
    return ";".join(sorted(set(frame["selected_model"].dropna().astype(str))))


def _first_next_action(diagnostics: dict[str, Any]) -> str:
    steps = diagnostics.get("next_diagnostic_steps")
    if isinstance(steps, list) and steps:
        return str(steps[0])
    return ""


def _overall_status(checks: pd.DataFrame) -> str:
    statuses = set(checks["status"].astype(str)) if not checks.empty and "status" in checks.columns else set()
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "status" not in frame.columns:
        return {"pass": 0, "warn": 0, "fail": 0}
    values = frame["status"].astype(str)
    return {status: int(values.eq(status).sum()) for status in ("pass", "warn", "fail")}


def _drift_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {status: sum(1 for row in rows if row.get("status") == status) for status in ("pass", "warn", "fail")}


def _drift_row(signal: str, scope: str, status: str, current: str, baseline: str, delta: str, note: str) -> dict[str, Any]:
    return {
        "signal": signal,
        "scope": scope,
        "status": status,
        "current_value": current,
        "baseline_value": baseline,
        "delta": delta,
        "note": note,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "modified_at_utc": ""}
    stat = path.stat()
    return {"exists": True, "bytes": int(stat.st_size), "modified_at_utc": _timestamp_utc(stat.st_mtime)}


def _mtime_utc(path: Path) -> str:
    if not path.exists():
        return ""
    return _timestamp_utc(path.stat().st_mtime)


def _timestamp_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _format_number(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{float(number):.6g}"
