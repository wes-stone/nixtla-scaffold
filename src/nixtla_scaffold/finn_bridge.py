from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd

from nixtla_scaffold.comparisons import ForecastComparisonResult, write_forecast_comparison
from nixtla_scaffold.comparisons import (
    FORECAST_COMPARISON_LLM_CONTEXT_OUTPUT,
    FORECAST_COMPARISON_MANIFEST_OUTPUT,
    FORECAST_COMPARISON_OUTPUT,
    FORECAST_COMPARISON_REPORT_OUTPUT,
    FORECAST_COMPARISON_SUMMARY_OUTPUT,
    FORECAST_COMPARISON_WORKBOOK_OUTPUT,
)
from nixtla_scaffold.external import build_external_forecast_metadata, canonicalize_external_forecasts, load_external_forecasts
from nixtla_scaffold.external_scoring import (
    EXTERNAL_BACKTEST_LONG_OUTPUT,
    EXTERNAL_MODEL_METRICS_OUTPUT,
    EXTERNAL_SCORING_MANIFEST_OUTPUT,
    ExternalForecastScoreResult,
    write_external_forecast_scores,
)

FINN_BRIDGE_SCHEMA_VERSION = "nixtla_scaffold.finn_bridge.v1"
FINN_REPORTING_DIR = "finn"
FINN_FORECAST_OUTPUT = "finn_forecasts.csv"
FINN_MANIFEST_OUTPUT = "finn_manifest.json"
FINN_RUNNER_TEMPLATE_OUTPUT = "finn_runner_template.R"
FINN_INSTALL_HINTS = (
    "Direct FINN execution requires R/Rscript plus the finnts R package. "
    "On Windows, install R first (for example with winget install RProject.R), "
    "restart the terminal so Rscript is on PATH, then install finnts in R before using `nixtla-scaffold finn run --runner ...`."
)


@dataclass(frozen=True)
class FINNCheckResult:
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FINNRunResult:
    forecasts: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FINNComparisonResult:
    run: FINNRunResult
    comparison: ForecastComparisonResult
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FINNScoreResult:
    run: FINNRunResult
    scores: ExternalForecastScoreResult
    manifest: dict[str, Any]


def check_finn_environment(*, rscript: str = "Rscript") -> FINNCheckResult:
    """Check whether Rscript and the finnts R package are available."""

    rscript_path = _resolve_rscript(rscript)
    r_version = _run_command([rscript_path, "--version"], timeout_seconds=30)
    package_version = None
    package_status = "not_checked"
    if r_version["returncode"] == 0:
        package = _run_command(
            [rscript_path, "-e", "cat(as.character(utils::packageVersion('finnts')))"],
            timeout_seconds=30,
        )
        package_status = "available" if package["returncode"] == 0 else "missing"
        package_version = package["stdout"].strip() if package["returncode"] == 0 else None
    manifest = {
        "schema_version": FINN_BRIDGE_SCHEMA_VERSION,
        "operation": "check",
        "rscript": rscript_path,
        "rscript_available": r_version["returncode"] == 0,
        "rscript_version": (r_version["stderr"] or r_version["stdout"]).strip(),
        "finnts_package_status": package_status,
        "finnts_package_version": package_version,
        "available": r_version["returncode"] == 0 and package_status == "available",
        "install_hint": FINN_INSTALL_HINTS if r_version["returncode"] != 0 or package_status != "available" else "",
        "checked_at_utc": _now_utc(),
        "advisory_only": True,
    }
    return FINNCheckResult(manifest=manifest)


def canonicalize_finn_forecasts(
    source: str | Path | pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    sheet: str | int | None = None,
    format: str = "auto",
    model_name: str = "FINN",
    source_id: str = "finn",
    id_col: str = "unique_id",
    time_col: str = "ds",
    value_col: str = "yhat",
    model_col: str = "model",
    forecast_origin_col: str = "cutoff",
) -> FINNRunResult:
    """Canonicalize FINN-produced forecasts into the scaffold external forecast contract."""

    source_path = Path(source) if isinstance(source, (str, Path)) else None
    forecasts = load_external_forecasts(
        source,
        sheet=sheet,
        format=format,
        model_name=model_name,
        source_id=source_id,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
        model_col=model_col,
        forecast_origin_col=forecast_origin_col,
    )
    forecasts["source_system"] = "FINN"
    forecasts["advisory_only"] = True
    manifest = _build_finn_manifest(
        operation="canonicalize",
        forecasts=forecasts,
        source=source_path,
        options={
            "format": format,
            "model_name": model_name,
            "source_id": source_id,
            "id_col": id_col,
            "time_col": time_col,
            "value_col": value_col,
            "model_col": model_col,
            "forecast_origin_col": forecast_origin_col,
        },
    )
    result = FINNRunResult(forecasts=forecasts, manifest=manifest)
    if output_dir is not None:
        result_to_directory(result, output_dir)
    return result


def run_finn_bridge(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    runner: str | Path | None = None,
    rscript: str = "Rscript",
    raw_output: str = "finn_raw_forecast.csv",
    seed: int = 123,
    model_name: str = "FINN",
    source_id: str = "finn",
    extra_args: tuple[str, ...] = (),
) -> FINNRunResult:
    """Run a user-supplied FINN R runner or write a template when no runner is provided."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    template_path = out / FINN_RUNNER_TEMPLATE_OUTPUT
    template_path.write_text(build_finn_runner_template(), encoding="utf-8")

    raw_path = out / raw_output
    base_manifest = {
        "schema_version": FINN_BRIDGE_SCHEMA_VERSION,
        "operation": "run",
        "input": str(input_file),
        "input_sha256": _sha256(input_file),
        "output_dir": str(out),
        "runner_template": FINN_RUNNER_TEMPLATE_OUTPUT,
        "runner": str(runner) if runner is not None else None,
        "raw_output": raw_output,
        "seed": seed,
        "model_name": model_name,
        "source_id": source_id,
        "extra_args": list(extra_args),
        "advisory_only": True,
        "created_at_utc": _now_utc(),
    }
    if runner is None:
        manifest = {
            **base_manifest,
            "status": "template_only",
            "status_reason": "no --runner provided; edit finn_runner_template.R or pass a FINN R runner script",
            "outputs": {"runner_template": FINN_RUNNER_TEMPLATE_OUTPUT, "manifest": FINN_MANIFEST_OUTPUT},
        }
        (out / FINN_MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        return FINNRunResult(forecasts=pd.DataFrame(columns=["unique_id", "ds", "model", "yhat"]), manifest=manifest)

    runner_path = Path(runner)
    if not runner_path.exists():
        raise FileNotFoundError(runner_path)
    command = [
        _resolve_rscript(rscript),
        str(runner_path),
        "--input",
        str(input_file),
        "--output",
        str(raw_path),
        "--seed",
        str(seed),
        *extra_args,
    ]
    command_result = _run_command(command, timeout_seconds=3600)
    if command_result["returncode"] != 0:
        manifest = {
            **base_manifest,
            "status": "failed",
            "command": command,
            "returncode": command_result["returncode"],
            "stdout": command_result["stdout"],
            "stderr": command_result["stderr"],
            "outputs": {"runner_template": FINN_RUNNER_TEMPLATE_OUTPUT, "manifest": FINN_MANIFEST_OUTPUT},
        }
        (out / FINN_MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        raise RuntimeError(f"FINN runner failed with exit code {command_result['returncode']}; see {out / FINN_MANIFEST_OUTPUT}")
    if not raw_path.exists():
        raise FileNotFoundError(f"FINN runner completed but did not write expected output: {raw_path}")

    canonical = canonicalize_finn_forecasts(raw_path, model_name=model_name, source_id=source_id)
    manifest = {
        **base_manifest,
        **canonical.manifest,
        "operation": "run",
        "status": "completed",
        "command": command,
        "returncode": command_result["returncode"],
        "stdout": command_result["stdout"],
        "stderr": command_result["stderr"],
        "outputs": {
            "raw_output": raw_output,
            "forecasts": FINN_FORECAST_OUTPUT,
            "manifest": FINN_MANIFEST_OUTPUT,
            "runner_template": FINN_RUNNER_TEMPLATE_OUTPUT,
        },
    }
    result = FINNRunResult(forecasts=canonical.forecasts, manifest=manifest)
    result_to_directory(result, out)
    return result


def result_to_directory(result: FINNRunResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_csv(out / FINN_FORECAST_OUTPUT, index=False)
    (out / FINN_MANIFEST_OUTPUT).write_text(json.dumps(result.manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def compare_finn_forecasts(
    run_dir: str | Path,
    finn_forecasts: str | Path,
    output_dir: str | Path | None = None,
    *,
    scaffold_model: str | None = None,
    **kwargs: Any,
) -> FINNComparisonResult:
    out = Path(output_dir) if output_dir is not None else Path(run_dir) / FINN_REPORTING_DIR
    finn = canonicalize_finn_forecasts(finn_forecasts, output_dir=out, **kwargs)
    comparison = write_forecast_comparison(
        run_dir,
        finn.forecasts,
        output_dir=out,
        external_format="long",
        scaffold_model=scaffold_model,
    )
    manifest = {
        **finn.manifest,
        "operation": "compare",
        "scaffold_run_dir": str(Path(run_dir)),
        "scaffold_model": scaffold_model or "selected",
        "comparison": comparison.manifest,
        "outputs": {
            **finn.manifest.get("outputs", {}),
            "forecast_comparison": FORECAST_COMPARISON_OUTPUT,
            "forecast_comparison_summary": FORECAST_COMPARISON_SUMMARY_OUTPUT,
            "forecast_comparison_workbook": FORECAST_COMPARISON_WORKBOOK_OUTPUT,
            "forecast_comparison_report": FORECAST_COMPARISON_REPORT_OUTPUT,
            "forecast_comparison_llm_context": FORECAST_COMPARISON_LLM_CONTEXT_OUTPUT,
            "forecast_comparison_manifest": FORECAST_COMPARISON_MANIFEST_OUTPUT,
        },
    }
    (out / FINN_MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    register_finn_reporting_artifacts(run_dir, out, manifest)
    return FINNComparisonResult(run=finn, comparison=comparison, manifest=manifest)


def score_finn_forecasts(
    finn_forecasts: str | Path,
    actuals: str | Path,
    output_dir: str | Path | None = None,
    *,
    run_dir: str | Path | None = None,
    actuals_sheet: str | int | None = None,
    actual_id_col: str = "unique_id",
    actual_time_col: str = "ds",
    actual_value_col: str = "y",
    season_length: int = 1,
    requested_horizon: int | None = None,
    **kwargs: Any,
) -> FINNScoreResult:
    if output_dir is None and run_dir is None:
        raise ValueError("output_dir is required unless run_dir is supplied")
    out = Path(output_dir) if output_dir is not None else Path(run_dir) / FINN_REPORTING_DIR
    finn = canonicalize_finn_forecasts(finn_forecasts, output_dir=out, **kwargs)
    scores = write_external_forecast_scores(
        finn.forecasts,
        actuals,
        out,
        external_format="long",
        actuals_sheet=actuals_sheet,
        actual_id_col=actual_id_col,
        actual_time_col=actual_time_col,
        actual_value_col=actual_value_col,
        season_length=season_length,
        requested_horizon=requested_horizon,
    )
    manifest = {
        **finn.manifest,
        "operation": "score",
        "actuals_source": str(Path(actuals)),
        "scoring": scores.manifest,
        "outputs": {
            **finn.manifest.get("outputs", {}),
            "external_backtest_long": EXTERNAL_BACKTEST_LONG_OUTPUT,
            "external_model_metrics": EXTERNAL_MODEL_METRICS_OUTPUT,
            "external_scoring_manifest": EXTERNAL_SCORING_MANIFEST_OUTPUT,
        },
    }
    (out / FINN_MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    if run_dir is not None:
        register_finn_reporting_artifacts(run_dir, out, manifest)
    return FINNScoreResult(run=finn, scores=scores, manifest=manifest)


def register_finn_reporting_artifacts(run_dir: str | Path, output_dir: str | Path, manifest: dict[str, Any]) -> None:
    """Attach FINN advisory artifacts to a scaffold run manifest for reports and agents."""

    run_path = Path(run_dir)
    out = Path(output_dir)
    if not run_path.exists():
        return
    try:
        relative_output = out.resolve().relative_to(run_path.resolve())
    except ValueError:
        return
    relative_prefix = _posix_path(relative_output)
    known_outputs = {
        "forecasts": FINN_FORECAST_OUTPUT,
        "manifest": FINN_MANIFEST_OUTPUT,
        "runner_template": FINN_RUNNER_TEMPLATE_OUTPUT,
        "forecast_comparison": FORECAST_COMPARISON_OUTPUT,
        "forecast_comparison_summary": FORECAST_COMPARISON_SUMMARY_OUTPUT,
        "forecast_comparison_workbook": FORECAST_COMPARISON_WORKBOOK_OUTPUT,
        "forecast_comparison_report": FORECAST_COMPARISON_REPORT_OUTPUT,
        "forecast_comparison_llm_context": FORECAST_COMPARISON_LLM_CONTEXT_OUTPUT,
        "forecast_comparison_manifest": FORECAST_COMPARISON_MANIFEST_OUTPUT,
        "external_backtest_long": EXTERNAL_BACKTEST_LONG_OUTPUT,
        "external_model_metrics": EXTERNAL_MODEL_METRICS_OUTPUT,
        "external_scoring_manifest": EXTERNAL_SCORING_MANIFEST_OUTPUT,
    }
    attached = {
        f"finn_{key}": f"{relative_prefix}/{filename}" if relative_prefix else filename
        for key, filename in known_outputs.items()
        if (out / filename).exists()
    }
    if not attached:
        return
    _update_json_mapping(run_path / "manifest.json", ("outputs",), attached)
    _update_json_mapping(run_path / "llm_context.json", ("artifact_index",), attached)
    _update_control_pane_state(run_path / "control_pane_state.json", attached, manifest)


def build_finn_runner_template() -> str:
    return """#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

input_path <- get_arg("--input")
output_path <- get_arg("--output", "finn_raw_forecast.csv")
seed <- as.integer(get_arg("--seed", "123"))
set.seed(seed)

if (is.null(input_path)) stop("--input is required")
if (!requireNamespace("finnts", quietly = TRUE)) {
  stop("Install the FINN/finnts R package before running this template")
}

# Replace this block with your preferred FINN workflow. The expected output is a
# CSV with at least unique_id, ds, model, and yhat columns. Cutoff/backtest rows
# can be included with a cutoff column for later score workflows.
history <- read.csv(input_path)
stop("Edit this template to call finnts::forecast_time_series() or the FINN agent workflow, then write output_path")
"""


def _build_finn_manifest(
    *,
    operation: str,
    forecasts: pd.DataFrame,
    source: Path | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    metadata = build_external_forecast_metadata(forecasts)
    return {
        "schema_version": FINN_BRIDGE_SCHEMA_VERSION,
        "operation": operation,
        "source": str(source) if source is not None else "dataframe",
        "source_sha256": _sha256(source) if source is not None and source.exists() else None,
        "metadata": metadata,
        "options": options,
        "advisory_only": True,
        "created_at_utc": _now_utc(),
        "outputs": {"forecasts": FINN_FORECAST_OUTPUT, "manifest": FINN_MANIFEST_OUTPUT},
    }


def _run_command(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": 124, "stdout": exc.stdout or "", "stderr": exc.stderr or "command timed out"}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _resolve_rscript(rscript: str) -> str:
    resolved = shutil.which(rscript)
    if resolved:
        return resolved
    explicit = Path(rscript)
    if explicit.exists():
        return str(explicit)
    if rscript.lower() not in {"rscript", "rscript.exe"}:
        return rscript
    candidates: list[Path] = []
    for root in (
        Path("C:\\Program Files\\R"),
        Path.home() / "AppData" / "Local" / "Programs" / "R",
    ):
        if root.exists():
            candidates.extend(root.glob("R-*\\bin\\Rscript.exe"))
            candidates.extend(root.glob("R-*\\bin\\x64\\Rscript.exe"))
            candidates.extend(root.glob("*\\bin\\Rscript.exe"))
            candidates.extend(root.glob("*\\bin\\x64\\Rscript.exe"))
    unique = sorted({path.resolve() for path in candidates if path.exists()}, reverse=True)
    return str(unique[0]) if unique else rscript


def _update_json_mapping(path: Path, key_path: tuple[str, ...], values: dict[str, str]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    target = data
    for key in key_path:
        target = target.setdefault(key, {})
    if isinstance(target, dict):
        target.update(values)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _update_control_pane_state(path: Path, outputs: dict[str, str], manifest: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    artifacts = data.setdefault("artifacts", {})
    if isinstance(artifacts, dict):
        artifacts.update(outputs)
    feature_map = data.setdefault("feature_map", [])
    if isinstance(feature_map, list):
        feature_map = [
            row
            for row in feature_map
            if not (isinstance(row, dict) and str(row.get("mechanism", "")).lower() == "finn advisory bridge")
        ]
        feature_map.append(
            {
                "mechanism": "FINN advisory bridge",
                "status": "available",
                "artifact": "finn/finn_manifest.json / finn/external_model_metrics.csv",
                "advisory_only": bool(manifest.get("advisory_only", True)),
            }
        )
        data["feature_map"] = feature_map
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _posix_path(path: Path) -> str:
    text = path.as_posix()
    return "" if text == "." else text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
