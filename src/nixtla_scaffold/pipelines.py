from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nixtla_scaffold.connectors import read_query_result_source
from nixtla_scaffold.data import canonicalize_forecast_frame
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.ops import write_operational_receipts
from nixtla_scaffold.presets import forecast_spec_preset
from nixtla_scaffold.refresh import write_refresh_artifacts
from nixtla_scaffold.schema import ForecastSpec, forecast_spec_from_dict

PIPELINE_SCHEMA_VERSION = "nixtla_scaffold.pipeline.v1"
PIPELINE_MANIFEST_FILE = "pipeline_manifest.json"
PIPELINE_SUMMARY_FILE = "pipeline_summary.md"

_OUTPUT_REF_RE = re.compile(r"\{output:([^}]+)\}")
_SPEC_FIELD_NAMES = set(ForecastSpec.__dataclass_fields__)


def run_pipeline(
    config: str | Path,
    output_dir: str | Path | None = None,
    *,
    forecast: bool = True,
) -> dict[str, Any]:
    """Run script-backed extracts into one canonical forecast input."""

    return _run_pipeline(config, output_dir, forecast=forecast, previous_run=None)


def refresh_pipeline(
    config: str | Path,
    previous_run: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Rerun a source pipeline and refresh from a previous forecast manifest spec."""

    return _run_pipeline(config, output_dir, forecast=True, previous_run=previous_run)


def _run_pipeline(
    config: str | Path,
    output_dir: str | Path | None,
    *,
    forecast: bool,
    previous_run: str | Path | None,
) -> dict[str, Any]:
    config_path = Path(config).resolve()
    config_dir = config_path.parent.resolve()
    payload = _load_pipeline_config(config_path)
    out = (_default_output_dir(payload) if output_dir is None else Path(output_dir)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = _base_manifest(payload, config_path, out, previous_run=previous_run)
    output_refs: dict[str, Path] = {}
    try:
        _run_extracts(payload, config_dir=config_dir, output_dir=out, output_refs=output_refs, manifest=manifest)
        final_source = _run_transform(payload, config_dir=config_dir, output_dir=out, output_refs=output_refs, manifest=manifest)
        canonical_path = _write_canonical_input(payload, final_source, output_dir=out, manifest=manifest)
        output_refs["forecast_input"] = canonical_path
        forecast_result = None
        if forecast and _forecast_enabled(payload):
            forecast_result = _write_forecast_run(payload, canonical_path, output_dir=out, previous_run=previous_run, manifest=manifest)
        manifest["status"] = "succeeded"
        manifest["completed_at_utc"] = _utc_now()
        summary_path = _write_pipeline_summary(out, manifest)
        _write_manifest(out, manifest)
        if forecast_result is not None:
            _attach_pipeline_manifest(Path(forecast_result["forecast_output"]), out / PIPELINE_MANIFEST_FILE, summary_path, manifest)
        return manifest
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_pipeline_summary(out, manifest)
        _write_manifest(out, manifest)
        raise


def _load_pipeline_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("pipeline config must be a YAML mapping")
    schema_version = payload.get("schema_version", PIPELINE_SCHEMA_VERSION)
    if schema_version != PIPELINE_SCHEMA_VERSION:
        raise ValueError(f"unsupported pipeline schema_version: {schema_version}")
    extracts = payload.get("extracts", [])
    if not isinstance(extracts, list) or not extracts:
        raise ValueError("pipeline config requires a non-empty extracts list")
    for step in extracts:
        if not isinstance(step, dict):
            raise ValueError("each pipeline extract must be a mapping")
    transform = payload.get("transform")
    if transform is not None and not isinstance(transform, dict):
        raise ValueError("pipeline transform must be a mapping when provided")
    return payload


def _base_manifest(payload: dict[str, Any], config_path: Path, output_dir: Path, *, previous_run: str | Path | None) -> dict[str, Any]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": "running",
        "name": str(payload.get("name") or config_path.stem),
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir),
        "previous_run": str(previous_run) if previous_run is not None else "",
        "config": _file_info(config_path.resolve()),
        "steps": [],
        "outputs": {},
        "canonical_input": {},
        "forecast": {},
    }


def _run_extracts(
    payload: dict[str, Any],
    *,
    config_dir: Path,
    output_dir: Path,
    output_refs: dict[str, Path],
    manifest: dict[str, Any],
) -> None:
    for index, step in enumerate(payload.get("extracts", []), start=1):
        name = _step_name(step, default=f"extract_{index}")
        outputs = _step_outputs(step, name=name, output_dir=output_dir)
        _ensure_no_output_conflict(output_refs, outputs)
        output_refs.update(outputs)
        result = _run_script_step(
            step,
            name=name,
            role="extract",
            config_dir=config_dir,
            output_dir=output_dir,
            output_refs=output_refs,
            expected_outputs=outputs,
        )
        manifest["steps"].append(result)
        if result["status"] != "succeeded":
            raise RuntimeError(f"pipeline extract '{name}' failed with exit code {result['exit_code']}")


def _run_transform(
    payload: dict[str, Any],
    *,
    config_dir: Path,
    output_dir: Path,
    output_refs: dict[str, Path],
    manifest: dict[str, Any],
) -> Path:
    transform = payload.get("transform")
    if transform is None:
        canonical = payload.get("canonical", {})
        source_key = canonical.get("source_output") if isinstance(canonical, dict) else None
        if source_key:
            try:
                return output_refs[str(source_key)]
            except KeyError as exc:
                raise ValueError(f"canonical.source_output references unknown output: {source_key}") from exc
        if len(output_refs) == 1:
            return next(iter(output_refs.values()))
        known = ", ".join(sorted(output_refs))
        raise ValueError(f"pipeline with multiple extracts and no transform must set canonical.source_output; known outputs: {known}")

    name = _step_name(transform, default="transform")
    output = _resolve_output_path(output_dir, transform.get("output", "prepared/forecast_input.csv"))
    outputs = {name: output, "forecast_input": output}
    _ensure_no_output_conflict(output_refs, outputs)
    transform_refs = dict(output_refs)
    transform_refs.update(outputs)
    result = _run_script_step(
        transform,
        name=name,
        role="transform",
        config_dir=config_dir,
        output_dir=output_dir,
        output_refs=transform_refs,
        expected_outputs=outputs,
    )
    manifest["steps"].append(result)
    if result["status"] != "succeeded":
        raise RuntimeError(f"pipeline transform '{name}' failed with exit code {result['exit_code']}")
    output_refs.update(outputs)
    return output


def _run_script_step(
    step: Mapping[str, Any],
    *,
    name: str,
    role: str,
    config_dir: Path,
    output_dir: Path,
    output_refs: Mapping[str, Path],
    expected_outputs: Mapping[str, Path],
) -> dict[str, Any]:
    script = _resolve_config_path(config_dir, _required_str(step, "script", step_name=name))
    if not script.exists():
        raise FileNotFoundError(script)
    if script.suffix.lower() != ".py":
        raise ValueError(f"pipeline {role} '{name}' must reference a Python script: {script}")
    query_files = _query_files(step, config_dir)
    args = _resolve_args(step.get("args", []), config_dir=config_dir, output_dir=output_dir, output_refs=output_refs, query_files=query_files)
    command = [sys.executable, str(script), *args]
    started = _utc_now()
    result = _execute_script(command, cwd=output_dir, timeout_seconds=_timeout_seconds(step))
    completed = _utc_now()
    outputs = {key: _describe_output(path) for key, path in expected_outputs.items()}
    status = "succeeded" if result["exit_code"] == 0 and all(item["exists"] for item in outputs.values()) else "failed"
    if status == "failed" and result["exit_code"] == 0:
        missing = [key for key, item in outputs.items() if not item["exists"]]
        result["stderr"] = (result["stderr"] + f"\nMissing declared output(s): {', '.join(missing)}").strip()
    return {
        "name": name,
        "role": role,
        "status": status,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "script": _file_info(script),
        "query_files": [_file_info(path) for path in query_files],
        "command": command,
        "exit_code": result["exit_code"],
        "stdout_tail": result["stdout"][-4000:],
        "stderr_tail": result["stderr"][-4000:],
        "outputs": outputs,
    }


def _write_canonical_input(payload: dict[str, Any], source: Path, *, output_dir: Path, manifest: dict[str, Any]) -> Path:
    canonical_config = payload.get("canonical", {})
    if canonical_config is None:
        canonical_config = {}
    if not isinstance(canonical_config, dict):
        raise ValueError("pipeline canonical config must be a mapping")
    canonical_path = _resolve_output_path(output_dir, canonical_config.get("output", "prepared/forecast_input.csv"))
    frame = read_query_result_source(source)
    canonical = canonicalize_forecast_frame(
        frame,
        id_col=str(canonical_config.get("id_col", "unique_id")),
        time_col=str(canonical_config.get("time_col", "ds")),
        target_col=str(canonical_config.get("target_col", "y")),
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(canonical_path, index=False)
    metadata = {
        "source_kind": "pipeline",
        "pipeline_name": manifest["name"],
        "pipeline_manifest": str(Path(manifest["output_dir"]) / PIPELINE_MANIFEST_FILE),
        "source_output": str(source),
        "output": str(canonical_path),
        "columns": {
            "id_col": str(canonical_config.get("id_col", "unique_id")),
            "time_col": str(canonical_config.get("time_col", "ds")),
            "target_col": str(canonical_config.get("target_col", "y")),
        },
        "rows": int(len(canonical)),
        "series_count": int(canonical["unique_id"].nunique()),
        "start": canonical["ds"].min().date().isoformat(),
        "end": canonical["ds"].max().date().isoformat(),
    }
    metadata_path = canonical_path.with_suffix(".source.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    manifest["canonical_input"] = {
        **metadata,
        "metadata_file": str(metadata_path),
        "file": _describe_output(canonical_path),
    }
    manifest["outputs"]["canonical_input"] = str(canonical_path)
    manifest["outputs"]["canonical_source_metadata"] = str(metadata_path)
    return canonical_path


def _write_forecast_run(
    payload: dict[str, Any],
    canonical_path: Path,
    *,
    output_dir: Path,
    previous_run: str | Path | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    forecast_config = payload.get("forecast", {}) or {}
    if not isinstance(forecast_config, dict):
        raise ValueError("pipeline forecast config must be a mapping")
    forecast_dir = _resolve_output_path(output_dir, forecast_config.get("output", "forecast"))
    spec = _refresh_spec(previous_run) if previous_run is not None else _forecast_spec_from_config(forecast_config)
    run = run_forecast(canonical_path, spec)
    run_output = run.to_directory(forecast_dir)
    challenger_payload = None
    if any(challenger.enabled for challenger in spec.challengers):
        from nixtla_scaffold.challengers import run_challengers

        challenger_payload = run_challengers(run_output, challengers=spec.challengers)
    refresh_result = None
    if previous_run is not None:
        refresh_result = write_refresh_artifacts(previous_run, run_output)
    manifest["forecast"] = {
        "enabled": True,
        "mode": "refresh" if previous_run is not None else "run",
        "forecast_output": str(run_output),
        "spec": spec.to_dict(),
        "refresh": refresh_result or {},
        "challengers": challenger_payload or {},
    }
    manifest["outputs"]["forecast_run"] = str(run_output)
    return manifest["forecast"]


def _forecast_spec_from_config(forecast_config: Mapping[str, Any]) -> ForecastSpec:
    preset = forecast_config.get("preset")
    base = forecast_spec_preset(str(preset)) if preset else ForecastSpec()
    overrides = {key: value for key, value in forecast_config.items() if key in _SPEC_FIELD_NAMES}
    spec_data = base.to_dict() | overrides
    spec_data["id_col"] = "unique_id"
    spec_data["time_col"] = "ds"
    spec_data["target_col"] = "y"
    return forecast_spec_from_dict(spec_data)


def _refresh_spec(previous_run: str | Path | None) -> ForecastSpec:
    if previous_run is None:
        raise ValueError("previous_run is required for pipeline refresh")
    manifest_path = Path(previous_run) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = forecast_spec_from_dict(manifest.get("spec", {}))
    return forecast_spec_from_dict(spec.to_dict() | {"id_col": "unique_id", "time_col": "ds", "target_col": "y"})


def _attach_pipeline_manifest(forecast_dir: Path, pipeline_manifest_path: Path, pipeline_summary_path: Path, manifest: dict[str, Any]) -> None:
    appendix = forecast_dir / "appendix"
    appendix.mkdir(parents=True, exist_ok=True)
    manifest_target = appendix / "source_pipeline_manifest.json"
    manifest_target.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    summary_target = appendix / "source_pipeline_summary.md"
    summary_target.write_text(pipeline_summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    run_manifest_path = forecast_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest.setdefault("outputs", {})["source_pipeline_manifest"] = "appendix/source_pipeline_manifest.json"
    run_manifest["outputs"]["source_pipeline_summary"] = "appendix/source_pipeline_summary.md"
    run_manifest["source_pipeline"] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "name": manifest.get("name"),
        "pipeline_manifest": str(pipeline_manifest_path),
        "pipeline_summary": str(pipeline_summary_path),
        "canonical_input": manifest.get("canonical_input", {}).get("output", ""),
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, default=str) + "\n", encoding="utf-8")
    write_operational_receipts(forecast_dir)


def _write_pipeline_summary(output_dir: Path, manifest: dict[str, Any]) -> Path:
    summary_path = output_dir / PIPELINE_SUMMARY_FILE
    manifest.setdefault("outputs", {})["pipeline_summary"] = str(summary_path)
    summary_path.write_text(_pipeline_summary_markdown(manifest), encoding="utf-8")
    return summary_path


def _pipeline_summary_markdown(manifest: Mapping[str, Any]) -> str:
    output_dir = Path(str(manifest.get("output_dir", "")))
    lines = [
        f"# Source pipeline: {_md_text(manifest.get('name', 'pipeline'))}",
        "",
        f"Status: `{_md_text(manifest.get('status', 'unknown'))}`",
        "",
        "## Flow",
        "",
        "```mermaid",
        *_pipeline_mermaid_lines(manifest),
        "```",
        "",
        "## Run summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Config | `{_md_cell(_display_path(manifest.get('config_path', ''), output_dir))}` |",
        f"| Output directory | `{_md_cell(output_dir)}` |",
        f"| Generated | `{_md_cell(manifest.get('generated_at_utc', ''))}` |",
        f"| Completed | `{_md_cell(manifest.get('completed_at_utc', ''))}` |",
        "",
        "## Steps",
        "",
        "| # | Role | Name | Status | Outputs |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, step in enumerate(manifest.get("steps", []), start=1):
        if isinstance(step, Mapping):
            lines.append(
                f"| {index} | {_md_cell(step.get('role', ''))} | {_md_cell(step.get('name', ''))} | "
                f"{_md_cell(step.get('status', ''))} | {_md_cell(_step_outputs_summary(step))} |"
            )
    if not manifest.get("steps"):
        lines.append("|  |  |  |  | No script steps completed. |")

    canonical = manifest.get("canonical_input", {})
    if isinstance(canonical, Mapping) and canonical:
        lines.extend(
            [
                "",
                "## Canonical input",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| File | `{_md_cell(_display_path(canonical.get('output', ''), output_dir))}` |",
                f"| Rows | {_md_cell(canonical.get('rows', ''))} |",
                f"| Series | {_md_cell(canonical.get('series_count', ''))} |",
                f"| Date range | {_md_cell(canonical.get('start', ''))} to {_md_cell(canonical.get('end', ''))} |",
            ]
        )

    forecast = manifest.get("forecast", {})
    if isinstance(forecast, Mapping) and forecast:
        spec = forecast.get("spec", {}) if isinstance(forecast.get("spec"), Mapping) else {}
        lines.extend(
            [
                "",
                "## Forecast",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Mode | {_md_cell(forecast.get('mode', ''))} |",
                f"| Output | `{_md_cell(_display_path(forecast.get('forecast_output', ''), output_dir))}` |",
                f"| Horizon | {_md_cell(spec.get('horizon', ''))} |",
                f"| Frequency | `{_md_cell(spec.get('freq', ''))}` |",
                f"| Model policy | `{_md_cell(spec.get('model_policy', ''))}` |",
            ]
        )

    error = manifest.get("error", {})
    if isinstance(error, Mapping) and error:
        lines.extend(
            [
                "",
                "## Error",
                "",
                f"- Type: `{_md_text(error.get('type', ''))}`",
                f"- Message: {_md_text(error.get('message', ''))}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _pipeline_mermaid_lines(manifest: Mapping[str, Any]) -> list[str]:
    lines = ["flowchart LR", '  config["Pipeline config"]']
    extract_ids: list[str] = []
    transform_ids: list[str] = []
    last_step_ids: list[str] = ["config"]
    for index, step in enumerate(manifest.get("steps", []), start=1):
        if not isinstance(step, Mapping):
            continue
        step_id = f"step{index}"
        role = str(step.get("role", "step"))
        lines.append(f'  {step_id}["{_mermaid_step_label(step)}"]')
        if role == "extract":
            lines.append(f"  config --> {step_id}")
            extract_ids.append(step_id)
        elif role == "transform":
            inputs = extract_ids or last_step_ids
            for input_id in inputs:
                lines.append(f"  {input_id} --> {step_id}")
            transform_ids.append(step_id)
        else:
            for input_id in last_step_ids:
                lines.append(f"  {input_id} --> {step_id}")
        last_step_ids = [step_id]

    canonical = manifest.get("canonical_input", {})
    if isinstance(canonical, Mapping) and canonical:
        lines.append(f'  canonical["{_mermaid_canonical_label(canonical)}"]')
        for input_id in (transform_ids[-1:] or extract_ids or last_step_ids):
            lines.append(f"  {input_id} --> canonical")
        forecast = manifest.get("forecast", {})
        if isinstance(forecast, Mapping) and forecast:
            lines.append(f'  forecast["{_mermaid_forecast_label(forecast)}"]')
            lines.append("  canonical --> forecast")
        else:
            lines.append('  prepared["Prepared input"]')
            lines.append("  canonical --> prepared")
    else:
        terminal = "failed" if manifest.get("status") == "failed" else "pending"
        lines.append(f'  terminal["{terminal}: no canonical input"]')
        for input_id in last_step_ids:
            lines.append(f"  {input_id} --> terminal")
    return lines


def _mermaid_step_label(step: Mapping[str, Any]) -> str:
    parts = [f"{step.get('role', 'step')}: {step.get('name', '')}", str(step.get("status", "unknown"))]
    output_summary = _step_outputs_summary(step)
    if output_summary:
        parts.append(output_summary)
    return _mermaid_label("<br/>".join(parts))


def _mermaid_canonical_label(canonical: Mapping[str, Any]) -> str:
    return _mermaid_label(f"canonical input<br/>{canonical.get('rows', '?')} rows<br/>{canonical.get('series_count', '?')} series")


def _mermaid_forecast_label(forecast: Mapping[str, Any]) -> str:
    spec = forecast.get("spec", {}) if isinstance(forecast.get("spec"), Mapping) else {}
    return _mermaid_label(f"forecast run<br/>h={spec.get('horizon', '?')}<br/>{forecast.get('mode', 'run')}")


def _step_outputs_summary(step: Mapping[str, Any]) -> str:
    outputs = step.get("outputs", {})
    if not isinstance(outputs, Mapping) or not outputs:
        return ""
    parts: list[str] = []
    for name, output in outputs.items():
        if isinstance(output, Mapping) and "rows" in output:
            parts.append(f"{name} ({output.get('rows')} rows)")
        else:
            parts.append(str(name))
    return ", ".join(parts)


def _display_path(value: Any, base: Path) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value))
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _mermaid_label(value: str) -> str:
    return str(value).replace('"', "'").replace("\n", "<br/>")


def _md_cell(value: Any) -> str:
    return _md_text(value).replace("|", "\\|")


def _md_text(value: Any) -> str:
    return str(value).replace("\n", " ").strip()


def _step_outputs(step: Mapping[str, Any], *, name: str, output_dir: Path) -> dict[str, Path]:
    has_output = "output" in step
    has_outputs = "outputs" in step
    if has_output and has_outputs:
        raise ValueError(f"pipeline step '{name}' cannot define both output and outputs")
    if has_outputs:
        outputs = step.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"pipeline step '{name}' outputs must be a non-empty mapping")
        return {str(key): _resolve_output_path(output_dir, value) for key, value in outputs.items()}
    if has_output:
        return {name: _resolve_output_path(output_dir, step.get("output"))}
    raise ValueError(f"pipeline step '{name}' requires output or outputs")


def _resolve_args(
    args: Any,
    *,
    config_dir: Path,
    output_dir: Path,
    output_refs: Mapping[str, Path],
    query_files: Sequence[Path],
) -> list[str]:
    if args is None:
        return []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("pipeline step args must be a list of strings")
    return [
        _resolve_arg_token(item, config_dir=config_dir, output_dir=output_dir, output_refs=output_refs, query_files=query_files)
        for item in args
    ]


def _resolve_arg_token(
    value: str,
    *,
    config_dir: Path,
    output_dir: Path,
    output_refs: Mapping[str, Path],
    query_files: Sequence[Path],
) -> str:
    token = value.replace("{config_dir}", str(config_dir)).replace("{output_dir}", str(output_dir))
    if "{query_file}" in token:
        if not query_files:
            raise ValueError("pipeline arg uses {query_file}, but the step has no query_file")
        token = token.replace("{query_file}", str(query_files[0]))

    def output_replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            return str(output_refs[key])
        except KeyError as exc:
            known = ", ".join(sorted(output_refs))
            raise ValueError(f"unknown output reference {{{{output:{key}}}}}; known outputs: {known}") from exc

    return _OUTPUT_REF_RE.sub(output_replace, token)


def _query_files(step: Mapping[str, Any], config_dir: Path) -> list[Path]:
    raw = step.get("query_file", step.get("query_files", []))
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        values = raw
    else:
        raise ValueError("pipeline query_file/query_files must be a string or list of strings")
    paths = [_resolve_config_path(config_dir, value) for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return paths


def _execute_script(command: list[str], *, cwd: Path, timeout_seconds: int | None) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"exit_code": 124, "stdout": exc.stdout or "", "stderr": f"script timed out after {timeout_seconds} seconds"}
    return {"exit_code": int(completed.returncode), "stdout": completed.stdout, "stderr": completed.stderr}


def _describe_output(path: Path) -> dict[str, Any]:
    info = _file_info(path)
    if not path.exists():
        return info
    try:
        frame = read_query_result_source(path)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        info["read_error"] = str(exc)
        return info
    info["rows"] = int(len(frame))
    info["columns"] = [str(column) for column in frame.columns]
    if "unique_id" in frame.columns:
        info["series_count"] = int(frame["unique_id"].nunique())
    date_col = _date_column(frame)
    if date_col:
        dates = _parse_date_values(frame[date_col])
        if not dates.empty:
            info["date_column"] = date_col
            info["start"] = dates.min().date().isoformat()
            info["end"] = dates.max().date().isoformat()
    return info


def _file_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    info: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists and path.is_file():
        info["bytes"] = path.stat().st_size
        info["sha256"] = _sha256(path)
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_column(frame: pd.DataFrame) -> str | None:
    preferred = [
        column
        for column in frame.columns
        if str(column).lower() == "ds" or any(token in str(column).lower() for token in ("date", "month", "period", "time"))
    ]
    for column in preferred:
        parsed = _parse_date_values(frame[column])
        if not parsed.empty:
            return str(column)
    return None


def _parse_date_values(values: pd.Series) -> pd.Series:
    raw = values.dropna()
    if raw.empty:
        return pd.Series(dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(raw, errors="coerce").dropna()

    text = raw.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    yyyymmdd = text.str.fullmatch(r"\d{8}")
    if yyyymmdd.any():
        parsed = pd.to_datetime(text.where(yyyymmdd), format="%Y%m%d", errors="coerce").dropna()
        if not parsed.empty:
            return parsed

    if pd.api.types.is_numeric_dtype(raw) or text.str.fullmatch(r"\d+").all():
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(raw, errors="coerce").dropna()


def _resolve_config_path(config_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("path value cannot be empty")
    path = Path(str(value))
    return path if path.is_absolute() else config_dir / path


def _resolve_output_path(output_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("output path value cannot be empty")
    path = Path(str(value))
    return path if path.is_absolute() else output_dir / path


def _required_str(step: Mapping[str, Any], key: str, *, step_name: str) -> str:
    value = step.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pipeline step '{step_name}' requires {key}")
    return value


def _step_name(step: Mapping[str, Any], *, default: str) -> str:
    name = step.get("name", default)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("pipeline step name cannot be blank")
    return name


def _timeout_seconds(step: Mapping[str, Any]) -> int | None:
    value = step.get("timeout_seconds")
    if value is None:
        return 300
    seconds = int(value)
    if seconds < 1:
        raise ValueError("pipeline timeout_seconds must be >= 1")
    return seconds


def _forecast_enabled(payload: Mapping[str, Any]) -> bool:
    forecast = payload.get("forecast", {})
    if forecast is None:
        return True
    if not isinstance(forecast, dict):
        raise ValueError("pipeline forecast config must be a mapping")
    return bool(forecast.get("enabled", True))


def _ensure_no_output_conflict(existing: Mapping[str, Path], new: Mapping[str, Path]) -> None:
    conflicts = sorted(set(existing) & set(new))
    if conflicts:
        raise ValueError(f"duplicate pipeline output reference(s): {', '.join(conflicts)}")


def _default_output_dir(payload: Mapping[str, Any]) -> Path:
    name = _slug(str(payload.get("name") or "pipeline"))
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"pipeline_{name}_{stamp}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or "pipeline"


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PIPELINE_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
