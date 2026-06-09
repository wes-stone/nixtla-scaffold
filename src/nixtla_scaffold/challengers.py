"""Generic external challenger lane run beside the canonical scaffold pipeline.

Challengers are spec-driven external engines (FINN/finnts first) that produce
advisory forecasts which are compared and cutoff-scored against the canonical
run without ever mutating ``forecast.csv``. Engines plug in through
``register_challenger_engine`` so future engines reuse the same lifecycle,
artifacts, soft-fail semantics, and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import pandas as pd

from nixtla_scaffold.finn_bridge import (
    check_finn_environment,
    compare_finn_forecasts,
    score_finn_forecasts,
    _resolve_rscript,
    _run_command,
)
from nixtla_scaffold.schema import ChallengerSpec, forecast_spec_from_dict

CHALLENGERS_SCHEMA_VERSION = "nixtla_scaffold.challengers.v1"
CHALLENGER_STATUS_OUTPUT = "challenger_status.json"
CHALLENGER_AGENT_BRIEF_OUTPUT = "agent_brief.json"
CHALLENGER_LEADERBOARD_OUTPUT = "appendix/challenger_leaderboard.csv"
FINN_SPEC_RUNNER_OUTPUT = "finn_spec_runner.R"
FINN_PARAMS_OUTPUT = "finn_params.json"
FINN_INPUT_OUTPUT = "finn_input.csv"
FINN_RAW_FUTURE_OUTPUT = "finn_raw_future.csv"
FINN_RAW_BACKTEST_OUTPUT = "finn_raw_backtest.csv"

_LEADERBOARD_METRIC_COLUMNS = ["rmse", "mae", "wape", "mase", "rmsse", "bias", "abs_bias", "observations"]


class ChallengerSkip(RuntimeError):
    """Raised when a challenger cannot run; carries a remediation hint."""

    def __init__(self, reason: str, remediation: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.remediation = remediation


@dataclass(frozen=True)
class ChallengerForecasts:
    """Canonical long forecasts produced by an engine (unique_id, ds, model, yhat, cutoff)."""

    future: pd.DataFrame
    backtest: pd.DataFrame


class ChallengerEngine(Protocol):
    name: str

    def check_environment(self, spec: ChallengerSpec) -> dict[str, Any]: ...

    def run(
        self,
        *,
        history: pd.DataFrame,
        spec: ChallengerSpec,
        run_spec: dict[str, Any],
        output_dir: Path,
    ) -> ChallengerForecasts: ...


_ENGINE_REGISTRY: dict[str, Callable[[], ChallengerEngine]] = {}


def register_challenger_engine(name: str, factory: Callable[[], ChallengerEngine]) -> None:
    _ENGINE_REGISTRY[str(name).strip().lower()] = factory


def available_challenger_engines() -> tuple[str, ...]:
    return tuple(sorted(_ENGINE_REGISTRY))


def get_challenger_engine(name: str) -> ChallengerEngine:
    key = str(name).strip().lower()
    if key not in _ENGINE_REGISTRY:
        raise ChallengerSkip(
            f"unknown challenger engine '{name}'",
            remediation=f"register the engine with register_challenger_engine or use one of: {', '.join(available_challenger_engines()) or '(none)'}",
        )
    return _ENGINE_REGISTRY[key]()


def run_challengers(
    run_dir: str | Path,
    challengers: tuple[ChallengerSpec, ...] | None = None,
) -> dict[str, Any]:
    """Run every enabled challenger against a completed scaffold run.

    Soft-fail by default: failures are recorded in ``<source_id>/challenger_status.json``
    and the pipeline continues unless the spec sets ``on_error="fail"``.
    """

    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"scaffold run manifest not found: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_spec = run_manifest.get("spec", {}) if isinstance(run_manifest.get("spec"), dict) else {}
    if challengers is None:
        challengers = forecast_spec_from_dict(run_spec).challengers

    history_path = run_path / "appendix" / "history.csv"
    statuses: list[dict[str, Any]] = []
    for challenger in challengers:
        statuses.append(_run_single_challenger(run_path, challenger, run_spec, history_path))

    payload = {
        "schema_version": CHALLENGERS_SCHEMA_VERSION,
        "run_dir": str(run_path),
        "challengers": statuses,
        "completed": sum(1 for status in statuses if status["status"] == "completed"),
        "skipped": sum(1 for status in statuses if status["status"] in {"skipped", "failed", "disabled"}),
        "created_at_utc": _now_utc(),
    }
    if any(status["status"] == "completed" for status in statuses):
        leaderboard = build_challenger_leaderboard(run_path)
        if not leaderboard.empty:
            leaderboard_path = run_path / CHALLENGER_LEADERBOARD_OUTPUT
            leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
            leaderboard.to_csv(leaderboard_path, index=False)
            payload["challenger_leaderboard"] = CHALLENGER_LEADERBOARD_OUTPUT
            _register_run_output(run_path, "challenger_leaderboard", CHALLENGER_LEADERBOARD_OUTPUT)
    return payload


def _run_single_challenger(
    run_path: Path,
    challenger: ChallengerSpec,
    run_spec: dict[str, Any],
    history_path: Path,
) -> dict[str, Any]:
    out = run_path / challenger.source_id
    status: dict[str, Any] = {
        "schema_version": CHALLENGERS_SCHEMA_VERSION,
        "engine": challenger.engine,
        "source_id": challenger.source_id,
        "model_name": challenger.model_name,
        "on_error": challenger.on_error,
        "spec": challenger.to_dict(),
        "advisory_only": True,
        "created_at_utc": _now_utc(),
    }
    if not challenger.enabled:
        status.update({"status": "disabled", "reason": "challenger disabled in spec", "remediation": "set challengers[].enabled = true to run this engine"})
        _write_status(run_path, out, status)
        return status

    started = time.perf_counter()
    try:
        engine = get_challenger_engine(challenger.engine)
        environment = engine.check_environment(challenger)
        status["environment"] = environment
        if not environment.get("available", False):
            raise ChallengerSkip(
                environment.get("reason", f"{challenger.engine} environment is not available"),
                remediation=environment.get("install_hint", ""),
            )
        if not history_path.exists():
            raise ChallengerSkip(
                f"canonical history not found at {history_path}",
                remediation="regenerate the run with the current scaffold so appendix/history.csv exists",
            )
        history = pd.read_csv(history_path)
        out.mkdir(parents=True, exist_ok=True)
        forecasts = engine.run(history=history, spec=challenger, run_spec=run_spec, output_dir=out)
        artifacts = _attach_challenger_artifacts(run_path, out, challenger, run_spec, history_path, forecasts)
        status.update(
            {
                "status": "completed",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "future_rows": int(len(forecasts.future)),
                "backtest_rows": int(len(forecasts.backtest)),
                "outputs": artifacts,
            }
        )
    except ChallengerSkip as skip:
        status.update(
            {
                "status": "skipped",
                "reason": str(skip),
                "remediation": skip.remediation,
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        )
        if challenger.on_error == "fail":
            _write_status(run_path, out, status)
            raise
    except Exception as error:  # soft-fail lane: record and continue unless on_error=fail
        status.update(
            {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "remediation": "inspect challenger_status.json and the engine logs under the challenger output folder",
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        )
        if challenger.on_error == "fail":
            _write_status(run_path, out, status)
            raise
    _write_status(run_path, out, status)
    _write_agent_brief(run_path, out, challenger, status)
    return status


def _attach_challenger_artifacts(
    run_path: Path,
    out: Path,
    challenger: ChallengerSpec,
    run_spec: dict[str, Any],
    history_path: Path,
    forecasts: ChallengerForecasts,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    common = {"model_name": challenger.model_name, "source_id": challenger.source_id, "format": "long"}
    if not forecasts.future.empty:
        compare_finn_forecasts(run_path, forecasts.future, output_dir=out, **common)
        artifacts["forecast_comparison"] = str(out / "forecast_comparison.csv")
        artifacts["manifest"] = str(out / "finn_manifest.json")
    if not forecasts.backtest.empty:
        score_finn_forecasts(
            forecasts.backtest,
            history_path,
            output_dir=out,
            run_dir=run_path,
            season_length=int(run_spec.get("season_length") or 1),
            requested_horizon=int(run_spec.get("horizon") or 0) or None,
            **common,
        )
        artifacts["external_model_metrics"] = str(out / "external_model_metrics.csv")
        artifacts["external_backtest_long"] = str(out / "external_backtest_long.csv")
    return artifacts


def build_challenger_leaderboard(run_dir: str | Path) -> pd.DataFrame:
    """Merge native backtest metrics with challenger cutoff-scored metrics into one lane-labeled table."""

    run_path = Path(run_dir)
    frames: list[pd.DataFrame] = []
    native_path = run_path / "audit" / "backtest_metrics.csv"
    if native_path.exists():
        native = pd.read_csv(native_path)
        if not native.empty:
            native["lane"] = "native"
            native["source_id"] = "scaffold"
            native["comparable"] = native.get("cv_horizon_matches_requested", True)
            native["cutoff_count"] = native.get("cv_windows")
            frames.append(native)
    for metrics_path in sorted(run_path.glob("*/external_model_metrics.csv")):
        external = pd.read_csv(metrics_path)
        if external.empty:
            continue
        external["lane"] = "challenger"
        if "source_id" not in external.columns:
            external["source_id"] = metrics_path.parent.name
        if "scoring_evidence_status" in external.columns:
            external["comparable"] = external["scoring_evidence_status"] == "external_cutoff_scored"
        else:
            external["comparable"] = False
        frames.append(external)
    if not frames:
        return pd.DataFrame()
    columns = ["unique_id", "model", "lane", "source_id", "comparable", "cutoff_count", *_LEADERBOARD_METRIC_COLUMNS]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    leaderboard = combined[columns].copy()
    leaderboard = leaderboard.sort_values(["unique_id", "rmse"], kind="stable").reset_index(drop=True)
    leaderboard["lane_rank"] = leaderboard.groupby(["unique_id", "lane"])["rmse"].rank(method="first")
    leaderboard["overall_rank"] = leaderboard.groupby("unique_id")["rmse"].rank(method="first")
    return leaderboard


def _write_status(run_path: Path, out: Path, status: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / CHALLENGER_STATUS_OUTPUT).write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")
    relative = f"{out.name}/{CHALLENGER_STATUS_OUTPUT}"
    _register_run_output(run_path, f"{status['source_id']}_challenger_status", relative)


def _write_agent_brief(run_path: Path, out: Path, challenger: ChallengerSpec, status: dict[str, Any]) -> None:
    brief = {
        "schema_version": CHALLENGERS_SCHEMA_VERSION,
        "engine": challenger.engine,
        "source_id": challenger.source_id,
        "status": status.get("status"),
        "reason": status.get("reason", ""),
        "remediation": status.get("remediation", ""),
        "advisory_only": True,
        "comparable_evidence": f"{challenger.source_id}/external_model_metrics.csv"
        if (out / "external_model_metrics.csv").exists()
        else None,
        "directional_comparison": f"{challenger.source_id}/forecast_comparison.csv"
        if (out / "forecast_comparison.csv").exists()
        else None,
        "leaderboard": CHALLENGER_LEADERBOARD_OUTPUT if (run_path / CHALLENGER_LEADERBOARD_OUTPUT).exists() else None,
        "guidance": [
            "Challenger forecasts are advisory and never mutate forecast.csv.",
            "Use external_model_metrics.csv (cutoff-scored) for apples-to-apples accuracy claims; forecast_comparison.csv is directional only.",
            "appendix/challenger_leaderboard.csv merges native and challenger lanes; champion selection ignores challenger rows.",
        ],
        "next_commands": [
            f"nixtla-scaffold finn pipeline --run {run_path}",
            f"nixtla-scaffold report --run {run_path}",
        ],
        "created_at_utc": _now_utc(),
    }
    (out / CHALLENGER_AGENT_BRIEF_OUTPUT).write_text(json.dumps(brief, indent=2, default=str) + "\n", encoding="utf-8")
    _register_run_output(run_path, f"{challenger.source_id}_agent_brief", f"{out.name}/{CHALLENGER_AGENT_BRIEF_OUTPUT}")


def _register_run_output(run_path: Path, key: str, relative_path: str) -> None:
    manifest_path = run_path / "manifest.json"
    llm_context_path = run_path / "llm_context.json"
    for path, mapping_key in ((manifest_path, "outputs"), (llm_context_path, "artifact_index")):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping = data.get(mapping_key)
        if not isinstance(mapping, dict):
            mapping = {}
            data[mapping_key] = mapping
        mapping[key] = relative_path
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


class FinnChallengerEngine:
    """FINN/finnts engine: generates an auditable R runner from the spec and runs it."""

    name = "finn"

    def check_environment(self, spec: ChallengerSpec) -> dict[str, Any]:
        manifest = check_finn_environment(rscript=spec.rscript).manifest
        manifest["reason"] = (
            ""
            if manifest.get("available")
            else "R/Rscript or the finnts package is not available"
        )
        return manifest

    def run(
        self,
        *,
        history: pd.DataFrame,
        spec: ChallengerSpec,
        run_spec: dict[str, Any],
        output_dir: Path,
    ) -> ChallengerForecasts:
        output_dir.mkdir(parents=True, exist_ok=True)
        date_type = _finn_date_type(str(run_spec.get("freq") or ""), history)
        params = {
            "schema_version": CHALLENGERS_SCHEMA_VERSION,
            "horizon": int(run_spec.get("horizon") or 12),
            "date_type": date_type,
            "models_to_run": list(spec.models),
            "back_test_scenarios": spec.back_test_scenarios,
            "back_test_spacing": spec.back_test_spacing,
            "forecast_approach": spec.forecast_approach,
            "run_ensemble_models": spec.run_ensemble_models,
            "feature_selection": spec.feature_selection,
            "seed": spec.seed,
            "run_name": "scaffold_challenger",
            "extra": {key: value for key, value in spec.extra},
        }
        params_path = output_dir / FINN_PARAMS_OUTPUT
        params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        runner_path = output_dir / FINN_SPEC_RUNNER_OUTPUT
        runner_path.write_text(build_finn_spec_runner(), encoding="utf-8")
        input_path = output_dir / FINN_INPUT_OUTPUT
        history[["unique_id", "ds", "y"]].to_csv(input_path, index=False)
        future_path = output_dir / FINN_RAW_FUTURE_OUTPUT
        backtest_path = output_dir / FINN_RAW_BACKTEST_OUTPUT

        command = [
            _resolve_rscript(spec.rscript),
            str(runner_path),
            "--input",
            str(input_path),
            "--params",
            str(params_path),
            "--future-output",
            str(future_path),
            "--backtest-output",
            str(backtest_path),
        ]
        result = _run_command(command, timeout_seconds=spec.timeout_seconds)
        (output_dir / "finn_spec_runner_log.json").write_text(
            json.dumps({"command": command, **result}, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if result["returncode"] != 0:
            stderr_tail = "\n".join(str(result.get("stderr", "")).splitlines()[-12:])
            raise RuntimeError(f"FINN spec runner failed with exit code {result['returncode']}:\n{stderr_tail}")
        if not future_path.exists():
            raise RuntimeError(f"FINN spec runner completed but did not write {future_path}")

        history_ds = pd.to_datetime(history["ds"])
        future = _canonicalize_engine_frame(pd.read_csv(future_path), spec, history_ds)
        future["cutoff"] = history_ds.max()
        backtest = pd.DataFrame()
        if backtest_path.exists():
            backtest = _canonicalize_engine_frame(pd.read_csv(backtest_path), spec, history_ds, with_cutoff_from_horizon=True)
        return ChallengerForecasts(future=future, backtest=backtest)


def _finn_date_type(freq: str, history: pd.DataFrame) -> str:
    code = (freq or "").upper()
    if not code:
        inferred = pd.infer_freq(pd.to_datetime(history["ds"]).drop_duplicates().sort_values())
        code = (inferred or "M").upper()
    if code.startswith(("A", "Y")):
        return "year"
    if code.startswith("Q"):
        return "quarter"
    if code.startswith("M"):
        return "month"
    if code.startswith("W"):
        return "week"
    return "day"


def _period_alias(history_ds: pd.Series) -> str | None:
    inferred = pd.infer_freq(history_ds.drop_duplicates().sort_values())
    if not inferred:
        return None
    base = inferred.split("-")[0].upper()
    for prefix, alias in (("A", "Y"), ("Y", "Y"), ("Q", "Q"), ("M", "M"), ("W", "W"), ("B", "D"), ("D", "D")):
        if base.startswith(prefix):
            return alias
    return None


def _history_period_convention(history_ds: pd.Series, alias: str) -> str:
    periods = history_ds.dt.to_period(alias)
    if (history_ds.dt.normalize() == periods.dt.end_time.dt.normalize()).all():
        return "end"
    if (history_ds.dt.normalize() == periods.dt.start_time.dt.normalize()).all():
        return "start"
    return "end"


def _snap_periods(periods: pd.Series, convention: str) -> pd.Series:
    if convention == "start":
        return periods.dt.start_time.dt.normalize()
    return periods.dt.end_time.dt.normalize()


def _canonicalize_engine_frame(
    frame: pd.DataFrame,
    spec: ChallengerSpec,
    history_ds: pd.Series,
    *,
    with_cutoff_from_horizon: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "model", "yhat", "cutoff"])
    rename = {column: column.lower() for column in frame.columns}
    out = frame.rename(columns=rename)
    if "date" in out.columns and "ds" not in out.columns:
        out = out.rename(columns={"date": "ds"})
    dates = pd.to_datetime(out["ds"])
    alias = _period_alias(history_ds)
    if alias:
        convention = _history_period_convention(history_ds, alias)
        periods = dates.dt.to_period(alias)
        out["ds"] = _snap_periods(periods, convention)
        if with_cutoff_from_horizon and "horizon" in out.columns:
            horizon = pd.to_numeric(out["horizon"], errors="coerce").fillna(1).astype(int)
            out["cutoff"] = _snap_periods(periods - horizon, convention)
    else:
        out["ds"] = dates
        if with_cutoff_from_horizon and "horizon" in out.columns:
            raise ChallengerSkip(
                "could not infer a period alias from canonical history to label challenger backtest cutoffs",
                remediation="provide an explicit freq in the spec so cutoffs can be derived apples-to-apples",
            )
    out["unique_id"] = out["unique_id"].astype(str)
    out["model"] = out.get("model", pd.Series(["selected"] * len(out))).fillna("selected").astype(str)
    out["model"] = out["model"].map(lambda value: _clean_model_label(spec.model_name, value))
    out["yhat"] = pd.to_numeric(out["yhat"], errors="coerce")
    columns = ["unique_id", "ds", "model", "yhat"]
    if "cutoff" in out.columns:
        columns.append("cutoff")
    return out[columns].dropna(subset=["yhat"]).reset_index(drop=True)


def _clean_model_label(prefix: str, value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value).strip()) or "selected"
    if cleaned.startswith(f"{prefix}_"):
        return cleaned
    return f"{prefix}_{cleaned}"


def build_finn_spec_runner() -> str:
    """Static, auditable R runner driven entirely by finn_params.json."""

    return """#!/usr/bin/env Rscript
# Generated by nixtla-scaffold challengers; behavior is driven by finn_params.json.
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

input_path <- get_arg("--input")
params_path <- get_arg("--params")
future_output <- get_arg("--future-output", "finn_raw_future.csv")
backtest_output <- get_arg("--backtest-output", "finn_raw_backtest.csv")
if (is.null(input_path)) stop("--input is required")
if (is.null(params_path)) stop("--params is required")
if (nzchar(Sys.getenv("R_LIBS_USER"))) {
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
}
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install the jsonlite R package")
if (!requireNamespace("finnts", quietly = TRUE)) stop("Install the FINN/finnts R package")
suppressPackageStartupMessages(library(finnts))

params <- jsonlite::fromJSON(params_path)
set.seed(as.integer(params$seed))

history <- read.csv(input_path)
history$Date <- as.Date(cut(as.Date(history$ds), breaks = params$date_type))
input_data <- data.frame(
  id = as.character(history$unique_id),
  Date = history$Date,
  value = as.numeric(history$y),
  stringsAsFactors = FALSE
)

fcst_args <- list(
  input_data = input_data,
  combo_variables = c("id"),
  target_variable = "value",
  date_type = params$date_type,
  forecast_horizon = as.numeric(params$horizon),
  forecast_approach = params$forecast_approach,
  run_ensemble_models = isTRUE(params$run_ensemble_models),
  feature_selection = isTRUE(params$feature_selection),
  return_data = TRUE,
  run_name = params$run_name
)
if (length(params$models_to_run) > 0) fcst_args$models_to_run <- params$models_to_run
if (!is.null(params$back_test_scenarios)) fcst_args$back_test_scenarios <- as.numeric(params$back_test_scenarios)
if (!is.null(params$back_test_spacing)) fcst_args$back_test_spacing <- as.numeric(params$back_test_spacing)
for (key in names(params$extra)) fcst_args[[key]] <- params$extra[[key]]

result <- do.call(finnts::forecast_time_series, fcst_args)

final_fcst <- as.data.frame(result$final_fcst)
future <- final_fcst[final_fcst$Type != "Historical", , drop = FALSE]
write.csv(
  data.frame(
    unique_id = future$Combo,
    ds = as.character(future$Date),
    model = ifelse(is.na(future$Model) | future$Model == "", "selected", future$Model),
    yhat = as.numeric(future$value),
    stringsAsFactors = FALSE
  ),
  future_output,
  row.names = FALSE
)

backtest <- as.data.frame(result$back_test_data)
if (nrow(backtest) > 0) {
  write.csv(
    data.frame(
      unique_id = backtest$Combo,
      ds = as.character(backtest$Date),
      model = ifelse(is.na(backtest$Model) | backtest$Model == "", "selected", backtest$Model),
      yhat = as.numeric(backtest$FCST),
      horizon = as.integer(backtest$Horizon),
      stringsAsFactors = FALSE
    ),
    backtest_output,
    row.names = FALSE
  )
}
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


register_challenger_engine("finn", FinnChallengerEngine)
