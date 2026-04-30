from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.models import (
    ModelResult,
    _adaptive_cv_params,
    _empty_backtest_metrics,
    _empty_backtest_predictions,
    _empty_model_explainability,
    _empty_model_weights,
    _error_scale_map,
    _future_dates,
    _merge_prediction_frames,
    _metrics_from_cv,
)
from nixtla_scaffold.schema import CustomModelSpec, DataProfile, ForecastSpec


CUSTOM_MODEL_SCHEMA_VERSION = "nixtla_scaffold.custom_model.v1"

CUSTOM_MODEL_CONTRACT_COLUMNS = [
    "schema_version",
    "name",
    "model",
    "family",
    "source_id",
    "invocation_type",
    "callable_path",
    "callable_object_ref",
    "script_path",
    "timeout_seconds",
    "extra_args_json",
    "history_contract",
    "future_grid_contract",
    "output_contract",
    "leakage_guard",
    "status",
    "invocation_count",
    "error",
    "notes",
]

CUSTOM_MODEL_INVOCATION_COLUMNS = [
    "schema_version",
    "model",
    "source_id",
    "invocation_id",
    "invocation_kind",
    "invocation_type",
    "cutoff",
    "horizon",
    "history_rows",
    "future_rows",
    "output_rows",
    "status",
    "error",
    "script_path",
    "used_python_executable",
    "python_executable",
    "command_json",
]


class CustomModelExecutionError(RuntimeError):
    def __init__(self, message: str, *, invocation_rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.invocation_rows = list(invocation_rows)


def append_custom_model_result(
    base_result: ModelResult,
    *,
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    """Append the optional custom challenger to the model tournament."""

    if not spec.custom_models:
        return base_result

    custom = spec.custom_models[0]
    try:
        if spec.transform.enabled:
            raise ValueError("custom models do not support target transforms or normalization_factor_col in v1; run on raw y units instead")
        if custom.model_name in set(base_result.forecast.columns):
            raise ValueError(f"custom model name {custom.model_name!r} conflicts with an existing model column")
        custom_result = forecast_with_custom_model(history, profile, spec, custom)
    except Exception as exc:
        invocation_rows = exc.invocation_rows if isinstance(exc, CustomModelExecutionError) else []
        warning = f"custom model {custom.model_name} failed and was excluded; built-in candidates remain ({exc})"
        contracts = _custom_contract_frame(custom, status="failed", invocation_count=len(invocation_rows), error=str(exc))
        if not invocation_rows:
            invocation_rows = [
                {
                    "model": custom.model_name,
                    "source_id": custom.source_id,
                    "invocation_id": "custom_failure",
                    "invocation_kind": "setup",
                    "invocation_type": custom.invocation_type,
                    "cutoff": "",
                    "horizon": spec.horizon,
                    "history_rows": len(history),
                    "future_rows": 0,
                    "output_rows": 0,
                    "status": "failed",
                    "error": str(exc),
                    "script_path": custom.script_path or "",
                    "used_python_executable": False,
                    "python_executable": "",
                    "command_json": "",
                }
            ]
        invocations = _custom_invocation_frame(invocation_rows)
        return _replace_model_result(
            base_result,
            warnings=base_result.warnings + (warning,),
            custom_model_contracts=_concat_frames(base_result.custom_model_contracts, contracts),
            custom_model_invocations=_concat_frames(base_result.custom_model_invocations, invocations),
            model_policy_resolution=_custom_policy_resolution(
                base_result.model_policy_resolution,
                custom=custom,
                ran=False,
                reason_if_not_ran=str(exc),
                contributed_models=[],
            ),
        )

    merged = _merge_custom_challenger(base_result, custom_result)
    return _replace_model_result(
        merged,
        model_policy_resolution=_custom_policy_resolution(
            base_result.model_policy_resolution,
            custom=custom,
            ran=True,
            reason_if_not_ran="",
            contributed_models=[custom.model_name],
        ),
    )


def forecast_with_custom_model(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
    custom: CustomModelSpec,
) -> ModelResult:
    invocation_rows: list[dict[str, Any]] = []
    try:
        forecast = _custom_future_forecast(history, profile, spec, custom, invocation_rows)
        metrics, backtest_predictions = _custom_backtest(history, profile, spec, custom, invocation_rows)
    except Exception as exc:
        raise CustomModelExecutionError(str(exc), invocation_rows=invocation_rows) from exc
    warnings_out: list[str] = [
        (
            f"custom model {custom.model_name} ran as an opt-in challenger; "
            "it received cutoff-limited history plus an exact future grid, returned point forecasts only, "
            "and was scored independently outside WeightedEnsemble"
        )
    ]
    if backtest_predictions.empty:
        warnings_out.append(f"custom model {custom.model_name} did not produce rolling-origin backtest metrics")
    contracts = _custom_contract_frame(custom, status="succeeded", invocation_count=len(invocation_rows), error="")
    invocations = _custom_invocation_frame(invocation_rows)
    return ModelResult(
        forecast=forecast,
        backtest_metrics=metrics,
        backtest_predictions=backtest_predictions,
        engine="custom",
        model_weights=_empty_model_weights(),
        model_explainability=_empty_model_explainability(),
        custom_model_contracts=contracts,
        custom_model_invocations=invocations,
        warnings=tuple(warnings_out),
    )


def _merge_custom_challenger(base_result: ModelResult, custom_result: ModelResult) -> ModelResult:
    forecast = _merge_prediction_frames(base_result.forecast, custom_result.forecast, keys=["unique_id", "ds"])
    backtest_predictions = _merge_prediction_frames(
        base_result.backtest_predictions,
        custom_result.backtest_predictions,
        keys=["unique_id", "ds", "cutoff"],
    )
    backtest_metrics = _concat_frames(base_result.backtest_metrics, custom_result.backtest_metrics)
    explainability = _concat_frames(base_result.model_explainability, custom_result.model_explainability)
    contracts = _concat_frames(base_result.custom_model_contracts, custom_result.custom_model_contracts)
    invocations = _concat_frames(base_result.custom_model_invocations, custom_result.custom_model_invocations)
    warnings = tuple(dict.fromkeys(list(base_result.warnings) + list(custom_result.warnings)))
    return ModelResult(
        forecast=forecast,
        backtest_metrics=backtest_metrics,
        backtest_predictions=backtest_predictions,
        engine=f"{base_result.engine}+custom",
        model_weights=base_result.model_weights,
        model_explainability=explainability,
        custom_model_contracts=contracts,
        custom_model_invocations=invocations,
        warnings=warnings,
        model_policy_resolution=base_result.model_policy_resolution,
    )


def _custom_future_forecast(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
    custom: CustomModelSpec,
    invocation_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    future_grid = _future_grid_from_history(history, profile.freq, spec.horizon)
    cutoff = pd.to_datetime(history["ds"]).max()
    validated = _invoke_and_validate(
        custom,
        history=history,
        future_grid=future_grid,
        horizon=spec.horizon,
        freq=profile.freq,
        cutoff=cutoff,
        levels=spec.levels,
        invocation_kind="future",
        invocation_rows=invocation_rows,
    )
    return validated.rename(columns={"yhat": custom.model_name})[["unique_id", "ds", custom.model_name]]


def _custom_backtest(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
    custom: CustomModelSpec,
    invocation_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_by_group: dict[tuple[pd.Timestamp, int], list[pd.DataFrame]] = {}
    metadata_rows: list[dict[str, Any]] = []
    season = profile.season_length
    for uid, grp in history.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds").reset_index(drop=True)
        n_obs = len(grp)
        h, n_windows, step_size = _adaptive_cv_params(n_obs, spec.horizon, season, strict=spec.strict_cv_horizon)
        if h < 1 or n_windows < 1:
            continue
        for window in range(n_windows, 0, -1):
            cutoff_idx = n_obs - 1 - window * step_size - (h - step_size)
            if cutoff_idx < 0:
                continue
            train = grp.iloc[: cutoff_idx + 1]
            test = grp.iloc[cutoff_idx + 1 : cutoff_idx + 1 + h]
            if train.empty or test.empty:
                continue
            cutoff = pd.Timestamp(train["ds"].max())
            expected = pd.DataFrame(
                {
                    "unique_id": str(uid),
                    "ds": pd.to_datetime(test["ds"]).to_numpy(),
                    "cutoff": cutoff,
                    "horizon_step": np.arange(1, len(test) + 1),
                    "y": pd.to_numeric(test["y"], errors="coerce").to_numpy(dtype="float64"),
                }
            )
            expected_by_group.setdefault((cutoff, len(test)), []).append(expected)
            metadata_rows.append(
                {
                    "unique_id": str(uid),
                    "requested_horizon": spec.horizon,
                    "selection_horizon": h,
                    "cv_windows": 1,
                    "cv_step_size": step_size,
                    "cv_horizon_matches_requested": h == spec.horizon,
                }
            )

    cv_frames: list[pd.DataFrame] = []
    for (cutoff, horizon), frames in sorted(expected_by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        expected = pd.concat(frames, ignore_index=True)
        future_grid = expected.drop(columns=["y"])
        grid_ids = set(future_grid["unique_id"].astype(str))
        cutoff_limited_history = history[
            (pd.to_datetime(history["ds"]) <= cutoff)
            & (history["unique_id"].astype(str).isin(grid_ids))
        ].copy()
        validated = _invoke_and_validate(
            custom,
            history=cutoff_limited_history,
            future_grid=future_grid,
            horizon=horizon,
            freq=profile.freq,
            cutoff=cutoff,
            levels=spec.levels,
            invocation_kind="backtest",
            invocation_rows=invocation_rows,
        )
        cv = expected[["unique_id", "ds", "cutoff", "y"]].merge(validated, on=["unique_id", "ds"], how="left")
        cv = cv.rename(columns={"yhat": custom.model_name})
        cv_frames.append(cv[["unique_id", "ds", "cutoff", "y", custom.model_name]])

    if not cv_frames:
        return _empty_backtest_metrics(), _empty_backtest_predictions()

    backtest_predictions = pd.concat(cv_frames, ignore_index=True).sort_values(["unique_id", "cutoff", "ds"]).reset_index(drop=True)
    metrics = _metrics_from_cv(backtest_predictions, scales_by_series=_error_scale_map(history, season))
    if not metrics.empty and metadata_rows:
        metadata = pd.DataFrame(metadata_rows)
        metadata = (
            metadata.groupby("unique_id", as_index=False)
            .agg(
                requested_horizon=("requested_horizon", "first"),
                selection_horizon=("selection_horizon", "first"),
                cv_windows=("cv_windows", "sum"),
                cv_step_size=("cv_step_size", "first"),
                cv_horizon_matches_requested=("cv_horizon_matches_requested", "first"),
            )
        )
        metrics = metrics.merge(metadata, on="unique_id", how="left")
    return metrics, backtest_predictions


def _invoke_and_validate(
    custom: CustomModelSpec,
    *,
    history: pd.DataFrame,
    future_grid: pd.DataFrame,
    horizon: int,
    freq: str,
    cutoff: pd.Timestamp,
    levels: tuple[int, ...],
    invocation_kind: str,
    invocation_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    invocation_id = f"{invocation_kind}_{len(invocation_rows) + 1}"
    model_grid = future_grid.drop(columns=["y"], errors="ignore").copy()
    context = _custom_context(
        custom,
        future_grid=model_grid,
        horizon=horizon,
        freq=freq,
        cutoff=cutoff,
        levels=levels,
        invocation_kind=invocation_kind,
    )
    command_metadata: dict[str, Any] = {}
    output_rows = 0
    try:
        raw, command_metadata = _invoke_custom(custom, history=history, future_grid=model_grid, context=context)
        output_rows = len(raw)
        validated = _validate_custom_output(raw, expected_grid=model_grid, custom=custom)
    except Exception as exc:
        invocation_rows.append(
            _invocation_row(
                custom,
                invocation_id=invocation_id,
                invocation_kind=invocation_kind,
                cutoff=cutoff,
                horizon=horizon,
                history_rows=len(history),
                future_rows=len(model_grid),
                output_rows=output_rows,
                status="failed",
                error=str(exc),
                command_metadata=command_metadata,
            )
        )
        raise
    invocation_rows.append(
        _invocation_row(
            custom,
            invocation_id=invocation_id,
            invocation_kind=invocation_kind,
            cutoff=cutoff,
            horizon=horizon,
            history_rows=len(history),
            future_rows=len(model_grid),
            output_rows=len(validated),
            status="succeeded",
            error="",
            command_metadata=command_metadata,
        )
    )
    return validated


def _invoke_custom(
    custom: CustomModelSpec,
    *,
    history: pd.DataFrame,
    future_grid: pd.DataFrame,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if custom.script_path is not None:
        return _invoke_script(custom, history=history, future_grid=future_grid, context=context)

    func = custom.callable if custom.callable is not None else _resolve_callable(str(custom.callable_path))
    result = func(
        history[["unique_id", "ds", "y"]].copy(),
        horizon=int(context["horizon"]),
        freq=str(context["freq"]),
        cutoff=pd.Timestamp(context["cutoff"]),
        levels=tuple(context["levels"]),
        context=context,
    )
    return pd.DataFrame(result), {
        "command_json": "",
        "script_path": "",
        "used_python_executable": False,
        "python_executable": "",
    }


def _invoke_script(
    custom: CustomModelSpec,
    *,
    history: pd.DataFrame,
    future_grid: pd.DataFrame,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    script = Path(str(custom.script_path)).expanduser()
    if not script.exists():
        raise FileNotFoundError(f"custom script not found: {script}")
    script = script.resolve()

    with tempfile.TemporaryDirectory(prefix="nixtla_custom_model_") as tmp:
        tmp_path = Path(tmp)
        history_path = tmp_path / "history.csv"
        future_grid_path = tmp_path / "future_grid.csv"
        context_path = tmp_path / "context.json"
        output_path = tmp_path / "forecast.csv"
        history[["unique_id", "ds", "y"]].to_csv(history_path, index=False)
        future_grid.to_csv(future_grid_path, index=False)
        context_path.write_text(json.dumps(_json_safe(context), indent=2, default=str) + "\n", encoding="utf-8")

        used_python = script.suffix.lower() == ".py"
        command = ([sys.executable, str(script)] if used_python else [str(script)]) + [
            "--history",
            str(history_path),
            "--future-grid",
            str(future_grid_path),
            "--context",
            str(context_path),
            "--output",
            str(output_path),
            *list(custom.extra_args),
        ]
        completed = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=custom.timeout_seconds,
            check=False,
            cwd=tmp_path,
        )
        metadata = {
            "command_json": json.dumps(command, default=str),
            "script_path": str(script),
            "used_python_executable": used_python,
            "python_executable": sys.executable if used_python else "",
        }
        if completed.returncode != 0:
            message = _script_failure_message(completed)
            raise RuntimeError(f"custom script failed with exit code {completed.returncode}: {message}")
        if not output_path.exists():
            raise RuntimeError("custom script did not write the required --output forecast CSV")
        return pd.read_csv(output_path), metadata


def _validate_custom_output(raw: pd.DataFrame, *, expected_grid: pd.DataFrame, custom: CustomModelSpec) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("custom model returned no forecast rows")
    required = {"unique_id", "ds", "yhat"}
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError(f"custom model output missing required column(s): {missing_cols}")
    interval_cols = [col for col in raw.columns if str(col).startswith(("yhat_lo", "yhat_hi")) or "-lo-" in str(col) or "-hi-" in str(col)]
    if interval_cols:
        raise ValueError(f"custom model intervals are not supported in v1; remove columns {interval_cols}")

    output = raw[["unique_id", "ds", "yhat"]].copy()
    output["unique_id"] = output["unique_id"].astype(str)
    output["ds"] = pd.to_datetime(output["ds"], errors="coerce")
    if output["ds"].isna().any():
        raise ValueError("custom model output contains invalid ds values")
    output["yhat"] = pd.to_numeric(output["yhat"], errors="coerce")
    finite = output["yhat"].notna() & pd.Series(np.isfinite(output["yhat"].to_numpy(dtype="float64")), index=output.index)
    if not bool(finite.all()):
        raise ValueError("custom model output yhat values must be finite numeric values")
    duplicate_mask = output.duplicated(["unique_id", "ds"], keep=False)
    if duplicate_mask.any():
        sample = output.loc[duplicate_mask, ["unique_id", "ds"]].head(5).to_dict("records")
        raise ValueError(f"custom model output contains duplicate unique_id/ds rows: {sample}")

    expected = expected_grid[["unique_id", "ds", "cutoff"]].copy()
    expected["unique_id"] = expected["unique_id"].astype(str)
    expected["ds"] = pd.to_datetime(expected["ds"])
    expected["cutoff"] = pd.to_datetime(expected["cutoff"])
    invalid_dates = expected[expected["ds"] <= expected["cutoff"]]
    if not invalid_dates.empty:
        sample = invalid_dates[["unique_id", "ds", "cutoff"]].head(5).to_dict("records")
        raise ValueError(f"custom model future grid has rows on/before cutoff: {sample}")

    expected_keys = _key_set(expected)
    output_keys = _key_set(output)
    missing_keys = sorted(expected_keys - output_keys)[:5]
    extra_keys = sorted(output_keys - expected_keys)[:5]
    if missing_keys or extra_keys:
        raise ValueError(
            "custom model output must exactly match the scaffold future grid; "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    merged = expected[["unique_id", "ds"]].merge(output, on=["unique_id", "ds"], how="left")
    merged["model"] = custom.model_name
    return merged[["unique_id", "ds", "yhat"]]


def _future_grid_from_history(history: pd.DataFrame, freq: str, horizon: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for uid, grp in history.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds")
        cutoff = pd.Timestamp(grp["ds"].max())
        future = _future_dates(cutoff, freq, horizon)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": str(uid),
                    "ds": future,
                    "cutoff": cutoff,
                    "horizon_step": np.arange(1, len(future) + 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _custom_context(
    custom: CustomModelSpec,
    *,
    future_grid: pd.DataFrame,
    horizon: int,
    freq: str,
    cutoff: pd.Timestamp,
    levels: tuple[int, ...],
    invocation_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": CUSTOM_MODEL_SCHEMA_VERSION,
        "model_name": custom.model_name,
        "source_id": custom.source_id,
        "invocation_kind": invocation_kind,
        "horizon": int(horizon),
        "freq": freq,
        "cutoff": str(pd.Timestamp(cutoff)),
        "max_cutoff": str(pd.Timestamp(cutoff)),
        "target_scale": "raw_y",
        "levels": list(levels),
        "future_grid": _records(future_grid),
        "output_contract": {
            "required_columns": ["unique_id", "ds", "yhat"],
            "point_forecasts_only": True,
            "must_match_future_grid_exactly": True,
        },
        "leakage_guard": "history is filtered to rows with ds <= cutoff for every backtest invocation",
    }


def _resolve_callable(path: str) -> Any:
    if not path:
        raise ValueError("custom callable path is required")
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        module_name, attr_path = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_path.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"custom callable path {path!r} did not resolve to a callable")
    return target


def _custom_contract_frame(
    custom: CustomModelSpec,
    *,
    status: str,
    invocation_count: int,
    error: str,
) -> pd.DataFrame:
    payload = custom.to_dict()
    return pd.DataFrame(
        [
            {
                "schema_version": CUSTOM_MODEL_SCHEMA_VERSION,
                "name": custom.name,
                "model": custom.model_name,
                "family": "custom",
                "source_id": custom.source_id,
                "invocation_type": custom.invocation_type,
                "callable_path": custom.callable_path or "",
                "callable_object_ref": payload.get("callable_object_ref", ""),
                "script_path": custom.script_path or "",
                "timeout_seconds": custom.timeout_seconds,
                "extra_args_json": json.dumps(list(custom.extra_args)),
                "history_contract": "raw-unit history columns unique_id, ds, y; backtests receive only rows with ds <= cutoff for future-grid series",
                "future_grid_contract": "scaffold supplies exact unique_id/ds future grid with cutoff and horizon_step",
                "output_contract": "return unique_id, ds, yhat; no intervals in v1; output rows must exactly match future grid; scored independently outside WeightedEnsemble",
                "leakage_guard": "cutoff-limited history for each rolling-origin invocation",
                "status": status,
                "invocation_count": invocation_count,
                "error": error,
                "notes": custom.notes,
            }
        ],
        columns=CUSTOM_MODEL_CONTRACT_COLUMNS,
    )


def _invocation_row(
    custom: CustomModelSpec,
    *,
    invocation_id: str,
    invocation_kind: str,
    cutoff: pd.Timestamp,
    horizon: int,
    history_rows: int,
    future_rows: int,
    output_rows: int,
    status: str,
    error: str,
    command_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": custom.model_name,
        "source_id": custom.source_id,
        "invocation_id": invocation_id,
        "invocation_kind": invocation_kind,
        "invocation_type": custom.invocation_type,
        "cutoff": str(pd.Timestamp(cutoff)) if str(cutoff) else "",
        "horizon": int(horizon),
        "history_rows": int(history_rows),
        "future_rows": int(future_rows),
        "output_rows": int(output_rows),
        "status": status,
        "error": error,
        "script_path": command_metadata.get("script_path", custom.script_path or ""),
        "used_python_executable": bool(command_metadata.get("used_python_executable", False)),
        "python_executable": command_metadata.get("python_executable", ""),
        "command_json": command_metadata.get("command_json", ""),
    }


def _custom_invocation_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=CUSTOM_MODEL_INVOCATION_COLUMNS)
    out.insert(0, "schema_version", CUSTOM_MODEL_SCHEMA_VERSION)
    return out.reindex(columns=CUSTOM_MODEL_INVOCATION_COLUMNS)


def _custom_policy_resolution(
    resolution: dict[str, Any],
    *,
    custom: CustomModelSpec,
    ran: bool,
    reason_if_not_ran: str,
    contributed_models: list[str],
) -> dict[str, Any]:
    out = dict(resolution or {})
    families = [dict(row) for row in out.get("families", []) if row.get("family") != "custom"]
    families.append(
        {
            "family": "custom",
            "requested": True,
            "eligible": True,
            "ran": bool(ran),
            "reason_if_not_ran": "" if ran else reason_if_not_ran,
            "contributed_models": contributed_models,
        }
    )
    out["families"] = families
    return out


def _replace_model_result(result: ModelResult, **updates: Any) -> ModelResult:
    values = {
        "forecast": result.forecast,
        "backtest_metrics": result.backtest_metrics,
        "backtest_predictions": result.backtest_predictions,
        "engine": result.engine,
        "model_weights": result.model_weights,
        "model_explainability": result.model_explainability,
        "custom_model_contracts": result.custom_model_contracts,
        "custom_model_invocations": result.custom_model_invocations,
        "warnings": result.warnings,
        "model_policy_resolution": result.model_policy_resolution,
    }
    values.update(updates)
    return ModelResult(**values)


def _concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame()
    return pd.concat(available, ignore_index=True)


def _key_set(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return {(str(row["unique_id"]), pd.Timestamp(row["ds"])) for row in frame[["unique_id", "ds"]].to_dict("records")}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_safe(frame.to_dict("records"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _script_failure_message(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    combined = "\n".join(part for part in [stderr, stdout] if part)
    if not combined:
        return "no stdout/stderr"
    if len(combined) > 1000:
        return combined[:500] + "\n...\n" + combined[-500:]
    return combined
