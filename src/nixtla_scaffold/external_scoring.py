from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.external import (
    ExternalForecastFormat,
    build_external_forecast_metadata,
    canonicalize_external_forecasts,
    load_external_forecasts,
)


EXTERNAL_SCORE_SCHEMA_VERSION = "nixtla_scaffold.external_forecast_score.v1"

EXTERNAL_BACKTEST_LONG_OUTPUT = "external_backtest_long.csv"
EXTERNAL_MODEL_METRICS_OUTPUT = "external_model_metrics.csv"
EXTERNAL_SCORING_MANIFEST_OUTPUT = "external_scoring_manifest.json"


@dataclass(frozen=True)
class ExternalForecastScoreResult:
    """Leakage-safe scoring artifacts for historical cutoff-labeled external forecasts."""

    backtest_long: pd.DataFrame
    model_metrics: pd.DataFrame
    manifest: dict[str, Any]
    external_forecasts: pd.DataFrame
    actuals: pd.DataFrame

    def to_directory(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backtest_long.to_csv(out / EXTERNAL_BACKTEST_LONG_OUTPUT, index=False)
        self.model_metrics.to_csv(out / EXTERNAL_MODEL_METRICS_OUTPUT, index=False)
        (out / EXTERNAL_SCORING_MANIFEST_OUTPUT).write_text(json.dumps(self.manifest, indent=2, default=str) + "\n", encoding="utf-8")
        return out


def score_external_forecasts(
    external_forecasts: str | Path | pd.DataFrame,
    actuals: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None = None,
    actuals_sheet: str | int | None = None,
    external_format: ExternalForecastFormat = "auto",
    external_model_name: str | None = None,
    external_source_id: str | None = None,
    external_id_col: str = "unique_id",
    external_time_col: str = "ds",
    external_value_col: str = "yhat",
    external_model_col: str = "model",
    actual_id_col: str = "unique_id",
    actual_time_col: str = "ds",
    actual_value_col: str = "y",
    season_length: int = 1,
    requested_horizon: int | None = None,
) -> ExternalForecastScoreResult:
    """Score historical external forecast snapshots against actual outcomes.

    External rows must carry `cutoff` / `forecast_origin` labels. Future-only
    external forecasts remain comparison-only and fail closed here.
    """

    _validate_scoring_parameters(season_length=season_length, requested_horizon=requested_horizon)
    external = _load_external_for_scoring(
        external_forecasts,
        sheet=sheet,
        external_format=external_format,
        external_model_name=external_model_name,
        external_source_id=external_source_id,
        external_id_col=external_id_col,
        external_time_col=external_time_col,
        external_value_col=external_value_col,
        external_model_col=external_model_col,
    )
    actual_frame = _load_actuals_for_scoring(
        actuals,
        sheet=actuals_sheet,
        actual_id_col=actual_id_col,
        actual_time_col=actual_time_col,
        actual_value_col=actual_value_col,
    )
    backtest_long = build_external_backtest_long(external, actual_frame, season_length=season_length)
    if not backtest_long["scoring_status"].eq("scored").any():
        raise ValueError(_no_matched_actuals_message(backtest_long, actual_frame))
    model_metrics = build_external_model_metrics(backtest_long, requested_horizon=requested_horizon)
    manifest = build_external_scoring_manifest(
        external_forecasts,
        actuals,
        external,
        actual_frame,
        backtest_long,
        model_metrics,
        sheet=sheet,
        actuals_sheet=actuals_sheet,
        external_format=external_format,
        external_model_name=external_model_name,
        external_source_id=external_source_id,
        season_length=season_length,
        requested_horizon=requested_horizon,
    )
    return ExternalForecastScoreResult(
        backtest_long=backtest_long,
        model_metrics=model_metrics,
        manifest=manifest,
        external_forecasts=external,
        actuals=actual_frame,
    )


def write_external_forecast_scores(
    external_forecasts: str | Path | pd.DataFrame,
    actuals: str | Path | pd.DataFrame,
    output_dir: str | Path,
    **kwargs: Any,
) -> ExternalForecastScoreResult:
    """Score historical external forecasts and write scoring artifacts."""

    result = score_external_forecasts(external_forecasts, actuals, **kwargs)
    manifest = dict(result.manifest)
    manifest["output_dir"] = str(Path(output_dir))
    result = ExternalForecastScoreResult(
        backtest_long=result.backtest_long,
        model_metrics=result.model_metrics,
        manifest=manifest,
        external_forecasts=result.external_forecasts,
        actuals=result.actuals,
    )
    result.to_directory(output_dir)
    return result


def build_external_backtest_long(
    external: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    season_length: int = 1,
) -> pd.DataFrame:
    """Join cutoff-labeled external forecasts to actuals and compute row errors."""

    _validate_positive_int_param(season_length, "season_length")
    _require_cutoff_labeled_external(external)
    actual_frame = _prepare_actuals(actuals)
    external_frame = _prepare_external_for_scoring(external)
    merged = external_frame.merge(actual_frame.rename(columns={"y": "y_actual"}), on=["unique_id", "ds"], how="left")

    y_actual = pd.to_numeric(merged["y_actual"], errors="coerce")
    yhat = pd.to_numeric(merged["yhat"], errors="coerce")
    scored = y_actual.notna() & yhat.notna()
    merged["scoring_status"] = np.where(scored, "scored", "missing_actual")
    merged["record_type"] = "external_backtest"
    merged["h"] = merged["horizon_step"]
    merged["error"] = np.where(scored, y_actual - yhat, np.nan)
    merged["forecast_error"] = np.where(scored, yhat - y_actual, np.nan)
    merged["abs_error"] = np.abs(merged["error"])
    merged["squared_error"] = merged["error"] ** 2
    merged["pct_error"] = np.where(scored & y_actual.ne(0), merged["error"] / y_actual, np.nan)
    merged["is_external_forecast"] = True
    merged["external_is_actual"] = False
    merged["is_backtested"] = scored
    merged["backtest_status"] = np.where(scored, "scored_against_actuals", "missing_actual_unscored")
    merged["scoring_evidence_status"] = np.where(scored, "external_cutoff_scored", "missing_actual_unscored")

    scales = _row_scales(merged, actual_frame, season_length=season_length)
    merged["mase_scale"] = scales["mase_scale"]
    merged["rmsse_scale"] = scales["rmsse_scale"]
    merged["scale_basis"] = scales["scale_basis"]
    merged["effective_season_length"] = scales["effective_season_length"]
    merged["scaled_abs_error"] = merged["abs_error"] / merged["mase_scale"]
    merged["scaled_squared_error"] = merged["squared_error"] / (merged["rmsse_scale"] ** 2)

    ordered = [column for column in _backtest_long_columns() if column in merged.columns]
    extras = [column for column in merged.columns if column not in ordered]
    return merged[ordered + extras].sort_values(_backtest_sort_columns(merged)).reset_index(drop=True)


def build_external_model_metrics(
    backtest_long: pd.DataFrame,
    *,
    requested_horizon: int | None = None,
) -> pd.DataFrame:
    """Aggregate scored external forecast rows using scaffold metric semantics."""

    if requested_horizon is not None:
        _validate_positive_int_param(requested_horizon, "requested_horizon")
    if backtest_long.empty:
        return pd.DataFrame(columns=_model_metrics_columns())
    scored = backtest_long[backtest_long["scoring_status"].eq("scored")].copy()
    if scored.empty:
        return pd.DataFrame(columns=_model_metrics_columns())

    if "scenario_name" not in scored.columns:
        scored["scenario_name"] = ""
    scored["scenario_name"] = scored["scenario_name"].fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    group_cols = ["unique_id", "model", "source_id", "scenario_name"]
    for keys, group in scored.groupby(group_cols, dropna=False, sort=True):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=True))
        errors = pd.to_numeric(group["error"], errors="coerce")
        forecast_errors = pd.to_numeric(group["forecast_error"], errors="coerce")
        actual = pd.to_numeric(group["y_actual"], errors="coerce")
        abs_error = pd.to_numeric(group["abs_error"], errors="coerce")
        squared_error = pd.to_numeric(group["squared_error"], errors="coerce")
        denom = float(actual.abs().sum())
        horizon_step = pd.to_numeric(group["horizon_step"], errors="coerce")
        selection_horizon = int(horizon_step.max()) if horizon_step.notna().any() else None
        requested = int(requested_horizon) if requested_horizon is not None else selection_horizon
        scaled_abs = pd.to_numeric(group["scaled_abs_error"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        scaled_squared = pd.to_numeric(group["scaled_squared_error"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rmse = float(np.sqrt(squared_error.mean())) if squared_error.notna().any() else None
        mae = float(abs_error.mean()) if abs_error.notna().any() else None
        bias = float(forecast_errors.sum() / denom) if denom else None
        scale_basis_counts = group["scale_basis"].astype(str).value_counts().to_dict() if "scale_basis" in group.columns else {}
        rows.append(
            {
                **key_map,
                "family": "external",
                "rmse": rmse,
                "mae": mae,
                "wape": float(abs_error.sum() / denom) if denom else None,
                "mase": float(scaled_abs.mean()) if scaled_abs.notna().any() else None,
                "rmsse": float(np.sqrt(scaled_squared.mean())) if scaled_squared.notna().any() else None,
                "bias": bias,
                "abs_bias": abs(bias) if bias is not None else None,
                "observations": int(errors.notna().sum()),
                "cutoff_count": int(group["cutoff"].nunique()),
                "scale_basis_distribution": json.dumps({str(key): int(value) for key, value in sorted(scale_basis_counts.items())}),
                "start": _date_or_none(group["ds"], "min"),
                "end": _date_or_none(group["ds"], "max"),
                "min_horizon_step": int(horizon_step.min()) if horizon_step.notna().any() else None,
                "max_horizon_step": selection_horizon,
                "requested_horizon": requested,
                "selection_horizon": selection_horizon,
                "cv_windows": int(group["cutoff"].nunique()),
                "cv_step_size": None,
                "cv_horizon_matches_requested": bool(selection_horizon == requested) if selection_horizon is not None and requested is not None else None,
                "is_backtested": True,
                "backtest_status": "scored_against_actuals",
                "scoring_evidence_status": "external_cutoff_scored",
                "metric_definition": (
                    "Metrics are computed only from external rows with cutoff < ds and matched actual y values. "
                    "bias = sum(yhat - y_actual) / sum(abs(y_actual)); positive bias means the external forecast overstated actuals."
                ),
            }
        )
    return pd.DataFrame(rows, columns=_model_metrics_columns()).sort_values(["unique_id", "source_id", "model", "scenario_name"]).reset_index(drop=True)


def build_external_scoring_manifest(
    external_source: str | Path | pd.DataFrame,
    actuals_source: str | Path | pd.DataFrame,
    external: pd.DataFrame,
    actuals: pd.DataFrame,
    backtest_long: pd.DataFrame,
    model_metrics: pd.DataFrame,
    *,
    sheet: str | int | None = None,
    actuals_sheet: str | int | None = None,
    external_format: ExternalForecastFormat = "auto",
    external_model_name: str | None = None,
    external_source_id: str | None = None,
    season_length: int = 1,
    requested_horizon: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for external forecast scoring."""

    status_counts = _value_counts(backtest_long, "scoring_status")
    scored_rows = int(status_counts.get("scored", 0))
    missing_rows = int(status_counts.get("missing_actual", 0))
    return {
        "schema_version": EXTERNAL_SCORE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scoring_scope": "external_cutoff_actual_scoring",
        "inputs": {
            "external_source": str(external_source) if not isinstance(external_source, pd.DataFrame) else "dataframe",
            "external_sheet": sheet,
            "external_format": external_format,
            "external_model_name": external_model_name,
            "external_source_id": external_source_id,
            "actuals_source": str(actuals_source) if not isinstance(actuals_source, pd.DataFrame) else "dataframe",
            "actuals_sheet": actuals_sheet,
            "season_length": int(season_length),
            "requested_horizon": requested_horizon,
        },
        "rows": {
            "external_rows": int(len(external)),
            "actual_rows": int(len(actuals)),
            "backtest_long_rows": int(len(backtest_long)),
            "scored_rows": scored_rows,
            "missing_actual_rows": missing_rows,
            "metric_rows": int(len(model_metrics)),
            "series_count": int(backtest_long["unique_id"].astype(str).nunique()) if not backtest_long.empty else 0,
            "cutoff_count": int(backtest_long["cutoff"].nunique()) if "cutoff" in backtest_long.columns else 0,
        },
        "date_ranges": {
            "external_target_start": _date_or_none(external["ds"], "min") if "ds" in external.columns else None,
            "external_target_end": _date_or_none(external["ds"], "max") if "ds" in external.columns else None,
            "actual_start": _date_or_none(actuals["ds"], "min") if "ds" in actuals.columns else None,
            "actual_end": _date_or_none(actuals["ds"], "max") if "ds" in actuals.columns else None,
        },
        "external": build_external_forecast_metadata(external),
        "scoring_status_distribution": status_counts,
        "metric_status": "scored" if scored_rows else "no_matched_actuals",
        "scale_basis_distribution": _value_counts(backtest_long, "scale_basis"),
        "guardrails": [
            "Only cutoff-labeled external forecasts can be scored.",
            "Rows must pass cutoff < ds before scoring, preventing same-period or future-origin leakage.",
            "External forecasts remain imported yhat values; actuals come only from the actuals input.",
            "Metrics are computed only for rows with matched actuals; missing actual rows stay unscored diagnostics.",
            "MASE/RMSSE scales are computed from actual history available at each cutoff; rows disclose seasonal vs naive-fallback scale basis.",
            "bias = sum(yhat - y_actual) / sum(abs(y_actual)); positive bias means external forecasts overstated actuals.",
            "This scoring workflow is separate from the directional compare command and does not alter scaffold model selection.",
        ],
        "outputs": {
            "backtest_long": EXTERNAL_BACKTEST_LONG_OUTPUT,
            "model_metrics": EXTERNAL_MODEL_METRICS_OUTPUT,
            "manifest": EXTERNAL_SCORING_MANIFEST_OUTPUT,
        },
    }


def _load_external_for_scoring(
    external_forecasts: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None,
    external_format: ExternalForecastFormat,
    external_model_name: str | None,
    external_source_id: str | None,
    external_id_col: str,
    external_time_col: str,
    external_value_col: str,
    external_model_col: str,
) -> pd.DataFrame:
    if isinstance(external_forecasts, pd.DataFrame):
        return canonicalize_external_forecasts(
            external_forecasts,
            format=external_format,
            model_name=external_model_name,
            source_id=external_source_id,
            id_col=external_id_col,
            time_col=external_time_col,
            value_col=external_value_col,
            model_col=external_model_col,
        )
    return load_external_forecasts(
        external_forecasts,
        sheet=sheet,
        format=external_format,
        model_name=external_model_name,
        source_id=external_source_id,
        id_col=external_id_col,
        time_col=external_time_col,
        value_col=external_value_col,
        model_col=external_model_col,
    )


def _validate_scoring_parameters(*, season_length: int, requested_horizon: int | None) -> None:
    _validate_positive_int_param(season_length, "season_length")
    if requested_horizon is not None:
        _validate_positive_int_param(requested_horizon, "requested_horizon")


def _validate_positive_int_param(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _no_matched_actuals_message(backtest_long: pd.DataFrame, actuals: pd.DataFrame) -> str:
    return (
        "external forecast scoring found no rows with matching actuals. "
        "Join key is unique_id + ds. "
        f"External target date range: {_date_range_text(backtest_long['ds'])}; "
        f"actual date range: {_date_range_text(actuals['ds'])}. "
        f"External series sample: {_sample_values(backtest_long, 'unique_id')}; "
        f"actual series sample: {_sample_values(actuals, 'unique_id')}. "
        f"Unmatched external rows sample: {_sample_records(backtest_long, ['unique_id', 'ds', 'cutoff', 'model', 'source_id'])}. "
        "Confirm actuals include realized target dates, matching unique_id values, and the correct "
        "--actual-id-col/--actual-time-col/--actual-target-col mappings."
    )


def _load_actuals_for_scoring(
    actuals: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None,
    actual_id_col: str,
    actual_time_col: str,
    actual_value_col: str,
) -> pd.DataFrame:
    return load_forecast_dataset(
        actuals,
        sheet=sheet,
        id_col=actual_id_col,
        time_col=actual_time_col,
        target_col=actual_value_col,
    )


def _require_cutoff_labeled_external(external: pd.DataFrame) -> None:
    if "cutoff" not in external.columns or external["cutoff"].isna().all():
        raise ValueError("external forecast scoring requires cutoff or forecast_origin labels; future-only forecasts are comparison-only")
    if "comparison_evidence_status" in external.columns:
        statuses = set(external["comparison_evidence_status"].astype(str))
        if statuses != {"historical_cutoff_labeled_unscored"}:
            raise ValueError(f"external forecast scoring requires historical cutoff-labeled rows; found {sorted(statuses)}")


def _prepare_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "y"}
    missing = sorted(required - set(actuals.columns))
    if missing:
        raise ValueError(f"actuals are missing required column(s): {missing}")
    out = actuals.copy()
    out["unique_id"] = _clean_text(out["unique_id"], "unique_id", frame_name="actuals")
    out["ds"] = _coerce_datetime(out["ds"], "ds", frame_name="actuals")
    out["y"] = _coerce_numeric(out["y"], "y", frame_name="actuals")
    duplicated = out.duplicated(["unique_id", "ds"], keep=False)
    if duplicated.any():
        sample = out.loc[duplicated, ["unique_id", "ds"]].head(5).to_dict("records")
        raise ValueError(f"actuals have duplicate unique_id/ds rows for external scoring: {sample}")
    return out[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _prepare_external_for_scoring(external: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "model", "source_id", "cutoff", "yhat", "horizon_step"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"external forecasts are missing required scoring column(s): {missing}")
    out = external.copy()
    out["unique_id"] = _clean_text(out["unique_id"], "unique_id", frame_name="external forecasts")
    out["model"] = _clean_text(out["model"], "model", frame_name="external forecasts")
    out["source_id"] = _clean_text(out["source_id"], "source_id", frame_name="external forecasts")
    out["ds"] = _coerce_datetime(out["ds"], "ds", frame_name="external forecasts")
    out["cutoff"] = _coerce_datetime(out["cutoff"], "cutoff", frame_name="external forecasts")
    out["yhat"] = _coerce_numeric(out["yhat"], "yhat", frame_name="external forecasts")
    out["horizon_step"] = _coerce_positive_int(out["horizon_step"], "horizon_step", frame_name="external forecasts")
    invalid = out["cutoff"] >= out["ds"]
    if invalid.any():
        sample = out.loc[invalid, ["unique_id", "model", "source_id", "cutoff", "ds"]].head(5).to_dict("records")
        raise ValueError(f"external scoring requires cutoff before ds: {sample}")
    keep = [
        "unique_id",
        "cutoff",
        "ds",
        "model",
        "source_id",
        "scenario_name",
        "yhat",
        "horizon_step",
        "comparison_evidence_status",
        "owner",
        "model_version",
        "notes",
    ]
    return out[[column for column in keep if column in out.columns]]


def _row_scales(scored_frame: pd.DataFrame, actuals: pd.DataFrame, *, season_length: int) -> pd.DataFrame:
    actuals_by_series = {uid: group.sort_values("ds") for uid, group in actuals.groupby("unique_id", sort=False)}
    rows: list[dict[str, float | None]] = []
    for row in scored_frame.itertuples(index=False):
        history = actuals_by_series.get(str(row.unique_id))
        if history is None:
            rows.append(
                {
                    "mase_scale": None,
                    "rmsse_scale": None,
                    "scale_basis": "missing_series_history",
                    "effective_season_length": None,
                }
            )
            continue
        rows.append(_cutoff_scale(history, pd.Timestamp(row.cutoff), season_length=season_length))
    return pd.DataFrame(rows, index=scored_frame.index)


def _cutoff_scale(history: pd.DataFrame, cutoff: pd.Timestamp, *, season_length: int) -> dict[str, float | None]:
    lag = max(1, int(season_length or 1))
    past = history[pd.to_datetime(history["ds"]) <= cutoff].sort_values("ds")
    values = pd.to_numeric(past["y"], errors="coerce").dropna().to_numpy(dtype="float64")
    active_lag = lag if lag > 1 and len(values) > lag else 1
    if len(values) <= active_lag:
        return {
            "mase_scale": None,
            "rmsse_scale": None,
            "scale_basis": "insufficient_history",
            "effective_season_length": None,
        }
    diff = values[active_lag:] - values[:-active_lag]
    abs_scale = float(np.nanmean(np.abs(diff)))
    rmsse_scale = float(np.sqrt(np.nanmean(diff ** 2)))
    return {
        "mase_scale": abs_scale if np.isfinite(abs_scale) and abs_scale > 0 else None,
        "rmsse_scale": rmsse_scale if np.isfinite(rmsse_scale) and rmsse_scale > 0 else None,
        "scale_basis": "seasonal" if active_lag == lag and lag > 1 else ("naive_fallback" if lag > 1 else "naive"),
        "effective_season_length": active_lag,
    }


def _clean_text(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = values.astype("string").str.strip()
    missing = out.isna() | out.eq("")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} {frame_name} rows have blank {column} values")
    return out.astype(str)


def _coerce_datetime(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} {frame_name} rows have invalid {column} dates")
    return out


def _coerce_numeric(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    invalid = out.isna() | ~np.isfinite(out.to_numpy(dtype="float64"))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} {frame_name} rows have missing or non-numeric {column} values")
    return out.astype(float)


def _coerce_positive_int(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    invalid = out.isna() | (out < 1) | (out % 1 != 0)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} {frame_name} rows have invalid {column} values")
    return out.astype(int)


def _date_or_none(values: pd.Series, method: str) -> str | None:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    value = dates.min() if method == "min" else dates.max()
    return value.date().isoformat()


def _date_range_text(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return "empty"
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}


def _sample_values(frame: pd.DataFrame, column: str, *, limit: int = 5) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())[:limit]


def _sample_records(frame: pd.DataFrame, columns: list[str], *, limit: int = 5) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return []
    sample = frame.loc[:, available].head(limit).copy()
    for column in ("ds", "cutoff"):
        if column in sample.columns:
            sample[column] = pd.to_datetime(sample[column], errors="coerce").dt.date.astype(str)
    return sample.to_dict("records")


def _backtest_long_columns() -> list[str]:
    return [
        "record_type",
        "unique_id",
        "cutoff",
        "ds",
        "model",
        "source_id",
        "scenario_name",
        "y_actual",
        "yhat",
        "horizon_step",
        "h",
        "error",
        "forecast_error",
        "abs_error",
        "squared_error",
        "pct_error",
        "mase_scale",
        "rmsse_scale",
        "scale_basis",
        "effective_season_length",
        "scaled_abs_error",
        "scaled_squared_error",
        "scoring_status",
        "scoring_evidence_status",
        "is_external_forecast",
        "external_is_actual",
        "is_backtested",
        "backtest_status",
    ]


def _backtest_sort_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["unique_id", "source_id", "model", "cutoff", "ds"]
    if "scenario_name" in frame.columns:
        columns.insert(3, "scenario_name")
    return [column for column in columns if column in frame.columns]


def _model_metrics_columns() -> list[str]:
    return [
        "unique_id",
        "model",
        "source_id",
        "scenario_name",
        "family",
        "rmse",
        "mae",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "abs_bias",
        "observations",
        "cutoff_count",
        "scale_basis_distribution",
        "start",
        "end",
        "min_horizon_step",
        "max_horizon_step",
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
        "is_backtested",
        "backtest_status",
        "scoring_evidence_status",
        "metric_definition",
    ]
