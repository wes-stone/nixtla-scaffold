from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from nixtla_scaffold.data import read_tabular_source


ExternalForecastFormat = Literal["auto", "long", "wide"]

EXTERNAL_FORECAST_SCHEMA_VERSION = "nixtla_scaffold.external_forecast.v1"

_METADATA_COLUMNS = {
    "currency",
    "known_as_of",
    "model_version",
    "normalization_label",
    "notes",
    "owner",
    "scenario_name",
    "sheet",
    "source_file",
    "source_id",
    "source_system",
    "unit_label",
    "version",
}


def load_external_forecasts(
    source: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None = None,
    format: ExternalForecastFormat = "auto",
    model_name: str | None = None,
    source_id: str | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    value_col: str = "yhat",
    model_col: str = "model",
    forecast_origin_col: str = "cutoff",
) -> pd.DataFrame:
    """Load an external finance forecast into a safe comparison contract.

    The returned frame is for comparison only. It never represents observed
    actuals and should not be fed into the normal forecast data loader as `y`.
    """

    source_path = Path(source) if isinstance(source, (str, Path)) else None
    raw = read_tabular_source(source, sheet=sheet)
    inferred_source_id = source_id or (source_path.stem if source_path is not None else "external")
    inferred_model_name = model_name or (source_path.stem if source_path is not None else None)
    return canonicalize_external_forecasts(
        raw,
        format=format,
        model_name=inferred_model_name,
        source_id=inferred_source_id,
        source_file=str(source_path) if source_path is not None else None,
        sheet=sheet,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
        model_col=model_col,
        forecast_origin_col=forecast_origin_col,
    )


def canonicalize_external_forecasts(
    frame: pd.DataFrame,
    *,
    format: ExternalForecastFormat = "auto",
    model_name: str | None = None,
    source_id: str | None = None,
    source_file: str | None = None,
    sheet: str | int | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    value_col: str = "yhat",
    model_col: str = "model",
    forecast_origin_col: str = "cutoff",
) -> pd.DataFrame:
    """Canonicalize imported finance forecasts for optional comparison.

    Long input requires date and forecast value columns. A model column is
    required unless `model_name` is supplied. Wide input melts date-like columns
    into `ds` rows and preserves non-date columns as metadata.
    """

    if format not in {"auto", "long", "wide"}:
        raise ValueError("external forecast format must be one of: auto, long, wide")
    _ensure_frame(frame)
    _ensure_unique_column_names(frame)
    _validate_optional_label("model_name", model_name)
    _validate_optional_label("source_id", source_id)

    resolved = _resolve_format(frame, format=format, time_col=time_col, value_col=value_col)
    if resolved == "wide":
        out = _wide_to_long(
            frame,
            model_name=model_name,
            id_col=id_col,
            model_col=model_col,
        )
    else:
        out = _long_to_canonical(
            frame,
            model_name=model_name,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            model_col=model_col,
            forecast_origin_col=forecast_origin_col,
        )

    out = _finalize_external_frame(out, source_id=source_id, source_file=source_file, sheet=sheet)
    return out


def build_external_forecast_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    """Return deterministic metadata for an already-canonical external forecast frame."""

    _ensure_frame(frame)
    if not {"unique_id", "ds", "model", "yhat"}.issubset(frame.columns):
        raise ValueError("external forecast metadata requires canonical columns: unique_id, ds, model, yhat")
    dates = pd.to_datetime(frame["ds"], errors="coerce")
    return {
        "schema_version": EXTERNAL_FORECAST_SCHEMA_VERSION,
        "rows": int(len(frame)),
        "series_count": int(frame["unique_id"].astype(str).nunique()),
        "model_count": int(frame["model"].astype(str).nunique()),
        "models": sorted(frame["model"].astype(str).unique().tolist()),
        "sources": sorted(frame["source_id"].astype(str).unique().tolist()) if "source_id" in frame.columns else [],
        "start": dates.min().date().isoformat() if dates.notna().any() else None,
        "end": dates.max().date().isoformat() if dates.notna().any() else None,
        "evidence_status_distribution": _value_counts(frame, "comparison_evidence_status"),
        "is_backtested": bool(frame.get("is_backtested", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "is_backtested_rationale": (
            "External forecasts are imported yhat values, not scaffold model candidates or actuals. "
            "They remain unbacktested until an explicit scoring workflow joins cutoff-labeled forecasts "
            "to actual values and reports accuracy metrics."
        ),
    }


def _resolve_format(
    frame: pd.DataFrame,
    *,
    format: ExternalForecastFormat,
    time_col: str,
    value_col: str,
) -> Literal["long", "wide"]:
    if format in {"long", "wide"}:
        return format
    has_long_columns = time_col in frame.columns and value_col in frame.columns
    if has_long_columns:
        return "long"
    if _date_header_map(frame.columns):
        return "wide"
    raise ValueError(
        "could not infer external forecast format. Provide long columns "
        f"'{time_col}' and '{value_col}', or pass format='wide' with date-like forecast columns."
    )


def _long_to_canonical(
    frame: pd.DataFrame,
    *,
    model_name: str | None,
    id_col: str,
    time_col: str,
    value_col: str,
    model_col: str,
    forecast_origin_col: str,
) -> pd.DataFrame:
    if time_col not in frame.columns:
        raise ValueError(f"missing external forecast date column '{time_col}'")
    if value_col not in frame.columns:
        raise ValueError(f"missing external forecast value column '{value_col}'")

    out = frame.copy()
    rename: dict[str, str] = {}
    if id_col in out.columns and id_col != "unique_id":
        rename[id_col] = "unique_id"
    if time_col != "ds":
        rename[time_col] = "ds"
    if value_col != "yhat":
        rename[value_col] = "yhat"
    if model_col in out.columns and model_col != "model":
        rename[model_col] = "model"
    elif model_col not in out.columns and "external_model" in out.columns:
        rename["external_model"] = "model"
    if forecast_origin_col in out.columns and forecast_origin_col != "cutoff":
        rename[forecast_origin_col] = "cutoff"
    elif forecast_origin_col not in out.columns and "forecast_origin" in out.columns:
        rename["forecast_origin"] = "cutoff"
    out = out.rename(columns=rename)

    if "unique_id" not in out.columns:
        out["unique_id"] = "series_1"
    if "model" not in out.columns:
        if model_name is None:
            raise ValueError("external forecast model is required; pass model_name or include a model column")
        out["model"] = model_name
    return out


def _wide_to_long(
    frame: pd.DataFrame,
    *,
    model_name: str | None,
    id_col: str,
    model_col: str,
) -> pd.DataFrame:
    date_map = _date_header_map(frame.columns)
    if not date_map:
        raise ValueError("wide external forecast input has no date-like forecast columns")
    _reject_duplicate_parsed_dates(date_map)

    out = frame.copy()
    if id_col in out.columns and id_col != "unique_id":
        out = out.rename(columns={id_col: "unique_id"})
    if "unique_id" not in out.columns:
        out["unique_id"] = "series_1"
    if model_col in out.columns and model_col != "model":
        out = out.rename(columns={model_col: "model"})
    if "model" not in out.columns:
        if model_name is None:
            raise ValueError("wide external forecast model is required; pass model_name or include a model column")
        out["model"] = model_name

    id_vars = [column for column in out.columns if column not in date_map]
    melted = out.melt(id_vars=id_vars, value_vars=list(date_map), var_name="source_column", value_name="yhat")
    melted["ds"] = melted["source_column"].map(date_map)
    return melted


def _finalize_external_frame(
    frame: pd.DataFrame,
    *,
    source_id: str | None,
    source_file: str | None,
    sheet: str | int | None,
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        raise ValueError("external forecast input is empty; at least one forecast row is required")
    missing = [column for column in ("unique_id", "ds", "model", "yhat") if column not in out.columns]
    if missing:
        raise ValueError(f"external forecast is missing required canonical column(s): {missing}")

    out["unique_id"] = _clean_required_text(out["unique_id"], "unique_id")
    out["model"] = _clean_required_text(out["model"], "model")
    out["ds"] = _coerce_datetime(out["ds"], "ds")
    out["yhat"] = _coerce_numeric(out["yhat"], "yhat")

    cutoff_present = "cutoff" in out.columns and out["cutoff"].notna().any()
    if cutoff_present:
        out["cutoff"] = _coerce_datetime(out["cutoff"], "cutoff")
    elif "cutoff" in out.columns:
        out = out.drop(columns=["cutoff"])

    if source_id is not None:
        out["source_id"] = source_id
    elif "source_id" not in out.columns:
        out["source_id"] = "external"
    out["source_id"] = _clean_required_text(out["source_id"], "source_id")

    if "horizon_step" not in out.columns:
        out["horizon_step"] = _horizon_steps(out, has_cutoff=cutoff_present)
    else:
        out["horizon_step"] = _coerce_positive_int(out["horizon_step"], "horizon_step")
    if source_file is not None:
        out["source_file"] = source_file
    if sheet is not None:
        out["sheet"] = str(sheet)
    if cutoff_present:
        _validate_cutoff_precedes_target(out)

    out["family"] = "external"
    out["is_external_forecast"] = True
    out["external_forecast_validation_required"] = True
    out["comparison_evidence_status"] = "historical_cutoff_labeled_unscored" if cutoff_present else "future_only_unscored"
    out["is_backtested"] = False
    out["backtest_status"] = "not_backtested"
    out["status_message"] = (
        "Historical cutoff labels are present but forecast origin has not been independently verified "
        "and rows have not been scored against actuals. Do not claim backtest accuracy yet."
        if cutoff_present
        else "Future-only external forecast; compare directionally only. Do not use as actuals or make accuracy/backtest claims."
    )

    _reject_duplicate_external_rows(out, has_cutoff=cutoff_present)
    return _order_external_columns(out).sort_values(_sort_columns(out)).reset_index(drop=True)


def _clean_required_text(values: pd.Series, column: str) -> pd.Series:
    out = values.astype("string").str.strip()
    missing = out.isna() | out.eq("")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} external forecast rows have blank {column} values")
    return out.astype(str)


def _coerce_datetime(values: pd.Series, column: str) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} external forecast rows have invalid {column} dates")
    return out


def _coerce_numeric(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        out = pd.to_numeric(values, errors="coerce")
    else:
        cleaned = values.astype("string").str.strip()
        cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        cleaned = cleaned.str.replace(r"[\$,]", "", regex=True)
        out = pd.to_numeric(cleaned.replace({"": pd.NA}), errors="coerce")
    invalid = out.isna() | ~np.isfinite(out.to_numpy(dtype="float64"))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} external forecast rows have missing or non-numeric {column} values")
    return out.astype(float)


def _coerce_positive_int(values: pd.Series, column: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    invalid = out.isna() | (out < 1) | (out % 1 != 0)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} external forecast rows have invalid {column} values")
    return out.astype(int)


def _horizon_steps(frame: pd.DataFrame, *, has_cutoff: bool) -> pd.Series:
    group_cols = ["unique_id", "model"]
    if "source_id" in frame.columns:
        group_cols.append("source_id")
    if has_cutoff:
        group_cols.append("cutoff")
    if "scenario_name" in frame.columns:
        group_cols.append("scenario_name")
    ordered = frame.sort_values(group_cols + ["ds"]).copy()
    steps = ordered.groupby(group_cols, sort=False).cumcount() + 1
    return steps.reindex(frame.index).astype(int)


def _reject_duplicate_external_rows(frame: pd.DataFrame, *, has_cutoff: bool) -> None:
    key = ["unique_id", "ds", "model", "source_id"]
    if has_cutoff:
        key.append("cutoff")
    if "scenario_name" in frame.columns:
        key.append("scenario_name")
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, key].head(5).to_dict("records")
        raise ValueError(f"duplicate external forecast rows for key {key}: {sample}")


def _validate_cutoff_precedes_target(frame: pd.DataFrame) -> None:
    invalid = frame["cutoff"] >= frame["ds"]
    if invalid.any():
        sample = frame.loc[invalid, ["unique_id", "model", "cutoff", "ds"]].head(5).to_dict("records")
        raise ValueError(
            "external forecast rows with cutoff labels must have cutoff before ds to avoid leakage risk: "
            f"{sample}"
        )


def _date_header_map(columns: pd.Index) -> dict[Any, pd.Timestamp]:
    parsed: dict[Any, pd.Timestamp] = {}
    for column in columns:
        if str(column) in _METADATA_COLUMNS or str(column) in {"unique_id", "model", "external_model", "ds", "yhat"}:
            continue
        timestamp = _parse_date_header(column)
        if timestamp is not None:
            parsed[column] = timestamp
    return parsed


def _parse_date_header(value: Any) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value)
    if isinstance(value, (int, float)) and np.isfinite(value):
        if 20000 <= float(value) <= 80000:
            return pd.to_datetime(float(value), unit="D", origin="1899-12-30")
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        if len(text) == 8 and text[:2] in {"19", "20"}:
            parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
            return None if pd.isna(parsed) else parsed
        numeric = float(text)
        if 20000 <= numeric <= 80000:
            return pd.to_datetime(numeric, unit="D", origin="1899-12-30")
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _reject_duplicate_parsed_dates(date_map: dict[Any, pd.Timestamp]) -> None:
    normalized = [value.normalize() for value in date_map.values()]
    if len(set(normalized)) == len(normalized):
        return
    rows = [
        {"column": str(column), "parsed_date": timestamp.date().isoformat()}
        for column, timestamp in date_map.items()
        if normalized.count(timestamp.normalize()) > 1
    ]
    raise ValueError(f"wide external forecast has duplicate parsed date columns: {rows}")


def _ensure_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("external forecast input must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("external forecast input is empty; at least one row is required")


def _ensure_unique_column_names(frame: pd.DataFrame) -> None:
    labels = [str(column) for column in frame.columns]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"external forecast input has duplicate column names: {duplicates}")


def _validate_optional_label(name: str, value: str | None) -> None:
    if value is not None and not str(value).strip():
        raise ValueError(f"{name} cannot be blank")


def _order_external_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "unique_id",
        "ds",
        "model",
        "yhat",
        "cutoff",
        "horizon_step",
        "scenario_name",
        "comparison_evidence_status",
        "is_backtested",
        "backtest_status",
        "status_message",
        "family",
        "is_external_forecast",
        "external_forecast_validation_required",
        "source_id",
        "source_file",
        "sheet",
        "source_column",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extras]


def _sort_columns(frame: pd.DataFrame) -> list[str]:
    cols = ["unique_id", "model"]
    if "source_id" in frame.columns:
        cols.append("source_id")
    if "cutoff" in frame.columns:
        cols.append("cutoff")
    if "scenario_name" in frame.columns:
        cols.append("scenario_name")
    cols.append("ds")
    return cols


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}
