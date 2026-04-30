from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nixtla_scaffold.schema import ForecastSpec


def load_forecast_dataset(
    source: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None = None,
    spec: ForecastSpec | None = None,
    id_col: str | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Load and canonicalize data into unique_id/ds/y long form."""

    spec = spec or ForecastSpec()
    id_col = id_col or spec.id_col
    time_col = time_col or spec.time_col
    target_col = target_col or spec.target_col

    raw = read_tabular_source(source, sheet=sheet)

    return canonicalize_forecast_frame(
        raw,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )


def read_tabular_source(source: str | Path | pd.DataFrame, *, sheet: str | int | None = None) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".xls":
        raise ValueError("legacy .xls files are not supported; save as .xlsx or CSV before forecasting")
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet or 0, keep_default_na=False, na_values=[""])
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def canonicalize_forecast_frame(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Return a canonical frame while preserving extra columns."""

    if time_col not in df.columns:
        raise ValueError(_missing_column_message("time", time_col, df.columns))
    if target_col not in df.columns:
        raise ValueError(_missing_column_message("target", target_col, df.columns))

    out = df.copy()
    rename: dict[str, str] = {}
    if id_col in out.columns and id_col != "unique_id":
        rename[id_col] = "unique_id"
    if time_col != "ds":
        rename[time_col] = "ds"
    if target_col != "y":
        rename[target_col] = "y"
    out = out.rename(columns=rename)

    if "unique_id" not in out.columns:
        _reject_implicit_multi_series(out, time_col="ds")
        out["unique_id"] = "series_1"

    missing_id = out["unique_id"].isna() | out["unique_id"].astype(str).str.strip().eq("")
    if missing_id.any():
        raise ValueError(f"{int(missing_id.sum())} rows have missing unique_id values")
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    raw_y = out["y"].copy()
    out["y"] = pd.to_numeric(raw_y, errors="coerce")

    if out["ds"].isna().any():
        bad = int(out["ds"].isna().sum())
        raise ValueError(f"{bad} rows have invalid dates in 'ds'")
    invalid_y = int(out["y"].isna().sum() - raw_y.isna().sum())
    if invalid_y:
        raise ValueError(f"{invalid_y} rows have non-numeric values in 'y'")
    if out.empty:
        raise ValueError("forecast input is empty; at least one row is required")

    ordered_cols = ["unique_id", "ds", "y"]
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    out = out[ordered_cols + extra_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    empty_series = out.groupby("unique_id")["y"].apply(lambda values: values.notna().sum() == 0)
    if empty_series.any():
        bad_ids = empty_series[empty_series].index.astype(str).tolist()
        raise ValueError(f"series with no usable numeric y values: {bad_ids}")
    return out


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Small helper for MCP/agent outputs that arrive as JSON-like records."""

    return canonicalize_forecast_frame(pd.DataFrame.from_records(records))


def _missing_column_message(role: str, expected: str, columns: pd.Index) -> str:
    detected = [str(col) for col in columns]
    suggestion = _suggest_column(role, detected)
    hint = f" Try --{role}-col {suggestion}." if suggestion else ""
    return f"missing required {role} column '{expected}'. Detected columns: {detected}.{hint}"


def _suggest_column(role: str, columns: list[str]) -> str | None:
    candidates = {
        "time": ("ds", "date", "month", "week", "period", "time"),
        "target": ("y", "revenue", "arr", "amount", "value", "actual", "metric"),
    }.get(role, ())
    for column in columns:
        normalized = column.lower().replace("_", " ").replace("-", " ")
        if any(token in normalized for token in candidates):
            return column
    return None


def _reject_implicit_multi_series(df: pd.DataFrame, *, time_col: str) -> None:
    if not df[time_col].duplicated().any():
        return

    candidate_cols = [
        col
        for col in df.columns
        if col not in {time_col, "ds", "y"} and not pd.api.types.is_numeric_dtype(df[col])
    ]
    hints = f" Candidate id columns: {candidate_cols}." if candidate_cols else ""
    raise ValueError(
        "unique_id is missing but timestamps repeat, so the input looks like multiple series. "
        "Pass --id-col or rename the series identifier to unique_id."
        + hints
    )

