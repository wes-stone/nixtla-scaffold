from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from nixtla_scaffold.schema import DriverEvent, TargetTransform, TransformSpec


def add_fiscal_calendar(frame: pd.DataFrame, *, fiscal_year_start_month: int = 2) -> pd.DataFrame:
    """Add GitHub-style fiscal calendar fields without changing the grain."""

    if fiscal_year_start_month < 1 or fiscal_year_start_month > 12:
        raise ValueError("fiscal_year_start_month must be between 1 and 12")
    out = frame.copy()
    if "ds" not in out.columns:
        raise ValueError("fiscal calendar requires a 'ds' date column")
    ds = pd.to_datetime(out["ds"], errors="coerce")
    if ds.isna().any():
        raise ValueError(f"{int(ds.isna().sum())} rows have invalid dates in 'ds'")

    fiscal_month = ((ds.dt.month - fiscal_year_start_month) % 12) + 1
    out["fiscal_year"] = ds.dt.year + (ds.dt.month >= fiscal_year_start_month).astype(int)
    out["fiscal_quarter"] = ((fiscal_month - 1) // 3) + 1
    out["fiscal_month"] = fiscal_month
    out["fiscal_period"] = out["fiscal_year"].astype(str) + "-M" + fiscal_month.astype(str).str.zfill(2)
    return out


def normalize_by_factor(
    frame: pd.DataFrame,
    *,
    factor_col: str,
    target_col: str = "y",
    output_col: str = "y_adjusted",
) -> pd.DataFrame:
    """Create an adjusted target by dividing observed values by a positive normalization factor."""

    if target_col not in frame.columns:
        raise ValueError(f"missing target column '{target_col}'")
    if factor_col not in frame.columns:
        raise ValueError(f"missing normalization factor column '{factor_col}'")
    out = frame.copy()
    factor = pd.to_numeric(out[factor_col], errors="coerce")
    if factor.isna().any():
        raise ValueError(f"{int(factor.isna().sum())} rows have non-numeric normalization factors in '{factor_col}'")
    if (factor <= 0).any():
        raise ValueError(f"normalization factors in '{factor_col}' must be positive")
    out[output_col] = pd.to_numeric(out[target_col], errors="coerce") / factor
    return out


TARGET_TRANSFORM_AUDIT_COLUMNS = [
    "unique_id",
    "ds",
    "y_raw",
    "normalization_factor_col",
    "normalization_factor",
    "y_adjusted",
    "target_transform",
    "y_modeled",
    "output_scale",
    "notes",
]


def prepare_modeling_target(
    frame: pd.DataFrame,
    transform: TransformSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply finance normalization and target transforms before model fitting."""

    if not transform.enabled:
        return frame.copy(), pd.DataFrame(columns=TARGET_TRANSFORM_AUDIT_COLUMNS), []
    required = {"unique_id", "ds", "y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"target transformation requires columns {required}")

    out = frame.copy()
    raw = pd.to_numeric(out["y"], errors="coerce").astype("float64")
    adjusted = raw.copy()
    factor_values = pd.Series(np.nan, index=out.index, dtype="float64")
    warnings_out: list[str] = []
    notes: list[str] = []
    output_scale = "raw_units"

    if transform.normalization_factor_col:
        factor_col = transform.normalization_factor_col
        if factor_col not in out.columns:
            raise ValueError(f"missing normalization factor column '{factor_col}'")
        factor_values = pd.to_numeric(out[factor_col], errors="coerce").astype("float64")
        if factor_values.isna().any():
            raise ValueError(f"{int(factor_values.isna().sum())} rows have non-numeric normalization factors in '{factor_col}'")
        if (factor_values <= 0).any():
            raise ValueError(f"normalization factors in '{factor_col}' must be positive")
        adjusted = raw / factor_values
        output_scale = "normalized_units"
        label = f" ({transform.normalization_label})" if transform.normalization_label else ""
        notes.append(f"raw y divided by {factor_col}{label}")
        warnings_out.append(
            f"target normalization applied using factor column '{factor_col}'; forecasts are in normalized units unless future factors are supplied externally"
        )

    modeled = _apply_target_transform_values(adjusted, transform.target)
    if transform.target != "none":
        notes.append(f"{transform.target} target transform")
        warnings_out.append(
            f"target transform '{transform.target}' applied for modeling; model outputs and backtests are inverse-transformed for reporting"
        )

    out["y"] = modeled
    audit = pd.DataFrame(
        {
            "unique_id": out["unique_id"].astype(str),
            "ds": pd.to_datetime(out["ds"]),
            "y_raw": raw,
            "normalization_factor_col": transform.normalization_factor_col or "",
            "normalization_factor": factor_values,
            "y_adjusted": adjusted,
            "target_transform": transform.target,
            "y_modeled": modeled,
            "output_scale": output_scale,
            "notes": "; ".join(notes),
        }
    )
    return out, audit[TARGET_TRANSFORM_AUDIT_COLUMNS], warnings_out


def inverse_target_transform_frame(
    frame: pd.DataFrame,
    transform: TargetTransform,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Inverse a monotonic target transform for forecast/backtest columns."""

    if frame.empty or transform == "none":
        return frame.copy()
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").astype("float64")
        out[column] = _inverse_target_transform_values(values, transform)
    return out


def _apply_target_transform_values(values: pd.Series, transform: TargetTransform) -> pd.Series:
    if transform == "none":
        return values
    if transform == "log":
        invalid = values.notna() & (values <= 0)
        if invalid.any():
            raise ValueError("log target transform requires all modeled target values to be > 0")
        return np.log(values)
    if transform == "log1p":
        invalid = values.notna() & (values <= -1)
        if invalid.any():
            raise ValueError("log1p target transform requires all modeled target values to be > -1")
        return np.log1p(values)
    raise ValueError(f"unsupported target transform: {transform}")


def _inverse_target_transform_values(values: pd.Series, transform: TargetTransform) -> pd.Series:
    if transform == "log":
        return np.exp(values)
    if transform == "log1p":
        return np.expm1(values)
    if transform == "none":
        return values
    raise ValueError(f"unsupported target transform: {transform}")


def label_anomalies(
    frame: pd.DataFrame,
    *,
    group_col: str = "unique_id",
    target_col: str = "y",
    score_col: str = "anomaly_score",
    label_col: str = "anomaly_label",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Label robust z-score anomalies per series using median absolute deviation."""

    if threshold <= 0:
        raise ValueError("anomaly threshold must be positive")
    required = {group_col, target_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"anomaly labeling requires columns {required}")

    out = frame.copy()
    scores = pd.Series(0.0, index=out.index, dtype="float64")
    for _, idx in out.groupby(group_col, sort=False).groups.items():
        values = pd.to_numeric(out.loc[idx, target_col], errors="coerce")
        median = float(values.median())
        mad = float((values - median).abs().median())
        if not np.isfinite(mad) or mad == 0:
            continue
        scores.loc[idx] = 0.6745 * (values - median) / mad
    out[score_col] = scores
    out[label_col] = scores.abs() >= threshold
    return out


def apply_event_adjustments(
    forecast: pd.DataFrame,
    events: Sequence[DriverEvent],
    *,
    value_col: str = "yhat",
    output_col: str = "yhat_scenario",
) -> pd.DataFrame:
    """Apply auditable additive or multiplicative event assumptions to a forecast."""

    if not events:
        return forecast.copy()
    required = {"unique_id", "ds", value_col}
    if not required.issubset(forecast.columns):
        raise ValueError(f"event adjustments require columns {required}")

    out = forecast.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["event_adjustment"] = 0.0
    out["event_names"] = ""
    adjusted = pd.to_numeric(out[value_col], errors="coerce").astype("float64")
    baseline = adjusted.copy()
    scenario_intervals = _scenario_interval_baselines(out, value_col=value_col)
    event_warnings: list[str] = []
    for event in events:
        mask = _event_mask(out, event)
        if not mask.any():
            event_warnings.append(
                f"event '{event.name}' matched 0 forecast rows; no adjustment applied "
                f"(window {event.start} to {event.end or event.start}, affected_unique_ids={list(event.affected_unique_ids)})"
            )
            continue
        weighted = event.magnitude * event.confidence
        if event.effect == "multiplicative":
            factor = 1 + weighted
            adjusted.loc[mask] = adjusted.loc[mask] * factor
            for interval_values in scenario_intervals.values():
                interval_values.loc[mask] = interval_values.loc[mask] * factor
        elif event.effect == "additive":
            adjusted.loc[mask] = adjusted.loc[mask] + weighted
            for interval_values in scenario_intervals.values():
                interval_values.loc[mask] = interval_values.loc[mask] + weighted
        else:
            raise ValueError(f"unsupported event effect: {event.effect}")
        out.loc[mask, "event_names"] = out.loc[mask, "event_names"].apply(
            lambda names: event.name if not names else f"{names}; {event.name}"
        )
    out[output_col] = adjusted.round(10)
    out["event_adjustment"] = (out[output_col] - baseline).round(10)
    _add_scenario_interval_columns(out, scenario_intervals=scenario_intervals, output_col=output_col)
    out.attrs["event_warnings"] = event_warnings
    return out


def _event_mask(frame: pd.DataFrame, event: DriverEvent) -> pd.Series:
    start = pd.Timestamp(event.start)
    end = pd.Timestamp(event.end) if event.end else start
    mask = frame["ds"].between(start, end)
    if event.affected_unique_ids:
        affected_ids = [str(unique_id) for unique_id in event.affected_unique_ids]
        mask &= frame["unique_id"].astype(str).isin(affected_ids)
    return mask


def _scenario_interval_baselines(out: pd.DataFrame, *, value_col: str) -> dict[str, pd.Series]:
    intervals: dict[str, pd.Series] = {}
    for column in list(out.columns):
        if column.startswith(f"{value_col}_lo_") or column.startswith(f"{value_col}_hi_"):
            intervals[column] = pd.to_numeric(out[column], errors="coerce").astype("float64").copy()
    return intervals


def _add_scenario_interval_columns(
    out: pd.DataFrame,
    *,
    scenario_intervals: dict[str, pd.Series],
    output_col: str,
) -> None:
    scenario_prefix = output_col.removesuffix("_scenario")
    point = pd.to_numeric(out[output_col], errors="coerce")
    for column, values in scenario_intervals.items():
        if "_lo_" in column:
            level = column.rsplit("_", 1)[-1]
            hi_source = column.replace("_lo_", "_hi_")
            if hi_source not in scenario_intervals:
                continue
            lo = values
            hi = scenario_intervals[hi_source]
            lower = pd.concat([lo, hi, point], axis=1).min(axis=1, skipna=True)
            out[f"{scenario_prefix}_scenario_lo_{level}"] = lower.round(10)
        elif "_hi_" in column:
            level = column.rsplit("_", 1)[-1]
            lo_source = column.replace("_hi_", "_lo_")
            if lo_source not in scenario_intervals:
                continue
            lo = scenario_intervals[lo_source]
            hi = values
            upper = pd.concat([lo, hi, point], axis=1).max(axis=1, skipna=True)
            out[f"{scenario_prefix}_scenario_hi_{level}"] = upper.round(10)
