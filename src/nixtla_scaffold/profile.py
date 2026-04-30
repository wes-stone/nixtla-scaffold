from __future__ import annotations

from calendar import month_abbr
from dataclasses import replace

import numpy as np
import pandas as pd

from nixtla_scaffold.schema import DataProfile, ForecastSpec, SeriesProfile


def infer_frequency(df: pd.DataFrame) -> str:
    dates = pd.Series(pd.to_datetime(df["ds"]).dropna().sort_values().unique())
    if len(dates) < 2:
        raise ValueError("could not infer frequency from fewer than 2 distinct timestamps; pass --freq explicitly")
    if len(dates) == 2:
        short_freq = _infer_two_point_frequency(dates)
        if short_freq is not None:
            return short_freq
        raise ValueError("could not infer frequency from 2 irregular timestamps; pass --freq explicitly")

    inferred = pd.infer_freq(dates)
    if inferred:
        return _normalize_freq(inferred, dates)

    month_period_diffs = dates.dt.to_period("M").astype("int64").diff().dropna()
    if not month_period_diffs.empty and (month_period_diffs % 3 == 0).all() and month_period_diffs.median() >= 3:
        if dates.dt.is_month_end.mean() >= 0.8:
            return _quarter_end_alias(pd.Timestamp(dates.iloc[0]))
        if (dates.dt.day == 1).mean() >= 0.8:
            return _quarter_start_alias(pd.Timestamp(dates.iloc[0]))
    if not month_period_diffs.empty and month_period_diffs.max() <= 3:
        if dates.dt.is_month_end.mean() >= 0.8:
            return "ME"
        if (dates.dt.day == 1).mean() >= 0.8:
            return "MS"

    deltas = dates.diff().dropna().dt.days
    if deltas.empty:
        raise ValueError("could not infer frequency; pass --freq explicitly")
    median_days = float(deltas.median())
    if 27 <= median_days <= 32:
        return "ME" if dates.dt.is_month_end.mean() >= 0.8 else "MS"
    if 6 <= median_days <= 8:
        return "W"
    if 88 <= median_days <= 93:
        return "QE"
    return "D"


def season_length_for_freq(freq: str, min_obs: int) -> int:
    normalized = freq.upper()
    if normalized.startswith("W"):
        return 52
    if normalized.startswith("Q"):
        return 4
    if normalized in {"ME", "M", "MS"}:
        return 12
    if normalized.startswith("B"):
        return 5
    if normalized.startswith("D"):
        return 7
    return 1


def profile_dataset(df: pd.DataFrame, spec: ForecastSpec | None = None) -> DataProfile:
    spec = spec or ForecastSpec()
    required = {"unique_id", "ds", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"expected canonical columns {required}, got {set(df.columns)}")
    if df.empty:
        raise ValueError("forecast input is empty; at least one row is required")

    frame = df.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    freq = spec.freq or infer_frequency(frame)
    counts = frame.groupby("unique_id").size()
    min_obs = int(counts.min()) if not counts.empty else 0
    season_length = spec.season_length or season_length_for_freq(freq, min_obs)

    duplicate_rows = int(frame.duplicated(["unique_id", "ds"]).sum())
    null_y = int(frame["y"].isna().sum())
    zero_y = int((frame["y"] == 0).sum())
    negative_y = int((frame["y"] < 0).sum())

    series_profiles: list[SeriesProfile] = []
    missing_total = 0
    warnings: list[str] = []
    for uid, grp in frame.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds")
        missing = _missing_timestamps(grp, freq)
        missing_total += missing
        row_count = int(len(grp))
        readiness, series_warnings = _readiness(row_count, season_length, spec.horizon)
        if missing:
            series_warnings.append(f"{missing} missing timestamps at inferred frequency {freq}")
        series_profiles.append(
            SeriesProfile(
                unique_id=str(uid),
                rows=row_count,
                start=_date_str(grp["ds"].min()),
                end=_date_str(grp["ds"].max()),
                missing_timestamps=missing,
                null_y=int(grp["y"].isna().sum()),
                zero_y=int((grp["y"] == 0).sum()),
                negative_y=int((grp["y"] < 0).sum()),
                readiness=readiness,
                warnings=tuple(series_warnings),
            )
        )

    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate unique_id/ds rows detected")
    if null_y:
        warnings.append(f"{null_y} null y values detected")
    if missing_total:
        warnings.append(f"{missing_total} missing timestamps detected across all series")
    if negative_y:
        warnings.append(f"{negative_y} negative y values detected")
    if min_obs and min_obs < max(4, spec.horizon):
        warnings.append("one or more series has limited history relative to the forecast horizon")

    return DataProfile(
        rows=int(len(frame)),
        series_count=int(frame["unique_id"].nunique()),
        freq=freq,
        season_length=season_length,
        start=_date_str(frame["ds"].min()) if len(frame) else None,
        end=_date_str(frame["ds"].max()) if len(frame) else None,
        min_obs_per_series=min_obs,
        max_obs_per_series=int(counts.max()) if not counts.empty else 0,
        duplicate_rows=duplicate_rows,
        missing_timestamps=missing_total,
        null_y=null_y,
        zero_y=zero_y,
        negative_y=negative_y,
        data_freshness=_date_str(frame["ds"].max()) if len(frame) else None,
        warnings=tuple(warnings),
        series=tuple(series_profiles),
    )


def repair_time_index(
    df: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """Make each series regular and impute missing y values with an explicit policy."""

    warnings: list[str] = []
    if df.empty:
        raise ValueError("forecast input is empty; at least one row is required")
    duplicates = int(df.duplicated(["unique_id", "ds"]).sum())
    if duplicates:
        raise ValueError(f"{duplicates} duplicate unique_id/ds rows detected; aggregate or deduplicate before forecasting")

    frames: list[pd.DataFrame] = []
    for uid, grp in df.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds")
        dates = pd.date_range(grp["ds"].min(), grp["ds"].max(), freq=profile.freq)
        regular = pd.DataFrame({"ds": dates})
        regular["unique_id"] = str(uid)
        regular = regular.merge(grp[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
        missing_before = int(regular["y"].isna().sum())
        if spec.fill_method == "drop":
            regular = regular.dropna(subset=["y"])
        elif spec.fill_method == "ffill":
            regular["y"] = regular["y"].ffill().bfill()
        elif spec.fill_method == "zero":
            regular["y"] = regular["y"].fillna(0)
        elif spec.fill_method == "interpolate":
            regular["y"] = regular["y"].interpolate(method="linear").ffill().bfill()
        else:
            raise ValueError(f"unsupported fill_method: {spec.fill_method}")
        if missing_before and spec.fill_method != "drop":
            warnings.append(f"{uid}: imputed {missing_before} values with fill_method={spec.fill_method}")
        elif missing_before:
            warnings.append(f"{uid}: dropped {missing_before} structurally missing timestamps")
        if regular["y"].notna().sum() == 0:
            raise ValueError(f"{uid}: no usable numeric y values remain after repair")
        frames.append(regular)

    out = pd.concat(frames, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out[["unique_id", "ds", "y"]], warnings


def _normalize_freq(freq: str, dates: pd.Series) -> str:
    if freq in {"M", "ME"} or freq.startswith("M"):
        return "ME" if dates.dt.is_month_end.mean() >= 0.8 else "MS"
    if freq.startswith("W"):
        return freq
    if freq.startswith("QS"):
        return "QS" if freq == "QS-OCT" else freq
    if freq.startswith("QE"):
        return "QE" if freq == "QE-DEC" else freq
    if freq.startswith("Q"):
        if dates.dt.is_quarter_start.mean() >= 0.8:
            return _quarter_start_alias(pd.Timestamp(dates.iloc[0]))
        return _quarter_end_alias(pd.Timestamp(dates.iloc[0]))
    if freq.startswith("B"):
        return "B"
    if freq.startswith("D"):
        return "D"
    return freq


def _infer_two_point_frequency(dates: pd.Series) -> str | None:
    first = pd.Timestamp(dates.iloc[0])
    second = pd.Timestamp(dates.iloc[1])
    if first.is_month_end and second.is_month_end:
        month_delta = (second.year - first.year) * 12 + second.month - first.month
        if month_delta == 1:
            return "ME"
        if month_delta == 3:
            return _quarter_end_alias(first)
    if first.day == 1 and second.day == 1:
        month_delta = (second.year - first.year) * 12 + second.month - first.month
        if month_delta == 1:
            return "MS"
        if month_delta == 3:
            return _quarter_start_alias(first)
    day_delta = int((second - first).days)
    if day_delta == 1:
        return "D"
    if day_delta == 7:
        return "W"
    return None


def _quarter_start_alias(timestamp: pd.Timestamp) -> str:
    anchor = _quarter_anchor_month(timestamp)
    return "QS" if anchor == "OCT" else f"QS-{anchor}"


def _quarter_end_alias(timestamp: pd.Timestamp) -> str:
    anchor = _quarter_anchor_month(timestamp)
    return "QE" if anchor == "DEC" else f"QE-{anchor}"


def _quarter_anchor_month(timestamp: pd.Timestamp) -> str:
    cycle_month = ((timestamp.month - 1) % 3) + 1
    anchor_month = ((cycle_month - 4) % 12) + 1
    return month_abbr[anchor_month].upper()


def _missing_timestamps(grp: pd.DataFrame, freq: str) -> int:
    if len(grp) < 2:
        return 0
    expected = pd.date_range(grp["ds"].min(), grp["ds"].max(), freq=freq)
    observed = pd.Index(pd.to_datetime(grp["ds"]))
    return int(len(expected.difference(observed)))


def _readiness(rows: int, season_length: int, horizon: int) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if rows < 2:
        return "insufficient", ["fewer than 2 observations"]
    if rows < 4:
        return "limited", ["fewer than 4 observations; baseline-only forecast recommended"]
    if rows < max(2 * season_length, horizon + 2):
        warnings.append("history is short for seasonal backtesting")
        return "usable", warnings
    return "rich", warnings


def _date_str(value: pd.Timestamp | float | None) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value).date().isoformat()

