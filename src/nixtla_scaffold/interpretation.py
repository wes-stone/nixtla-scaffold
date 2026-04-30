from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def backtest_windows_frame(run: Any) -> pd.DataFrame:
    if run.backtest_predictions.empty or "cutoff" not in run.backtest_predictions.columns:
        return pd.DataFrame(columns=["unique_id", "cutoff", "test_start", "test_end", "horizon_rows", "models"])
    rows: list[dict[str, Any]] = []
    model_cols = _model_columns(run.backtest_predictions, extra_exclude={"cutoff", "y"})
    for (uid, cutoff), grp in run.backtest_predictions.groupby(["unique_id", "cutoff"], sort=True):
        rows.append(
            {
                "unique_id": uid,
                "cutoff": cutoff,
                "test_start": pd.to_datetime(grp["ds"]).min().date().isoformat(),
                "test_end": pd.to_datetime(grp["ds"]).max().date().isoformat(),
                "horizon_rows": int(len(grp)),
                "models": "; ".join(model_cols),
            }
        )
    return pd.DataFrame(rows)


def seasonality_profile_frame(run: Any) -> pd.DataFrame:
    freq = str(run.profile.freq).upper()
    if run.history.empty:
        return pd.DataFrame(columns=["unique_id", "season_position", "mean_y", "index_vs_series_mean", "observations"])
    frame = run.history.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    if freq.startswith("D") or freq.startswith("B"):
        frame["season_position"] = frame["ds"].dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    elif freq.startswith("W"):
        frame["season_position"] = frame["ds"].dt.isocalendar().week.astype(int).astype(str)
        order = None
    elif freq.startswith("Q"):
        frame["season_position"] = "Q" + frame["ds"].dt.quarter.astype(str)
        order = ["Q1", "Q2", "Q3", "Q4"]
    elif freq in {"ME", "M", "MS"}:
        frame["season_position"] = frame["ds"].dt.month_name().str[:3]
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    else:
        frame["season_position"] = "all"
        order = ["all"]

    grouped = (
        frame.groupby(["unique_id", "season_position"], sort=False)
        .agg(mean_y=("y", "mean"), observations=("y", "count"))
        .reset_index()
    )
    series_mean = frame.groupby("unique_id")["y"].mean().rename("series_mean").reset_index()
    grouped = grouped.merge(series_mean, on="unique_id", how="left")
    grouped["index_vs_series_mean"] = grouped["mean_y"] / grouped["series_mean"].replace(0, np.nan)
    if order:
        grouped["season_position"] = pd.Categorical(grouped["season_position"], categories=order, ordered=True)
        grouped = grouped.sort_values(["unique_id", "season_position"])
        grouped["season_position"] = grouped["season_position"].astype(str)
    return grouped[["unique_id", "season_position", "mean_y", "index_vs_series_mean", "observations"]].reset_index(drop=True)


def seasonality_summary_frame(run: Any) -> pd.DataFrame:
    profile = seasonality_profile_frame(run)
    if profile.empty:
        return pd.DataFrame(columns=["unique_id", "seasonality_strength", "peak_position", "trough_position", "interpretation"])
    rows: list[dict[str, Any]] = []
    for uid, grp in profile.groupby("unique_id", sort=True):
        values = pd.to_numeric(grp["index_vs_series_mean"], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            strength = np.nan
        else:
            strength = float(finite.max() - finite.min())
        peak_idx = values.idxmax() if np.isfinite(values).any() else grp.index[0]
        trough_idx = values.idxmin() if np.isfinite(values).any() else grp.index[0]
        label = "weak"
        if np.isfinite(strength) and strength >= 0.30:
            label = "strong"
        elif np.isfinite(strength) and strength >= 0.12:
            label = "moderate"
        rows.append(
            {
                "unique_id": uid,
                "seasonality_strength": strength,
                "peak_position": grp.loc[peak_idx, "season_position"],
                "trough_position": grp.loc[trough_idx, "season_position"],
                "interpretation": f"{label} recurring seasonal pattern based on average seasonal index spread",
            }
        )
    return pd.DataFrame(rows)


SEASONALITY_DIAGNOSTIC_COLUMNS = [
    "unique_id",
    "frequency",
    "season_length",
    "history_rows",
    "cycle_count",
    "complete_cycles",
    "candidate_periods",
    "seasonality_strength",
    "trend_strength",
    "remainder_strength",
    "credibility_label",
    "credibility_reason",
    "warning",
    "interpretation",
]


def seasonality_diagnostics_frame(run: Any) -> pd.DataFrame:
    """One-row-per-series evidence for whether seasonality is credible."""

    if run.history.empty:
        return pd.DataFrame(columns=SEASONALITY_DIAGNOSTIC_COLUMNS)
    summary = seasonality_summary_frame(run)
    decomposition = seasonality_decomposition_frame(run)
    rows: list[dict[str, Any]] = []
    season_length = int(run.profile.season_length or 1)
    freq = str(run.profile.freq)
    for uid, grp in run.history.groupby("unique_id", sort=True):
        history_rows = int(len(grp))
        cycle_count = float(history_rows / season_length) if season_length > 0 else 0.0
        complete_cycles = int(np.floor(cycle_count))
        summary_row = summary[summary["unique_id"].astype(str) == str(uid)]
        seasonality_strength = _safe_float(summary_row["seasonality_strength"].iloc[0]) if not summary_row.empty else None
        trend_strength = _linear_trend_strength(grp.sort_values("ds")["y"])
        series_decomp = decomposition[decomposition["unique_id"].astype(str) == str(uid)]
        remainder_strength = _remainder_strength(series_decomp)
        label, reason, warning = _seasonality_credibility(
            freq=freq,
            season_length=season_length,
            cycle_count=cycle_count,
            seasonality_strength=seasonality_strength,
        )
        rows.append(
            {
                "unique_id": uid,
                "frequency": freq,
                "season_length": season_length,
                "history_rows": history_rows,
                "cycle_count": cycle_count,
                "complete_cycles": complete_cycles,
                "candidate_periods": _candidate_period_label(freq, season_length),
                "seasonality_strength": seasonality_strength,
                "trend_strength": trend_strength,
                "remainder_strength": remainder_strength,
                "credibility_label": label,
                "credibility_reason": reason,
                "warning": warning,
                "interpretation": _seasonality_diagnostic_interpretation(
                    label=label,
                    cycle_count=cycle_count,
                    seasonality_strength=seasonality_strength,
                    trend_strength=trend_strength,
                    warning=warning,
                ),
            }
        )
    return pd.DataFrame(rows, columns=SEASONALITY_DIAGNOSTIC_COLUMNS)


SEASONALITY_DECOMPOSITION_COLUMNS = [
    "unique_id",
    "ds",
    "season_position",
    "observed",
    "trend",
    "seasonal",
    "remainder",
]


def seasonality_decomposition_frame(run: Any) -> pd.DataFrame:
    """Simple additive decomposition for report evidence when enough cycles exist."""

    if run.history.empty:
        return pd.DataFrame(columns=SEASONALITY_DECOMPOSITION_COLUMNS)
    season_length = int(run.profile.season_length or 1)
    if season_length <= 1:
        return pd.DataFrame(columns=SEASONALITY_DECOMPOSITION_COLUMNS)
    rows: list[pd.DataFrame] = []
    for uid, grp in run.history.groupby("unique_id", sort=True):
        ordered = grp.sort_values("ds").copy()
        if len(ordered) < 2 * season_length:
            continue
        ordered["ds"] = pd.to_datetime(ordered["ds"])
        ordered["observed"] = pd.to_numeric(ordered["y"], errors="coerce")
        if ordered["observed"].notna().sum() < 2 * season_length:
            continue
        trend = (
            ordered["observed"]
            .rolling(window=season_length, center=True, min_periods=max(2, season_length // 2))
            .mean()
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
        )
        season_position = _season_position(ordered["ds"], str(run.profile.freq))
        detrended = ordered["observed"] - trend
        seasonal_index = detrended.groupby(season_position).mean()
        seasonal_index = seasonal_index - seasonal_index.mean()
        seasonal = season_position.map(seasonal_index)
        out = pd.DataFrame(
            {
                "unique_id": str(uid),
                "ds": ordered["ds"],
                "season_position": season_position.astype(str),
                "observed": ordered["observed"],
                "trend": trend,
                "seasonal": seasonal,
                "remainder": ordered["observed"] - trend - seasonal,
            }
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=SEASONALITY_DECOMPOSITION_COLUMNS)
    return pd.concat(rows, ignore_index=True)[SEASONALITY_DECOMPOSITION_COLUMNS]


def build_interpretation_payload(run: Any) -> dict[str, Any]:
    windows = backtest_windows_frame(run)
    seasonality = seasonality_summary_frame(run)
    seasonality_diagnostics = seasonality_diagnostics_frame(run)
    naive_comparison = _naive_comparison(run)
    return {
        "backtesting": {
            "window_count": int(len(windows)),
            "prediction_rows": int(len(run.backtest_predictions)),
            "windows": _records(windows),
            "notes": [
                "Each cutoff is a historical training endpoint; rows after the cutoff are forecast-vs-actual holdout periods.",
                "Compare backtest_long.csv with audit/backtest_windows.csv to understand model behavior over time.",
            ],
        },
        "naive_comparison": naive_comparison,
        "seasonality": {
            "frequency": run.profile.freq,
            "season_length": run.profile.season_length,
            "summary": _records(seasonality),
            "diagnostics": _records(seasonality_diagnostics),
            "notes": [
                "Seasonality profile uses simple historical seasonal averages for interpretability.",
                "Use this as a sanity check, not proof that a seasonal model is always best.",
            ],
        },
    }


def format_interpretation_markdown(run: Any) -> str:
    windows = backtest_windows_frame(run)
    seasonality = seasonality_summary_frame(run)
    seasonality_diagnostics = seasonality_diagnostics_frame(run)
    lines = ["# Forecast interpretation", "", "## Backtesting windows", ""]
    if windows.empty:
        lines.append("- No backtest windows were produced; history may be too short for rolling-origin evaluation.")
    else:
        for row in windows.head(12).to_dict("records"):
            lines.append(
                f"- `{row['unique_id']}` cutoff `{row['cutoff']}` tested {row['test_start']} to {row['test_end']} "
                f"({row['horizon_rows']} rows)."
            )
        if len(windows) > 12:
            lines.append(f"- Plus {len(windows) - 12} additional windows in `audit/backtest_windows.csv`.")
    lines.extend(["", "## Naive comparison", ""])
    naive_comp = run.interpretation().get("naive_comparison", {})
    for note in naive_comp.get("notes", []):
        lines.append(f"- {note}")
    for row in naive_comp.get("series", [])[:12]:
        status = "beats naive" if row.get("beats_naive") else "underperforms naive"
        improvement = f"{row['improvement_vs_naive']:.1%}" if row.get("improvement_vs_naive") is not None else "N/A"
        metric_name = str(row.get("metric_name", "rmse")).upper()
        lines.append(
            f"- `{row['unique_id']}`: selected `{row['selected_model']}` ({metric_name} {row.get('selected_metric', 'N/A')}) "
            f"vs `{row.get('naive_model', 'Naive')}` ({metric_name} {row.get('naive_metric', 'N/A')}) - {status} ({improvement})"
        )
    lines.extend(["", "## Seasonality interpretation", ""])
    if seasonality.empty:
        lines.append("- No seasonality summary was produced.")
    else:
        for row in seasonality.head(12).to_dict("records"):
            lines.append(
                f"- `{row['unique_id']}`: {row['interpretation']}; peak `{row['peak_position']}`, "
                f"trough `{row['trough_position']}`, strength {row['seasonality_strength']:.2f}."
            )
    lines.extend(["", "## Seasonality credibility", ""])
    if seasonality_diagnostics.empty:
        lines.append("- No seasonality diagnostics were produced.")
    else:
        for row in seasonality_diagnostics.head(12).to_dict("records"):
            warning = f" Warning: {row['warning']}" if row.get("warning") else ""
            lines.append(
                f"- `{row['unique_id']}`: {row['credibility_label']} credibility; "
                f"{row['cycle_count']:.1f} cycle(s), strength {_safe_display(row.get('seasonality_strength'))}, "
                f"trend {_safe_pct(row.get('trend_strength'))}.{warning}"
            )
    return "\n".join(lines) + "\n"


def _season_position(ds: pd.Series, freq: str) -> pd.Series:
    normalized = freq.upper()
    if normalized.startswith("D") or normalized.startswith("B"):
        return ds.dt.day_name()
    if normalized.startswith("W"):
        return ds.dt.isocalendar().week.astype(int).astype(str)
    if normalized.startswith("Q"):
        return "Q" + ds.dt.quarter.astype(str)
    if normalized in {"ME", "M", "MS"}:
        return ds.dt.month_name().str[:3]
    return pd.Series(["all"] * len(ds), index=ds.index)


def _linear_trend_strength(values: pd.Series) -> float | None:
    y = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(y) < 3:
        return None
    t = np.arange(len(y), dtype="float64")
    slope, intercept = np.polyfit(t, y, 1)
    trend = slope * t + intercept
    residual = y - trend
    trend_var = float(np.var(trend))
    residual_var = float(np.var(residual))
    total = trend_var + residual_var
    if not np.isfinite(total) or total <= 0:
        return 0.0
    return max(0.0, min(1.0, trend_var / total))


def _remainder_strength(decomposition: pd.DataFrame) -> float | None:
    if decomposition.empty:
        return None
    observed = pd.to_numeric(decomposition["observed"], errors="coerce")
    remainder = pd.to_numeric(decomposition["remainder"], errors="coerce")
    observed_var = float(observed.var(ddof=0)) if observed.notna().any() else 0.0
    remainder_var = float(remainder.var(ddof=0)) if remainder.notna().any() else 0.0
    if observed_var <= 0 or not np.isfinite(observed_var) or not np.isfinite(remainder_var):
        return None
    return max(0.0, min(1.0, remainder_var / observed_var))


def _seasonality_credibility(
    *,
    freq: str,
    season_length: int,
    cycle_count: float,
    seasonality_strength: float | None,
) -> tuple[str, str, str | None]:
    if season_length <= 1:
        return "not_applicable", "No repeating seasonal period was inferred for this frequency.", None
    period = _seasonal_period_name(freq, season_length)
    if cycle_count < 2:
        warning = f"Only {cycle_count:.1f} {period} cycle(s); need at least 2 before trusting seasonal claims."
        return "low", "Too few complete seasonal cycles.", warning
    if seasonality_strength is None or not np.isfinite(seasonality_strength) or seasonality_strength < 0.12:
        return "low", "Enough cycles may exist, but the seasonal index spread is weak.", None
    if cycle_count < 4:
        return "medium", "At least two cycles exist, but seasonal evidence is still limited.", None
    return "high", "Multiple seasonal cycles and visible seasonal index spread support the pattern.", None


def _candidate_period_label(freq: str, season_length: int) -> str:
    return f"{season_length} ({_seasonal_period_name(freq, season_length)})"


def _seasonal_period_name(freq: str, season_length: int) -> str:
    normalized = freq.upper()
    if normalized in {"ME", "M", "MS"} and season_length == 12:
        return "annual"
    if normalized.startswith("Q") and season_length == 4:
        return "annual"
    if (normalized.startswith("D") or normalized.startswith("B")) and season_length == 7:
        return "weekly"
    if normalized.startswith("W") and season_length == 52:
        return "annual"
    return f"length-{season_length}"


def _seasonality_diagnostic_interpretation(
    *,
    label: str,
    cycle_count: float,
    seasonality_strength: float | None,
    trend_strength: float | None,
    warning: str | None,
) -> str:
    parts = [
        f"{label} seasonality credibility",
        f"{cycle_count:.1f} observed cycle(s)",
        f"seasonal strength {_safe_display(seasonality_strength)}",
        f"trend strength {_safe_pct(trend_strength)}",
    ]
    if warning:
        parts.append(warning)
    return "; ".join(parts)


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_display(value: Any) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.2f}"


def _safe_pct(value: Any) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.0%}"


def _model_columns(frame: pd.DataFrame, *, extra_exclude: set[str] | None = None) -> list[str]:
    excluded = {"unique_id", "ds"} | (extra_exclude or set())
    return [
        col
        for col in frame.columns
        if col not in excluded and "-lo-" not in col and "-hi-" not in col
    ]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.to_dict("records")


def _naive_comparison(run: Any) -> dict[str, Any]:
    """Compare the selected champion model against naive benchmarks per series."""
    if run.backtest_metrics.empty or run.model_selection.empty:
        return {"series": [], "notes": ["No backtest metrics available for naive comparison."]}
    naive_models = {"Naive", "SeasonalNaive"}
    naive_metrics = run.backtest_metrics[run.backtest_metrics["model"].isin(naive_models)].copy()
    if naive_metrics.empty:
        return {"series": [], "notes": ["No naive benchmark metrics found in backtest results."]}
    metric = "rmse" if "rmse" in run.backtest_metrics.columns and "rmse" in run.model_selection.columns else "wape"
    tie_breakers = [col for col in ["mae", "wape", "bias"] if col in naive_metrics.columns and col != metric]
    best_naive = naive_metrics.sort_values(["unique_id", metric, *tie_breakers]).groupby("unique_id", as_index=False).first()
    best_naive = best_naive.rename(columns={"model": "naive_model", metric: "naive_metric"})
    merged = run.model_selection[["unique_id", "selected_model", metric]].rename(columns={metric: "selected_metric"}).merge(
        best_naive[["unique_id", "naive_model", "naive_metric"]], on="unique_id", how="left"
    )
    merged["beats_naive"] = (
        merged["selected_metric"].notna()
        & merged["naive_metric"].notna()
        & (merged["selected_metric"] <= merged["naive_metric"])
    )
    merged["improvement_vs_naive"] = np.where(
        merged["naive_metric"].notna() & (merged["naive_metric"] > 0),
        1.0 - merged["selected_metric"] / merged["naive_metric"],
        np.nan,
    )
    records = []
    for row in merged.to_dict("records"):
        records.append({
            "unique_id": row["unique_id"],
            "selected_model": row["selected_model"],
            "metric_name": metric,
            "selected_metric": _safe_round(row.get("selected_metric")),
            "naive_model": row.get("naive_model"),
            "naive_metric": _safe_round(row.get("naive_metric")),
            "beats_naive": bool(row.get("beats_naive", False)),
            "improvement_vs_naive": _safe_round(row.get("improvement_vs_naive")),
        })
    beats_count = int(merged["beats_naive"].sum())
    total = int(len(merged))
    notes = [
        f"{beats_count} of {total} series beat or match the naive benchmark on backtested {metric.upper()}.",
    ]
    if beats_count < total:
        notes.append(
            "Series where the champion does NOT beat naive may benefit from simpler models, "
            "additional history, or explicit event/driver assumptions."
        )
    return {"series": records, "notes": notes}


def _safe_round(value: Any, decimals: int = 6) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), decimals)
