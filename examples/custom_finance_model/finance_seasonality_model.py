from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def forecast(
    history: pd.DataFrame,
    *,
    horizon: int,
    freq: str,
    cutoff: pd.Timestamp,
    levels: tuple[int, ...],
    context: dict,
) -> pd.DataFrame:
    """Callable contract for the scaffold custom-model runner."""

    future_grid = pd.DataFrame(context["future_grid"])
    return build_finance_forecast(history, future_grid=future_grid)


def build_finance_forecast(
    history: pd.DataFrame,
    *,
    future_grid: pd.DataFrame,
    annual_target: float | None = None,
    growth_window: int = 6,
    seasonality_years: int = 2,
) -> pd.DataFrame:
    """Average recent MoM growth into an annual total, then allocate by monthly seasonality."""

    prepared = _prepare_history(history)
    grid = future_grid[["unique_id", "ds"]].copy()
    grid["unique_id"] = grid["unique_id"].astype(str)
    grid["ds"] = pd.to_datetime(grid["ds"], errors="raise")

    frames: list[pd.DataFrame] = []
    for uid, grid_part in grid.groupby("unique_id", sort=True):
        series = prepared[prepared["unique_id"] == uid].sort_values("ds").copy()
        forecast_part = grid_part.sort_values("ds").copy()
        if series.empty:
            forecast_part["yhat"] = 0.0
            frames.append(forecast_part)
            continue

        month_shares = _historical_month_shares(series, seasonality_years=seasonality_years)
        annual_total = float(annual_target) if annual_target is not None else _project_next_year_total(series, growth_window=growth_window)
        forecast_part["month"] = forecast_part["ds"].dt.month
        forecast_part["yhat"] = forecast_part["month"].map(month_shares).astype(float) * annual_total
        forecast_part = forecast_part.drop(columns=["month"])
        frames.append(forecast_part)

    output = pd.concat(frames, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    output["yhat"] = pd.to_numeric(output["yhat"], errors="coerce")
    if not np.isfinite(output["yhat"].to_numpy(dtype="float64")).all():
        raise ValueError("finance custom model produced non-finite yhat values")
    return output[["unique_id", "ds", "yhat"]]


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "y"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")
    prepared = history[["unique_id", "ds", "y"]].copy()
    prepared["unique_id"] = prepared["unique_id"].astype(str)
    prepared["ds"] = pd.to_datetime(prepared["ds"], errors="raise")
    prepared["y"] = pd.to_numeric(prepared["y"], errors="raise").astype(float)
    return prepared.dropna(subset=["unique_id", "ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _historical_month_shares(series: pd.DataFrame, *, seasonality_years: int) -> dict[int, float]:
    lookback = max(int(seasonality_years), 1) * 12
    series = series.copy()
    series["year"] = series["ds"].dt.year
    series["month"] = series["ds"].dt.month
    complete_years = (
        series.groupby("year")["month"]
        .nunique()
        .loc[lambda counts: counts >= 12]
        .index.to_list()
    )
    if complete_years:
        selected_years = sorted(complete_years)[-max(int(seasonality_years), 1) :]
        recent = series[series["year"].isin(selected_years)].copy()
    else:
        recent = series.tail(lookback).copy()
    recent["year"] = recent["ds"].dt.year
    recent["month"] = recent["ds"].dt.month
    recent["year_total"] = recent.groupby("year")["y"].transform("sum")
    valid = recent["year_total"].abs() > 1e-9
    shares = pd.Series(1.0 / 12.0, index=range(1, 13), dtype="float64")
    if valid.any():
        monthly = (
            recent.loc[valid]
            .assign(month_share=lambda frame: frame["y"] / frame["year_total"])
            .groupby("month")["month_share"]
            .median()
        )
        shares.update(monthly)
    shares = shares.clip(lower=0.0)
    total = float(shares.sum())
    if total <= 0 or not np.isfinite(total):
        return {month: 1.0 / 12.0 for month in range(1, 13)}
    shares = shares / total
    return {int(month): float(value) for month, value in shares.items()}


def _project_next_year_total(series: pd.DataFrame, *, growth_window: int) -> float:
    values = series["y"].astype(float).to_numpy()
    if len(values) == 0:
        return 0.0
    base = float(np.nansum(values[-min(len(values), 12) :]))
    if len(values) < 2 or abs(float(values[-1])) <= 1e-9:
        return base

    pct = pd.Series(values).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if pct.empty:
        mom_growth = 0.0
    else:
        mom_growth = float(pct.tail(max(int(growth_window), 1)).median())
        mom_growth = float(np.clip(mom_growth, -0.2, 0.2))

    last_value = float(values[-1])
    projected = [last_value * ((1.0 + mom_growth) ** step) for step in range(1, 13)]
    projected_total = float(np.nansum(projected))
    return projected_total if np.isfinite(projected_total) and abs(projected_total) > 1e-9 else base


def _main() -> None:
    parser = argparse.ArgumentParser(description="Simple finance custom model for nixtla-scaffold.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--future-grid", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annual-target", type=float, default=None, help="Optional calendar-year annual target to allocate by historical month shares.")
    parser.add_argument("--growth-window", type=int, default=6, help="Recent periods used for median MoM growth when no annual target is supplied.")
    parser.add_argument("--seasonality-years", type=int, default=2, help="Most recent years used for month-of-year share estimates.")
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    future_grid = pd.read_csv(args.future_grid)
    _ = json.loads(Path(args.context).read_text(encoding="utf-8"))
    output = build_finance_forecast(
        history,
        future_grid=future_grid,
        annual_target=args.annual_target,
        growth_window=args.growth_window,
        seasonality_years=args.seasonality_years,
    )
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    _main()
