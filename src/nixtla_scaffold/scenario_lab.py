from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.interpretation import backtest_windows_frame, seasonality_summary_frame
from nixtla_scaffold.schema import ForecastSpec


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    archetype: str
    frequency: str
    train_rows: int
    holdout_rows: int
    series_count: int
    status: str
    engine: str
    selected_models: str
    wape: float | None
    naive_wape: float | None
    error_metric: str
    accuracy_score: float
    validity_score: float
    ease_score: float
    explainability_score: float
    composite_score: float
    warnings_count: int
    feedback: str


def run_scenario_lab(
    *,
    count: int = 100,
    output_dir: str | Path = "runs/scenario_lab",
    model_policy: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [_run_one_scenario(idx, model_policy=model_policy, seed=seed) for idx in range(count)]
    frame = pd.DataFrame(asdict(result) for result in results)
    summary = _summary_payload(frame, count=count, model_policy=model_policy, seed=seed)
    recommendations = _recommendations(frame)
    frame.to_csv(out / "scenario_scores.csv", index=False)
    (out / "scenario_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "scenario_recommendations.json").write_text(
        json.dumps(recommendations, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "recommendations": recommendations, "output_dir": str(out)}


def _run_one_scenario(index: int, *, model_policy: str, seed: int) -> ScenarioResult:
    scenario = _make_scenario(index, seed=seed)
    train = scenario["train"]
    holdout = scenario["holdout"]
    freq = scenario["freq"]
    horizon = scenario["horizon"]
    archetype = scenario["archetype"]
    spec = ForecastSpec(horizon=horizon, freq=freq, model_policy=model_policy, verbose=False)
    try:
        run = run_forecast(train, spec)
        merged = run.forecast.merge(holdout, on=["unique_id", "ds"], how="inner")
        scale_by_series = _fallback_scale_by_series(train)
        if not merged.empty:
            merged["fallback_scale"] = merged["unique_id"].map(scale_by_series).fillna(1.0)
            wape, error_metric = _holdout_error(merged["yhat"], merged["y_actual"], merged["fallback_scale"])
        else:
            wape, error_metric = np.nan, "unmatched_holdout"
        naive_wape = _naive_holdout_wape(train, holdout, freq)
        accuracy_score = _accuracy_score(wape, naive_wape)
        validity_score = _validity_score(run, expected_rows=len(holdout), matched_rows=len(merged))
        ease_score = _ease_score(run, scenario)
        explainability_score = _explainability_score(run)
        composite = 0.35 * validity_score + 0.35 * accuracy_score + 0.20 * ease_score + 0.10 * explainability_score
        selected_models = "; ".join(sorted(run.model_selection["selected_model"].dropna().astype(str).unique()))
        return ScenarioResult(
            scenario_id=f"s{index + 1:03d}",
            archetype=archetype,
            frequency=freq,
            train_rows=int(len(train)),
            holdout_rows=int(len(holdout)),
            series_count=int(train["unique_id"].nunique()),
            status="passed",
            engine=run.engine,
            selected_models=selected_models,
            wape=_finite_or_none(wape),
            naive_wape=_finite_or_none(naive_wape),
            error_metric=error_metric,
            accuracy_score=round(accuracy_score, 2),
            validity_score=round(validity_score, 2),
            ease_score=round(ease_score, 2),
            explainability_score=round(explainability_score, 2),
            composite_score=round(float(composite), 2),
            warnings_count=len(run.warnings),
            feedback=_feedback(archetype, wape, naive_wape, error_metric, run.warnings),
        )
    except Exception as exc:
        return ScenarioResult(
            scenario_id=f"s{index + 1:03d}",
            archetype=archetype,
            frequency=freq,
            train_rows=int(len(train)),
            holdout_rows=int(len(holdout)),
            series_count=int(train["unique_id"].nunique()),
            status="crashed",
            engine="none",
            selected_models="",
            wape=None,
            naive_wape=None,
            error_metric="crash",
            accuracy_score=0.0,
            validity_score=0.0,
            ease_score=0.0,
            explainability_score=0.0,
            composite_score=0.0,
            warnings_count=0,
            feedback=f"Crash: {type(exc).__name__}: {exc}",
        )


def _make_scenario(index: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + index)
    archetype_idx = index % 20
    variant = index // 20
    horizon = [3, 4, 6, 8, 12][variant % 5]
    specs = [
        ("monthly_trend_seasonal", "ME", 48, horizon),
        ("monthly_flat_noise", "ME", 30, horizon),
        ("monthly_step_holdout", "ME", 36, horizon),
        ("future_launch_shock", "ME", 36, horizon),
        ("intermittent_demand", "ME", 36, horizon),
        ("short_monthly", "ME", 8 + variant, min(3, horizon)),
        ("daily_weekly_seasonal", "D", 90, 14),
        ("business_day", "B", 70, 10),
        ("weekly_trend", "W-SUN", 72, 8),
        ("quarterly", "QE", 20, 4),
        ("fiscal_quarter_start", "QS-NOV", 16, 4),
        ("missing_internal_month", "ME", 36, horizon),
        ("outlier_in_train", "ME", 36, horizon),
        ("negative_cost_series", "ME", 30, horizon),
        ("three_series_panel", "ME", 30, horizon),
        ("many_sparse_panel", "ME", 14, 3),
        ("exponential_growth", "ME", 36, horizon),
        ("declining_trend", "ME", 36, horizon),
        ("high_noise", "ME", 36, horizon),
        ("level_shift_in_train", "ME", 36, horizon),
    ]
    archetype, freq, train_periods, h = specs[archetype_idx]
    start = "2022-01-31" if freq == "ME" else "2022-01-01"
    if freq == "QS-NOV":
        start = "2022-02-01"
    dates = pd.date_range(start, periods=train_periods + h, freq=freq)

    if archetype == "three_series_panel":
        full = pd.concat(
            [
                _series_frame(f"Series_{letter}", dates, _values("monthly_trend_seasonal", len(dates), rng, offset=idx * 20))
                for idx, letter in enumerate(["A", "B", "C"])
            ]
        ).reset_index(drop=True)
    elif archetype == "many_sparse_panel":
        frames = []
        for series_idx in range(8):
            series_dates = dates[series_idx % 3 :]
            values = _values("short_monthly", len(series_dates), rng, offset=series_idx * 5)
            frames.append(_series_frame(f"SKU_{series_idx + 1}", series_dates, values))
        full = pd.concat(frames, ignore_index=True)
    else:
        full = _series_frame(archetype, dates, _values(archetype, len(dates), rng))

    train_cutoff = full.groupby("unique_id")["ds"].transform(lambda values: values.sort_values().iloc[-h])
    train = full[full["ds"] < train_cutoff].copy()
    holdout = full[full["ds"] >= train_cutoff].copy().rename(columns={"y": "y_actual"})
    if archetype == "missing_internal_month" and len(train) > 10:
        train = train.drop(train.index[[5, 11]]).reset_index(drop=True)
    return {"archetype": archetype, "freq": freq, "horizon": h, "train": train.reset_index(drop=True), "holdout": holdout.reset_index(drop=True)}


def _values(archetype: str, n: int, rng: np.random.Generator, *, offset: float = 0) -> np.ndarray:
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 2.5, n)
    base = 100 + offset
    if archetype in {"monthly_trend_seasonal", "three_series_panel"}:
        return base + 2.2 * t + seasonal + noise
    if archetype == "monthly_flat_noise":
        return base + noise
    if archetype == "monthly_step_holdout":
        return base + 1.5 * t + seasonal + np.where(t >= n - 4, 35, 0) + noise
    if archetype == "future_launch_shock":
        return base + 1.2 * t + seasonal + np.where(t >= n - 6, 60, 0) + noise
    if archetype == "intermittent_demand":
        values = np.where(rng.random(n) < 0.72, 0, base + rng.normal(0, 15, n))
        return np.maximum(values, 0)
    if archetype == "short_monthly":
        return base + 3 * t + noise
    if archetype == "daily_weekly_seasonal":
        return base + 0.4 * t + 12 * np.sin(2 * np.pi * t / 7) + noise
    if archetype == "business_day":
        return base + 0.5 * t + 5 * np.sin(2 * np.pi * t / 5) + noise
    if archetype == "weekly_trend":
        return base + 1.1 * t + 6 * np.sin(2 * np.pi * t / 52) + noise
    if archetype in {"quarterly", "fiscal_quarter_start"}:
        return base + 4 * t + 14 * np.sin(2 * np.pi * t / 4) + noise
    if archetype == "missing_internal_month":
        return base + 2.0 * t + seasonal + noise
    if archetype == "outlier_in_train":
        values = base + 2.0 * t + seasonal + noise
        values[n // 2] += 90
        return values
    if archetype == "negative_cost_series":
        return -(base + 1.8 * t + seasonal + noise)
    if archetype == "many_sparse_panel":
        return base + 1.5 * t + noise
    if archetype == "exponential_growth":
        return base * np.power(1.035, t) + noise
    if archetype == "declining_trend":
        return base + 140 - 2.6 * t + seasonal + noise
    if archetype == "high_noise":
        return base + 1.6 * t + seasonal + rng.normal(0, 18, n)
    if archetype == "level_shift_in_train":
        return base + 1.0 * t + seasonal + np.where(t >= n // 2, 45, 0) + noise
    return base + t + noise


def _series_frame(unique_id: str, dates: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": unique_id, "ds": dates, "y": values})


def _wape(pred: pd.Series, actual: pd.Series) -> float:
    denom = float(np.nansum(np.abs(actual.to_numpy(dtype="float64"))))
    if denom == 0:
        return np.nan
    return float(np.nansum(np.abs(pred.to_numpy(dtype="float64") - actual.to_numpy(dtype="float64"))) / denom)


def _holdout_error(pred: pd.Series, actual: pd.Series, fallback_scale: pd.Series) -> tuple[float, str]:
    actual_values = actual.to_numpy(dtype="float64")
    pred_values = pred.to_numpy(dtype="float64")
    denom = float(np.nansum(np.abs(actual_values)))
    metric = "wape"
    if denom == 0:
        fallback_values = fallback_scale.to_numpy(dtype="float64")
        denom = float(np.nansum(np.maximum(fallback_values, 1.0)))
        metric = "scale_normalized_absolute_error"
    if denom == 0:
        return np.nan, metric
    return float(np.nansum(np.abs(pred_values - actual_values)) / denom), metric


def _naive_holdout_wape(train: pd.DataFrame, holdout: pd.DataFrame, freq: str) -> float:
    rows: list[pd.DataFrame] = []
    scale_by_series = _fallback_scale_by_series(train)
    for uid, grp in train.groupby("unique_id", sort=True):
        last_value = float(grp.sort_values("ds")["y"].iloc[-1])
        actual = holdout[holdout["unique_id"] == uid].copy()
        actual["naive"] = last_value
        actual["fallback_scale"] = scale_by_series.get(uid, 1.0)
        rows.append(actual)
    merged = pd.concat(rows, ignore_index=True)
    error, _ = _holdout_error(merged["naive"], merged["y_actual"], merged["fallback_scale"])
    return error


def _fallback_scale_by_series(train: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    for uid, grp in train.groupby("unique_id", sort=True):
        y = pd.to_numeric(grp["y"], errors="coerce").dropna().to_numpy(dtype="float64")
        magnitudes = np.abs(y[np.isfinite(y)])
        nonzero = magnitudes[magnitudes > 0]
        scale = float(np.nanmedian(nonzero)) if len(nonzero) else 1.0
        scales[str(uid)] = max(scale, 1.0)
    return scales


def _accuracy_score(wape: float, naive_wape: float) -> float:
    if not np.isfinite(wape):
        return 0.0
    absolute = max(0.0, 100.0 * (1.0 - min(wape, 1.25) / 1.25))
    relative = 50.0
    if np.isfinite(naive_wape):
        relative = 100.0 if wape <= naive_wape else max(0.0, 100.0 * (naive_wape / max(wape, 1e-9)))
    return float(0.7 * absolute + 0.3 * relative)


def _validity_score(run: Any, *, expected_rows: int, matched_rows: int) -> float:
    score = 0.0
    score += 25 if len(run.forecast) == expected_rows else max(0, 25 - abs(len(run.forecast) - expected_rows) * 3)
    score += 20 if matched_rows == expected_rows else max(0, 20 - (expected_rows - matched_rows) * 4)
    score += 15 if run.forecast["yhat"].notna().all() else 0
    score += 15 if not run.model_selection.empty else 0
    score += 10 if run.diagnostics()["status"] == "success" else 0
    score += 10 if run.interpretation()["seasonality"]["summary"] is not None else 0
    score += 5 if run.manifest()["outputs"] else 0
    return float(min(100, score))


def _ease_score(run: Any, scenario: dict[str, Any]) -> float:
    score = 100.0
    score -= min(25, len(run.warnings) * 4)
    if scenario["archetype"] in {"missing_internal_month", "many_sparse_panel", "short_monthly"}:
        score -= 6
    if scenario["freq"] in {"QS-NOV", "QE-OCT"}:
        score -= 4
    return float(max(0, score))


def _explainability_score(run: Any) -> float:
    score = 0.0
    score += 20 if not run.model_selection.empty else 0
    score += 20 if not run.backtest_metrics.empty else 0
    score += 20 if not run.backtest_predictions.empty and not backtest_windows_frame(run).empty else 0
    score += 20 if not seasonality_summary_frame(run).empty else 0
    score += 20 if run.diagnostics()["next_diagnostic_steps"] else 0
    return float(score)


def _feedback(archetype: str, wape: float, naive_wape: float, error_metric: str, warnings: list[str]) -> str:
    notes: list[str] = []
    if error_metric == "scale_normalized_absolute_error":
        notes.append("zero-actual holdout scored with scale-normalized absolute error")
    if np.isfinite(wape) and wape > 0.25:
        notes.append("accuracy weak on holdout; inspect event/regressor assumptions")
    if np.isfinite(naive_wape) and np.isfinite(wape) and wape > naive_wape:
        notes.append("underperformed naive holdout baseline")
    if archetype in {"future_launch_shock", "monthly_step_holdout"}:
        notes.append("known future shock needs explicit DriverEvent or MLForecast regressor path")
    if archetype == "intermittent_demand":
        notes.append("intermittent demand would benefit from Croston/ADIDA/IMAPA policy")
    if archetype == "many_sparse_panel":
        notes.append("sparse panel needs pooling or hierarchy-guided fallback")
    if warnings:
        notes.append(f"{len(warnings)} warning(s) emitted")
    return "; ".join(notes) if notes else "solid"


def _summary_payload(frame: pd.DataFrame, *, count: int, model_policy: str, seed: int) -> dict[str, Any]:
    passed = frame[frame["status"] == "passed"]
    return {
        "count": count,
        "model_policy": model_policy,
        "seed": seed,
        "passed": int(len(passed)),
        "crashed": int((frame["status"] == "crashed").sum()),
        "composite_score": float(frame["composite_score"].mean()) if len(frame) else 0.0,
        "accuracy_score": float(frame["accuracy_score"].mean()) if len(frame) else 0.0,
        "validity_score": float(frame["validity_score"].mean()) if len(frame) else 0.0,
        "ease_score": float(frame["ease_score"].mean()) if len(frame) else 0.0,
        "explainability_score": float(frame["explainability_score"].mean()) if len(frame) else 0.0,
        "median_wape": float(passed["wape"].median()) if len(passed) else None,
        "underperformed_naive": int(((passed["wape"] > passed["naive_wape"]) & passed["naive_wape"].notna()).sum()) if len(passed) else 0,
        "lowest_scoring_archetypes": (
            frame.groupby("archetype")["composite_score"].mean().sort_values().head(8).round(2).to_dict()
            if len(frame)
            else {}
        ),
    }


def _recommendations(frame: pd.DataFrame) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    feedback = " ".join(frame["feedback"].dropna().astype(str).tolist()).lower()
    if "intermittent" in feedback:
        recommendations.append(
            {
                "priority": "high",
                "area": "model policy",
                "recommendation": "Add intermittent-demand model policy using StatsForecast Croston/ADIDA/IMAPA/TSB candidates for sparse zero-heavy finance series.",
            }
        )
    if "shock" in feedback or "regressor" in feedback:
        recommendations.append(
            {
                "priority": "high",
                "area": "drivers",
                "recommendation": "Promote setup MCP regressor search into an MLForecast known-future-driver workflow with leakage checks and scenario fallback.",
            }
        )
    if "sparse panel" in feedback:
        recommendations.append(
            {
                "priority": "medium",
                "area": "hierarchy",
                "recommendation": "Add hierarchy-guided sparse child fallback and later full HierarchicalForecast reconciliation.",
            }
        )
    underperformed = frame[(frame["status"] == "passed") & (frame["wape"] > frame["naive_wape"])]
    if len(underperformed):
        recommendations.append(
            {
                "priority": "medium",
                "area": "selection",
                "recommendation": "Expose holdout-style champion-vs-naive comparisons in interpretation artifacts and warn when selected model underperforms naive.",
            }
        )
    if not recommendations:
        recommendations.append({"priority": "low", "area": "general", "recommendation": "No dominant failure class found; continue expanding real-data scenario coverage."})
    return recommendations


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
