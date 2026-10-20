"""Configuration evidence, not a replacement forecasting or validation engine."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from nixtla_scaffold.candidates import parameter_value
from nixtla_scaffold.schema import ForecastRun, ForecastSpec


CONFIGURATION_VERSION = 1
REFERENCES = {
    "native_models": "https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html",
    "mfles": "https://nixtlaverse.nixtla.io/statsforecast/docs/models/mfles.html",
    "multiple_seasonalities": "https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/multipleseasonalities.html",
    "validation": "https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html",
    "decomposition": "https://otexts.com/fpppy/03-decomposition.html",
}


def seasonal_evidence(group: pd.DataFrame, period: int) -> dict:
    """Compare cycle-normalized phases in disjoint early/recent blocks.

    This is an explicitly heuristic shape diagnostic, not an STL strength,
    hypothesis test, change-point probability or forecast-accuracy estimate.
    """
    ordered = group.sort_values("ds")
    y = pd.to_numeric(ordered.y, errors="coerce").to_numpy(dtype=float)
    count = len(y) // period
    result = {"period": period, "complete_cycles": count, "status": "insufficient_cycles",
              "strength_proxy": None, "shape_drift": None, "shape_changed": False, "phases": []}
    if period < 2 or count < 4:
        return result
    cycles = y[-count * period:].reshape(count, period)
    valid = np.isfinite(cycles).all(axis=1)
    if not valid.all():
        return {**result, "status": "missing_observations"}
    if np.any(cycles < 0) or np.any(cycles.mean(axis=1) <= 0):
        return {**result, "status": "relative_shape_unavailable_nonpositive_cycles"}
    normalized = cycles / cycles.mean(axis=1, keepdims=True)
    shape = np.median(normalized, axis=0)
    variance = float(np.var(normalized))
    strength = max(0., min(1., 1 - float(np.var(normalized - shape)) / variance)) if variance > 0 else 0.
    result.update(strength_proxy=strength, status=(
        "supported_heuristic" if strength >= .3 and np.ptp(shape) >= .1 else "weak_heuristic"
    ))
    if count >= 16:
        early, recent = normalized[:8], normalized[-8:]
        early_shape, recent_shape = np.median(early, axis=0), np.median(recent, axis=0)
        drift = float(np.mean(np.abs(recent_shape - early_shape)))
        noise = float(np.median(np.abs(np.concatenate((early - early_shape, recent - recent_shape)))))
        threshold = max(.1, 4 * noise / np.sqrt(8))
        result.update(shape_drift=drift, shape_changed=drift > threshold,
                      drift_threshold=threshold, reference_cycles=8, recent_cycles=8)
        if period <= 31:
            dates = pd.to_datetime(ordered.ds).iloc[-count * period:][:period]
            result["phases"] = [{
                "phase": index + 1, "example_date": date.isoformat(),
                "early_index": float(early_shape[index]), "recent_index": float(recent_shape[index]),
            } for index, date in enumerate(dates)]
    return result


def configuration_diagnostics(history: pd.DataFrame, spec: ForecastSpec, *, freq: str, season: int) -> list[dict]:
    from coreforecast.seasonal import find_season_length

    rows = []
    periods = list(dict.fromkeys([season, *spec.model_configuration.seasonal_periods]))
    if freq == "D":
        periods = list(dict.fromkeys([*periods, 7, 365]))
    for uid, group in history.groupby("unique_id", sort=True):
        group = group.sort_values("ds")
        available_rows = len(group)
        if spec.evaluation.training_window:
            group = group.tail(spec.evaluation.training_window)
        dates = pd.DatetimeIndex(pd.to_datetime(group.ds))
        regular = dates.equals(pd.date_range(dates[0], periods=len(dates), freq=freq))
        y = pd.to_numeric(group.y, errors="coerce").to_numpy(dtype=float)
        eligible = regular and np.isfinite(y).all()
        max_lag = min(4096, max(periods), len(y) // 4)
        detected = None
        if eligible and max_lag >= 2:
            detected = int(find_season_length(np.ascontiguousarray(y), max_lag)) if np.ptp(y) > 0 else 0
        evidence = [seasonal_evidence(group, period) for period in periods if period > 1] if regular else []
        changed = [item["period"] for item in evidence if item["shape_changed"]]
        primary = next((item for item in evidence if item["period"] == season), {})
        rows.append({
            "unique_id": str(uid), "as_of": dates[-1].isoformat(), "history_rows": len(group),
            "available_history_rows": available_rows,
            "frequency": freq, "primary_period": season,
            "period_source": "explicit" if spec.season_length is not None else "calendar_prior",
            "detected_period": detected, "detector_max_period": max_lag,
            "detector_status": "measured" if detected is not None else "unavailable",
            "primary_evidence": primary.get("status", "nonseasonal" if season == 1 else "unavailable"),
            "shape_changed": bool(changed), "changed_periods": changed, "period_evidence": evidence,
            "training_window": spec.evaluation.training_window,
            "recommendation": (
                "Compare a frozen recent-history or additive-seasonality treatment on identical origins; do not shorten every model by default."
                if changed else
                "Retain the declared recipe; compare against nonseasonal/simple baselines. A calendar prior is not proof of seasonality."
            ),
        })
    return rows


def _cycle_jump(group: pd.DataFrame, predicted: pd.Series, period: int) -> dict:
    y = pd.to_numeric(group.sort_values("ds").y, errors="coerce").to_numpy(dtype=float)
    future = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)[:period]
    count = min(27, len(y) // period)
    result = {"jump_status": "unavailable", "next_cycle_change_pct": None}
    if period < 2 or count < 9 or len(future) != period or not np.isfinite(future).all():
        return result
    cycles = y[-count * period:].reshape(count, period)
    if not np.isfinite(cycles).all() or (cycles < 0).any():
        return result
    totals = cycles.sum(axis=1)
    if np.any(totals <= 0):
        return result
    growth = totals[1:] / totals[:-1] - 1
    change = float(future.sum() / totals[-1] - 1)
    lower, upper = np.quantile(growth, [.05, .95])
    admission_lower, admission_upper = min(0., float(lower)), max(0., float(upper))
    return {
        "jump_status": "outside_recent_range" if change < lower or change > upper else "inside_recent_range",
        "next_cycle_change_pct": 100 * change,
        "observed_change_p05_pct": 100 * float(lower), "observed_change_p95_pct": 100 * float(upper),
        "admission_lower_change_pct": 100 * admission_lower,
        "admission_upper_change_pct": 100 * admission_upper,
        "admission_jump": bool(change < admission_lower or change > admission_upper),
        "historical_changes": len(growth), "cycle_observations": period,
        "next_cycle_total": float(future.sum()), "last_cycle_total": float(totals[-1]),
    }


def candidate_admission(history, forecasts, spec, *, freq, season) -> list[dict]:
    from nixtla_scaffold.models import model_columns

    diagnostics = {row["unique_id"]: row for row in configuration_diagnostics(
        history, spec, freq=freq, season=season,
    )}
    rows = []
    for uid, group in history.groupby("unique_id", sort=True):
        uid = str(uid)
        paths = forecasts.loc[forecasts.unique_id.astype(str).eq(uid)].sort_values("ds")
        reference = group.sort_values("ds")
        if spec.evaluation.training_window:
            reference = reference.tail(spec.evaluation.training_window)
        for model in model_columns(paths):
            jump = _cycle_jump(reference, paths[model], season)
            review = diagnostics[uid]["shape_changed"] and jump.get("admission_jump", False)
            rows.append({
                "unique_id": uid, "model": model, "cutoff": pd.Timestamp(reference.ds.max()).isoformat(),
                "policy": spec.selection_policy.configuration_guard,
                "review_required": bool(review),
                "eligible": not review or spec.selection_policy.configuration_guard == "warn",
                "reason": "changing_seasonal_shape_and_unusual_cycle_jump" if review else "no_configuration_exclusion",
                **jump,
            })
    return rows


def screen_candidates(history, forecasts, metrics, spec, *, freq, season):
    """Filter only the selector's view; retain every original path and score."""
    from nixtla_scaffold.models import model_columns

    admission = candidate_admission(history, forecasts, spec, freq=freq, season=season)
    screened, scores = forecasts.copy(), metrics.copy()
    excluded = {(row["unique_id"], row["model"]) for row in admission if not row["eligible"]}
    for uid, model in excluded:
        screened.loc[screened.unique_id.astype(str).eq(uid), model] = np.nan
    for uid, group in screened.groupby("unique_id", sort=True):
        if not any(np.isfinite(group[model]).all() for model in model_columns(group)):
            raise ValueError(f"{uid}: no finite configuration-admissible forecast candidate; "
                             "review the saved recipe or explicitly choose configuration_guard='warn'. "
                             "No rejected candidate was silently substituted.")
    if not scores.empty:
        scores["configuration_eligible"] = [
            (str(uid), str(model)) not in excluded for uid, model in zip(scores.unique_id, scores.model)
        ]
        scores["selection_eligible"] = scores.get("selection_eligible", True) & scores.configuration_eligible
    return screened, scores, admission


def build_configuration_report(run: ForecastRun) -> dict:
    from nixtla_scaffold.models import model_columns

    history = run.observed_history if not run.observed_history.empty else run.history
    series = configuration_diagnostics(history, run.spec, freq=run.profile.freq, season=run.profile.season_length)
    evidence = run.model_policy_resolution.get("candidate_evidence", {})
    recipes = evidence.get("recipes", {})
    fits = {(row["unique_id"], row["model"]): row for row in evidence.get("fits", [])
            if row.get("stage", "future") == "future"}
    models, selected_reviews = [], []
    relevant = {"season_length", "model", "damped", "phi", "fourier_order", "multiplicative",
                "test_size", "n_windows", "step_size", "metric", "window_size", "lags", "max_depth"}
    by_series = {row["unique_id"]: row for row in series}
    admission = {(row["unique_id"], row["model"]): row
                 for row in run.model_policy_resolution.get("configuration_admission", [])}
    for uid, group in history.groupby("unique_id", sort=True):
        uid = str(uid)
        diagnostic = by_series[uid]
        paths = run.all_models.loc[run.all_models.unique_id.astype(str).eq(uid)].sort_values("ds")
        chosen = run.forecast.loc[run.forecast.unique_id.astype(str).eq(uid)].sort_values("ds")
        chosen_name = str(chosen.model.iloc[0]) if not chosen.empty else ""
        for model in model_columns(paths):
            fitted = fits.get((uid, model), {})
            recipe = recipes.get(fitted.get("recipe_id"), {})
            definition = recipe.get("estimator", {})
            params = definition.get("parameters", {})
            if model == "WeightedEnsemble":
                params = {"metric": "daily_rmse", "model": "legacy_inverse_error_policy"}
            native = fitted.get("fitted_structure", {})
            mfles = native.get("mfles", {})
            selected = model == chosen_name
            reference = group.sort_values("ds")
            if run.spec.evaluation.training_window:
                reference = reference.tail(run.spec.evaluation.training_window)
            jump = _cycle_jump(reference, chosen.yhat if selected else paths[model], run.profile.season_length)
            review = diagnostic["shape_changed"] and jump.get("admission_jump", False)
            eligible = admission.get((uid, model), {}).get(
                "eligible", not review or run.spec.selection_policy.configuration_guard == "warn",
            )
            if review and selected:
                selected_reviews.append(uid)
            models.append({
                "unique_id": uid, "model": model, "is_selected": selected,
                "recipe_id": fitted.get("recipe_id"), "fit_status": fitted.get("status", "derived_from_components" if model == "WeightedEnsemble" else "unavailable"),
                "estimator_class": definition.get("class", definition.get("implementation", "unavailable")),
                "settings": {key: value for key, value in params.items() if key in relevant},
                "feature_recipe": recipe.get("execution", {}).get("feature_recipe", {}),
                "fitted_method": native.get("method", native.get("components")),
                "internal_transform": mfles.get("internal_transform", "not_reported"),
                "fitted_multiplicative": mfles.get("multiplicative"),
                "selected_native_config": mfles.get("selected_config"),
                "native_config_status": mfles.get("selected_config_status", "not_reported"),
                "native_search_metric": mfles.get("search_metric", params.get("metric")),
                "outer_objective": run.spec.selection_policy.objective,
                "review_required": bool(review), "selection_eligible": bool(eligible),
                "admission_policy": run.spec.selection_policy.configuration_guard, **jump,
            })
    return {
        "schema_version": CONFIGURATION_VERSION, "status": "review_required" if selected_reviews else "diagnostic_only",
        "review_series": sorted(set(selected_reviews)), "declared": run.spec.model_configuration.to_dict(),
        "external_transform": run.spec.transform.to_dict(), "series": series, "models": models,
        "references": REFERENCES,
        "scope": "Historical-only diagnostics. The declared configuration guard controls candidate admission, not point clipping or adoption.",
        "limitations": [
            "Calendar frequency is not a detected seasonal period. CoreForecast's bounded detector can return 0 and is not a multiple-seasonality search.",
            "Phase strength and drift are descriptive heuristics, not significance tests. Nonpositive cycles cannot support relative-shape diagnostics.",
            "The recent 5th-95th percentile cycle-change range is a plausibility comparator, not a prediction interval or business constraint.",
            "AutoETS/AutoARIMA optimize native structural criteria; AutoMFLES optimizes its recorded inner metric, not the outer monthly objective.",
            "Changing seasonal shape plus an unusual selected path requires review. Passing this diagnostic does not establish accuracy.",
        ],
    }


def configuration_frames(report: dict) -> dict[str, pd.DataFrame]:
    series, seasons, phases = [], [], []
    for row in report.get("series", []):
        series.append({key: value for key, value in row.items() if key != "period_evidence"})
        for item in row["period_evidence"]:
            seasons.append({"unique_id": row["unique_id"], **{key: value for key, value in item.items() if key != "phases"}})
            phases.extend({"unique_id": row["unique_id"], "period": item["period"], **phase} for phase in item["phases"])
    frames = {"series": series, "seasonality": seasons, "phases": phases, "models": report.get("models", [])}
    return {name: pd.DataFrame([{
        key: json.dumps(parameter_value(value), sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    } for row in rows]) for name, rows in frames.items()}


def mfles_additive_spec(spec: ForecastSpec) -> ForecastSpec:
    return replace(spec, model_configuration=replace(spec.model_configuration, mfles_seasonality="additive"))


def mfles_identified_fourier_spec(spec: ForecastSpec) -> ForecastSpec:
    periods = spec.model_configuration.seasonal_periods or (spec.season_length or 1,)
    if min(periods) < 3:
        raise ValueError("Resolve a seasonal period >= 3 before testing a nonredundant Fourier basis")
    return replace(spec, model_configuration=replace(
        spec.model_configuration, mfles_fourier_order=min(3, (min(periods) - 1) // 2),
    ))
