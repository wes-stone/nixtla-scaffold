from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd

from nixtla_scaffold.model_families import model_family
from nixtla_scaffold.schema import DataProfile, ForecastSpec


MLFORECAST_MIN_OBS = 30


@dataclass(frozen=True)
class ModelResult:
    forecast: pd.DataFrame
    backtest_metrics: pd.DataFrame
    backtest_predictions: pd.DataFrame
    engine: str
    model_weights: pd.DataFrame
    model_explainability: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_invocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: tuple[str, ...] = ()
    model_policy_resolution: dict[str, Any] = field(default_factory=dict)


StatsForecastFactory = tuple[str, Callable[[], Any]]


@dataclass(frozen=True)
class MLForecastIntervalPlan:
    n_windows: int
    lags: tuple[int, ...]
    dropped_lags: tuple[int, ...]


def forecast_with_policy(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    if spec.model_policy == "baseline":
        return _with_policy_resolution(forecast_with_baselines(history, profile, spec), profile, spec)
    if spec.model_policy == "mlforecast":
        return _with_policy_resolution(forecast_with_mlforecast(history, profile, spec), profile, spec)

    # For "auto", "statsforecast", or "all": run statsforecast first
    sf_result = _try_statsforecast(history, profile, spec)

    # For "all" or "auto" with enough data, also run MLForecast and merge
    ml_override: dict[str, Any] | None = None
    if spec.model_policy in ("all", "auto"):
        if profile.min_obs_per_series < MLFORECAST_MIN_OBS:
            reason = f"min_history_below_threshold: min observations {profile.min_obs_per_series} < {MLFORECAST_MIN_OBS}"
            ml_override = {"eligible": False, "ran": False, "reason_if_not_ran": reason, "contributed_models": []}
            if spec.model_policy == "all":
                sf_result = _append_warnings(
                    sf_result,
                    (
                        "MLForecast skipped for model_policy='all' because "
                        f"min history {profile.min_obs_per_series} < {MLFORECAST_MIN_OBS}"
                    ),
                )
        else:
            try:
                ml_result = forecast_with_mlforecast(history, profile, spec)
            except ImportError as exc:
                if spec.model_policy == "all":
                    raise ImportError(f"MLForecast requested by model_policy='all' but unavailable: {exc}") from exc
                ml_override = {
                    "eligible": True,
                    "ran": False,
                    "reason_if_not_ran": f"not_installed: {exc}",
                    "contributed_models": [],
                }
                sf_result = _append_warnings(sf_result, f"MLForecast unavailable; continuing without MLForecast ({exc})")
            except Exception as exc:
                if spec.model_policy == "all":
                    raise RuntimeError(f"MLForecast requested by model_policy='all' but failed: {exc}") from exc
                ml_override = {
                    "eligible": True,
                    "ran": False,
                    "reason_if_not_ran": f"runtime_failure: {exc}",
                    "contributed_models": [],
                }
                sf_result = _append_warnings(sf_result, f"MLForecast failed; continuing without MLForecast ({exc})")
            else:
                if ml_result.engine == "mlforecast":
                    sf_result = _merge_model_results(sf_result, ml_result, history=history, profile=profile, spec=spec)
                    ml_models = _non_ensemble_model_columns(ml_result.forecast)
                    if not ml_models:
                        reason = "produced_no_candidates"
                        if spec.model_policy == "all":
                            raise RuntimeError("MLForecast requested by model_policy='all' but produced no candidate models")
                        ml_override = {"eligible": True, "ran": False, "reason_if_not_ran": reason, "contributed_models": []}
                        sf_result = _append_warnings(sf_result, "MLForecast produced no candidate models; continuing without MLForecast")
                else:
                    reason = f"runtime_failure: unexpected engine {ml_result.engine}"
                    if spec.model_policy == "all":
                        raise RuntimeError(f"MLForecast requested by model_policy='all' but returned {ml_result.engine}")
                    ml_override = {"eligible": True, "ran": False, "reason_if_not_ran": reason, "contributed_models": []}
                    sf_result = _append_warnings(sf_result, f"MLForecast returned {ml_result.engine}; continuing without MLForecast")

    return _with_policy_resolution(sf_result, profile, spec, {"mlforecast": ml_override} if ml_override else None)


def _try_statsforecast(history: pd.DataFrame, profile: DataProfile, spec: ForecastSpec) -> ModelResult:
    try:
        return forecast_with_statsforecast(history, profile, spec)
    except ImportError as exc:
        if spec.model_policy == "statsforecast":
            raise
        result = forecast_with_baselines(history, profile, spec)
        return ModelResult(
            forecast=result.forecast,
            backtest_metrics=result.backtest_metrics,
            backtest_predictions=result.backtest_predictions,
            engine="baseline",
            model_weights=result.model_weights,
            model_explainability=result.model_explainability,
            warnings=result.warnings + (f"StatsForecast unavailable; used baseline engine ({exc})",),
        )
    except Exception as exc:
        if spec.model_policy == "statsforecast":
            raise
        result = forecast_with_baselines(history, profile, spec)
        return ModelResult(
            forecast=result.forecast,
            backtest_metrics=result.backtest_metrics,
            backtest_predictions=result.backtest_predictions,
            engine="baseline",
            model_weights=result.model_weights,
            model_explainability=result.model_explainability,
            warnings=result.warnings + (f"StatsForecast failed; used baseline engine ({exc})",),
        )


def _append_warnings(result: ModelResult, *messages: str) -> ModelResult:
    return ModelResult(
        forecast=result.forecast,
        backtest_metrics=result.backtest_metrics,
        backtest_predictions=result.backtest_predictions,
        engine=result.engine,
        model_weights=result.model_weights,
        model_explainability=result.model_explainability,
        custom_model_contracts=result.custom_model_contracts,
        custom_model_invocations=result.custom_model_invocations,
        warnings=result.warnings + tuple(message for message in messages if message),
        model_policy_resolution=result.model_policy_resolution,
    )


def _merge_model_results(
    primary: ModelResult,
    secondary: ModelResult,
    *,
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    """Merge MLForecast columns into the primary (StatsForecast) result."""
    base_primary_forecast = _drop_weighted_ensemble_columns(primary.forecast)
    base_secondary_forecast = _drop_weighted_ensemble_columns(secondary.forecast)
    merged_forecast = _merge_prediction_frames(base_primary_forecast, base_secondary_forecast, keys=["unique_id", "ds"])

    base_primary_bt = _drop_weighted_ensemble_columns(primary.backtest_predictions)
    base_secondary_bt = _drop_weighted_ensemble_columns(secondary.backtest_predictions)
    merged_bt = _merge_prediction_frames(base_primary_bt, base_secondary_bt, keys=["unique_id", "ds", "cutoff"])

    prior_metrics = pd.concat(
        [
            primary.backtest_metrics[primary.backtest_metrics["model"].astype(str) != "WeightedEnsemble"],
            secondary.backtest_metrics[secondary.backtest_metrics["model"].astype(str) != "WeightedEnsemble"],
        ],
        ignore_index=True,
    ) if not primary.backtest_metrics.empty or not secondary.backtest_metrics.empty else _empty_backtest_metrics()

    scales = _error_scale_map(history, profile.season_length)
    merged_metrics = _restore_cv_metadata(
        _metrics_from_cv(merged_bt, scales_by_series=scales),
        prior_metrics,
    )
    weight_warnings: list[str] = []
    if spec.weighted_ensemble:
        model_weights, weight_warnings = _model_weights_from_common_support(merged_bt)
        merged_forecast = _add_weighted_ensemble_forecast(merged_forecast, model_weights)
        merged_bt = _add_weighted_ensemble_to_cv(merged_bt, common_support=True)
        merged_metrics = _restore_cv_metadata(
            _metrics_from_cv(merged_bt, scales_by_series=scales),
            prior_metrics,
        )
    else:
        model_weights = _empty_model_weights()

    combined_warnings = list(primary.warnings) + list(secondary.warnings) + weight_warnings
    combined_warnings.extend(_weighted_ensemble_warnings(spec, model_weights))
    explainability = pd.concat([primary.model_explainability, secondary.model_explainability], ignore_index=True)
    custom_contracts = _concat_optional_frames(primary.custom_model_contracts, secondary.custom_model_contracts)
    custom_invocations = _concat_optional_frames(primary.custom_model_invocations, secondary.custom_model_invocations)
    return ModelResult(
        forecast=merged_forecast,
        backtest_metrics=merged_metrics,
        backtest_predictions=merged_bt,
        engine=f"{primary.engine}+{secondary.engine}",
        model_weights=model_weights,
        model_explainability=explainability,
        custom_model_contracts=custom_contracts,
        custom_model_invocations=custom_invocations,
        warnings=tuple(dict.fromkeys(combined_warnings)),
    )


def _concat_optional_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame()
    return pd.concat(available, ignore_index=True)


def rebuild_result_metrics_on_output_scale(
    result: ModelResult,
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    """Recalculate metrics and weights after forecast columns move to a new scale."""

    if result.backtest_predictions.empty:
        return result

    base_cv = result.backtest_predictions.drop(columns=["WeightedEnsemble"], errors="ignore")
    base_forecast = result.forecast.drop(columns=["WeightedEnsemble"], errors="ignore")
    scales = _error_scale_map(history, profile.season_length)
    base_metrics = _restore_cv_metadata(
        _metrics_from_cv(base_cv, scales_by_series=scales),
        result.backtest_metrics,
    )
    common_support = spec.weighted_ensemble and _has_classical_and_ml_models(_non_ensemble_model_columns(base_cv))
    weight_warnings: list[str] = []
    if spec.weighted_ensemble and common_support:
        model_weights, weight_warnings = _model_weights_from_common_support(base_cv)
    elif spec.weighted_ensemble:
        model_weights = _model_weights_from_metrics(base_metrics)
    else:
        model_weights = _empty_model_weights()
    forecast = _add_weighted_ensemble_forecast(base_forecast, model_weights) if spec.weighted_ensemble else base_forecast
    backtest_predictions = _add_weighted_ensemble_to_cv(base_cv, common_support=common_support) if spec.weighted_ensemble else base_cv
    metrics = _restore_cv_metadata(
        _metrics_from_cv(backtest_predictions, scales_by_series=scales),
        result.backtest_metrics,
    )
    combined_warnings = list(result.warnings) + weight_warnings
    combined_warnings.extend(_weighted_ensemble_warnings(spec, model_weights))
    return ModelResult(
        forecast=forecast,
        backtest_metrics=metrics,
        backtest_predictions=backtest_predictions,
        engine=result.engine,
        model_weights=model_weights,
        model_explainability=result.model_explainability,
        custom_model_contracts=result.custom_model_contracts,
        custom_model_invocations=result.custom_model_invocations,
        warnings=tuple(dict.fromkeys(combined_warnings)),
        model_policy_resolution=result.model_policy_resolution,
    )


def _restore_cv_metadata(metrics: pd.DataFrame, previous_metrics: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
    ]
    if metrics.empty or previous_metrics.empty:
        return metrics
    available_cols = [col for col in metadata_cols if col in previous_metrics.columns]
    if not available_cols:
        return metrics
    keyed = previous_metrics[["unique_id", "model", *available_cols]].drop_duplicates(["unique_id", "model"])
    out = metrics.merge(keyed, on=["unique_id", "model"], how="left")
    by_series = previous_metrics[["unique_id", *available_cols]].drop_duplicates("unique_id")
    fallback = metrics[["unique_id"]].merge(by_series, on="unique_id", how="left")
    for col in available_cols:
        out[col] = out[col].fillna(fallback[col])
    return out


def _with_policy_resolution(
    result: ModelResult,
    profile: DataProfile,
    spec: ForecastSpec,
    overrides: dict[str, dict[str, Any] | None] | None = None,
) -> ModelResult:
    resolution = _build_model_policy_resolution(result, profile, spec, overrides or {})
    return ModelResult(
        forecast=result.forecast,
        backtest_metrics=result.backtest_metrics,
        backtest_predictions=result.backtest_predictions,
        engine=result.engine,
        model_weights=result.model_weights,
        model_explainability=result.model_explainability,
        custom_model_contracts=result.custom_model_contracts,
        custom_model_invocations=result.custom_model_invocations,
        warnings=result.warnings,
        model_policy_resolution=resolution,
    )


def _build_model_policy_resolution(
    result: ModelResult,
    profile: DataProfile,
    spec: ForecastSpec,
    overrides: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    policy = spec.model_policy
    requested = {
        "baseline": policy == "baseline",
        "statsforecast": policy in {"auto", "all", "statsforecast"},
        "mlforecast": policy in {"auto", "all", "mlforecast"},
    }
    eligible = {
        "baseline": requested["baseline"] or result.engine == "baseline",
        "statsforecast": requested["statsforecast"],
        "mlforecast": (policy == "mlforecast") or (requested["mlforecast"] and profile.min_obs_per_series >= MLFORECAST_MIN_OBS),
    }
    contributed = _contributed_models_by_family(result.forecast)
    families: list[dict[str, Any]] = []
    for family in ["baseline", "statsforecast", "mlforecast"]:
        override = overrides.get(family)
        if override is not None:
            family_eligible = bool(override.get("eligible", eligible[family]))
            ran = bool(override.get("ran", False))
            reason = str(override.get("reason_if_not_ran", ""))
            models = list(override.get("contributed_models", []))
        else:
            models = contributed.get(family, [])
            family_eligible = eligible[family]
            ran = bool(models)
            if not requested[family]:
                reason = "not_requested"
            elif not family_eligible and family == "mlforecast":
                reason = f"min_history_below_threshold: min observations {profile.min_obs_per_series} < {MLFORECAST_MIN_OBS}"
            elif not ran:
                reason = "produced_no_candidates"
            else:
                reason = ""
        families.append(
            {
                "family": family,
                "requested": bool(requested[family]),
                "eligible": bool(family_eligible),
                "ran": bool(ran),
                "reason_if_not_ran": "" if ran else reason,
                "contributed_models": models,
            }
        )
    return {"model_policy": policy, "families": families}


def _contributed_models_by_family(forecast: pd.DataFrame) -> dict[str, list[str]]:
    contributed: dict[str, list[str]] = {"baseline": [], "statsforecast": [], "mlforecast": []}
    for model in _non_ensemble_model_columns(forecast):
        family = model_family(model)
        if family in contributed:
            contributed[family].append(model)
    return {family: sorted(set(models)) for family, models in contributed.items()}


def forecast_with_baselines(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    forecast = _baseline_predictions(history, profile.freq, profile.season_length, spec.horizon)
    metrics, backtest_predictions = _baseline_backtest(history, profile, spec)
    model_weights = _model_weights_from_metrics(metrics) if spec.weighted_ensemble else _empty_model_weights()
    forecast = _add_weighted_ensemble_forecast(forecast, model_weights)
    run_warnings = _weighted_ensemble_warnings(spec, model_weights)
    run_warnings.extend(_cv_horizon_warnings(metrics))
    return ModelResult(
        forecast=forecast,
        backtest_metrics=metrics,
        backtest_predictions=backtest_predictions,
        engine="baseline",
        model_weights=model_weights,
        model_explainability=_empty_model_explainability(),
        warnings=tuple(run_warnings),
    )


def forecast_with_statsforecast(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    from statsforecast.utils import ConformalIntervals

    models = _statsforecast_model_factories(history, profile, spec)

    interval_kwargs: dict[str, Any] = {}
    interval_windows = _interval_windows(profile.min_obs_per_series, spec.horizon, profile.season_length)
    run_warnings: list[str] = []
    run_warnings.append(f"StatsForecast ladder: {len(models)} models ({', '.join(alias for alias, _ in models)})")
    if spec.levels and interval_windows >= 2:
        interval_kwargs["level"] = list(spec.levels)
        interval_kwargs["prediction_intervals"] = ConformalIntervals(h=spec.horizon, n_windows=interval_windows)
    elif spec.levels:
        run_warnings.append("prediction intervals skipped because history is too short for conformal windows")

    forecast, forecast_warnings = _statsforecast_forecast_resilient(
        models,
        history=history,
        freq=profile.freq,
        horizon=spec.horizon,
        verbose=spec.verbose,
        interval_kwargs=interval_kwargs,
    )
    run_warnings.extend(forecast_warnings)
    forecast = _add_zero_forecast_for_intermittent(forecast, history)

    metrics, backtest_warnings, backtest_predictions = _statsforecast_backtest(models, history, profile, spec)
    model_weights = _model_weights_from_metrics(metrics) if spec.weighted_ensemble else _empty_model_weights()
    forecast = _add_weighted_ensemble_forecast(forecast, model_weights)
    run_warnings.extend(backtest_warnings)
    run_warnings.extend(_weighted_ensemble_warnings(spec, model_weights))
    return ModelResult(
        forecast=forecast,
        backtest_metrics=metrics,
        backtest_predictions=backtest_predictions,
        engine="statsforecast",
        model_weights=model_weights,
        model_explainability=_empty_model_explainability(),
        warnings=tuple(run_warnings),
    )


def _statsforecast_model_factories(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> list[StatsForecastFactory]:
    from statsforecast.models import (
        ADIDA,
        IMAPA,
        AutoARIMA,
        AutoETS,
        AutoTheta,
        CrostonClassic,
        CrostonOptimized,
        CrostonSBA,
        DynamicOptimizedTheta,
        HistoricAverage,
        Holt,
        HoltWinters,
        MSTL,
        Naive,
        OptimizedTheta,
        RandomWalkWithDrift,
        SeasonalExponentialSmoothingOptimized,
        SeasonalNaive,
        SeasonalWindowAverage,
        SimpleExponentialSmoothingOptimized,
        TSB,
        Theta,
        WindowAverage,
    )

    try:
        from statsforecast.models import AutoMFLES, MFLES
    except ImportError:  # pragma: no cover - older statsforecast versions
        AutoMFLES = None
        MFLES = None

    season = profile.season_length
    min_obs = profile.min_obs_per_series
    has_intermittent = _has_intermittent_demand(history)

    # --- Tier 1: Always-on baselines (any history length) ---
    models: list[StatsForecastFactory] = [
        ("Naive", lambda: Naive(alias="Naive")),
        ("HistoricAverage", lambda: HistoricAverage(alias="HistoricAverage")),
        ("RandomWalkWithDrift", lambda: RandomWalkWithDrift(alias="RandomWalkWithDrift")),
        ("WindowAverage", lambda: WindowAverage(window_size=max(1, min(season, min_obs)), alias="WindowAverage")),
        ("SES", lambda: SimpleExponentialSmoothingOptimized(alias="SES")),
    ]

    # --- Tier 2: Seasonal baselines (need season > 1) ---
    if season > 1:
        models.append(("SeasonalNaive", lambda: SeasonalNaive(season_length=season, alias="SeasonalNaive")))
        models.append(
            (
                "SeasonalWindowAverage",
                lambda: SeasonalWindowAverage(
                    season_length=season,
                    window_size=max(1, min(2, min_obs // season)),
                    alias="SeasonalWindowAverage",
                ),
            )
        )
        models.append(
            (
                "SeasonalExpSmoothing",
                lambda: SeasonalExponentialSmoothingOptimized(season_length=season, alias="SeasonalExpSmoothing"),
            )
        )

    # --- Tier 3: Intermittent demand models (sparse zero-heavy non-negative series) ---
    if has_intermittent:
        models.extend([
            ("CrostonClassic", lambda: CrostonClassic(alias="CrostonClassic")),
            ("CrostonOptimized", lambda: CrostonOptimized(alias="CrostonOptimized")),
            ("CrostonSBA", lambda: CrostonSBA(alias="CrostonSBA")),
            ("ADIDA", lambda: ADIDA(alias="ADIDA")),
            ("IMAPA", lambda: IMAPA(alias="IMAPA")),
            ("TSB", lambda: TSB(alpha_d=0.3, alpha_p=0.3, alias="TSB")),
        ])

    # --- Tier 4: Classical decomposition + state space (need ≥6 obs) ---
    if min_obs >= 6:
        models.extend([
            ("AutoETS", lambda: AutoETS(season_length=season, alias="AutoETS")),
            ("AutoARIMA", lambda: AutoARIMA(season_length=season, alias="AutoARIMA")),
            ("Holt", lambda: Holt(season_length=season, alias="Holt")),
        ])
        if season > 1 and min_obs >= 2 * season:
            models.append(("HoltWinters", lambda: HoltWinters(season_length=season, alias="HoltWinters")))

    # --- Tier 5: Theta family (need ≥8 obs) ---
    if min_obs >= 8:
        models.extend([
            ("AutoTheta", lambda: AutoTheta(season_length=season, alias="AutoTheta")),
            ("Theta", lambda: Theta(season_length=season, alias="Theta")),
            ("OptimizedTheta", lambda: OptimizedTheta(season_length=season, alias="OptimizedTheta")),
            ("DynamicOptimizedTheta", lambda: DynamicOptimizedTheta(season_length=season, alias="DynamicOptimizedTheta")),
        ])

    # --- Tier 6: MSTL decomposition + complex seasonal (need ≥2 full seasons) ---
    if min_obs >= 2 * season and season > 1:
        models.append(("MSTL", lambda: MSTL(season_length=season, alias="MSTL")))
        models.append(
            (
                "MSTL_AutoARIMA",
                lambda: MSTL(season_length=season, trend_forecaster=AutoARIMA(season_length=1), alias="MSTL_AutoARIMA"),
            )
        )

    # --- Tier 7: boosted decomposition (need enough data for internal validation) ---
    if min_obs >= max(12, 2 * season if season > 1 else 12):
        if MFLES is not None:
            models.append(("MFLES", lambda: MFLES(season_length=season if season > 1 else None, alias="MFLES")))
        if AutoMFLES is not None:
            test_size, n_windows, step_size = _adaptive_cv_params(
                min_obs,
                spec.horizon,
                season,
                strict=spec.strict_cv_horizon,
            )
            if test_size >= 1:
                auto_windows = max(1, min(2, n_windows if n_windows >= 1 else 1))
                auto_step = step_size if step_size >= 1 else None
                models.append(
                    (
                        "AutoMFLES",
                        lambda: AutoMFLES(
                            test_size=test_size,
                            season_length=season if season > 1 else None,
                            n_windows=auto_windows,
                            step_size=auto_step,
                            alias="AutoMFLES",
                        ),
                    )
                )
    return models


def _statsforecast_forecast_resilient(
    models: list[StatsForecastFactory],
    *,
    history: pd.DataFrame,
    freq: str,
    horizon: int,
    verbose: bool,
    interval_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    from statsforecast import StatsForecast

    warnings_out: list[str] = []
    df = history[["unique_id", "ds", "y"]]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            forecast = StatsForecast(
                models=[factory() for _, factory in models],
                freq=freq,
                n_jobs=1,
                verbose=verbose,
            ).forecast(df=df, h=horizon, **interval_kwargs)
        warnings_out.extend(_format_caught_warnings(caught))
        return _normalize_statsforecast_output(forecast), warnings_out
    except Exception as exc:
        warnings_out.append(f"StatsForecast full ladder failed; retrying candidate-by-candidate ({exc})")

    frames: list[pd.DataFrame] = []
    for alias, factory in models:
        candidate_frame, candidate_warnings = _forecast_candidate_resilient(
            alias,
            factory,
            history=history,
            freq=freq,
            horizon=horizon,
            verbose=verbose,
            interval_kwargs=interval_kwargs,
        )
        if not candidate_frame.empty:
            frames.append(candidate_frame)
        warnings_out.extend(candidate_warnings)
    if not frames:
        raise RuntimeError("no StatsForecast candidate model able to be fitted")
    return _merge_statsforecast_frames(frames, keys=["unique_id", "ds"]), sorted(set(warnings_out))


def _forecast_candidate_resilient(
    alias: str,
    factory: Callable[[], Any],
    *,
    history: pd.DataFrame,
    freq: str,
    horizon: int,
    verbose: bool,
    interval_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    from statsforecast import StatsForecast

    warnings_out: list[str] = []
    df = history[["unique_id", "ds", "y"]]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            forecast = StatsForecast(
                models=[factory()],
                freq=freq,
                n_jobs=1,
                verbose=verbose,
            ).forecast(df=df, h=horizon, **interval_kwargs)
        warnings_out.extend(_format_caught_warnings(caught))
        return _normalize_statsforecast_output(forecast), warnings_out
    except Exception as full_exc:
        series_count = history["unique_id"].nunique()
        if series_count > 1:
            frames: list[pd.DataFrame] = []
            skipped: list[str] = []
            for uid, grp in history.groupby("unique_id", sort=True):
                frame, group_warnings = _forecast_candidate_resilient(
                    alias,
                    factory,
                    history=grp,
                    freq=freq,
                    horizon=horizon,
                    verbose=verbose,
                    interval_kwargs=interval_kwargs,
                )
                if frame.empty:
                    skipped.append(str(uid))
                else:
                    frames.append(frame)
                warnings_out.extend(group_warnings)
            if frames:
                warnings_out.append(
                    f"StatsForecast candidate {alias} failed on full panel; kept series-level successes ({full_exc})"
                )
                if skipped:
                    warnings_out.append(f"StatsForecast candidate {alias} skipped for series {skipped}")
                return _merge_statsforecast_frames(frames, keys=["unique_id", "ds"]), sorted(set(warnings_out))
            warnings_out.append(f"StatsForecast candidate {alias} failed and was skipped for all series ({full_exc})")
            return pd.DataFrame(), sorted(set(warnings_out))

        if not interval_kwargs:
            warnings_out.append(f"StatsForecast candidate {alias} failed and was skipped ({full_exc})")
            return pd.DataFrame(), warnings_out
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                forecast = StatsForecast(
                    models=[factory()],
                    freq=freq,
                    n_jobs=1,
                    verbose=verbose,
                ).forecast(df=df, h=horizon)
            warnings_out.extend(_format_caught_warnings(caught))
            warnings_out.append(
                f"StatsForecast candidate {alias} could not produce conformal intervals; kept point forecast only ({full_exc})"
            )
            return _normalize_statsforecast_output(forecast), warnings_out
        except Exception as point_exc:
            warnings_out.append(f"StatsForecast candidate {alias} failed and was skipped ({point_exc})")
            return pd.DataFrame(), warnings_out


def forecast_with_mlforecast(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> ModelResult:
    """MLForecast engine using sklearn/LightGBM regressors with lag/date features."""

    season = profile.season_length
    min_obs = profile.min_obs_per_series

    # Build lag list: recent lags + seasonal lags
    lags = list(range(1, min(13, min_obs // 2) + 1))
    if season > 1:
        lags.extend([season, 2 * season])
    lags = sorted(set(lag for lag in lags if lag < min_obs))
    if not lags:
        lags = [1]

    h, n_windows, step_size = _adaptive_cv_params(min_obs, spec.horizon, season, strict=spec.strict_cv_horizon)
    cv_lag_plan = _mlforecast_cv_lag_plan(
        min_obs=min_obs,
        horizon=h,
        n_windows=n_windows,
        step_size=step_size,
        lags=lags,
    )
    if cv_lag_plan is not None:
        lags = list(cv_lag_plan.lags)

    interval_plan = _mlforecast_interval_plan(
        min_obs=min_obs,
        horizon=spec.horizon,
        season_length=season,
        lags=lags,
        levels=spec.levels,
    )
    if interval_plan is not None:
        lags = list(interval_plan.lags)

    model_factories = _mlforecast_model_factories(min_obs=min_obs)
    if not model_factories:
        raise ImportError("MLForecast requires at least one installed sklearn or LightGBM regressor")

    date_features = ["month", "dayofweek"] if profile.freq.upper().startswith("D") or profile.freq.upper().startswith("B") else ["month"]

    run_warnings: list[str] = []
    run_warnings.append(
        "MLForecast ladder: "
        f"{len(model_factories)} models ({', '.join(alias for alias, _ in model_factories)}) "
        f"with lags={lags}, date_features={date_features}"
    )
    if interval_plan is not None:
        run_warnings.append(
            "MLForecast conformal intervals enabled: "
            f"levels={list(spec.levels)}, n_windows={interval_plan.n_windows}, h={spec.horizon}"
        )
        if interval_plan.dropped_lags:
            run_warnings.append(
                "MLForecast interval calibration dropped lag(s) "
                f"{list(interval_plan.dropped_lags)} so each conformal window has enough training rows"
            )
    elif spec.levels:
        run_warnings.append(
            "MLForecast prediction intervals skipped because history, horizon, and lag requirements "
            "cannot support at least two conformal calibration windows"
        )
    if cv_lag_plan is not None and cv_lag_plan.dropped_lags:
        run_warnings.append(
            "MLForecast CV alignment dropped lag(s) "
            f"{list(cv_lag_plan.dropped_lags)} so MLForecast uses the shared rolling-origin windows"
        )

    interval_factory: Callable[[], Any] | None = None
    if interval_plan is not None:
        from mlforecast.utils import PredictionIntervals

        interval_factory = lambda: PredictionIntervals(n_windows=interval_plan.n_windows, h=spec.horizon)

    forecast, model_explainability, fit_warnings = _mlforecast_fit_predict_resilient(
        model_factories,
        history=history,
        freq=profile.freq,
        lags=lags,
        date_features=date_features,
        horizon=spec.horizon,
        levels=spec.levels,
        prediction_intervals_factory=interval_factory,
    )
    run_warnings.extend(fit_warnings)

    forecast = forecast.reset_index() if "unique_id" not in forecast.columns else forecast.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast = forecast.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Cross-validation for backtest metrics
    metrics = _empty_backtest_metrics()
    backtest_predictions = _empty_backtest_predictions()
    cv_interval_factory: Callable[[], Any] | None = None
    cv_interval_windows = 0
    if spec.levels and n_windows >= 1 and h >= 1:
        shortest_train = min_obs - h - step_size * max(0, n_windows - 1)
        cv_interval_plan = _mlforecast_interval_plan(
            min_obs=shortest_train,
            horizon=h,
            season_length=season,
            lags=lags,
            levels=spec.levels,
            allow_lag_cap=False,
        )
        if cv_interval_plan is not None:
            from mlforecast.utils import PredictionIntervals

            cv_interval_windows = cv_interval_plan.n_windows
            cv_interval_factory = lambda: PredictionIntervals(n_windows=cv_interval_windows, h=h)
        elif interval_plan is not None:
            run_warnings.append(
                "MLForecast CV interval diagnostics skipped because rolling-origin training folds "
                "are too short for two conformal calibration windows at the selected CV horizon"
            )
    if n_windows >= 1 and h >= 1:
        try:
            cv, cv_warnings = _mlforecast_cross_validation_resilient(
                model_factories,
                history=history,
                freq=profile.freq,
                lags=lags,
                date_features=date_features,
                horizon=h,
                step_size=step_size,
                n_windows=n_windows,
                levels=spec.levels,
                prediction_intervals_factory=cv_interval_factory,
            )
            cv = cv.reset_index() if "unique_id" not in cv.columns else cv.copy()
            cv["ds"] = pd.to_datetime(cv["ds"])
            cv = cv.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            metrics = _attach_cv_metadata(
                _metrics_from_cv(cv, scales_by_series=_error_scale_map(history, season)),
                requested_horizon=spec.horizon,
                selection_horizon=h,
                n_windows=n_windows,
                step_size=step_size,
            )
            backtest_predictions = cv
            run_warnings.extend(cv_warnings)
            run_warnings.extend(_cv_horizon_warnings(metrics))
        except Exception as exc:
            run_warnings.append(f"MLForecast cross-validation failed: {exc}")

    return ModelResult(
        forecast=forecast,
        backtest_metrics=metrics,
        backtest_predictions=backtest_predictions,
        engine="mlforecast",
        model_weights=_empty_model_weights(),
        model_explainability=model_explainability,
        warnings=tuple(run_warnings),
    )


def _mlforecast_model_factories(*, min_obs: int) -> list[tuple[str, Callable[[], Any]]]:
    factories: list[tuple[str, Callable[[], Any]]] = []

    try:
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LinearRegression, Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        leaf_size = max(1, min(5, min_obs // 10))
        neighbor_count = max(1, min(5, min_obs // 6))
        factories.extend(
            [
                ("LinearRegression", lambda: LinearRegression()),
                ("Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                ("Ridge_Regularized", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
                ("BayesianRidge", lambda: make_pipeline(StandardScaler(), BayesianRidge())),
                (
                    "ElasticNet",
                    lambda: make_pipeline(
                        StandardScaler(),
                        ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000, random_state=42),
                    ),
                ),
                (
                    "Huber",
                    lambda: make_pipeline(
                        StandardScaler(),
                        HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000),
                    ),
                ),
                (
                    "RandomForest",
                    lambda: RandomForestRegressor(
                        n_estimators=80,
                        min_samples_leaf=leaf_size,
                        max_depth=8,
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
                (
                    "ExtraTrees",
                    lambda: ExtraTreesRegressor(
                        n_estimators=80,
                        min_samples_leaf=leaf_size,
                        max_depth=8,
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
                (
                    "GradientBoosting",
                    lambda: GradientBoostingRegressor(
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
                (
                    "HistGradientBoosting",
                    lambda: HistGradientBoostingRegressor(
                        max_iter=80,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
                ("KNeighbors", lambda: KNeighborsRegressor(n_neighbors=neighbor_count, weights="distance")),
            ]
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb

        factories.extend(
            [
                (
                    "LightGBM",
                    lambda: lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        num_leaves=31,
                        verbose=-1,
                        random_state=42,
                    ),
                ),
                (
                    "LightGBM_Conservative",
                    lambda: lgb.LGBMRegressor(
                        n_estimators=60,
                        learning_rate=0.05,
                        num_leaves=15,
                        min_child_samples=10,
                        verbose=-1,
                        random_state=42,
                    ),
                ),
                (
                    "LightGBM_Shallow",
                    lambda: lgb.LGBMRegressor(
                        n_estimators=80,
                        learning_rate=0.08,
                        num_leaves=7,
                        max_depth=3,
                        min_child_samples=8,
                        verbose=-1,
                        random_state=42,
                    ),
                ),
                (
                    "LightGBM_Robust",
                    lambda: lgb.LGBMRegressor(
                        objective="huber",
                        n_estimators=80,
                        learning_rate=0.06,
                        num_leaves=15,
                        min_child_samples=8,
                        verbose=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    return factories


def _mlforecast_interval_plan(
    *,
    min_obs: int,
    horizon: int,
    season_length: int,
    lags: list[int],
    levels: tuple[int, ...],
    allow_lag_cap: bool = True,
) -> MLForecastIntervalPlan | None:
    if not levels or horizon < 1 or min_obs < 1 or not lags:
        return None
    original_lags = tuple(sorted(set(lag for lag in lags if lag >= 1)))
    candidates: list[tuple[bool, int, int, MLForecastIntervalPlan]] = []
    max_windows = min(3, max(0, (min_obs - 2) // max(1, horizon)))
    for n_windows in range(2, max_windows + 1):
        lag_cap = min_obs - n_windows * horizon - 1
        if lag_cap < 1:
            continue
        kept_lags = tuple(lag for lag in original_lags if lag <= lag_cap)
        if not kept_lags:
            continue
        dropped_lags = tuple(lag for lag in original_lags if lag not in kept_lags)
        if dropped_lags and not allow_lag_cap:
            continue
        preserves_season = season_length <= 1 or season_length in kept_lags
        plan = MLForecastIntervalPlan(n_windows=n_windows, lags=kept_lags, dropped_lags=dropped_lags)
        candidates.append((preserves_season, n_windows, len(kept_lags), plan))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def _mlforecast_cv_lag_plan(
    *,
    min_obs: int,
    horizon: int,
    n_windows: int,
    step_size: int,
    lags: list[int],
) -> MLForecastIntervalPlan | None:
    if horizon < 1 or n_windows < 1 or not lags:
        return None
    original_lags = tuple(sorted(set(lag for lag in lags if lag >= 1)))
    shortest_train = min_obs - horizon - step_size * max(0, n_windows - 1)
    lag_cap = shortest_train - 1
    if lag_cap < 1:
        return None
    kept_lags = tuple(lag for lag in original_lags if lag <= lag_cap)
    dropped_lags = tuple(lag for lag in original_lags if lag not in kept_lags)
    if not kept_lags or not dropped_lags:
        return None
    return MLForecastIntervalPlan(n_windows=n_windows, lags=kept_lags, dropped_lags=dropped_lags)


def _mlforecast_fit_predict_resilient(
    model_factories: list[tuple[str, Callable[[], Any]]],
    *,
    history: pd.DataFrame,
    freq: str,
    lags: list[int],
    date_features: list[str],
    horizon: int,
    levels: tuple[int, ...],
    prediction_intervals_factory: Callable[[], Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    from mlforecast import MLForecast

    warnings_out: list[str] = []
    df = history[["unique_id", "ds", "y"]]
    predict_kwargs: dict[str, Any] = {}
    if prediction_intervals_factory is not None:
        predict_kwargs["level"] = list(levels)

    def fit_kwargs() -> dict[str, Any]:
        if prediction_intervals_factory is None:
            return {}
        return {"prediction_intervals": prediction_intervals_factory()}

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ml = MLForecast(
                models={alias: factory() for alias, factory in model_factories},
                freq=freq,
                lags=lags,
                date_features=date_features,
                num_threads=1,
            )
            ml.fit(df, **fit_kwargs())
            explainability = _mlforecast_feature_importance(ml)
            forecast = ml.predict(h=horizon, **predict_kwargs)
        warnings_out.extend(_format_caught_warnings(caught, prefix="MLForecast warning"))
        return _normalize_statsforecast_output(forecast), explainability, warnings_out
    except Exception as exc:
        warnings_out.append(f"MLForecast full ladder failed; retrying candidate-by-candidate ({exc})")

    frames: list[pd.DataFrame] = []
    explainability_frames: list[pd.DataFrame] = []
    for alias, factory in model_factories:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ml = MLForecast(
                    models={alias: factory()},
                    freq=freq,
                    lags=lags,
                    date_features=date_features,
                    num_threads=1,
                )
                ml.fit(df, **fit_kwargs())
                explainability_frames.append(_mlforecast_feature_importance(ml))
                forecast = ml.predict(h=horizon, **predict_kwargs)
            frames.append(_normalize_statsforecast_output(forecast))
            warnings_out.extend(_format_caught_warnings(caught, prefix="MLForecast warning"))
        except Exception as model_exc:
            if prediction_intervals_factory is None:
                warnings_out.append(f"MLForecast candidate {alias} failed and was skipped ({model_exc})")
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ml = MLForecast(
                        models={alias: factory()},
                        freq=freq,
                        lags=lags,
                        date_features=date_features,
                        num_threads=1,
                    )
                    ml.fit(df)
                    explainability_frames.append(_mlforecast_feature_importance(ml))
                    forecast = ml.predict(h=horizon)
                frames.append(_normalize_statsforecast_output(forecast))
                warnings_out.extend(_format_caught_warnings(caught, prefix="MLForecast warning"))
                warnings_out.append(
                    f"MLForecast candidate {alias} could not produce conformal intervals; kept point forecast only ({model_exc})"
                )
            except Exception as point_exc:
                warnings_out.append(f"MLForecast candidate {alias} failed and was skipped ({point_exc})")
    if not frames:
        raise RuntimeError("no MLForecast candidate model able to be fitted")
    explainability = (
        pd.concat(explainability_frames, ignore_index=True)
        if explainability_frames
        else _empty_model_explainability()
    )
    return _merge_statsforecast_frames(frames, keys=["unique_id", "ds"]), explainability, sorted(set(warnings_out))


def _mlforecast_cross_validation_resilient(
    model_factories: list[tuple[str, Callable[[], Any]]],
    *,
    history: pd.DataFrame,
    freq: str,
    lags: list[int],
    date_features: list[str],
    horizon: int,
    step_size: int,
    n_windows: int,
    levels: tuple[int, ...],
    prediction_intervals_factory: Callable[[], Any] | None,
) -> tuple[pd.DataFrame, list[str]]:
    from mlforecast import MLForecast

    warnings_out: list[str] = []
    df = history[["unique_id", "ds", "y"]]

    def cv_kwargs() -> dict[str, Any]:
        if prediction_intervals_factory is None:
            return {}
        return {"prediction_intervals": prediction_intervals_factory(), "level": list(levels)}

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cv = MLForecast(
                models={alias: factory() for alias, factory in model_factories},
                freq=freq,
                lags=lags,
                date_features=date_features,
                num_threads=1,
            ).cross_validation(
                df=df,
                h=horizon,
                step_size=step_size,
                n_windows=n_windows,
                **cv_kwargs(),
            )
        warnings_out.extend(_format_caught_warnings(caught, prefix="MLForecast warning"))
        return _normalize_statsforecast_output(cv), warnings_out
    except Exception as exc:
        warnings_out.append(f"MLForecast full-ladder cross-validation failed; retrying candidate-by-candidate ({exc})")

    frames: list[pd.DataFrame] = []
    for alias, factory in model_factories:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cv = MLForecast(
                    models={alias: factory()},
                    freq=freq,
                    lags=lags,
                    date_features=date_features,
                    num_threads=1,
                ).cross_validation(
                    df=df,
                    h=horizon,
                    step_size=step_size,
                    n_windows=n_windows,
                    **cv_kwargs(),
                )
            frames.append(_normalize_statsforecast_output(cv))
            warnings_out.extend(_format_caught_warnings(caught, prefix="MLForecast warning"))
        except Exception as model_exc:
            if prediction_intervals_factory is None:
                warnings_out.append(f"MLForecast candidate {alias} cross-validation failed and was skipped ({model_exc})")
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    cv = MLForecast(
                        models={alias: factory()},
                        freq=freq,
                        lags=lags,
                        date_features=date_features,
                        num_threads=1,
                    ).cross_validation(
                        df=df,
                        h=horizon,
                        step_size=step_size,
                        n_windows=n_windows,
                    )
                frames.append(_normalize_statsforecast_output(cv))
                warnings_out.extend(_format_caught_warnings(caught, prefix="MLForecast warning"))
                warnings_out.append(
                    f"MLForecast candidate {alias} CV intervals failed; kept point backtest only ({model_exc})"
                )
            except Exception as point_exc:
                warnings_out.append(f"MLForecast candidate {alias} cross-validation failed and was skipped ({point_exc})")
    if not frames:
        raise RuntimeError("no MLForecast candidate model able to be backtested")
    return _merge_statsforecast_frames(frames, keys=["unique_id", "ds", "cutoff"]), sorted(set(warnings_out))


def select_champions(
    all_models: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    model_cols = model_columns(all_models)
    available_by_series = _available_forecast_models_by_series(all_models, model_cols)
    fallback = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "selected_model": _default_model(available_by_series.get(str(uid)) or model_cols),
                "selection_reason": "default finite model; backtest not available for this series",
            }
            for uid in sorted(all_models["unique_id"].unique())
        ]
    )
    if metrics.empty:
        return fallback

    # Use RMSE as primary metric (penalizes large errors quadratically)
    # Fall back to MAE if RMSE is not available
    sort_col = "rmse" if "rmse" in metrics.columns else "mae"
    usable = metrics.dropna(subset=[sort_col]).copy()
    usable = _filter_metrics_to_available_forecasts(usable, available_by_series)
    if usable.empty:
        usable = metrics.dropna(subset=["mae"]).copy()
        sort_col = "mae"
        usable = _filter_metrics_to_available_forecasts(usable, available_by_series)
    if usable.empty:
        return fallback
    usable = usable.sort_values(["unique_id", sort_col, "mae", "abs_bias"])
    chosen = usable.groupby("unique_id", as_index=False).first()
    chosen = chosen.rename(columns={"model": "selected_model"})
    chosen["selection_reason"] = f"lowest backtested {sort_col.upper()}, tie-broken by MAE and bias"
    keep_cols = ["unique_id", "selected_model", "selection_reason"]
    for col in _selection_metric_columns(chosen):
        if col in chosen.columns:
            keep_cols.append(col)
    chosen = chosen[keep_cols]
    chosen = _apply_naive_guard(chosen, usable, sort_col=sort_col)
    chosen = _append_horizon_selection_reason(chosen)
    result_cols = ["unique_id", "selected_model", "selection_reason"]
    for col in _selection_metric_columns(chosen):
        if col in chosen.columns:
            result_cols.append(col)
    return fallback.merge(chosen, on="unique_id", how="left", suffixes=("_fallback", "")).assign(
        selected_model=lambda df: df["selected_model"].fillna(df["selected_model_fallback"]),
        selection_reason=lambda df: df["selection_reason"].fillna(df["selection_reason_fallback"]),
    )[result_cols]


def _append_horizon_selection_reason(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty or not {"selection_reason", "selection_horizon", "requested_horizon"}.issubset(selection.columns):
        return selection
    out = selection.copy()
    parts: list[str] = []
    for row in out.to_dict("records"):
        reason = str(row.get("selection_reason") or "")
        try:
            selected_h = int(row.get("selection_horizon"))
            requested_h = int(row.get("requested_horizon"))
        except (TypeError, ValueError):
            parts.append(reason)
            continue
        if selected_h >= requested_h:
            try:
                windows = int(row.get("cv_windows"))
            except (TypeError, ValueError):
                windows = None
            if windows is not None and windows < 2:
                suffix = f"validated through requested horizon {requested_h} with only {windows} CV window(s); planning claim limited"
            else:
                suffix = f"validated through requested horizon {requested_h}"
        else:
            suffix = f"validated through CV horizon {selected_h} of requested {requested_h}; later steps are directional"
        parts.append(f"{reason}; {suffix}" if suffix not in reason else reason)
    out["selection_reason"] = parts
    return out


def _available_forecast_models_by_series(all_models: pd.DataFrame, model_cols: list[str]) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for uid, grp in all_models.groupby("unique_id", sort=True):
        uid_key = str(uid)
        available[uid_key] = []
        for model in model_cols:
            values = pd.to_numeric(grp[model], errors="coerce")
            if values.notna().all() and np.isfinite(values.to_numpy(dtype="float64")).all():
                available[uid_key].append(model)
    return available


def _filter_metrics_to_available_forecasts(
    metrics: pd.DataFrame,
    available_by_series: dict[str, list[str]],
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    available_pairs = {
        (uid, model)
        for uid, models in available_by_series.items()
        for model in models
    }
    if not available_pairs:
        return metrics.iloc[0:0].copy()
    mask = [
        (str(row["unique_id"]), str(row["model"])) in available_pairs
        for row in metrics[["unique_id", "model"]].to_dict("records")
    ]
    return metrics.loc[mask].copy()


def _apply_naive_guard(chosen: pd.DataFrame, all_metrics: pd.DataFrame, *, sort_col: str = "rmse") -> pd.DataFrame:
    """If the selected model doesn't beat naive by ≥5% on the primary metric
    (RMSE by default), fall back to naive. Penalizes big errors more when using RMSE."""
    naive_models = {"Naive", "SeasonalNaive"}
    naive_metrics = all_metrics[all_metrics["model"].isin(naive_models)].copy()
    if naive_metrics.empty:
        return chosen
    metric_col = sort_col if sort_col in naive_metrics.columns else "mae"
    best_naive = naive_metrics.sort_values(["unique_id", metric_col, "mae"]).groupby("unique_id", as_index=False).first()
    rename_map = {"model": "naive_model"}
    keep_naive = ["unique_id", "naive_model"]
    for col in _selection_metric_columns(best_naive):
        if col in best_naive.columns:
            rename_map[col] = f"naive_{col}"
            keep_naive.append(f"naive_{col}")
    best_naive = best_naive.rename(columns=rename_map)[keep_naive]
    merged = chosen.merge(best_naive, on="unique_id", how="left")
    naive_metric_col = f"naive_{metric_col}"
    margin = 0.05
    guard_mask = (
        merged["selected_model"].notna()
        & ~merged["selected_model"].isin(naive_models)
        & merged[metric_col].notna()
        & merged[naive_metric_col].notna()
        & (merged[naive_metric_col] > 0)
        & ((1.0 - merged[metric_col] / merged[naive_metric_col]) < margin)
    )
    if guard_mask.any():
        merged.loc[guard_mask, "selected_model"] = merged.loc[guard_mask, "naive_model"]
        for col in _selection_metric_columns(merged):
            nc = f"naive_{col}"
            if col in merged.columns and nc in merged.columns:
                merged.loc[guard_mask, col] = merged.loc[guard_mask, nc]
        merged.loc[guard_mask, "selection_reason"] = (
            f"naive guard: selected model did not beat naive by ≥5% on backtested {metric_col.upper()}; "
            "fell back to best naive baseline per benchmark guardrail"
        )
    result_cols = ["unique_id", "selected_model", "selection_reason"]
    for col in _selection_metric_columns(merged):
        if col in merged.columns:
            result_cols.append(col)
    return merged[result_cols]


def build_selected_forecast(all_models: pd.DataFrame, selection: pd.DataFrame, levels: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = selection.set_index("unique_id")["selected_model"].to_dict()
    for _, row in all_models.iterrows():
        uid = row["unique_id"]
        model = selected.get(uid)
        if model is None or model not in row.index:
            continue
        out: dict[str, Any] = {
            "unique_id": uid,
            "ds": row["ds"],
            "yhat": row[model],
            "model": model,
        }
        for level in levels:
            lo_col = f"{model}-lo-{level}"
            hi_col = f"{model}-hi-{level}"
            if lo_col in row.index and hi_col in row.index:
                out[f"yhat_lo_{level}"] = row[lo_col]
                out[f"yhat_hi_{level}"] = row[hi_col]
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def model_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in {"unique_id", "ds"} and "-lo-" not in col and "-hi-" not in col
    ]


def _non_ensemble_model_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in model_columns(frame) if col not in {"cutoff", "y", "WeightedEnsemble"}]


def _drop_weighted_ensemble_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    drop_cols = [
        col
        for col in frame.columns
        if col == "WeightedEnsemble" or str(col).startswith("WeightedEnsemble-lo-") or str(col).startswith("WeightedEnsemble-hi-")
    ]
    return frame.drop(columns=drop_cols, errors="ignore")


def _merge_prediction_frames(primary: pd.DataFrame, secondary: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    if primary.empty:
        return secondary.copy()
    if secondary.empty:
        return primary.copy()
    payload_cols = [col for col in secondary.columns if col not in set(keys) and col not in primary.columns]
    if not payload_cols:
        return primary.copy()
    return primary.merge(secondary[keys + payload_cols], on=keys, how="left")


def _default_model(model_cols: list[str]) -> str:
    preferred = ["AutoARIMA", "AutoETS", "SeasonalNaive", "RandomWalkWithDrift", "Naive", "WindowAverage", "HistoricAverage"]
    for model in preferred:
        if model in model_cols:
            return model
    if not model_cols:
        raise ValueError("no model forecast columns available for selection")
    return model_cols[0]


def _statsforecast_backtest(
    models: list[StatsForecastFactory],
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    for uid, grp in history.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds")
        n = len(grp)
        season = profile.season_length
        h, n_windows, step_size = _adaptive_cv_params(n, spec.horizon, season, strict=spec.strict_cv_horizon)
        if n_windows < 1:
            all_warnings.append(f"{uid}: skipped backtest — only {n} obs, need ≥{h + h + 1} for 1 window")
            continue
        interval_kwargs: dict[str, Any] = {}
        if spec.levels:
            shortest_train = n - h - step_size * max(0, n_windows - 1)
            cv_interval_windows = _interval_windows(shortest_train, h, season)
            if cv_interval_windows >= 2:
                from statsforecast.utils import ConformalIntervals

                interval_kwargs = {
                    "level": list(spec.levels),
                    "prediction_intervals": ConformalIntervals(h=h, n_windows=cv_interval_windows),
                }
                all_warnings.append(
                    f"{uid}: StatsForecast CV conformal intervals enabled "
                    f"(levels={list(spec.levels)}, h={h}, n_windows={cv_interval_windows})"
                )
            else:
                all_warnings.append(
                    f"{uid}: StatsForecast CV interval diagnostics skipped because the shortest rolling-origin "
                    "training fold cannot support two conformal windows"
                )
        try:
            cv, cv_warnings = _statsforecast_cross_validation_resilient(
                models,
                history=grp,
                freq=profile.freq,
                horizon=h,
                step_size=step_size,
                n_windows=n_windows,
                verbose=spec.verbose,
                interval_kwargs=interval_kwargs,
            )
        except Exception as exc:
            all_warnings.append(f"{uid}: StatsForecast backtest failed for all candidate models ({exc})")
            continue
        cv = _add_zero_forecast_to_cv(cv, grp)
        if spec.weighted_ensemble:
            cv = _add_weighted_ensemble_to_cv(cv)
        metrics = _attach_cv_metadata(
            _metrics_from_cv(cv, scales_by_series=_error_scale_map(grp, season)),
            requested_horizon=spec.horizon,
            selection_horizon=h,
            n_windows=n_windows,
            step_size=step_size,
        )
        metric_frames.append(metrics)
        prediction_frames.append(cv)
        all_warnings.extend(cv_warnings)
        all_warnings.append(f"{uid}: backtest h={h}, n_windows={n_windows}, step_size={step_size} (from {n} obs, season={season})")
        if h < spec.horizon:
            all_warnings.append(
                f"{uid}: model selection used CV horizon {h}, shorter than requested forecast horizon {spec.horizon}; "
                "treat long-horizon steps as less validated or rerun with strict_cv_horizon=True"
            )

    if not metric_frames:
        return (
            _empty_backtest_metrics(),
            all_warnings,
            _empty_backtest_predictions(),
        )
    return pd.concat(metric_frames, ignore_index=True), sorted(set(all_warnings)), pd.concat(prediction_frames, ignore_index=True)


def _statsforecast_cross_validation_resilient(
    models: list[StatsForecastFactory],
    *,
    history: pd.DataFrame,
    freq: str,
    horizon: int,
    step_size: int,
    n_windows: int,
    verbose: bool,
    interval_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    from statsforecast import StatsForecast

    warnings_out: list[str] = []
    df = history[["unique_id", "ds", "y"]]
    interval_kwargs = interval_kwargs or {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            cv = StatsForecast(
                models=[factory() for _, factory in models],
                freq=freq,
                n_jobs=1,
                verbose=verbose,
            ).cross_validation(
                df=df,
                h=horizon,
                step_size=step_size,
                n_windows=n_windows,
                **interval_kwargs,
            )
        warnings_out.extend(_format_caught_warnings(caught))
        return _normalize_statsforecast_output(cv), warnings_out
    except Exception as exc:
        uid = str(history["unique_id"].iloc[0]) if not history.empty else "series"
        warnings_out.append(f"{uid}: StatsForecast full-ladder backtest failed; retrying candidate-by-candidate ({exc})")

    frames: list[pd.DataFrame] = []
    for alias, factory in models:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                cv = StatsForecast(
                    models=[factory()],
                    freq=freq,
                    n_jobs=1,
                    verbose=verbose,
                ).cross_validation(
                    df=df,
                    h=horizon,
                    step_size=step_size,
                    n_windows=n_windows,
                    **interval_kwargs,
                )
            frames.append(_normalize_statsforecast_output(cv))
            warnings_out.extend(_format_caught_warnings(caught))
        except Exception as exc:
            if not interval_kwargs:
                warnings_out.append(f"StatsForecast candidate {alias} backtest failed and was skipped ({exc})")
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    cv = StatsForecast(
                        models=[factory()],
                        freq=freq,
                        n_jobs=1,
                        verbose=verbose,
                    ).cross_validation(
                        df=df,
                        h=horizon,
                        step_size=step_size,
                        n_windows=n_windows,
                    )
                frames.append(_normalize_statsforecast_output(cv))
                warnings_out.extend(_format_caught_warnings(caught))
                warnings_out.append(
                    f"StatsForecast candidate {alias} CV intervals failed; kept point backtest only ({exc})"
                )
            except Exception as point_exc:
                warnings_out.append(f"StatsForecast candidate {alias} backtest failed and was skipped ({point_exc})")
    if not frames:
        raise RuntimeError("no StatsForecast candidate model able to be backtested")
    return _merge_statsforecast_frames(frames, keys=["unique_id", "ds", "cutoff"]), sorted(set(warnings_out))


def _baseline_backtest(
    history: pd.DataFrame,
    profile: DataProfile,
    spec: ForecastSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    season = profile.season_length
    for uid, grp in history.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds").reset_index(drop=True)
        n = len(grp)
        h, n_windows, step_size = _adaptive_cv_params(n, spec.horizon, season, strict=spec.strict_cv_horizon)
        if h < 1 or n_windows < 1:
            continue
        metadata_rows.append(
            {
                "unique_id": uid,
                "requested_horizon": spec.horizon,
                "selection_horizon": h,
                "cv_windows": n_windows,
                "cv_step_size": step_size,
                "cv_horizon_matches_requested": h == spec.horizon,
            }
        )
        for window in range(n_windows, 0, -1):
            cutoff_idx = n - 1 - window * step_size - (h - step_size)
            if cutoff_idx < 0:
                cutoff_idx = max(0, n - 1 - window * h)
            train = grp.iloc[: cutoff_idx + 1]
            test = grp.iloc[cutoff_idx + 1 : cutoff_idx + 1 + h]
            if train.empty or test.empty:
                continue
            forecast = _baseline_predictions(train, profile.freq, profile.season_length, len(test))
            forecast = forecast[forecast["unique_id"] == uid].copy()
            forecast["cutoff"] = train["ds"].max()
            forecast["y"] = test["y"].to_numpy()
            rows.append(forecast)
    if not rows:
        return _empty_backtest_metrics(), _empty_backtest_predictions()
    cv = pd.concat(rows, ignore_index=True)
    if spec.weighted_ensemble:
        cv = _add_weighted_ensemble_to_cv(cv)
    metrics = _metrics_from_cv(cv, scales_by_series=_error_scale_map(history, season))
    if metadata_rows:
        metrics = metrics.merge(pd.DataFrame(metadata_rows), on="unique_id", how="left")
    return metrics, cv


def _metrics_from_cv(
    cv: pd.DataFrame,
    *,
    include_weighted: bool = False,
    scales_by_series: dict[str, dict[str, float | None]] | None = None,
) -> pd.DataFrame:
    if include_weighted:
        cv = _add_weighted_ensemble_to_cv(cv)
    model_cols = [col for col in model_columns(cv) if col not in {"cutoff", "y"}]
    rows: list[dict[str, Any]] = []
    for uid, grp in cv.groupby("unique_id", sort=True):
        actual = grp["y"].to_numpy(dtype="float64")
        denom = float(np.nansum(np.abs(actual)))
        for model in model_cols:
            pred = grp[model].to_numpy(dtype="float64")
            err = pred - actual
            if not np.isfinite(err).any():
                continue
            mae = float(np.nanmean(np.abs(err)))
            rmse = float(np.sqrt(np.nanmean(err ** 2)))
            wape = float(np.nansum(np.abs(err)) / denom) if denom else np.nan
            bias = float(np.nansum(err) / denom) if denom else np.nan
            scales = (scales_by_series or {}).get(str(uid), {})
            mase_scale = scales.get("mase_scale")
            rmsse_scale = scales.get("rmsse_scale")
            rows.append(
                {
                    "unique_id": uid,
                    "model": model,
                    "rmse": rmse,
                    "mae": mae,
                    "wape": wape,
                    "mase": mae / mase_scale if mase_scale else np.nan,
                    "rmsse": rmse / rmsse_scale if rmsse_scale else np.nan,
                    "bias": bias,
                    "abs_bias": abs(bias) if not np.isnan(bias) else np.nan,
                    "observations": int(np.isfinite(err).sum()),
                }
            )
    if not rows:
        return _empty_backtest_metrics()
    return pd.DataFrame(rows)


def _empty_backtest_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "unique_id",
            "model",
            "rmse",
            "mae",
            "wape",
            "mase",
            "rmsse",
            "bias",
            "abs_bias",
            "observations",
            "requested_horizon",
            "selection_horizon",
            "cv_windows",
            "cv_step_size",
            "cv_horizon_matches_requested",
        ]
    )


def _attach_cv_metadata(
    metrics: pd.DataFrame,
    *,
    requested_horizon: int,
    selection_horizon: int,
    n_windows: int,
    step_size: int,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics.copy()
    out["requested_horizon"] = int(requested_horizon)
    out["selection_horizon"] = int(selection_horizon)
    out["cv_windows"] = int(n_windows)
    out["cv_step_size"] = int(step_size)
    out["cv_horizon_matches_requested"] = int(selection_horizon) == int(requested_horizon)
    return out


def _cv_horizon_warnings(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty or "cv_horizon_matches_requested" not in metrics.columns:
        return []
    warnings_out: list[str] = []
    matches = metrics["cv_horizon_matches_requested"].fillna(False).astype(bool)
    mismatched = metrics[~matches]
    if mismatched.empty:
        return []
    for row in mismatched[["unique_id", "requested_horizon", "selection_horizon"]].drop_duplicates().to_dict("records"):
        warnings_out.append(
            f"{row['unique_id']}: model selection used CV horizon {int(row['selection_horizon'])}, "
            f"shorter than requested forecast horizon {int(row['requested_horizon'])}; "
            "longer-horizon forecast steps are less validated"
        )
    return warnings_out


def _selection_metric_columns(frame: pd.DataFrame) -> list[str]:
    ordered = [
        "rmse",
        "wape",
        "mae",
        "mase",
        "rmsse",
        "bias",
        "abs_bias",
        "observations",
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
    ]
    return [column for column in ordered if column in frame.columns]


def _error_scale_map(history: pd.DataFrame, season_length: int) -> dict[str, dict[str, float | None]]:
    scales: dict[str, dict[str, float | None]] = {}
    lag = max(1, int(season_length or 1))
    for uid, grp in history.groupby("unique_id", sort=True):
        y = pd.to_numeric(grp.sort_values("ds")["y"], errors="coerce").dropna().to_numpy(dtype="float64")
        active_lag = lag if lag > 1 and len(y) > lag else 1
        if len(y) <= active_lag:
            scales[str(uid)] = {"mase_scale": None, "rmsse_scale": None}
            continue
        diff = y[active_lag:] - y[:-active_lag]
        abs_scale = float(np.nanmean(np.abs(diff)))
        squared_scale = float(np.sqrt(np.nanmean(diff ** 2)))
        scales[str(uid)] = {
            "mase_scale": abs_scale if np.isfinite(abs_scale) and abs_scale > 0 else None,
            "rmsse_scale": squared_scale if np.isfinite(squared_scale) and squared_scale > 0 else None,
        }
    return scales


def _add_weighted_ensemble_to_cv(cv: pd.DataFrame, *, common_support: bool = False) -> pd.DataFrame:
    out = cv.copy()
    model_cols = [col for col in model_columns(out) if col not in {"cutoff", "y", "WeightedEnsemble"}]
    if not model_cols or "y" not in out.columns or "cutoff" not in out.columns:
        return out
    use_common_support = common_support and _has_classical_and_ml_models(model_cols)
    out["WeightedEnsemble"] = np.nan
    for _, idx in out.groupby("unique_id", sort=True).groups.items():
        grp = out.loc[idx]
        ds_values = pd.to_datetime(grp["ds"])
        cutoff_values = pd.to_datetime(grp["cutoff"])
        cutoffs = sorted(cutoff_values.dropna().unique())
        for cutoff in cutoffs:
            cutoff_mask = cutoff_values == cutoff
            prior = grp.loc[ds_values <= cutoff]
            if prior.empty:
                continue
            if use_common_support:
                support = _common_support_rows(prior, model_cols)
                if support.empty:
                    continue
                weights = _weights_from_predictions(support, _complete_support_models(support, model_cols))
            else:
                weights = _weights_from_predictions(prior, model_cols)
            if not weights:
                continue
            current_index = grp.loc[cutoff_mask].index
            current = out.loc[current_index]
            out.loc[current_index, "WeightedEnsemble"] = _weighted_prediction_series(current, weights).to_numpy(dtype="float64")
    return out


def _model_weights_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return _empty_model_weights()
    rows: list[dict[str, Any]] = []
    metric_rows = metrics[metrics["model"] != "WeightedEnsemble"].copy()
    for uid, grp in metric_rows.groupby("unique_id", sort=True):
        weights = _weights_from_metric_rows(grp)
        for model, payload in weights.items():
            rows.append(
                {
                    "unique_id": uid,
                    "model": model,
                    "family": model_family(model),
                    "weight": payload["weight"],
                    "score_metric": payload["score_metric"],
                    "score_value": payload["score_value"],
                }
            )
    if not rows:
        return _empty_model_weights()
    return pd.DataFrame(rows).sort_values(["unique_id", "weight", "model"], ascending=[True, False, True]).reset_index(drop=True)


def _model_weights_from_common_support(cv: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if cv.empty or "y" not in cv.columns:
        return _empty_model_weights(), []
    rows: list[dict[str, Any]] = []
    warnings_out: list[str] = []
    model_cols = _non_ensemble_model_columns(cv)
    if not _has_classical_and_ml_models(model_cols):
        return _model_weights_from_metrics(_metrics_from_cv(cv)), []
    for uid, grp in cv.groupby("unique_id", sort=True):
        support = _common_support_rows(grp, model_cols)
        if support.empty:
            warnings_out.append(f"{uid}: combined model weights skipped because no common StatsForecast/MLForecast CV support was available")
            continue
        eligible_models = _complete_support_models(support, model_cols)
        for model in model_cols:
            if model not in eligible_models:
                warnings_out.append(
                    f"{uid}: {model} excluded from combined ensemble weights because it lacks complete common CV support"
                )
        weights = _weights_from_predictions(support, eligible_models)
        scores = _rmse_scores_from_predictions(support, eligible_models)
        for model, weight in weights.items():
            rows.append(
                {
                    "unique_id": uid,
                    "model": model,
                    "family": model_family(model),
                    "weight": weight,
                    "score_metric": "rmse",
                    "score_value": scores.get(model, np.nan),
                }
            )
    if not rows:
        return _empty_model_weights(), sorted(set(warnings_out))
    return (
        pd.DataFrame(rows).sort_values(["unique_id", "weight", "model"], ascending=[True, False, True]).reset_index(drop=True),
        sorted(set(warnings_out)),
    )


def _has_classical_and_ml_models(model_cols: list[str]) -> bool:
    families = {model_family(model) for model in model_cols}
    return bool(families & {"baseline", "statsforecast"}) and "mlforecast" in families


def _common_support_rows(frame: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    classical_cols = [model for model in model_cols if model_family(model) in {"baseline", "statsforecast"}]
    ml_cols = [model for model in model_cols if model_family(model) == "mlforecast"]
    if not classical_cols or not ml_cols:
        return frame.copy()
    classical_finite = _finite_any(frame, classical_cols)
    ml_finite = _finite_any(frame, ml_cols)
    return frame.loc[classical_finite & ml_finite].copy()


def _complete_support_models(support: pd.DataFrame, model_cols: list[str]) -> list[str]:
    eligible_models: list[str] = []
    if support.empty:
        return eligible_models
    for model in model_cols:
        if model not in support.columns:
            continue
        values = pd.to_numeric(support[model], errors="coerce")
        finite = values.notna() & pd.Series(np.isfinite(values.to_numpy(dtype="float64")), index=values.index)
        if len(values) == len(support) and bool(finite.all()):
            eligible_models.append(model)
    return eligible_models


def _finite_any(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=frame.index)
    available = [col for col in columns if col in frame.columns]
    if not available:
        return pd.Series(False, index=frame.index)
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(np.isfinite(values.to_numpy(dtype="float64")), index=values.index, columns=values.columns)
    return finite.any(axis=1)


def _add_weighted_ensemble_forecast(forecast: pd.DataFrame, model_weights: pd.DataFrame) -> pd.DataFrame:
    if model_weights.empty:
        return forecast
    out = forecast.copy()
    out["WeightedEnsemble"] = np.nan
    for uid, weights in model_weights.groupby("unique_id", sort=True):
        mask = out["unique_id"] == uid
        if not mask.any():
            continue
        model_weights = {str(row["model"]): float(row["weight"]) for row in weights.to_dict("records")}
        ensemble = _weighted_prediction_series(out.loc[mask], model_weights)
        if ensemble.notna().any():
            out.loc[mask, "WeightedEnsemble"] = ensemble.to_numpy(dtype="float64")
    return out


def _weighted_prediction_series(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weighted = pd.Series(0.0, index=frame.index, dtype="float64")
    total_weight = pd.Series(0.0, index=frame.index, dtype="float64")
    for model, weight in weights.items():
        if model not in frame.columns:
            continue
        values = pd.to_numeric(frame[model], errors="coerce").astype("float64")
        valid = values.notna() & pd.Series(np.isfinite(values.to_numpy(dtype="float64")), index=values.index)
        if not valid.any():
            continue
        weighted.loc[valid] += values.loc[valid] * float(weight)
        total_weight.loc[valid] += float(weight)
    return weighted / total_weight.replace(0.0, np.nan)


def _weights_from_predictions(grp: pd.DataFrame, model_cols: list[str]) -> dict[str, float]:
    """Use RMSE-based inverse weights (penalizes big errors more)."""
    scores = _rmse_scores_from_predictions(grp, model_cols)
    return _inverse_error_weights(scores)


def _rmse_scores_from_predictions(grp: pd.DataFrame, model_cols: list[str]) -> dict[str, float]:
    actual = pd.to_numeric(grp["y"], errors="coerce").to_numpy(dtype="float64")
    scores: dict[str, float] = {}
    for model in model_cols:
        if model not in grp.columns:
            continue
        pred = pd.to_numeric(grp[model], errors="coerce").to_numpy(dtype="float64")
        err = pred - actual
        if not np.isfinite(err).any():
            continue
        score = float(np.sqrt(np.nanmean(err ** 2)))
        if np.isfinite(score):
            scores[model] = max(score, 0.0)
    return scores


def _weights_from_metric_rows(grp: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    """Use RMSE as primary metric for weights, fall back to MAE."""
    scores: dict[str, float] = {}
    metric_names: dict[str, str] = {}
    for row in grp.to_dict("records"):
        model = str(row["model"])
        rmse = float(row["rmse"]) if pd.notna(row.get("rmse")) else np.nan
        mae = float(row["mae"]) if pd.notna(row.get("mae")) else np.nan
        if np.isfinite(rmse):
            scores[model] = max(rmse, 0.0)
            metric_names[model] = "rmse"
        elif np.isfinite(mae):
            scores[model] = max(mae, 0.0)
            metric_names[model] = "mae"
    weights = _inverse_error_weights(scores)
    return {
        model: {"weight": weight, "score_metric": metric_names[model], "score_value": scores[model]}
        for model, weight in weights.items()
    }


def _inverse_error_weights(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    zero_score_models = [model for model, score in scores.items() if score == 0]
    if zero_score_models:
        weight = 1.0 / len(zero_score_models)
        return {model: weight for model in zero_score_models}
    inverse = {model: 1.0 / score for model, score in scores.items() if score > 0 and np.isfinite(score)}
    total = sum(inverse.values())
    if total <= 0:
        return {}
    return {model: weight / total for model, weight in inverse.items()}


def _empty_model_weights() -> pd.DataFrame:
    return pd.DataFrame(columns=["unique_id", "model", "family", "weight", "score_metric", "score_value"])


def _empty_model_explainability() -> pd.DataFrame:
    return pd.DataFrame(columns=["engine", "model", "feature", "importance", "importance_type", "interpretation"])


def _mlforecast_feature_importance(ml: Any) -> pd.DataFrame:
    """Return global feature importances or coefficient magnitudes from fitted MLForecast models."""

    rows: list[dict[str, Any]] = []
    models = getattr(ml, "models_", {}) or {}
    fallback_features = list(getattr(getattr(ml, "ts", None), "features_order_", []) or getattr(getattr(ml, "ts", None), "features", []) or [])
    for model_name, estimator in models.items():
        importances, importance_type = _estimator_importances(estimator)
        if importances is None:
            continue
        features = list(getattr(_unwrap_pipeline_estimator(estimator), "feature_name_", None) or fallback_features)
        if not features:
            features = [f"feature_{idx}" for idx in range(len(importances))]
        for feature, importance in zip(features, importances, strict=False):
            rows.append(
                {
                    "engine": "mlforecast",
                    "model": str(model_name),
                    "feature": str(feature),
                    "importance": float(importance),
                    "importance_type": importance_type,
                    "interpretation": _feature_interpretation(str(feature)),
                }
            )
    if not rows:
        return _empty_model_explainability()
    return pd.DataFrame(rows).sort_values(["model", "importance", "feature"], ascending=[True, False, True]).reset_index(drop=True)


def _unwrap_pipeline_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "steps") and getattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator


def _estimator_importances(estimator: Any) -> tuple[np.ndarray | None, str]:
    core = _unwrap_pipeline_estimator(estimator)
    importances = getattr(core, "feature_importances_", None)
    if importances is not None:
        return np.asarray(importances, dtype="float64"), "tree_feature_importance"
    coefficients = getattr(core, "coef_", None)
    if coefficients is not None:
        values = np.asarray(coefficients, dtype="float64")
        if values.ndim > 1:
            values = np.nanmean(np.abs(values), axis=0)
        else:
            values = np.abs(values)
        return values, "absolute_coefficient"
    return None, ""


def _feature_interpretation(feature: str) -> str:
    if feature.startswith("lag"):
        lag = feature.removeprefix("lag")
        return f"lagged target value {lag} period(s) back"
    if feature == "month":
        return "calendar month seasonality feature"
    if feature == "dayofweek":
        return "day-of-week seasonality feature"
    return "MLForecast generated feature"


def _empty_backtest_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=["unique_id", "ds", "cutoff", "y"])


def _weighted_ensemble_warnings(spec: ForecastSpec, model_weights: pd.DataFrame) -> list[str]:
    if not spec.weighted_ensemble:
        return ["weighted ensemble disabled by ForecastSpec.weighted_ensemble=False"]
    if model_weights.empty:
        return ["weighted ensemble skipped because no finite backtest metrics were available"]
    # Ensemble-enabled is informational, not a warning — documented in model_weights.csv
    return []


def _format_caught_warnings(caught: list[warnings.WarningMessage], *, prefix: str = "StatsForecast warning") -> list[str]:
    messages = sorted({str(item.message) for item in caught if str(item.message)})
    return [f"{prefix}: {message}" for message in messages]


def _baseline_predictions(
    history: pd.DataFrame,
    freq: str,
    season_length: int,
    horizon: int,
) -> pd.DataFrame:
    forecasts = []
    for uid, grp in history.groupby("unique_id", sort=True):
        grp = grp.sort_values("ds")
        future = _future_dates(grp["ds"].max(), freq, horizon)
        y = grp["y"].to_numpy(dtype="float64")
        frame = pd.DataFrame({"unique_id": str(uid), "ds": future})
        frame["Naive"] = y[-1]
        frame["HistoricAverage"] = float(np.nanmean(y))
        frame["RandomWalkWithDrift"] = _random_walk_with_drift(y, horizon)
        frame["WindowAverage"] = _window_average(y, season_length)
        frame["SeasonalNaive"] = _seasonal_naive(y, season_length, horizon)
        if _is_intermittent_series(y):
            frame["ZeroForecast"] = 0.0
        forecasts.append(frame)
    return pd.concat(forecasts, ignore_index=True)


def _future_dates(last_date: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 1, freq=freq)[1:]


def _window_average(y: np.ndarray, season_length: int) -> float:
    window = max(1, min(len(y), season_length if season_length > 1 else 3))
    return float(np.nanmean(y[-window:]))


def _random_walk_with_drift(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) < 2:
        return np.repeat(y[-1], horizon)
    drift = (y[-1] - y[0]) / max(1, len(y) - 1)
    return y[-1] + drift * np.arange(1, horizon + 1)


def _seasonal_naive(y: np.ndarray, season_length: int, horizon: int) -> np.ndarray:
    if season_length <= 1 or len(y) < season_length:
        return np.repeat(y[-1], horizon)
    season = y[-season_length:]
    reps = int(np.ceil(horizon / len(season)))
    return np.tile(season, reps)[:horizon]


def _interval_windows(min_obs: int, horizon: int, season_length: int) -> int:
    min_train = max(8, season_length if season_length > 1 else 4)
    available = min_obs - min_train
    if available < horizon:
        return 0
    return max(0, min(3, available // max(1, horizon)))


def _adaptive_cv_params(
    n_obs: int,
    requested_horizon: int,
    season_length: int,
    *,
    strict: bool = False,
) -> tuple[int, int, int]:
    """Compute per-series backtest parameters adapted to history depth and seasonality.

    Returns (h, n_windows, step_size) where:
    - h: forecast horizon for each CV fold (may be smaller than requested for short series)
    - n_windows: number of rolling-origin folds
    - step_size: how far to slide the cutoff between folds

    Design principles (FPP ch. 5 + Nixtla cross_validation docs):
    - The test set should match the forecast horizon when possible
    - More windows = more reliable metrics, but each window needs ≥ season_length train obs
    - step_size = season_length gives non-overlapping seasonal folds (gold standard)
    - step_size = h gives non-overlapping horizon folds (simpler, still valid)
    - For short series, reduce windows before reducing horizon
    - For very short series (<2*season), reduce horizon to 1/4 of history
    """
    if n_obs < 3:
        return 0, 0, 1

    # Minimum training set: need enough context for models to learn
    # Short series: at least 2 obs for training
    # Normal: at least 4 obs or 1 season, whichever is smaller (but never more than half)
    if n_obs <= 6:
        min_train = 2
    elif season_length > 1:
        min_train = max(4, min(season_length, n_obs // 2))
    else:
        min_train = 4

    # Horizon: use requested when strict, otherwise cap at 1/4 of history for short series.
    h = requested_horizon if strict else min(requested_horizon, max(1, n_obs // 4))

    # Available observations after reserving min training
    available = n_obs - min_train

    if available < h:
        # Can't even do 1 window
        return h, 0, h

    # Step size: prefer season_length for seasonal alignment, fall back to h
    if season_length > 1 and season_length <= h:
        step_size = season_length
    else:
        step_size = max(1, h)

    # Max windows given available space and step size
    # Each window needs: min_train + (window_idx * step_size) + h <= n_obs
    max_windows = max(0, (available - h) // max(1, step_size) + 1)

    # Cap at 5 windows for rich history, 3 for moderate, 2 for limited
    if n_obs >= 10 * season_length and season_length > 1:
        cap = 5
    elif n_obs >= 4 * max(season_length, h):
        cap = 3
    else:
        cap = 2

    n_windows = min(cap, max_windows)
    if n_windows < 1:
        return h, 0, step_size

    return h, n_windows, step_size


def _normalize_statsforecast_output(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index() if "unique_id" not in frame.columns else frame.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _merge_statsforecast_frames(frames: list[pd.DataFrame], *, keys: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for frame in frames:
        if merged is None:
            merged = frame.copy()
            continue
        value_cols = [col for col in frame.columns if col not in keys and col not in merged.columns]
        if value_cols:
            merged = merged.merge(frame[keys + value_cols], on=keys, how="outer")
    if merged is None:
        return pd.DataFrame(columns=keys)
    return merged.sort_values(keys).reset_index(drop=True)


def _is_intermittent_series(y: np.ndarray) -> bool:
    """Detect intermittent demand: ≥35% zeros, non-negative, ≥6 observations."""
    if len(y) < 6 or np.any(y < 0):
        return False
    return float(np.mean(np.isclose(y, 0.0))) >= 0.35


def _has_intermittent_demand(history: pd.DataFrame) -> bool:
    """Check if any series in the panel has intermittent demand characteristics."""
    for _, grp in history.groupby("unique_id", sort=True):
        y = pd.to_numeric(grp["y"], errors="coerce").dropna().to_numpy(dtype="float64")
        if _is_intermittent_series(y):
            return True
    return False


def _add_zero_forecast_for_intermittent(forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add a ZeroForecast column for series detected as intermittent demand."""
    out = forecast.copy()
    intermittent_ids: set[str] = set()
    for uid, grp in history.groupby("unique_id", sort=True):
        y = pd.to_numeric(grp["y"], errors="coerce").dropna().to_numpy(dtype="float64")
        if _is_intermittent_series(y):
            intermittent_ids.add(str(uid))
    if intermittent_ids:
        out["ZeroForecast"] = np.where(out["unique_id"].astype(str).isin(intermittent_ids), 0.0, np.nan)
    return out


def _add_zero_forecast_to_cv(cv: pd.DataFrame, grp: pd.DataFrame) -> pd.DataFrame:
    """Add ZeroForecast column to cross-validation output if series is intermittent."""
    y = pd.to_numeric(grp["y"], errors="coerce").dropna().to_numpy(dtype="float64")
    if _is_intermittent_series(y):
        out = cv.copy()
        out["ZeroForecast"] = 0.0
        return out
    return cv


