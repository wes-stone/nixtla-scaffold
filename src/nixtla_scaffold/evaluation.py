"""Shared scoring rules for declared chronological comparison cohorts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from nixtla_scaffold.schema import DataProfile, EvaluationPolicy, ForecastSpec
    from nixtla_scaffold.models import ModelResult


def complete_metric_mean(frame: pd.DataFrame, column: str | None) -> float | None:
    """Equal-series mean, unavailable if any declared observation is undefined."""
    if column is None or frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
        return None
    if "unique_id" in frame:
        if frame["unique_id"].isna().any():
            return None
        values = values.groupby(frame["unique_id"], sort=False, observed=True).mean()
    result = float(values.mean())
    return result if np.isfinite(result) else None


@dataclass(frozen=True)
class Origin:
    unique_id: str
    cutoff: pd.Timestamp
    train_start: pd.Timestamp
    training_rows: int
    targets: tuple[pd.Timestamp, ...]

    def to_dict(self) -> dict:
        return {
            "unique_id": self.unique_id, "cutoff": self.cutoff.isoformat(),
            "train_start": self.train_start.isoformat(), "training_rows": self.training_rows,
            "targets": [value.isoformat() for value in self.targets],
        }


def cv_parameters(n: int, horizon: int, season: int, policy: EvaluationPolicy, *, strict: bool) -> tuple[int, int, int]:
    h = horizon if strict else min(horizon, max(1, n // 4))
    step = policy.step_size or h
    if policy.strategy == "none":
        return h, 0, step
    if policy.strategy != "rolling-origin":
        raise ValueError("Legacy evaluation must use its frozen parameter resolver")
    seasonal_minimum = max(policy.minimum_training_observations, policy.minimum_seasonal_cycles * season)
    capacity = min(n - h, policy.training_window or n)
    if capacity >= seasonal_minimum:
        minimum = seasonal_minimum
    elif capacity >= policy.minimum_training_observations:
        minimum = policy.minimum_training_observations
    else:
        minimum = policy.reference_minimum_training_observations
    if capacity < minimum:
        return h, 0, step
    return h, min(policy.max_windows, (n - minimum - h) // step + 1), step


def plan_origins(history: pd.DataFrame, spec: ForecastSpec, season: int) -> tuple[Origin, ...]:
    if spec.freq is None:
        raise ValueError("Resolve the data calendar before planning evaluation origins")
    origins = []
    for uid, group in history.groupby("unique_id", sort=True):
        dates = pd.DatetimeIndex(pd.to_datetime(group["ds"])).sort_values()
        if dates.hasnans or dates.has_duplicates:
            raise ValueError(f"{uid}: evaluation requires unique, non-null timestamps")
        if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq=spec.freq)):
            raise ValueError(f"{uid}: evaluation requires a complete declared calendar")
        h, windows, step = cv_parameters(len(dates), spec.horizon, season, spec.evaluation, strict=spec.strict_cv_horizon)
        for index in range(windows):
            stop = len(dates) - h - step * (windows - index - 1)
            start = max(0, stop - spec.evaluation.training_window) if spec.evaluation.training_window else 0
            origins.append(Origin(str(uid), dates[stop - 1], dates[start], stop - start, tuple(dates[stop:stop + h])))
    return tuple(origins)


def training_slice(history: pd.DataFrame, cutoff: pd.Timestamp | None, window: int | None) -> pd.DataFrame:
    selected = history if cutoff is None else history.loc[pd.to_datetime(history["ds"]).le(cutoff)]
    selected = selected.sort_values(["unique_id", "ds"])
    selected = selected.groupby("unique_id", sort=False).tail(window).copy() if window else selected.copy()
    if cutoff is not None and "available_at" in selected:
        released = pd.to_datetime(selected["available_at"], errors="coerce")
        withheld = released.isna() | released.gt(cutoff)
        for column in selected.columns.difference(["unique_id", "ds"]):
            selected[column] = selected[column].mask(withheld)
        selected.attrs["withheld_at_origin"] = int(withheld.sum())
    return selected


def disjoint_window_count(frame: pd.DataFrame, *, horizon: int | None = None) -> int:
    count, previous_end = 0, None
    for _, group in frame.groupby("cutoff", sort=True):
        if horizon is not None and (horizon <= 0 or group["ds"].nunique() != horizon):
            continue
        start, end = pd.to_datetime(group["ds"]).min(), pd.to_datetime(group["ds"]).max()
        if previous_end is None or start > previous_end:
            count += 1
            previous_end = end
    return count


def _support_metadata(group, model, spec, profile):
    finite = pd.Series(
        np.isfinite(pd.to_numeric(group[model], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(pd.to_numeric(group["y"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)),
        index=group.index,
    )
    support = group.loc[finite]
    metadata_group = group
    if model == "WeightedEnsemble" and not support.empty:
        metadata_group = group.loc[group["cutoff"].ge(support["cutoff"].min())]
    prefixes = []
    for cutoff, window in metadata_group.groupby("cutoff", sort=True):
        dates = set(pd.to_datetime(window.loc[finite.loc[window.index], "ds"]))
        prefix = 0
        for period in pd.date_range(cutoff, periods=spec.horizon + 1, freq=profile.freq)[1:]:
            if period not in dates:
                break
            prefix += 1
        prefixes.append(prefix)
    selected_horizon = min(prefixes) if prefixes else 0
    return finite, {
        "requested_horizon": spec.horizon,
        "selection_horizon": selected_horizon,
        "cv_windows": support["cutoff"].nunique(),
        "cv_disjoint_windows": disjoint_window_count(support, horizon=spec.horizon),
        "cv_step_size": spec.evaluation.step_size or int(group.groupby("cutoff").size().max()),
        "cv_horizon_matches_requested": selected_horizon == spec.horizon,
        "expected_observations": len(group),
    }


def frozen_comparison_result(result: ModelResult, history: pd.DataFrame, profile: DataProfile, spec: ForecastSpec) -> ModelResult:
    from nixtla_scaffold.models import _error_scale_map, _metrics_from_cv, model_columns, _empty_backtest_metrics
    cv = result.backtest_predictions
    if cv.empty:
        if result.model_policy_resolution.get("evaluation", {}).get("origins"):
            raise ValueError("Evaluation keys differ from the frozen origin plan; all planned targets are missing")
        return result
    keys = ["unique_id", "cutoff", "ds"]
    if cv[keys].isna().any().any() or cv.duplicated(keys).any():
        raise ValueError("Evaluation contains null or duplicate series/origin/target keys")
    evaluation = result.model_policy_resolution.get("evaluation", {})
    if "origins" in evaluation:
        expected = pd.MultiIndex.from_tuples([
            (origin["unique_id"], pd.Timestamp(origin["cutoff"]), pd.Timestamp(target))
            for origin in evaluation["origins"] for target in origin["targets"]
        ], names=keys)
        actual = pd.MultiIndex.from_frame(cv[keys])
        if len(expected.difference(actual)) or len(actual.difference(expected)):
            raise ValueError("Evaluation keys differ from the frozen origin plan; missing or extra targets cannot be rescored")
    scales = _error_scale_map(history, profile.season_length, cv)
    frames, receipts, warnings = [], [], []
    models = [name for name in model_columns(cv) if name not in {"cutoff", "y"}]
    for uid, group in cv.groupby("unique_id", sort=True):
        future = result.forecast.loc[result.forecast["unique_id"].astype(str).eq(str(uid))]
        metrics = _metrics_from_cv(group, scales_by_series=scales)
        eligible, missing = [], {}
        for model in models:
            finite, metadata = _support_metadata(group, model, spec, profile)
            missing[model] = int((~finite).sum())
            last_observed_date = pd.to_datetime(history.loc[history.unique_id.astype(str).eq(str(uid)), "ds"]).max()
            expected_future = pd.date_range(last_observed_date, periods=spec.horizon + 1, freq=profile.freq)[1:]
            future_complete = (
                model in future and len(future) == spec.horizon
                and pd.DatetimeIndex(pd.to_datetime(future["ds"])).sort_values().equals(expected_future)
                and np.isfinite(pd.to_numeric(future[model], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).all()
            )
            complete = bool(finite.all() and future_complete and model != "WeightedEnsemble")
            if complete:
                eligible.append(model)
            rows = metrics.loc[metrics["model"].eq(model)].copy()
            if rows.empty:
                continue
            for column, value in metadata.items():
                rows[column] = value
            rows["selection_eligible"] = complete
            rows["expected_observations"] = len(group)
            rows["comparison_support"] = (
                "component_full_schedule" if complete else
                "ensemble_later_advisory" if model == "WeightedEnsemble" else "incomplete_diagnostic"
            )
            frames.append(rows)
        missing = {model: count for model, count in missing.items() if count}
        receipts.append({
            "unique_id": str(uid), "models": eligible, "expected_rows": len(group),
            "paired_rows": len(group) if eligible else 0, "excluded_rows": 0 if eligible else len(group),
            "missing_rows_by_model": missing, "available_windows": group["cutoff"].nunique(),
            "paired_windows": group["cutoff"].nunique() if eligible else 0,
            "disjoint_windows": disjoint_window_count(group, horizon=spec.horizon) if eligible else 0,
            "planned_disjoint_windows": disjoint_window_count(group, horizon=spec.horizon),
            "excluded_models": [model for model in models if model not in eligible],
        })
        if missing:
            warnings.append(f"{uid}: incomplete candidates cannot shrink the frozen component cohort: {missing}. "
                            "Their partial scores are diagnostic only; ensemble warmup has a separate cohort.")
    return replace(
        result, backtest_metrics=pd.concat(frames, ignore_index=True) if frames else _empty_backtest_metrics(),
        warnings=(*result.warnings, *warnings),
        model_policy_resolution={**result.model_policy_resolution, "comparison": {
            "contract": "frozen_component_schedule_v1", "cohorts": receipts,
            "actual_vintage": "input snapshot; historical publication vintages are not certified",
            "scaled_error_reference": "observed training prefix through earliest planned cutoff; fixed across origins",
        }},
    )


def forecast_with_evaluation(
    history: pd.DataFrame, profile: DataProfile, spec: ForecastSpec, *,
    fit: Callable[[pd.DataFrame, DataProfile, ForecastSpec], ModelResult],
) -> ModelResult:
    """Refit each declared origin; inner engines never see an outer target row."""
    from nixtla_scaffold.custom_models import append_custom_model_result, _custom_policy_resolution
    from nixtla_scaffold.models import _metrics_from_cv, _error_scale_map, model_columns, _driver_model_cv_delta
    from nixtla_scaffold.drivers import DriverFeatureBundle
    from nixtla_scaffold.profile import profile_dataset, repair_time_index
    from nixtla_scaffold.exposure import scope_directory
    from nixtla_scaffold.candidates import merge_evidence, complete_evidence, contextualize_evidence
    from nixtla_scaffold.prediction_cache import PredictionCache, CacheError
    from nixtla_scaffold.anomalies import review_training

    resolved_spec = replace(spec, freq=profile.freq)
    origins = plan_origins(history, resolved_spec, profile.season_length)
    inner_policy = replace(spec.evaluation, strategy="none")
    custom_failed = False
    cache = None
    cache_reason = "disabled"
    anomaly_prefixes = []
    if spec.evaluation.cache_directory:
        if spec.custom_models or spec.challengers or spec.driver_snapshots or spec.regressors:
            cache_reason = "external_or_custom_inputs_require_fresh_admission_and_fit"
        else:
            cache = PredictionCache(spec.evaluation.cache_directory, spec.evaluation.cache_max_bytes)

    def fit_prefix(training, h, invocation_kind="future"):
        context = spec.context
        if context is not None and context.requested_horizon is not None:
            context = replace(context, requested_horizon=h)
        inner_spec = replace(spec, horizon=h, freq=profile.freq, context=context, evaluation=inner_policy,
                             weighted_ensemble=False, paired_validation=False, require_backtest=False,
                             custom_models=() if custom_failed else spec.custom_models)
        if spec.anomaly.mode == "adjust":
            # Inner native calibration would score cleaned holdouts. Calibrate only on raw outer errors.
            inner_spec = replace(inner_spec, levels=())
        inner_profile = profile_dataset(training, inner_spec)
        observed = training
        training, repair_warnings = repair_time_index(training, inner_profile, inner_spec)
        training, anomaly_review = review_training(training, inner_spec, inner_profile,
                                                  stage=invocation_kind, observed=observed)
        if invocation_kind != "future":
            anomaly_review["points"] = [row for row in anomaly_review["points"] if row["flagged"]]
        anomaly_prefixes.append(anomaly_review)
        inner_profile = profile_dataset(training, inner_spec)
        result = (cache.get_or_fit(cache.key(training, inner_profile, inner_spec, fit),
                                   lambda: fit(training, inner_profile, inner_spec)) if cache else
                  fit(training, inner_profile, inner_spec))
        result = replace(result, warnings=(*repair_warnings, *result.warnings))
        result = append_custom_model_result(result, history=training, profile=inner_profile, spec=inner_spec,
                                            invocation_kind=invocation_kind)
        evidence = complete_evidence(
            result.model_policy_resolution.get("candidate_evidence", {}), result.forecast, training, inner_spec,
            inner_profile, [*model_columns(result.forecast), *spec.model_allowlist], result.warnings,
        )
        diagnostics = {row["unique_id"]: row for row in inner_profile.configuration_diagnostics}
        for row in evidence.get("fits", []):
            row["stage"] = invocation_kind
            if invocation_kind != "future" and "native_fitted" in row:
                row["native_fitted"] = {**row["native_fitted"], "points": []}
            row["training_configuration"] = {
                key: value for key, value in diagnostics.get(row["unique_id"], {}).items()
                if key not in {"period_evidence", "recommendation"}
            }
        from nixtla_scaffold.configuration import candidate_admission
        from nixtla_scaffold.transformations import inverse_target_transform_frame
        admission = candidate_admission(
            inverse_target_transform_frame(observed, spec.transform.target, columns=["y"]),
            inverse_target_transform_frame(result.forecast, spec.transform.target,
                                           columns=model_columns(result.forecast)),
            inner_spec, freq=inner_profile.freq, season=inner_profile.season_length,
        )
        by_candidate = {(row["unique_id"], row["model"]): row for row in admission}
        for row in evidence.get("fits", []):
            row["configuration_admission"] = by_candidate.get((row["unique_id"], row["model"]), {})
        evidence = contextualize_evidence(evidence, {
                "target_transform": spec.transform.to_dict(),
                "training_window": spec.evaluation.training_window,
                "model_configuration": spec.model_configuration.to_dict(),
                "anomaly_policy": spec.anomaly.to_dict(),
                "calendar_policy": spec.calendar.to_dict(),
                "season_length_source": "explicit" if spec.season_length is not None else "calendar_prior",
                "point_semantics": "native_point" if spec.transform.target == "none" else
                "inverse_of_transformed_scale_point_not_bias_adjusted_raw_mean",
                "reporting_units": "normalized_units" if spec.transform.normalization_factor_col else "raw_units",
            })
        return replace(result, model_policy_resolution={**result.model_policy_resolution, "candidate_evidence": evidence})

    final = fit_prefix(training_slice(history, None, spec.evaluation.training_window), spec.horizon)
    final_features = _external_feature_recipes(final)
    if not final.custom_model_contracts.empty:
        custom_failed = bool(final.custom_model_contracts["status"].eq("failed").any())
    frames, receipts = [], []
    invocations = [final.custom_model_invocations] if not final.custom_model_invocations.empty else []
    candidate_evidence = [final.model_policy_resolution.get("candidate_evidence", {})]
    warnings = [message for message in final.warnings
                if not spec.weighted_ensemble or "weighted ensemble disabled" not in message]
    for h in sorted({len(origin.targets) for origin in origins if len(origin.targets) < spec.horizon}):
        warnings.append(f"Diagnostic CV horizon {h} is shorter than requested forecast horizon {spec.horizon}; later steps are directional only.")
    groups: dict[tuple[pd.Timestamp, int], list[Origin]] = {}
    for origin in origins:
        groups.setdefault((origin.cutoff, len(origin.targets)), []).append(origin)
    for (cutoff, h), cohort in sorted(groups.items()):
        ids = {origin.unique_id for origin in cohort}
        training = training_slice(history.loc[history["unique_id"].astype(str).isin(ids)], cutoff,
                                  spec.evaluation.training_window)
        grid = pd.DataFrame([{"unique_id": origin.unique_id, "ds": period, "cutoff": cutoff}
                             for origin in cohort for period in origin.targets])
        actuals = history[["unique_id", "ds", "y"]].copy()
        actuals["unique_id"] = actuals["unique_id"].astype(str)
        grid = grid.merge(actuals, on=["unique_id", "ds"], how="left", validate="one_to_one")
        try:
            fitted = fit_prefix(training, h, "backtest")
        except CacheError:
            raise
        except (RuntimeError, ValueError, ImportError) as exc:
            message = f"Evaluation origin {cutoff.isoformat()} failed: {type(exc).__name__}: {exc}"
            warnings.append(message)
            receipts.append({"cutoff": cutoff.isoformat(), "status": "failed", "reason": message,
                             "planned_rows": len(grid), "returned_rows": 0})
            frames.append(grid)
            continue
        predictions = fitted.forecast.copy()
        origin_features = _external_feature_recipes(fitted)
        for key in final_features.keys() | origin_features.keys():
            uid, model = key
            if final_features.get(key) == origin_features.get(key):
                continue
            if model not in predictions:
                continue
            predictions.loc[predictions["unique_id"].astype(str).eq(uid), model] = np.nan
            message = (
                f"{uid}/{model} at {cutoff.isoformat()}: external feature recipe differs from the deployed path; "
                "no substitute no-feature prediction is scored as driver evidence."
            )
            warnings.append(message)
            for row in fitted.model_policy_resolution["candidate_evidence"]["fits"]:
                if (row["unique_id"], row["model"]) == key:
                    row["status"] = "ineligible_feature_recipe"
                    row["diagnostics"] = [*row.get("diagnostics", []), message]
        candidate_evidence.append(fitted.model_policy_resolution.get("candidate_evidence", {}))
        predictions["unique_id"] = predictions["unique_id"].astype(str)
        keys = ["unique_id", "ds"]
        if predictions[keys].isna().any().any() or predictions.duplicated(keys).any():
            raise ValueError(f"Evaluation origin {cutoff}: predictions contain null or duplicate target keys")
        expected_keys = pd.MultiIndex.from_frame(grid[keys])
        returned_keys = pd.MultiIndex.from_frame(predictions[keys])
        if len(returned_keys.difference(expected_keys)):
            raise ValueError(f"Evaluation origin {cutoff}: predictions contain targets outside the frozen plan")
        missing_count = len(expected_keys.difference(returned_keys))
        if missing_count:
            warnings.append(f"Evaluation origin {cutoff}: {missing_count} planned prediction rows are missing; retained as unavailable evidence.")
        frames.append(grid.merge(predictions, on=["unique_id", "ds"], how="left", validate="one_to_one"))
        receipts.append({"cutoff": cutoff.isoformat(), "status": "incomplete" if missing_count else "completed",
                         "planned_rows": len(grid), "returned_rows": len(predictions),
                         "engine": fitted.engine, "warnings": list(fitted.warnings),
                         "withheld_by_publication_time": training.attrs.get("withheld_at_origin", 0)})
        if not fitted.custom_model_invocations.empty:
            calls = fitted.custom_model_invocations.copy()
            calls["invocation_id"] = cutoff.isoformat() + ":" + calls["invocation_id"].astype(str)
            invocations.append(calls)
            if calls["status"].eq("failed").any():
                custom_failed = True
        warnings.extend(f"{cutoff.isoformat()}: {message}" for message in fitted.warnings
                        if not spec.weighted_ensemble or "weighted ensemble disabled" not in message)
    cv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["unique_id", "ds", "cutoff", "y"])
    metrics = _metrics_from_cv(cv, scales_by_series=_error_scale_map(history, profile.season_length, cv))
    metadata = []
    for uid, group in cv.groupby("unique_id", sort=True):
        for model in metrics.loc[metrics.unique_id.eq(uid), "model"]:
            _, support = _support_metadata(group, model, spec, profile)
            metadata.append({"unique_id": uid, "model": model, **support})
    if metadata:
        metrics = metrics.merge(pd.DataFrame(metadata), on=["unique_id", "model"], how="left", validate="one_to_one")
    driver_scores = []
    for (uid, model), features in final_features.items():
        support = cv.loc[cv["unique_id"].astype(str).eq(uid)]
        if not features or model not in support or support.empty or not np.isfinite(support[[model, "y"]]).all().all():
            continue
        model_metrics = metrics.loc[metrics["unique_id"].astype(str).eq(uid) & metrics["model"].eq(model)]
        driver_scores.append(_driver_model_cv_delta(
            model_metrics, DriverFeatureBundle(feature_columns=features), cv_used_exogenous_features=True,
        ))
    if not origins:
        warnings.append("No eligible evaluation origins; future predictions are a directional baseline, not a validated winner.")
    invocation_frame = pd.concat(invocations, ignore_index=True) if invocations else final.custom_model_invocations
    contracts = final.custom_model_contracts.copy()
    if not invocation_frame.empty and not contracts.empty:
        for index, contract in contracts.iterrows():
            calls = invocation_frame.loc[invocation_frame["model"].eq(contract["model"])]
            contracts.at[index, "invocation_count"] = len(calls)
            if calls["status"].ne("succeeded").any() and contract["status"] == "succeeded":
                contracts.at[index, "status"] = "failed"
                contracts.at[index, "error"] = "; ".join(calls.loc[calls.status.eq("failed"), "error"].astype(str))
    if custom_failed and spec.custom_models:
        custom = spec.custom_models[0]
        final = replace(final, forecast=final.forecast.drop(columns=[custom.model_name], errors="ignore"),
                        engine=final.engine.removesuffix("+custom"),
                        model_policy_resolution=_custom_policy_resolution(
                            final.model_policy_resolution, custom=custom, ran=False,
                            reason_if_not_ran="Excluded after execution failure; remaining custom invocations stopped",
                            contributed_models=[],
                        ))
        warnings.append(f"{custom.model_name}: excluded after failure; remaining custom invocations stopped.")
    return replace(
        final, backtest_predictions=cv, backtest_metrics=metrics, warnings=tuple(warnings),
        driver_model_cv_delta=pd.concat(driver_scores, ignore_index=True) if driver_scores else final.driver_model_cv_delta,
        custom_model_invocations=invocation_frame, custom_model_contracts=contracts,
        model_policy_resolution={**final.model_policy_resolution,
                                 "anomalies": {"version": 1, "mode": spec.anomaly.mode,
                                               "actuals_changed": False, "prefixes": anomaly_prefixes},
                                 "prediction_cache": cache.receipt() if cache else {"status": cache_reason},
                                 "candidate_evidence": {**merge_evidence(*candidate_evidence),
                                                       "execution_context": candidate_evidence[0].get("execution_context", {})},
                                 "evaluation": {
            "version": 1, "policy": spec.evaluation.to_dict(), "requested_horizon": spec.horizon,
            "origins": [origin.to_dict() for origin in origins], "executions": receipts,
            "planned_rows": sum(len(origin.targets) for origin in origins),
            "exposure": "retrospective_or_tuning_not_untouched_confirmation",
            "exposure_scope": str(scope_directory(spec)),
            "exposure_scope_limitation": "Recorded runtime calls only; not a global data-access certificate.",
            "actual_vintage": "input_snapshot_historical_publication_availability_not_certified",
            "availability_rule": "available_at <= origin timestamp when supplied; otherwise historical availability unverified",
            "independence": "Overlapping origins are not independent confirmation.",
        }},
    )


def _external_feature_recipes(result: ModelResult) -> dict[tuple[str, str], tuple[str, ...]]:
    evidence = result.model_policy_resolution.get("candidate_evidence", {})
    recipes = evidence.get("recipes", {})
    return {
        (row["unique_id"], row["model"]): tuple(sorted(recipes[row["recipe_id"]]["execution"]["external_feature_columns"]))
        for row in evidence.get("fits", [])
        if row["status"] == "succeeded"
        and "external_feature_columns" in recipes[row["recipe_id"]]["execution"]
    }
