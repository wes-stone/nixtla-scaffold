from __future__ import annotations

from dataclasses import replace
import pandas as pd

from nixtla_scaffold.custom_models import append_custom_model_result
from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.drivers import audit_known_future_regressors
from nixtla_scaffold.hierarchy import hierarchy_metadata, reconcile_hierarchy_forecast, reconcile_hierarchy_forecast_both
from nixtla_scaffold.models import (
    ModelResult,
    build_selected_forecast,
    forecast_with_policy,
    rebuild_result_metrics_on_output_scale,
    select_champions,
    paired_comparison_result,
)
from nixtla_scaffold.profile import profile_dataset, repair_time_index
from nixtla_scaffold.schema import ForecastRun, ForecastSpec, TargetTransform
from nixtla_scaffold.transformations import apply_event_adjustments, inverse_target_transform_frame, prepare_modeling_target


def run_forecast(
    data: str | pd.DataFrame,
    spec: ForecastSpec | None = None,
    *,
    sheet: str | int | None = None,
) -> ForecastRun:
    """Run the simple load -> profile -> repair -> forecast -> select flow."""

    spec = spec or ForecastSpec()
    if spec.production is not None:
        from nixtla_scaffold.production import prepare_production_spec
        spec = prepare_production_spec(spec)
    if spec.anomaly.mode == "adjust" and spec.evaluation.strategy != "rolling-origin":
        raise ValueError("Anomaly adjustment requires rolling-origin evaluation; legacy/full-frame cleaning is not allowed")
    dataset = load_forecast_dataset(data, sheet=sheet, spec=spec)
    from nixtla_scaffold.exposure import record_exposure
    record_exposure(dataset, spec)
    modeling_dataset, transformation_audit, transform_warnings = prepare_modeling_target(dataset, spec.transform)
    profile = profile_dataset(modeling_dataset, spec)
    if spec.calendar.enabled and profile.freq not in {"D", "B"}:
        raise ValueError("Enhanced calendar features currently require daily or business-daily observations")
    if spec.selection_policy.objective == "monthly_landing" and profile.freq != "D":
        raise ValueError("Monthly landing selection requires daily flow observations")
    if spec.interval_calibration.aggregate_totals and (
        profile.freq != "D" or spec.events or spec.hierarchy_reconciliation != "none"
    ):
        raise ValueError("Aggregate intervals require daily flows and the unchanged, unreconciled statistical point path")

    warnings = list(transform_warnings) + list(profile.warnings)
    if "hierarchy_level" in dataset.columns:
        levels = sorted(str(level) for level in dataset["hierarchy_level"].dropna().unique())
        if spec.hierarchy_reconciliation == "none":
            warnings.append(
                f"hierarchy detected ({len(levels)} levels: {levels}); forecasts are independent per node "
                "and are not reconciled, so parent and child sums may differ"
            )
        else:
            warnings.append(
                f"hierarchy detected ({len(levels)} levels: {levels}); "
                f"reconciliation method '{spec.hierarchy_reconciliation}' requested"
            )
    history, repair_warnings = repair_time_index(modeling_dataset, profile, spec)
    warnings.extend(repair_warnings)
    if history.empty:
        raise ValueError("no usable rows remain after repair; adjust fill_method or input data")
    reporting_history = _inverse_history_target_transform(history, spec.transform.target)
    reporting_history = _restore_audited_reporting_target(
        reporting_history,
        transformation_audit,
    )
    repaired_profile = profile_dataset(reporting_history, spec)

    observed_history = pd.DataFrame()
    if spec.evaluation.strategy == "rolling-origin":
        from nixtla_scaffold.evaluation import forecast_with_evaluation
        # Calendar completion is structural; actual targets and repairs remain origin-local.
        evaluation_history = history[["unique_id", "ds"]].merge(
            modeling_dataset, on=["unique_id", "ds"], how="left", validate="one_to_one",
        )
        observed_history = _restore_audited_reporting_target(
            _inverse_history_target_transform(evaluation_history, spec.transform.target), transformation_audit,
        )
        model_result = forecast_with_evaluation(evaluation_history, repaired_profile, spec, fit=forecast_with_policy)
    else:
        model_result = forecast_with_policy(history, repaired_profile, spec)
        model_result = append_custom_model_result(model_result, history=history, profile=repaired_profile, spec=spec)
        from nixtla_scaffold.anomalies import review_training
        _, anomaly_review = review_training(history, spec, repaired_profile, stage="future", observed=modeling_dataset)
        model_result = replace(model_result, model_policy_resolution={
            **model_result.model_policy_resolution,
            "anomalies": {"version": 1, "mode": spec.anomaly.mode, "actuals_changed": False, "prefixes": [anomaly_review]},
        })
    if spec.anomaly.mode == "adjust":
        warnings.append("Anomaly adjustments affect training copies only; actuals and outer evaluation targets are unchanged. Native inner intervals are disabled; outer residual calibration needs sufficient untouched scoring targets.")
    if spec.cleaning.clean_outliers and spec.anomaly.mode == "off":
        warnings.append("Saved clean_outliers intent is not executed while anomaly.mode is off; use an explicit versioned anomaly treatment.")
    if spec.transform.target != "none":
        model_result = _inverse_model_result_target_transform(model_result, spec.transform.target)
    scoring_history = observed_history if not observed_history.empty else reporting_history
    if spec.selection_policy.clip_negative:
        from nixtla_scaffold.finance import clip_model_result
        model_result = clip_model_result(model_result)
    model_result = rebuild_result_metrics_on_output_scale(model_result, scoring_history, repaired_profile, spec)
    if spec.paired_validation:
        model_result = paired_comparison_result(model_result, scoring_history, repaired_profile, spec)
    if spec.interval_calibration.method != "native":
        from nixtla_scaffold.conformal import apply_experimental_intervals
        model_result = apply_experimental_intervals(model_result, scoring_history, spec, freq=repaired_profile.freq)
    from nixtla_scaffold.finance import apply_finance_objective
    model_result = apply_finance_objective(model_result, scoring_history, spec, freq=repaired_profile.freq)
    if spec.interval_calibration.aggregate_totals:
        if repaired_profile.freq != "D" or spec.events or spec.hierarchy_reconciliation != "none":
            raise ValueError("Aggregate intervals require daily flows and the unchanged, unreconciled statistical point path")
        from nixtla_scaffold.finance import aggregate_intervals
        from nixtla_scaffold.models import model_columns
        records = aggregate_intervals(
            scoring_history, model_result.backtest_predictions, model_result.forecast,
            [name for name in model_columns(model_result.forecast) if name != "WeightedEnsemble"],
            horizon=spec.horizon, levels=spec.levels, clip_negative=spec.selection_policy.clip_negative,
        )
        model_result = replace(model_result, model_policy_resolution={
            **model_result.model_policy_resolution, "aggregate_intervals": records,
        })
    warnings.extend(model_result.warnings)
    from nixtla_scaffold.configuration import screen_candidates
    selection_paths, selection_metrics, admission = screen_candidates(
        scoring_history, model_result.forecast, model_result.backtest_metrics, spec,
        freq=repaired_profile.freq, season=repaired_profile.season_length,
    )
    model_result = replace(model_result, backtest_metrics=selection_metrics)
    warnings.extend(
        f"{row['unique_id']}/{row['model']}: excluded from selection by the declared configuration guard; "
        "changing seasonal shape and an unusual next-cycle change. Original predictions and errors are retained."
        for row in admission if not row["eligible"]
    )
    if spec.require_backtest:
        eligible_metrics = model_result.backtest_metrics
        if "selection_eligible" in eligible_metrics:
            eligible_metrics = eligible_metrics.loc[eligible_metrics["selection_eligible"].fillna(False)]
        backtested_ids = set(eligible_metrics["unique_id"]) if not eligible_metrics.empty else set()
        missing_backtests = sorted(set(reporting_history["unique_id"]) - backtested_ids)
        if missing_backtests:
            raise ValueError(
                "require_backtest=True but backtest metrics could not be produced for "
                f"{missing_backtests}; add history, reduce horizon, or disable strict backtest mode"
            )

    selection = select_champions(selection_paths, selection_metrics, policy=spec.selection_policy)
    if any(not row["eligible"] for row in admission):
        excluded_ids = {row["unique_id"] for row in admission if not row["eligible"]}
        selection.loc[selection.unique_id.astype(str).isin(excluded_ids), "selection_reason"] += (
            "; configuration-review candidates excluded under declared policy"
        )
    from nixtla_scaffold.ensemble import selected_ensemble_comparisons
    ensemble_comparisons = selected_ensemble_comparisons(model_result.backtest_predictions, selection)
    forecast = build_selected_forecast(model_result.forecast, selection, spec.levels)
    metadata = hierarchy_metadata(dataset)
    if not metadata.empty:
        forecast = forecast.merge(metadata, on="unique_id", how="left")
    if spec.events:
        forecast = apply_event_adjustments(forecast, spec.events)
        warnings.extend(forecast.attrs.get("event_warnings", []))
        warnings.append(f"{len(spec.events)} driver/event scenario adjustment(s) applied to forecast")
    unreconciled_forecast = pd.DataFrame()
    hierarchy_reconciliation = pd.DataFrame()
    hierarchy_reconciliation_comparison = pd.DataFrame()
    if spec.hierarchy_reconciliation != "none":
        before_reconciliation = forecast.copy()
        if spec.hierarchy_reconciliation == "both":
            (
                forecast,
                hierarchy_reconciliation,
                hierarchy_reconciliation_comparison,
                reconciliation_warnings,
            ) = reconcile_hierarchy_forecast_both(forecast)
        else:
            forecast, hierarchy_reconciliation, reconciliation_warnings = reconcile_hierarchy_forecast(
                forecast,
                method=spec.hierarchy_reconciliation,
            )
        warnings.extend(reconciliation_warnings)
        if not hierarchy_reconciliation.empty:
            unreconciled_forecast = before_reconciliation
    driver_availability_audit, driver_warnings = audit_known_future_regressors(
        dataset,
        forecast,
        spec,
        forecast_origin=repaired_profile.end,
    )
    warnings.extend(driver_warnings)

    run = ForecastRun(
        history=reporting_history,
        observed_history=observed_history,
        forecast=forecast,
        all_models=model_result.forecast,
        model_selection=selection,
        backtest_metrics=model_result.backtest_metrics,
        backtest_predictions=model_result.backtest_predictions,
        profile=repaired_profile,
        spec=spec,
        model_weights=model_result.model_weights,
        model_explainability=model_result.model_explainability,
        transformation_audit=transformation_audit,
        driver_availability_audit=driver_availability_audit,
        driver_model_features=model_result.driver_model_features,
        driver_model_cv_delta=model_result.driver_model_cv_delta,
        custom_model_contracts=model_result.custom_model_contracts,
        custom_model_invocations=model_result.custom_model_invocations,
        unreconciled_forecast=unreconciled_forecast,
        hierarchy_reconciliation=hierarchy_reconciliation,
        hierarchy_reconciliation_comparison=hierarchy_reconciliation_comparison,
        warnings=list(dict.fromkeys(warnings)),
        engine=model_result.engine,
        model_policy_resolution={
            **model_result.model_policy_resolution,
            "ensemble_comparisons": ensemble_comparisons,
            "configuration_admission": admission,
            "deployed_path": {
                "point_status": "statistical_point_backtested" if spec.hierarchy_reconciliation == "none"
                                else "reconciled_point_requires_joint_policy_backtest",
                "scenario_status": "conditional_assumption_not_accuracy_evidence" if spec.events else "not_requested",
                "output_scale": "normalized_units" if spec.transform.normalization_factor_col else "raw_units",
                "postprocessing": spec.hierarchy_reconciliation,
            },
        },
    )
    from nixtla_scaffold.configuration import build_configuration_report
    configuration = build_configuration_report(run)
    run = replace(run, model_policy_resolution={**run.model_policy_resolution, "configuration": configuration},
                   warnings=[*run.warnings, *[
                       f"{uid}: selected forecast requires configuration review: changing seasonal shape and an unusual next-cycle change."
                       for uid in configuration["review_series"]
                   ]])
    if spec.production is not None:
        from nixtla_scaffold.production import qualify_production_run
        run = qualify_production_run(run)
    return run


def _inverse_history_target_transform(history: pd.DataFrame, transform: TargetTransform) -> pd.DataFrame:
    return inverse_target_transform_frame(history, transform, columns=["y"])


def _restore_audited_reporting_target(
    history: pd.DataFrame,
    transformation_audit: pd.DataFrame,
) -> pd.DataFrame:
    required = {"unique_id", "ds", "y_adjusted"}
    if history.empty or transformation_audit.empty or not required.issubset(transformation_audit.columns):
        return history
    audited = transformation_audit[["unique_id", "ds", "y_adjusted"]].copy()
    audited["unique_id"] = audited["unique_id"].astype(str)
    audited["ds"] = pd.to_datetime(audited["ds"])
    audited = audited.drop_duplicates(["unique_id", "ds"], keep="last")
    out = history.copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.merge(audited, on=["unique_id", "ds"], how="left", validate="many_to_one")
    out["y"] = pd.to_numeric(out["y_adjusted"], errors="coerce").fillna(out["y"])
    return out.drop(columns=["y_adjusted"])


def _inverse_model_result_target_transform(result: ModelResult, transform: TargetTransform) -> ModelResult:
    forecast_columns = _forecast_value_columns(result.forecast)
    backtest_columns = _forecast_value_columns(result.backtest_predictions)
    return ModelResult(
        forecast=inverse_target_transform_frame(result.forecast, transform, columns=forecast_columns),
        backtest_metrics=result.backtest_metrics,
        backtest_predictions=inverse_target_transform_frame(result.backtest_predictions, transform, columns=backtest_columns),
        engine=result.engine,
        model_weights=result.model_weights,
        model_explainability=result.model_explainability,
        driver_model_features=result.driver_model_features,
        driver_model_cv_delta=result.driver_model_cv_delta,
        custom_model_contracts=result.custom_model_contracts,
        custom_model_invocations=result.custom_model_invocations,
        warnings=result.warnings,
        model_policy_resolution=result.model_policy_resolution,
    )


def _forecast_value_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"unique_id", "ds", "cutoff", "model", "event_names"}
    return [column for column in frame.columns if column not in excluded]
