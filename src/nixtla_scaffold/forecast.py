from __future__ import annotations

import pandas as pd

from nixtla_scaffold.custom_models import append_custom_model_result
from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.drivers import audit_known_future_regressors
from nixtla_scaffold.hierarchy import hierarchy_metadata, reconcile_hierarchy_forecast
from nixtla_scaffold.models import (
    ModelResult,
    build_selected_forecast,
    forecast_with_policy,
    rebuild_result_metrics_on_output_scale,
    select_champions,
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
    dataset = load_forecast_dataset(data, sheet=sheet, spec=spec)
    modeling_dataset, transformation_audit, transform_warnings = prepare_modeling_target(dataset, spec.transform)
    profile = profile_dataset(modeling_dataset, spec)

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
    repaired_profile = profile_dataset(reporting_history, spec)

    model_result = forecast_with_policy(history, repaired_profile, spec)
    model_result = append_custom_model_result(model_result, history=history, profile=repaired_profile, spec=spec)
    if spec.transform.target != "none":
        model_result = _inverse_model_result_target_transform(model_result, spec.transform.target)
        model_result = rebuild_result_metrics_on_output_scale(model_result, reporting_history, repaired_profile, spec)
    warnings.extend(model_result.warnings)
    if spec.require_backtest:
        backtested_ids = set(model_result.backtest_metrics["unique_id"]) if not model_result.backtest_metrics.empty else set()
        missing_backtests = sorted(set(reporting_history["unique_id"]) - backtested_ids)
        if missing_backtests:
            raise ValueError(
                "require_backtest=True but backtest metrics could not be produced for "
                f"{missing_backtests}; add history, reduce horizon, or disable strict backtest mode"
            )

    selection = select_champions(model_result.forecast, model_result.backtest_metrics)
    forecast = build_selected_forecast(model_result.forecast, selection, spec.levels)
    forecast, shrinkage_warnings = _apply_shrinkage_toward_last_actual(forecast, reporting_history, model_result.backtest_predictions)
    warnings.extend(shrinkage_warnings)
    metadata = hierarchy_metadata(dataset)
    if not metadata.empty:
        forecast = forecast.merge(metadata, on="unique_id", how="left")
    if spec.events:
        forecast = apply_event_adjustments(forecast, spec.events)
        warnings.extend(forecast.attrs.get("event_warnings", []))
        warnings.append(f"{len(spec.events)} driver/event scenario adjustment(s) applied to forecast")
    unreconciled_forecast = pd.DataFrame()
    hierarchy_reconciliation = pd.DataFrame()
    if spec.hierarchy_reconciliation != "none":
        before_reconciliation = forecast.copy()
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

    return ForecastRun(
        history=reporting_history,
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
        custom_model_contracts=model_result.custom_model_contracts,
        custom_model_invocations=model_result.custom_model_invocations,
        unreconciled_forecast=unreconciled_forecast,
        hierarchy_reconciliation=hierarchy_reconciliation,
        warnings=list(dict.fromkeys(warnings)),
        engine=model_result.engine,
        model_policy_resolution=model_result.model_policy_resolution,
    )


def _inverse_history_target_transform(history: pd.DataFrame, transform: TargetTransform) -> pd.DataFrame:
    return inverse_target_transform_frame(history, transform, columns=["y"])


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
        custom_model_contracts=result.custom_model_contracts,
        custom_model_invocations=result.custom_model_invocations,
        warnings=result.warnings,
        model_policy_resolution=result.model_policy_resolution,
    )


def _forecast_value_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"unique_id", "ds", "cutoff", "model", "event_names"}
    return [column for column in frame.columns if column not in excluded]


def _apply_shrinkage_toward_last_actual(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """James-Stein-style shrinkage: blend forecast toward last actual value.

    alpha = model_var / (model_var + residual_var) where residual_var is
    estimated from backtest prediction errors. For series with high residual
    variance (e.g. distribution shift), alpha → 0 (trust last actual).
    For smooth series, alpha → 1 (trust model).
    """
    import numpy as np

    out = forecast.copy()
    warnings_out: list[str] = []
    if backtest_predictions.empty or "y" not in backtest_predictions.columns:
        return out, warnings_out

    last_actual: dict[str, float] = {}
    for uid, grp in history.groupby("unique_id", sort=True):
        last_actual[str(uid)] = float(grp.sort_values("ds")["y"].iloc[-1])

    model_cols_in_bt = [c for c in backtest_predictions.columns if c not in {"unique_id", "ds", "cutoff", "y"} and "-lo-" not in c and "-hi-" not in c]
    if not model_cols_in_bt:
        return out, warnings_out

    alpha_by_series: dict[str, float] = {}
    for uid, grp in backtest_predictions.groupby("unique_id", sort=True):
        actual = grp["y"].to_numpy(dtype="float64")
        if len(actual) < 2:
            alpha_by_series[str(uid)] = 1.0
            continue
        # Use the selected model's predictions if available, otherwise average
        errors = []
        for mc in model_cols_in_bt:
            if mc in grp.columns:
                pred = pd.to_numeric(grp[mc], errors="coerce").to_numpy(dtype="float64")
                valid = np.isfinite(pred) & np.isfinite(actual)
                if valid.any():
                    errors.extend((pred[valid] - actual[valid]).tolist())
        if not errors:
            alpha_by_series[str(uid)] = 1.0
            continue
        residual_var = float(np.var(errors))
        model_var = float(np.var(actual))
        if residual_var + model_var <= 0:
            alpha_by_series[str(uid)] = 1.0
        else:
            alpha_by_series[str(uid)] = float(np.clip(model_var / (model_var + residual_var), 0.3, 1.0))

    shrunk_count = 0
    for uid in out["unique_id"].unique():
        uid_str = str(uid)
        alpha = alpha_by_series.get(uid_str, 1.0)
        if alpha >= 0.99 or uid_str not in last_actual:
            continue
        mask = out["unique_id"] == uid
        yhat = pd.to_numeric(out.loc[mask, "yhat"], errors="coerce")
        last_val = last_actual[uid_str]
        shrunk_yhat = alpha * yhat + (1.0 - alpha) * last_val
        delta = shrunk_yhat - yhat
        out.loc[mask, "yhat"] = shrunk_yhat.to_numpy(dtype="float64")
        for level in _forecast_interval_levels(out):
            lo_col = f"yhat_lo_{level}"
            hi_col = f"yhat_hi_{level}"
            out.loc[mask, lo_col] = pd.to_numeric(out.loc[mask, lo_col], errors="coerce") + delta
            out.loc[mask, hi_col] = pd.to_numeric(out.loc[mask, hi_col], errors="coerce") + delta
        shrunk_count += 1

    if shrunk_count:
        _ensure_selected_intervals_contain_point(out)
        warnings_out.append(f"James-Stein shrinkage applied to {shrunk_count} series to reduce distribution shift risk")
    return out, warnings_out


def _forecast_interval_levels(frame: pd.DataFrame) -> list[str]:
    levels: list[str] = []
    for column in frame.columns:
        if column.startswith("yhat_lo_"):
            level = column.rsplit("_", 1)[-1]
            if f"yhat_hi_{level}" in frame.columns:
                levels.append(level)
    return sorted(set(levels), key=lambda item: int(item) if item.isdigit() else item)


def _ensure_selected_intervals_contain_point(frame: pd.DataFrame) -> None:
    yhat = pd.to_numeric(frame["yhat"], errors="coerce")
    for level in _forecast_interval_levels(frame):
        lo_col = f"yhat_lo_{level}"
        hi_col = f"yhat_hi_{level}"
        lo = pd.to_numeric(frame[lo_col], errors="coerce")
        hi = pd.to_numeric(frame[hi_col], errors="coerce")
        lower = pd.concat([lo, hi, yhat], axis=1).min(axis=1, skipna=True)
        upper = pd.concat([lo, hi, yhat], axis=1).max(axis=1, skipna=True)
        valid = lo.notna() & hi.notna() & yhat.notna()
        frame.loc[valid, lo_col] = lower.loc[valid]
        frame.loc[valid, hi_col] = upper.loc[valid]

