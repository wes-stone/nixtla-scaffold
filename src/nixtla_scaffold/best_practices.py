from __future__ import annotations

from typing import Any

import pandas as pd

from nixtla_scaffold.citations import fppy_source
from nixtla_scaffold.hierarchy import hierarchy_coherence


BEST_PRACTICE_DOCTRINE = [
    {
        "id": "tidy_data_checked",
        "principle": "Prepare, validate, and inspect time-series data before modeling.",
        "source": fppy_source("Chapter 5 tidy forecasting workflow", "https://otexts.com/fpppy/05-toolbox.html"),
        "modeling_hook": "profile_dataset + repair_time_index",
    },
    {
        "id": "benchmarks_included",
        "principle": "Use simple benchmark methods as a reference point for every forecast.",
        "source": fppy_source("Chapter 5 benchmark methods", "https://otexts.com/fpppy/05-toolbox.html"),
        "modeling_hook": "Naive, HistoricAverage, RandomWalkWithDrift, WindowAverage, SeasonalNaive",
    },
    {
        "id": "rolling_origin_evaluation",
        "principle": "Evaluate forecasts with time-ordered validation, not random splits.",
        "source": fppy_source("Chapter 5 forecast accuracy", "https://otexts.com/fpppy/05-toolbox.html"),
        "modeling_hook": "StatsForecast.cross_validation or baseline rolling-origin backtests",
    },
    {
        "id": "intervals_supported_by_history",
        "principle": "Prediction intervals should only be presented when the model and history support them.",
        "source": fppy_source("Chapter 13 practical forecasting", "https://otexts.com/fpppy/13-practical.html"),
        "modeling_hook": "conformal interval windows + effective_levels",
    },
    {
        "id": "horizon_claim_audited",
        "principle": "The evaluation horizon should match the business decision horizon, or the mismatch should be disclosed.",
        "source": fppy_source("Chapter 5 forecast accuracy", "https://otexts.com/fpppy/05-toolbox.html"),
        "modeling_hook": "requested_horizon/selection_horizon + horizon trust gates",
    },
    {
        "id": "hierarchy_coherence_visible",
        "principle": "Parent and child forecasts need coherence checks before being used together.",
        "source": fppy_source(
            "Chapter 11 hierarchical forecasting",
            "https://otexts.com/fpppy/11-hierarchical-forecasting.html",
        ),
        "modeling_hook": "hierarchy_coherence.csv diagnostics",
    },
    {
        "id": "drivers_are_audited_overlays",
        "principle": "Known future events should be explicit assumptions rather than hidden in the statistical model.",
        "source": fppy_source("Chapter 13 practical forecasting/event effects", "https://otexts.com/fpppy/13-practical.html"),
        "modeling_hook": "DriverEvent + yhat_scenario/event_adjustment columns",
    },
    {
        "id": "known_future_regressors_audited",
        "principle": "Regression-style forecast drivers require known future values or explicit scenarios and must not leak future target information.",
        "source": fppy_source("Chapter 10 dynamic regression models", "https://otexts.com/fpppy/10-dynamic-regression.html"),
        "modeling_hook": "KnownFutureRegressor + driver_availability_audit.csv",
    },
    {
        "id": "target_transform_audited",
        "principle": "Target transformations and finance normalization should be deliberate, reversible where possible, and auditable.",
        "source": fppy_source("Chapter 3 transformations and adjustments", "https://otexts.com/fpppy/03-decomposition.html"),
        "modeling_hook": "TransformSpec + audit/target_transform_audit.csv",
    },
    {
        "id": "weighted_ensemble_audited",
        "principle": "Weighted forecasts should be traceable to validation evidence, not hidden averaging.",
        "source": fppy_source(
            "Chapter 5 forecast accuracy/model combinations",
            "https://otexts.com/fpppy/05-toolbox.html",
        ),
        "modeling_hook": "WeightedEnsemble + audit/model_weights.csv inverse-error weights",
    },
]


def best_practice_receipts(run: Any) -> list[dict[str, str]]:
    """Return runtime evidence that FPPy/Nixtla research principles reached the model."""

    model_cols = _model_columns(run.all_models)
    receipts = [
        _receipt(
            "tidy_data_checked",
            "passed" if run.profile.duplicate_rows == 0 else "failed",
            f"{run.profile.rows} rows, {run.profile.series_count} series, freq={run.profile.freq}, duplicate_rows={run.profile.duplicate_rows}",
        ),
        _receipt(
            "benchmarks_included",
            "passed" if {"Naive", "HistoricAverage", "RandomWalkWithDrift", "WindowAverage"}.issubset(model_cols) else "warning",
            f"candidate model columns: {sorted(model_cols)}",
        ),
        _receipt(
            "rolling_origin_evaluation",
            "passed" if not run.backtest_metrics.empty else "warning",
            f"{len(run.backtest_metrics)} metric rows; selected models={len(run.model_selection)}",
        ),
        _receipt(
            "intervals_supported_by_history",
            _interval_receipt_status(run),
            _interval_receipt_evidence(run),
        ),
        _horizon_receipt(run),
        _hierarchy_receipt(run),
        _receipt(
            "drivers_are_audited_overlays",
            "passed" if run.spec.events else "not_applicable",
            f"{len(run.spec.events)} event(s); scenario column present={'yhat_scenario' in run.forecast.columns}",
        ),
        _known_future_regressor_receipt(run),
        _target_transform_receipt(run),
        _weighted_ensemble_receipt(run),
    ]
    doctrine = {item["id"]: item for item in BEST_PRACTICE_DOCTRINE}
    for receipt in receipts:
        receipt.update(
            {
                "principle": doctrine[receipt["id"]]["principle"],
                "source": doctrine[receipt["id"]]["source"],
                "modeling_hook": doctrine[receipt["id"]]["modeling_hook"],
            }
        )
    return receipts


def best_practice_receipts_frame(run: Any) -> pd.DataFrame:
    return pd.DataFrame(best_practice_receipts(run))


def _interval_receipt_status(run: Any) -> str:
    if not run.spec.levels:
        return "not_applicable"
    if not run.effective_levels():
        return "warning"
    selected = _selected_models(run)
    if not selected:
        return "warning"
    for uid, model in selected.items():
        if model == "WeightedEnsemble":
            return "warning"
        if not _has_selected_cv_interval_evidence(run, uid, model):
            return "warning"
    return "passed"


def _interval_receipt_evidence(run: Any) -> str:
    selected = _selected_models(run)
    selected_text = ", ".join(f"{uid}:{model}" for uid, model in selected.items()) or "none"
    return (
        f"requested={list(run.spec.levels)}, effective={run.effective_levels()}, "
        f"selected={selected_text}, horizon_matched_cv_interval_evidence={_selected_cv_interval_column_count(run)}"
    )


def _horizon_receipt(run: Any) -> dict[str, str]:
    selection = getattr(run, "model_selection", pd.DataFrame())
    if selection.empty:
        return _receipt("horizon_claim_audited", "warning", "no model_selection rows; no champion horizon evidence")
    required = {"unique_id", "selected_model", "selection_horizon", "requested_horizon", "cv_windows", "cv_horizon_matches_requested"}
    if not required.issubset(selection.columns):
        return _receipt("horizon_claim_audited", "warning", "model_selection missing CV horizon contract columns")
    rows = selection[list(required)].copy()
    selected = pd.to_numeric(rows["selection_horizon"], errors="coerce")
    requested = pd.to_numeric(rows["requested_horizon"], errors="coerce")
    windows = pd.to_numeric(rows["cv_windows"], errors="coerce")
    full_horizon = selected.notna() & requested.notna() & (selected >= requested) & windows.notna() & (windows >= 2)
    status = "passed" if bool(full_horizon.all()) else "warning"
    evidence = "; ".join(
        f"{row['unique_id']}:{row['selected_model']} h={row['selection_horizon']}/{row['requested_horizon']} windows={row['cv_windows']}"
        for row in rows.to_dict("records")
    )
    return _receipt("horizon_claim_audited", status, evidence)


def _selected_models(run: Any) -> dict[str, str]:
    if run.model_selection.empty or not {"unique_id", "selected_model"}.issubset(run.model_selection.columns):
        return {}
    return {
        str(row["unique_id"]): str(row["selected_model"])
        for row in run.model_selection[["unique_id", "selected_model"]].to_dict("records")
    }


def _has_selected_cv_interval_columns(run: Any, uid: str, model: str) -> bool:
    if run.backtest_predictions.empty:
        return False
    frame = run.backtest_predictions[run.backtest_predictions["unique_id"].astype(str) == uid]
    if frame.empty:
        return False
    for level in run.effective_levels():
        lo_col = f"{model}-lo-{level}"
        hi_col = f"{model}-hi-{level}"
        if lo_col in frame.columns and hi_col in frame.columns and frame[lo_col].notna().any() and frame[hi_col].notna().any():
            return True
    return False


def _has_selected_cv_interval_evidence(run: Any, uid: str, model: str) -> bool:
    if not _has_selected_cv_interval_columns(run, uid, model):
        return False
    if run.model_selection.empty or not {"unique_id", "selected_model"}.issubset(run.model_selection.columns):
        return False
    rows = run.model_selection[
        (run.model_selection["unique_id"].astype(str) == uid)
        & (run.model_selection["selected_model"].astype(str) == model)
    ]
    if rows.empty:
        return False
    row = rows.iloc[0]
    if {"selection_horizon", "requested_horizon"}.issubset(run.model_selection.columns):
        selection_horizon = pd.to_numeric(pd.Series([row.get("selection_horizon")]), errors="coerce").iloc[0]
        requested_horizon = pd.to_numeric(pd.Series([row.get("requested_horizon")]), errors="coerce").iloc[0]
        if pd.notna(selection_horizon) and pd.notna(requested_horizon) and selection_horizon < requested_horizon:
            return False
    return True


def _selected_cv_interval_column_count(run: Any) -> int:
    selected = _selected_models(run)
    return sum(1 for uid, model in selected.items() if _has_selected_cv_interval_evidence(run, uid, model))


def _hierarchy_receipt(run: Any) -> dict[str, str]:
    if "hierarchy_depth" not in run.forecast.columns:
        return _receipt("hierarchy_coherence_visible", "not_applicable", "no hierarchy metadata in forecast")
    coherence = hierarchy_coherence(run.forecast)
    if coherence.empty:
        return _receipt("hierarchy_coherence_visible", "warning", "hierarchy metadata exists, but no coherence rows were produced")
    max_gap = float(coherence["gap_pct"].abs().max(skipna=True))
    if getattr(run.spec, "hierarchy_reconciliation", "none") != "none" and max_gap <= 0.01:
        return _receipt(
            "hierarchy_coherence_visible",
            "passed",
            f"{len(coherence)} parent/child checks; max absolute gap_pct={max_gap:.2%}; reconciliation={run.spec.hierarchy_reconciliation}",
        )
    return _receipt(
        "hierarchy_coherence_visible",
        "warning",
        f"{len(coherence)} parent/child checks; max absolute gap_pct={max_gap:.2%}; forecasts are diagnostic-only until reconciled",
    )


def _weighted_ensemble_receipt(run: Any) -> dict[str, str]:
    if not run.spec.weighted_ensemble:
        return _receipt("weighted_ensemble_audited", "not_applicable", "weighted ensemble disabled")
    if run.model_weights.empty:
        return _receipt("weighted_ensemble_audited", "warning", "weighted ensemble requested, but no model weights were produced")
    return _receipt(
        "weighted_ensemble_audited",
        "passed",
        f"{len(run.model_weights)} model weight rows across {run.model_weights['unique_id'].nunique()} series",
    )


def _known_future_regressor_receipt(run: Any) -> dict[str, str]:
    if not getattr(run.spec, "regressors", ()):
        return _receipt("known_future_regressors_audited", "not_applicable", "no known-future regressors declared")
    audit = getattr(run, "driver_availability_audit", pd.DataFrame())
    if audit.empty or "audit_status" not in audit.columns:
        return _receipt("known_future_regressors_audited", "warning", "regressors declared, but driver_availability_audit.csv is empty")
    statuses = audit["audit_status"].astype(str).value_counts().sort_index().to_dict()
    if "failed" in statuses:
        status = "failed"
    elif "warning" in statuses:
        status = "warning"
    else:
        status = "passed"
    model_candidates = int((audit.get("mode", pd.Series(dtype=str)).astype(str) == "model_candidate").sum())
    evidence = (
        f"{len(run.spec.regressors)} regressor(s); audit_status={statuses}; "
        f"model_candidate_rows={model_candidates}; external_regressor_training_enabled=False"
    )
    return _receipt("known_future_regressors_audited", status, evidence)


def _target_transform_receipt(run: Any) -> dict[str, str]:
    transform = getattr(run.spec, "transform", None)
    enabled = bool(getattr(transform, "enabled", False))
    if not enabled:
        return _receipt("target_transform_audited", "not_applicable", "no target transform or finance normalization requested")
    if getattr(run, "transformation_audit", pd.DataFrame()).empty:
        return _receipt("target_transform_audited", "warning", "transform requested, but target_transform_audit is empty")
    target = getattr(transform, "target", "none")
    factor = getattr(transform, "normalization_factor_col", None)
    return _receipt(
        "target_transform_audited",
        "passed",
        f"target={target}; normalization_factor_col={factor or 'none'}; audit rows={len(run.transformation_audit)}",
    )


def _receipt(identifier: str, status: str, evidence: str) -> dict[str, str]:
    return {"id": identifier, "status": status, "evidence": evidence}


def _model_columns(frame: pd.DataFrame) -> set[str]:
    return {
        col
        for col in frame.columns
        if col not in {"unique_id", "ds"} and "-lo-" not in col and "-hi-" not in col
    }
