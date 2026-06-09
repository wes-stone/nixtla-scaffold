from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.model_families import model_family
from nixtla_scaffold.schema import ForecastRun

ENSEMBLE_LAB_SCHEMA_VERSION = "1.0"
ENSEMBLE_LAB_POLICIES = ("top_k_average", "family_diverse_average")
ENSEMBLE_BACKTEST_COLUMNS = [
    "schema_version",
    "policy",
    "unique_id",
    "cutoff",
    "ds",
    "y",
    "yhat",
    "error",
    "abs_error",
    "squared_error",
    "base_models",
    "base_model_count",
    "selection_source_cutoffs",
    "scoring_mode",
    "advisory_only",
]
ENSEMBLE_SELECTION_COLUMNS = [
    "schema_version",
    "policy",
    "unique_id",
    "rmse",
    "mae",
    "wape",
    "bias",
    "observations",
    "selected_advisory_policy",
    "scoring_mode",
    "advisory_only",
]
ENSEMBLE_POLICY_RECEIPT_COLUMNS = [
    "schema_version",
    "policy",
    "enabled",
    "max_models",
    "scoring_mode",
    "deployment_mode",
    "candidate_model_count",
    "backtest_rows",
    "forecast_rows",
    "advisory_only",
    "notes",
]
ENSEMBLE_FORECAST_COLUMNS = [
    "schema_version",
    "policy",
    "unique_id",
    "ds",
    "yhat",
    "base_models",
    "base_model_count",
    "selection_source",
    "deployment_mode",
    "advisory_only",
]

_META_COLUMNS = {
    "unique_id",
    "ds",
    "cutoff",
    "y",
    "model",
    "yhat",
    "lo",
    "hi",
    "horizon",
    "horizon_step",
    "row_horizon_status",
    "horizon_trust_state",
    "validated_through_horizon",
    "planning_eligible",
    "planning_eligibility_scope",
}


def build_ensemble_lab_artifacts(run: ForecastRun) -> dict[str, pd.DataFrame]:
    """Build advisory ensemble policy artifacts from an already-completed run."""

    policies = tuple(policy for policy in run.spec.ensemble.advisory_policies if policy in ENSEMBLE_LAB_POLICIES)
    empty = {
        "ensemble_backtest": pd.DataFrame(columns=ENSEMBLE_BACKTEST_COLUMNS),
        "ensemble_selection": pd.DataFrame(columns=ENSEMBLE_SELECTION_COLUMNS),
        "ensemble_policy_receipts": _policy_receipts(run, policies, 0, 0),
        "ensemble_forecast": pd.DataFrame(columns=ENSEMBLE_FORECAST_COLUMNS),
    }
    if not policies:
        return empty

    candidate_models = _candidate_model_columns(run.backtest_predictions)
    if not candidate_models:
        return empty

    backtest = _build_ensemble_backtest(run, policies, candidate_models)
    forecast = _build_ensemble_forecast(run, policies, candidate_models)
    selection = _build_ensemble_selection(run, backtest)
    receipts = _policy_receipts(run, policies, len(backtest), len(forecast), candidate_count=len(candidate_models))
    return {
        "ensemble_backtest": backtest,
        "ensemble_selection": selection,
        "ensemble_policy_receipts": receipts,
        "ensemble_forecast": forecast,
    }


def _build_ensemble_backtest(run: ForecastRun, policies: tuple[str, ...], candidate_models: list[str]) -> pd.DataFrame:
    cv = run.backtest_predictions.copy()
    if cv.empty or "cutoff" not in cv.columns or "y" not in cv.columns:
        return pd.DataFrame(columns=ENSEMBLE_BACKTEST_COLUMNS)

    cv["cutoff"] = pd.to_datetime(cv["cutoff"], errors="coerce")
    cv["ds"] = pd.to_datetime(cv["ds"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for unique_id, series_cv in cv.groupby("unique_id", dropna=False):
        series_cv = series_cv.sort_values(["cutoff", "ds"])
        cutoffs = [cutoff for cutoff in series_cv["cutoff"].dropna().drop_duplicates().sort_values()]
        for cutoff in cutoffs:
            prior = series_cv[series_cv["cutoff"] < cutoff]
            current = series_cv[series_cv["cutoff"] == cutoff]
            if prior.empty or current.empty:
                continue
            for policy in policies:
                selected_models = _select_policy_models(policy, prior, candidate_models, run.spec.ensemble.max_models)
                if not selected_models:
                    continue
                predictions = current[selected_models].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
                for idx, yhat in predictions.items():
                    if pd.isna(yhat):
                        continue
                    row = current.loc[idx]
                    y = pd.to_numeric(row.get("y"), errors="coerce")
                    if pd.isna(y):
                        continue
                    error = float(yhat) - float(y)
                    rows.append(
                        {
                            "schema_version": ENSEMBLE_LAB_SCHEMA_VERSION,
                            "policy": policy,
                            "unique_id": unique_id,
                            "cutoff": cutoff,
                            "ds": row.get("ds"),
                            "y": float(y),
                            "yhat": float(yhat),
                            "error": error,
                            "abs_error": abs(error),
                            "squared_error": error**2,
                            "base_models": "|".join(selected_models),
                            "base_model_count": len(selected_models),
                            "selection_source_cutoffs": int(prior["cutoff"].nunique()),
                            "scoring_mode": run.spec.ensemble.scoring,
                            "advisory_only": True,
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=ENSEMBLE_BACKTEST_COLUMNS)
    return pd.DataFrame(rows, columns=ENSEMBLE_BACKTEST_COLUMNS)


def _build_ensemble_forecast(run: ForecastRun, policies: tuple[str, ...], candidate_models: list[str]) -> pd.DataFrame:
    if run.all_models.empty:
        return pd.DataFrame(columns=ENSEMBLE_FORECAST_COLUMNS)
    forecast_models = [model for model in _candidate_model_columns(run.all_models) if model in candidate_models]
    if not forecast_models:
        return pd.DataFrame(columns=ENSEMBLE_FORECAST_COLUMNS)

    selection_source = _deployment_selection_source(run.backtest_predictions, run.spec.ensemble.deployment)
    rows: list[dict[str, Any]] = []
    for unique_id, future in run.all_models.groupby("unique_id", dropna=False):
        source = selection_source[selection_source["unique_id"] == unique_id] if "unique_id" in selection_source.columns else selection_source
        for policy in policies:
            selected_models = _select_policy_models(policy, source, forecast_models, run.spec.ensemble.max_models)
            if not selected_models:
                continue
            predictions = future[selected_models].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            for idx, yhat in predictions.items():
                if pd.isna(yhat):
                    continue
                row = future.loc[idx]
                rows.append(
                    {
                        "schema_version": ENSEMBLE_LAB_SCHEMA_VERSION,
                        "policy": policy,
                        "unique_id": unique_id,
                        "ds": row.get("ds"),
                        "yhat": float(yhat),
                        "base_models": "|".join(selected_models),
                        "base_model_count": len(selected_models),
                        "selection_source": run.spec.ensemble.deployment,
                        "deployment_mode": run.spec.ensemble.deployment,
                        "advisory_only": True,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=ENSEMBLE_FORECAST_COLUMNS)
    return pd.DataFrame(rows, columns=ENSEMBLE_FORECAST_COLUMNS)


def _build_ensemble_selection(run: ForecastRun, backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame(columns=ENSEMBLE_SELECTION_COLUMNS)
    rows: list[dict[str, Any]] = []
    grouped = backtest.groupby(["unique_id", "policy"], dropna=False)
    for (unique_id, policy), group in grouped:
        y = pd.to_numeric(group["y"], errors="coerce")
        errors = pd.to_numeric(group["error"], errors="coerce")
        valid = y.notna() & errors.notna()
        if not valid.any():
            continue
        y_valid = y[valid]
        errors_valid = errors[valid]
        abs_y_sum = float(y_valid.abs().sum())
        rows.append(
            {
                "schema_version": ENSEMBLE_LAB_SCHEMA_VERSION,
                "policy": policy,
                "unique_id": unique_id,
                "rmse": math.sqrt(float(np.mean(np.square(errors_valid)))),
                "mae": float(np.mean(np.abs(errors_valid))),
                "wape": float(np.abs(errors_valid).sum() / abs_y_sum) if abs_y_sum > 0 else np.nan,
                "bias": float(np.mean(errors_valid)),
                "observations": int(valid.sum()),
                "selected_advisory_policy": False,
                "scoring_mode": run.spec.ensemble.scoring,
                "advisory_only": True,
            }
        )
    selection = pd.DataFrame(rows, columns=ENSEMBLE_SELECTION_COLUMNS)
    if selection.empty:
        return selection
    for unique_id, group in selection.groupby("unique_id", dropna=False):
        valid = group.dropna(subset=["rmse", "mae"])
        if valid.empty:
            continue
        winner_idx = valid.sort_values(["rmse", "mae", "policy"], kind="mergesort").index[0]
        selection.loc[winner_idx, "selected_advisory_policy"] = True
    return selection


def _policy_receipts(
    run: ForecastRun,
    policies: tuple[str, ...],
    backtest_rows: int,
    forecast_rows: int,
    *,
    candidate_count: int | None = None,
) -> pd.DataFrame:
    candidate_count = candidate_count if candidate_count is not None else len(_candidate_model_columns(run.backtest_predictions))
    rows = []
    for policy in ENSEMBLE_LAB_POLICIES:
        enabled = policy in policies
        rows.append(
            {
                "schema_version": ENSEMBLE_LAB_SCHEMA_VERSION,
                "policy": policy,
                "enabled": enabled,
                "max_models": run.spec.ensemble.max_models,
                "scoring_mode": run.spec.ensemble.scoring,
                "deployment_mode": run.spec.ensemble.deployment,
                "candidate_model_count": candidate_count,
                "backtest_rows": backtest_rows if enabled else 0,
                "forecast_rows": forecast_rows if enabled else 0,
                "advisory_only": True,
                "notes": "prior-cutoff scoring; does not alter champion selection" if enabled else "not requested",
            }
        )
    return pd.DataFrame(rows, columns=ENSEMBLE_POLICY_RECEIPT_COLUMNS)


def _select_policy_models(policy: str, evidence: pd.DataFrame, candidate_models: Iterable[str], max_models: int) -> list[str]:
    metrics = _model_error_metrics(evidence, candidate_models)
    if metrics.empty:
        return []
    ranked = metrics.sort_values(["rmse", "mae", "model"], kind="mergesort")
    if policy == "top_k_average":
        return ranked["model"].head(max_models).tolist()
    if policy == "family_diverse_average":
        selected: list[str] = []
        for _, row in ranked.iterrows():
            family = row["family"]
            if any(model_family(model) == family for model in selected):
                continue
            selected.append(row["model"])
            if len(selected) >= max_models:
                break
        return selected
    return []


def _model_error_metrics(evidence: pd.DataFrame, candidate_models: Iterable[str]) -> pd.DataFrame:
    if evidence.empty or "y" not in evidence.columns:
        return pd.DataFrame(columns=["model", "family", "rmse", "mae"])
    y = pd.to_numeric(evidence["y"], errors="coerce")
    rows = []
    for model in candidate_models:
        if model not in evidence.columns:
            continue
        pred = pd.to_numeric(evidence[model], errors="coerce")
        valid = y.notna() & pred.notna()
        if not valid.any():
            continue
        errors = pred[valid] - y[valid]
        rows.append(
            {
                "model": model,
                "family": model_family(model),
                "rmse": math.sqrt(float(np.mean(np.square(errors)))),
                "mae": float(np.mean(np.abs(errors))),
            }
        )
    return pd.DataFrame(rows, columns=["model", "family", "rmse", "mae"])


def _candidate_model_columns(frame: pd.DataFrame) -> list[str]:
    models: list[str] = []
    for column in frame.columns:
        if column in _META_COLUMNS:
            continue
        if "-lo-" in column or "-hi-" in column:
            continue
        if column.endswith("_selected") or column == "WeightedEnsemble":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            models.append(column)
    return models


def _deployment_selection_source(backtest_predictions: pd.DataFrame, deployment_mode: str) -> pd.DataFrame:
    if deployment_mode != "last_cutoff" or backtest_predictions.empty or "cutoff" not in backtest_predictions.columns:
        return backtest_predictions
    frame = backtest_predictions.copy()
    frame["cutoff"] = pd.to_datetime(frame["cutoff"], errors="coerce")
    cutoff = frame["cutoff"].dropna().max()
    if pd.isna(cutoff):
        return frame
    prior = frame[frame["cutoff"] < cutoff]
    return prior if not prior.empty else frame[frame["cutoff"] == cutoff]
