"""Native MLForecast features admitted only under existing source contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nixtla_scaffold.drivers import DriverFeatureBundle
from nixtla_scaffold.schema import FeatureRecipeSpec, ForecastSpec


def fit_options(recipe: FeatureRecipeSpec, horizon: int) -> dict:
    if recipe.mlforecast_strategy == "recursive":
        return {}
    result = {"max_horizon": horizon}
    if recipe.horizon_features:
        if any(h > horizon for h, _ in recipe.horizon_features):
            raise ValueError("Direct horizon feature mapping exceeds this fit's horizon")
        result["horizon_features"] = {h: list(cols) for h, cols in recipe.horizon_features}
    return result


def native_transforms(history: pd.DataFrame, spec: ForecastSpec, bundle: DriverFeatureBundle, existing):
    recipe = spec.feature_recipe
    columns = set(recipe.lookup_lag_columns) | set(recipe.pooled_partition_columns)
    columns |= {c for _, cols in recipe.horizon_features for c in cols}
    if columns:
        if not spec.train_known_future_regressors:
            raise ValueError("Event, partition and horizon features require enabled, contracted regressors")
        declared = {r.value_col or r.name: r for r in spec.regressors}
        for column in sorted(columns):
            regressor = declared.get(column)
            if (regressor is None or regressor.source is None or regressor.mode != "model_candidate"
                    or regressor.availability not in {"calendar", "contracted", "scheduled"}
                    or column not in bundle.feature_columns):
                raise ValueError(f"Feature {column!r} requires an included as-of snapshot with known future availability")
        if spec.evaluation.strategy == "legacy":
            raise ValueError("Contracted advanced features require owned origin-local evaluation, not legacy CV")
    if recipe.pooled_series_ids:
        if len(recipe.pooled_series_ids) < 2 or set(history.unique_id.astype(str)) != set(recipe.pooled_series_ids):
            raise ValueError("Pooled transforms require every series in the exact declared multi-series cohort")
        units = spec.unit_label or (spec.context.units if spec.context else "")
        if not units or (spec.context and spec.context.units and spec.unit_label and spec.context.units != spec.unit_label):
            raise ValueError("Pooled transforms require one explicit, consistent target unit")
        if "units" in history and (history.units.isna().any() or set(history.units.astype(str)) != {units}):
            raise ValueError("Pooled transforms cannot mix incompatible or unknown target units")
        dates = [tuple(pd.to_datetime(g.ds).sort_values()) for _, g in history.groupby("unique_id")]
        if len(set(dates)) != 1:
            raise ValueError("Pooled transforms require aligned member calendars at every origin")
    for column in columns:
        for frame in (bundle.historical, bundle.future):
            if column not in frame or not np.isfinite(frame[column]).all():
                raise ValueError(f"Advanced feature {column!r} has missing or nonfinite input values")
    for column_set in (recipe.lookup_lag_columns, recipe.pooled_partition_columns):
        if not column_set:
            continue
        keys = ["unique_id", *column_set]
        counts = bundle.historical.groupby(keys, dropna=False).size().rename("matches").reset_index()
        coverage = bundle.future.merge(counts, on=keys, how="left", validate="many_to_one")
        if coverage.matches.isna().any():
            raise ValueError("An event/partition bucket has no observed historical match; no imputation or fallback is allowed")
    transforms = {lag: list(values) for lag, values in (existing or {}).items()}
    if recipe.lookup_lag_columns:
        from mlforecast.lag_transforms import LookupLag
        transforms.setdefault(1, []).append(LookupLag(partition_by=list(recipe.lookup_lag_columns)))
    if recipe.pooled_rolling_window:
        from mlforecast.lag_transforms import RollingMean
        transforms.setdefault(1, []).append(RollingMean(
            window_size=recipe.pooled_rolling_window, min_samples=1,
            global_=bool(recipe.pooled_series_ids),
            partition_by=list(recipe.pooled_partition_columns) or None,
        ))
    return transforms or None
