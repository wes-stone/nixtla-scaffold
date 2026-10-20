"""Bounded, JSON-safe candidate recipes and observed fitted-model evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.intervals import ml_interval_evidence, native_interval_evidence
from nixtla_scaffold.calendar_features import HolidayIndicator


def parameter_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return {"unavailable": "parameter nesting exceeds audit bound"}
    if isinstance(value, HolidayIndicator):
        return {"feature": value.__name__, "dates": parameter_value(value.dates, depth + 1),
                "shift_days": value.shift}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            return {"unavailable": "array exceeds audit bound", "length": len(value)}
        return [parameter_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): parameter_value(item, depth + 1) for key, item in value.items()}
    if type(value).__module__.split(".")[0] in {"statsforecast", "sklearn", "lightgbm", "mlforecast"}:
        return estimator_definition(value, depth + 1)
    return {"unavailable": f"unsupported parameter type {type(value).__module__}.{type(value).__qualname__}"}


def estimator_definition(estimator: Any, depth: int = 0) -> dict:
    if hasattr(estimator, "get_params"):
        parameters = estimator.get_params(deep=False)
    else:
        parameters = {
            name: getattr(estimator, name, {"unavailable": "constructor parameter not retained"})
            for name, argument in inspect.signature(type(estimator)).parameters.items()
            if argument.kind not in {argument.VAR_POSITIONAL, argument.VAR_KEYWORD}
        }
    return {
        "class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "parameters": {str(key): parameter_value(value, depth + 1) for key, value in parameters.items()},
    }


@dataclass(frozen=True)
class CandidateRecipe:
    model: str
    engine: str
    estimator: dict
    execution: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def merge_evidence(*values: dict) -> dict:
    recipes, fits = {}, []
    for value in values:
        recipes.update(value.get("recipes", {}))
        fits.extend(value.get("fits", []))
    return {"version": 1, "recipes": recipes, "fits": fits}


def contextualize_evidence(evidence: dict, context: dict) -> dict:
    """Context changes are part of recipe identity, not an unlinked annotation."""
    recipes, identities = {}, {}
    for identity, definition in evidence.get("recipes", {}).items():
        recipe = CandidateRecipe(**{**definition, "execution": {
            **definition["execution"], **parameter_value(context),
        }})
        identities[identity] = recipe.fingerprint
        recipes[recipe.fingerprint] = recipe.to_dict()
    return {**evidence, "recipes": recipes, "fits": [
        {**row, "recipe_id": identities[row["recipe_id"]]} for row in evidence.get("fits", [])
    ]}


def complete_evidence(evidence, forecast, history, spec, profile, requested, warnings):
    """Retain missing candidates and explicitly identify adapters without fitted metadata."""
    evidence = merge_evidence(evidence)
    recorded = {(row["unique_id"], row["model"]) for row in evidence["fits"]}
    for alias in sorted(set(requested) - {"WeightedEnsemble"}):
        recipe = CandidateRecipe(
            model=alias, engine="runtime",
            estimator={"unavailable": "adapter did not expose estimator constructor",
                       "reproduction": "saved spec, input snapshot and pinned runtime"},
            execution={"spec": spec.to_dict(), "frequency": profile.freq, "season_length": profile.season_length},
        )
        for uid, group in history.groupby("unique_id", sort=True):
            if (str(uid), alias) in recorded:
                continue
            values = forecast.loc[forecast.unique_id.astype(str).eq(str(uid)), alias] if alias in forecast else pd.Series(dtype=float)
            complete = len(values) == spec.horizon and np.isfinite(values).all()
            evidence["recipes"][recipe.fingerprint] = recipe.to_dict()
            evidence["fits"].append({
                "unique_id": str(uid), "model": alias, "cutoff": pd.to_datetime(group.ds).max().isoformat(),
                "recipe_id": recipe.fingerprint,
                "status": "succeeded" if complete else "not_produced" if values.empty else "incomplete_prediction",
                "fitted_structure": {}, "fitted_structure_status": "unavailable_from_adapter",
                "diagnostics": [] if complete else list(warnings),
            })
    return evidence


def native_fit_evidence(engine, aliases, forecast, history, *, horizon, freq, interval_kwargs):
    from nixtla_scaffold.native_fitted import fitted_diagnostics
    evidence = merge_evidence()
    summary_fields = {
        "method", "components", "aic", "aicc", "bic", "arma", "coef", "par",
        "sigma2", "n_params", "m", "description", "BoxCox_lambda", "p", "q", "k_vector",
        "alpha", "beta", "gamma", "phi",
    }
    for column, (alias, prototype) in enumerate(zip(aliases, engine.models)):
        recipe = CandidateRecipe(
            model=alias, engine="statsforecast", estimator=estimator_definition(prototype),
            execution={"frequency": freq, "horizon": horizon, "n_jobs": 1,
                       "interval_options": parameter_value(interval_kwargs)},
        )
        evidence["recipes"][recipe.fingerprint] = recipe.to_dict()
        for row, uid in enumerate(engine.uids):
            fitted = engine.fitted_[row, column]
            state = getattr(fitted, "model_", {})
            state = state if isinstance(state, dict) else {}
            structure = {key: parameter_value(state[key]) for key in sorted(summary_fields & state.keys())}
            trend = getattr(fitted, "trend_forecaster", None)
            if isinstance(getattr(trend, "model_", None), dict):
                structure["trend_forecaster"] = {
                    key: parameter_value(trend.model_[key]) for key in sorted(summary_fields & trend.model_.keys())
                }
            regressor = state.get("model")
            if regressor is not None and hasattr(regressor, "coef_"):
                structure["regression_coefficients"] = parameter_value(regressor.coef_)
                structure["regression_intercept"] = parameter_value(regressor.intercept_)
            if type(fitted).__name__ in {"MFLES", "AutoMFLES"} and regressor is not None:
                structure["mfles"] = _mfles_structure(fitted, regressor)
            subset = forecast.loc[forecast["unique_id"].astype(str).eq(str(uid))]
            complete = alias in subset and len(subset) == horizon and np.isfinite(subset[alias]).all()
            cutoff = pd.to_datetime(history.loc[history.unique_id.astype(str).eq(str(uid)), "ds"]).max()
            evidence["fits"].append({
                "unique_id": str(uid), "model": alias, "cutoff": cutoff.isoformat(),
                "recipe_id": recipe.fingerprint, "status": "succeeded" if complete else "incomplete_prediction",
                "fitted_structure": structure,
                "fitted_structure_status": "observed" if structure else "unavailable_from_estimator",
                "native_fitted": fitted_diagnostics(
                    fitted, history.loc[history.unique_id.astype(str).eq(str(uid))],
                ),
                "intervals": native_interval_evidence(
                    fitted, subset, history.loc[history.unique_id.astype(str).eq(str(uid))], alias,
                ),
            })
    return evidence


def _mfles_structure(fitted, state) -> dict:
    summary = {
        key: parameter_value(getattr(state, key))
        for key in ("multiplicative", "trend", "trend_penalty", "penalty", "seasonality")
        if hasattr(state, key)
    }
    summary["internal_transform"] = (
        "log_then_exp_without_mean_bias_adjustment" if getattr(state, "const", None) is not None
        else "standardize_then_inverse" if hasattr(state, "std") else "unavailable"
    )
    summary["selected_config_status"] = "not_applicable_fixed_recipe"
    if type(fitted).__name__ == "AutoMFLES":
        from statsforecast.mfles import default_configs
        grid = default_configs(fitted.season_length, fitted.config)
        scores = np.asarray(getattr(state, "metrics", []), dtype=float)
        summary["selected_config_status"] = "unavailable_from_native_search"
        if len(scores) == len(grid) and len(scores) and np.isfinite(scores).all():
            index = int(np.argmin(scores))
            summary.update(
                selected_config_status="reconstructed_from_native_grid_and_observed_scores",
                selected_config=parameter_value(grid[index]), search_metric=fitted.metric,
                search_score=float(scores[index]), search_candidates=len(grid),
            )
    return summary


def ml_fit_evidence(engine, forecast, history, *, execution):
    evidence = merge_evidence()
    for alias, fitted in engine.models_.items():
        definition = ({"class": "mlforecast.direct", "horizons": {
            str(h + 1): estimator_definition(estimator) for h, estimator in sorted(fitted.items())
        }} if isinstance(fitted, dict) else estimator_definition(fitted))
        recipe = CandidateRecipe(model=alias, engine="mlforecast", estimator=definition,
                                 execution=parameter_value(execution))
        evidence["recipes"][recipe.fingerprint] = recipe.to_dict()
        predictor = fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted
        summary = {name: parameter_value(getattr(predictor, name))
                   for name in ("coef_", "intercept_", "n_features_in_", "n_iter_", "best_iteration_")
                   if hasattr(predictor, name)}
        if isinstance(fitted, dict):
            summary = {"horizons": {str(h + 1): {
                name: parameter_value(getattr(estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator, name))
                for name in ("coef_", "intercept_", "n_features_in_")
                if hasattr(estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator, name)
            } for h, estimator in sorted(fitted.items())}}
        for uid, group in history.groupby("unique_id", sort=True):
            predicted = forecast.loc[forecast.unique_id.astype(str).eq(str(uid))]
            complete = alias in predicted and len(predicted) == execution["horizon"] and np.isfinite(predicted[alias]).all()
            evidence["fits"].append({
                "unique_id": str(uid), "model": alias, "cutoff": pd.to_datetime(group.ds).max().isoformat(),
                "recipe_id": recipe.fingerprint, "status": "succeeded" if complete else "incomplete_prediction",
                "fitted_structure": summary, "fitted_structure_status": "observed" if summary else "unavailable_from_estimator",
                "pooling": "one_global_estimator_over_supplied_training_series",
                "intervals": {
                    **ml_interval_evidence(engine, predicted, uid, alias),
                    "conditioning": "supplied_external_feature_paths" if execution["external_feature_columns"] else "target_history",
                },
            })
    return evidence


def baseline_evidence(forecast, history, profile, spec):
    evidence = merge_evidence()
    models = [name for name in forecast if name not in {"unique_id", "ds", "WeightedEnsemble"}]
    for alias in models:
        recipe = CandidateRecipe(
            model=alias, engine="baseline",
            estimator={"implementation": "nixtla_scaffold.models._baseline_predictions", "algorithm": alias},
            execution={"frequency": profile.freq, "horizon": spec.horizon, "season_length": profile.season_length},
        )
        evidence["recipes"][recipe.fingerprint] = recipe.to_dict()
        for uid, group in history.groupby("unique_id", sort=True):
            predicted = forecast.loc[forecast.unique_id.astype(str).eq(str(uid)), alias]
            evidence["fits"].append({
                "unique_id": str(uid), "model": alias, "cutoff": pd.to_datetime(group.ds).max().isoformat(),
                "recipe_id": recipe.fingerprint,
                "status": "succeeded" if len(predicted) == spec.horizon and np.isfinite(predicted).all() else "incomplete_prediction",
                "fitted_structure": {}, "fitted_structure_status": "not_applicable_closed_form",
            })
    return evidence
