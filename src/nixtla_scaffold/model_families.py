from __future__ import annotations

import re
from typing import Any


MLFORECAST_MODELS = {
    "LinearRegression",
    "Ridge",
    "Ridge_Regularized",
    "BayesianRidge",
    "ElasticNet",
    "Huber",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "KNeighbors",
    "LightGBM",
    "LightGBM_Conservative",
    "LightGBM_Shallow",
    "LightGBM_Robust",
}

BASELINE_MODELS = {
    "Naive",
    "HistoricAverage",
    "RandomWalkWithDrift",
    "WindowAverage",
    "SeasonalNaive",
    "SeasonalWindowAverage",
    "ZeroForecast",
}

STATSFORECAST_MODELS = {
    "SES",
    "SeasonalExpSmoothing",
    "AutoETS",
    "AutoARIMA",
    "Holt",
    "HoltWinters",
    "AutoTheta",
    "Theta",
    "OptimizedTheta",
    "DynamicOptimizedTheta",
    "MSTL",
    "MSTL_AutoARIMA",
    "MFLES",
    "AutoMFLES",
    "CrostonClassic",
    "CrostonOptimized",
    "CrostonSBA",
    "ADIDA",
    "IMAPA",
    "TSB",
}

MODEL_ALLOWLIST_CANDIDATES = BASELINE_MODELS | STATSFORECAST_MODELS | MLFORECAST_MODELS


def _model_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


_MODEL_ALIASES = {
    "arima": "AutoARIMA",
    "autoarima": "AutoARIMA",
    "ets": "AutoETS",
    "autoets": "AutoETS",
    "theta": "Theta",
    "autotheta": "AutoTheta",
    "optimizedtheta": "OptimizedTheta",
    "dynamicoptimizedtheta": "DynamicOptimizedTheta",
    "mstl": "MSTL",
    "mstldecomposition": "MSTL",
    "mstlarima": "MSTL_AutoARIMA",
    "arimamstl": "MSTL_AutoARIMA",
    "mstlautoarima": "MSTL_AutoARIMA",
    "autoarimamstl": "MSTL_AutoARIMA",
    "ses": "SES",
    "simpleexponentialsmoothing": "SES",
    "seasonalexponentialsmoothing": "SeasonalExpSmoothing",
    "seasonalexpsmoothing": "SeasonalExpSmoothing",
    "naive": "Naive",
    "historicaverage": "HistoricAverage",
    "historicalaverage": "HistoricAverage",
    "average": "HistoricAverage",
    "drift": "RandomWalkWithDrift",
    "randomwalkwithdrift": "RandomWalkWithDrift",
    "windowaverage": "WindowAverage",
    "seasonalnaive": "SeasonalNaive",
    "seasonalwindowaverage": "SeasonalWindowAverage",
    "zeroforecast": "ZeroForecast",
    "croston": "CrostonClassic",
    "crostonclassic": "CrostonClassic",
    "crostonoptimized": "CrostonOptimized",
    "crostonsba": "CrostonSBA",
    "adida": "ADIDA",
    "imapa": "IMAPA",
    "tsb": "TSB",
    "mfles": "MFLES",
    "automfles": "AutoMFLES",
    "linear": "LinearRegression",
    "linearregression": "LinearRegression",
    "ridge": "Ridge",
    "ridgeregularized": "Ridge_Regularized",
    "bayesianridge": "BayesianRidge",
    "elasticnet": "ElasticNet",
    "huber": "Huber",
    "randomforest": "RandomForest",
    "extratrees": "ExtraTrees",
    "gradientboosting": "GradientBoosting",
    "histgradientboosting": "HistGradientBoosting",
    "kneighbors": "KNeighbors",
    "knn": "KNeighbors",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "lightgbmconservative": "LightGBM_Conservative",
    "lightgbmshallow": "LightGBM_Shallow",
    "lightgbmrobust": "LightGBM_Robust",
}

_CANONICAL_MODEL_BY_KEY = {_model_key(name): name for name in MODEL_ALLOWLIST_CANDIDATES}
_MODEL_ALIASES = {**_MODEL_ALIASES, **_CANONICAL_MODEL_BY_KEY}


def canonical_model_name(name: Any) -> str | None:
    """Return the scaffold's canonical model name for a user-facing alias."""

    key = _model_key(name)
    if not key:
        return None
    return _MODEL_ALIASES.get(key)


def canonicalize_model_allowlist(names: tuple[Any, ...] | list[Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Canonicalize a user-provided model allowlist, preserving order."""

    canonical: list[str] = []
    unknown: list[str] = []
    for raw in names:
        raw_text = str(raw).strip()
        if not raw_text:
            unknown.append(raw_text)
            continue
        model = canonical_model_name(raw_text)
        if model is None:
            unknown.append(raw_text)
            continue
        if model not in canonical:
            canonical.append(model)
    return tuple(canonical), tuple(unknown)


def model_family(model: Any) -> str:
    name = str(model)
    if name == "WeightedEnsemble" or "Ensemble" in name:
        return "ensemble"
    if name.startswith("Custom_"):
        return "custom"
    if name in MLFORECAST_MODELS or name.startswith("LightGBM"):
        return "mlforecast"
    if name in BASELINE_MODELS:
        return "baseline"
    if name in STATSFORECAST_MODELS or name.startswith(("Auto", "Croston")):
        return "statsforecast"
    return "unknown"


def display_model_family(model: Any) -> str:
    return {
        "baseline": "Baseline",
        "statsforecast": "StatsForecast",
        "mlforecast": "MLForecast",
        "custom": "Custom",
        "ensemble": "Ensemble",
        "unknown": "Other",
    }.get(model_family(model), "Other")
