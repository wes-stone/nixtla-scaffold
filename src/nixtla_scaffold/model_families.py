from __future__ import annotations

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
