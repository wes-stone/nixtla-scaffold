from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from nixtla_scaffold.schema import ForecastSpec

ForecastPresetName = Literal["quick", "standard", "strict", "hierarchy", "finance"]
CANONICAL_PRESET_NAMES: tuple[str, ...] = ("quick", "standard", "strict", "hierarchy")
PRESET_ALIASES: dict[str, str] = {"finance": "standard"}
PRESET_NAMES: tuple[str, ...] = CANONICAL_PRESET_NAMES + tuple(PRESET_ALIASES)


_PRESET_SPECS: dict[str, ForecastSpec] = {
    "quick": ForecastSpec(horizon=6, model_policy="baseline", weighted_ensemble=True, verbose=False),
    "standard": ForecastSpec(horizon=6, model_policy="standard", weighted_ensemble=True, verbose=True),
    "strict": ForecastSpec(
        horizon=6,
        model_policy="standard",
        require_backtest=True,
        strict_cv_horizon=True,
        weighted_ensemble=True,
        verbose=True,
    ),
    "hierarchy": ForecastSpec(
        horizon=6,
        model_policy="standard",
        hierarchy_reconciliation="bottom_up",
        weighted_ensemble=True,
        verbose=True,
    ),
}

_PRESET_DESCRIPTIONS: dict[str, str] = {
    "quick": "Fast first pass for exploration: baseline ladder, intervals when available, and concise output.",
    "standard": "Default audit-ready forecast: standard model policy, trust/action artifacts, intervals, and full audit output.",
    "strict": "High-stakes mode: requires backtests and only selects champions from full-horizon CV windows.",
    "hierarchy": "Planning-coherent hierarchy mode: standard defaults plus bottom-up reconciliation unless overridden.",
}


def canonical_preset_name(name: ForecastPresetName | str) -> str:
    return PRESET_ALIASES.get(str(name), str(name))


def forecast_spec_preset(name: ForecastPresetName | str, **overrides: Any) -> ForecastSpec:
    """Return a ForecastSpec preset with explicit keyword overrides applied."""

    canonical = canonical_preset_name(name)
    if canonical not in _PRESET_SPECS:
        raise ValueError(f"forecast preset must be one of {list(PRESET_NAMES)}, got {name!r}")
    return replace(_PRESET_SPECS[canonical], **{key: value for key, value in overrides.items() if value is not None})


def preset_catalog() -> list[dict[str, Any]]:
    """Return analyst/agent-readable metadata for available forecast presets."""

    rows: list[dict[str, Any]] = []
    for name in CANONICAL_PRESET_NAMES:
        spec = _PRESET_SPECS[name]
        rows.append(
            {
                "name": name,
                "aliases": [alias for alias, target in PRESET_ALIASES.items() if target == name],
                "description": _PRESET_DESCRIPTIONS[name],
                "horizon": spec.horizon,
                "model_policy": spec.model_policy,
                "require_backtest": spec.require_backtest,
                "strict_cv_horizon": spec.strict_cv_horizon,
                "hierarchy_reconciliation": spec.hierarchy_reconciliation,
                "train_known_future_regressors": spec.train_known_future_regressors,
                "mlforecast_feature_policy": spec.mlforecast_feature_policy,
                "weighted_ensemble": spec.weighted_ensemble,
                "verbose": spec.verbose,
            }
        )
    return rows
