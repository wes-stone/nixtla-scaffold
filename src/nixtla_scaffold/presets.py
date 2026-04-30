from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from nixtla_scaffold.schema import ForecastSpec

ForecastPresetName = Literal["quick", "finance", "strict", "hierarchy"]
PRESET_NAMES: tuple[ForecastPresetName, ...] = ("quick", "finance", "strict", "hierarchy")


_PRESET_SPECS: dict[str, ForecastSpec] = {
    "quick": ForecastSpec(horizon=6, model_policy="baseline", weighted_ensemble=True, verbose=False),
    "finance": ForecastSpec(horizon=6, model_policy="auto", weighted_ensemble=True, verbose=True),
    "strict": ForecastSpec(
        horizon=6,
        model_policy="auto",
        require_backtest=True,
        strict_cv_horizon=True,
        weighted_ensemble=True,
        verbose=True,
    ),
    "hierarchy": ForecastSpec(
        horizon=6,
        model_policy="auto",
        hierarchy_reconciliation="bottom_up",
        weighted_ensemble=True,
        verbose=True,
    ),
}

_PRESET_DESCRIPTIONS: dict[str, str] = {
    "quick": "Fast first pass for exploration: baseline ladder, intervals when available, and concise output.",
    "finance": "Default serious finance baseline: StatsForecast/MLForecast when available, trust/action artifacts, intervals, and full audit output.",
    "strict": "High-stakes mode: requires backtests and only selects champions from full-horizon CV windows.",
    "hierarchy": "Planning-coherent hierarchy mode: finance defaults plus bottom-up reconciliation unless overridden.",
}


def forecast_spec_preset(name: ForecastPresetName | str, **overrides: Any) -> ForecastSpec:
    """Return a ForecastSpec preset with explicit keyword overrides applied."""

    if name not in _PRESET_SPECS:
        raise ValueError(f"forecast preset must be one of {list(PRESET_NAMES)}, got {name!r}")
    return replace(_PRESET_SPECS[str(name)], **{key: value for key, value in overrides.items() if value is not None})


def preset_catalog() -> list[dict[str, Any]]:
    """Return human/agent-readable metadata for available forecast presets."""

    rows: list[dict[str, Any]] = []
    for name in PRESET_NAMES:
        spec = _PRESET_SPECS[name]
        rows.append(
            {
                "name": name,
                "description": _PRESET_DESCRIPTIONS[name],
                "horizon": spec.horizon,
                "model_policy": spec.model_policy,
                "require_backtest": spec.require_backtest,
                "strict_cv_horizon": spec.strict_cv_horizon,
                "hierarchy_reconciliation": spec.hierarchy_reconciliation,
                "weighted_ensemble": spec.weighted_ensemble,
                "verbose": spec.verbose,
            }
        )
    return rows
