from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.mcp_contracts import signal_capabilities_for_family
from nixtla_scaffold.schema import ForecastContext, SignalNeed


SIGNAL_ARTIFACT_SCHEMA_VERSION = "nixtla_scaffold.signal_discovery.v1"
FINAL_SIGNAL_NEED_STATUSES = {"satisfied", "exhausted", "unavailable", "opted_out"}


def build_initial_signal_needs(
    *,
    target_semantics: str,
    grain: str,
    source_discovery_enabled: bool,
) -> tuple[SignalNeed, ...]:
    """Create diagnosis-first needs; an agent still chooses and executes the route."""

    status = "open" if source_discovery_enabled else "unavailable"
    target = str(target_semantics).strip() or "the target metric"
    target_grain = str(grain).strip() or "the forecast grain"
    definitions = (
        (
            "target-integrity",
            "target_integrity",
            f"Is {target} complete, consistently defined, and available at {target_grain}?",
            "A model cannot recover accuracy lost to mixed definitions, duplicate keys, missing periods, or structural breaks.",
            1,
            "Profile schema, keys, coverage, missingness, revisions, and known breaks before model experiments.",
        ),
        (
            "calendar-exposure",
            "calendar_exposure",
            f"Do deterministic calendar or exposure differences explain variation in {target}?",
            "Period length, fiscal calendars, holidays, and operating days can change observed totals without changing underlying demand.",
            2,
            "Test only deterministic or known-future calendar values aligned to the target grain.",
        ),
        (
            "plan-benchmark",
            "plan_benchmark",
            f"Which plan, budget, prior-year, or benchmark series should contextualize {target}?",
            "Benchmarks reveal decision gaps and definition mismatches even when they remain outside the statistical model.",
            3,
            "Keep plan and target separate from statistical yhat unless a validated future-value contract supports a regressor experiment.",
        ),
        (
            "business-driver",
            "operational_driver",
            f"Which leading operational or commercial mechanism could move {target} before it is observed?",
            "A plausible, timely leading signal can improve accuracy when it is available at every historical cutoff and future origin.",
            2,
            "Search bounded aggregates first; correlation alone cannot admit a regressor.",
        ),
        (
            "known-change",
            "known_change",
            f"Which launches, pricing changes, contracts, capacity limits, or headcount actions can change {target} over the horizon?",
            "Known future changes break the all-else-equal assumption and belong in explicit scenarios or validated regressors.",
            2,
            "Preserve event assumptions separately from historical accuracy evidence.",
        ),
    )
    return tuple(
        SignalNeed(
            need_id=need_id,
            signal_family=family,
            question=question,
            business_mechanism=mechanism,
            route_capabilities=signal_capabilities_for_family(family),
            priority=priority,
            status=status,
            next_probe="" if not source_discovery_enabled else next_probe,
            notes="" if source_discovery_enabled else "No authorized discovery route is enabled. This records unavailable enrichment, not a user opt-out.",
        )
        for need_id, family, question, mechanism, priority, next_probe in definitions
    )


def signal_discovery_summary(context: ForecastContext) -> dict[str, Any]:
    needs = context.signal_needs
    probes = context.signal_probes
    contracts = context.signal_contracts
    if not needs:
        status = "legacy_not_recorded"
        complete = True
    elif all(need.status in FINAL_SIGNAL_NEED_STATUSES for need in needs):
        status = "complete"
        complete = True
    else:
        status = "incomplete"
        complete = False
    need_status_counts = _counts(need.status for need in needs)
    probe_status_counts = _counts(probe.status for probe in probes)
    disposition_counts = _counts(contract.disposition for contract in contracts)
    return {
        "schema_version": SIGNAL_ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "complete": complete,
        "need_count": len(needs),
        "probe_count": len(probes),
        "contract_count": len(contracts),
        "need_status_counts": need_status_counts,
        "probe_status_counts": probe_status_counts,
        "disposition_counts": disposition_counts,
        "recorded_source_queries": context.source_query_count(),
        "unresolved_need_ids": [
            need.need_id for need in needs if need.status not in FINAL_SIGNAL_NEED_STATUSES
        ],
    }


def signal_artifact_payloads(context: ForecastContext) -> dict[str, Any]:
    return {
        "signal_needs": {
            "schema_version": SIGNAL_ARTIFACT_SCHEMA_VERSION,
            "summary": signal_discovery_summary(context),
            "needs": [need.to_dict() for need in context.signal_needs],
        },
        "signal_contracts": {
            "schema_version": SIGNAL_ARTIFACT_SCHEMA_VERSION,
            "contracts": [contract.to_dict() for contract in context.signal_contracts],
        },
        "signal_probes": [probe.to_dict() for probe in context.signal_probes],
    }


def write_signal_artifacts(context: ForecastContext, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = signal_artifact_payloads(context)
    needs_path = out / "signal_needs.json"
    probes_path = out / "signal_probe_ledger.jsonl"
    contracts_path = out / "signal_contracts.json"
    needs_path.write_text(
        json.dumps(payloads["signal_needs"], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    probe_text = "\n".join(
        json.dumps(probe, default=str) for probe in payloads["signal_probes"]
    )
    probes_path.write_text(probe_text + ("\n" if probe_text else ""), encoding="utf-8")
    contracts_path.write_text(
        json.dumps(payloads["signal_contracts"], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "signal_needs": needs_path,
        "signal_probe_ledger": probes_path,
        "signal_contracts": contracts_path,
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def screen_driver_relationships(
    tuning: pd.DataFrame, *, features: tuple[str, ...], horizon: int,
    season_length: int = 1, max_features: int = 8, max_lag: int = 12,
    minimum_pairs: int = 24,
) -> dict[str, Any]:
    """Bounded association diagnostics, never causal or training-admission proof."""
    if not 1 <= max_features <= 8 or not 1 <= max_lag <= 24 or minimum_pairs < 24:
        raise ValueError("Driver screening requires 1-8 features, 1-24 lags and at least 24 pairs")
    if horizon < 1 or season_length < 1:
        raise ValueError("Driver horizon and season length must be positive")
    if not {"unique_id", "ds", "y", *features}.issubset(tuning.columns):
        raise ValueError("Driver screening requires explicit target keys and supplied feature columns")
    if tuning[["unique_id", "ds"]].isna().any().any() or tuning.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Driver screening requires unique, complete entity/time keys")
    if len(tuning) > 100_000 or tuning.unique_id.nunique() > 50:
        raise ValueError("Driver screening exceeds 100,000 rows or 50 series; explicitly narrow the research scope")
    selected = tuple(dict.fromkeys(features))[:max_features]
    omitted = tuple(name for name in dict.fromkeys(features) if name not in selected)
    lags = range(-max_lag, max_lag + 1)
    search_count = len(selected) * len(lags) * tuning.unique_id.nunique()
    tests, cards = [], []
    for feature in selected:
        feature_tests = []
        for uid, group in tuning.sort_values("ds").groupby("unique_id", sort=True):
            dates = pd.DatetimeIndex(pd.to_datetime(group.ds))
            regular = len(dates) >= 3 and pd.infer_freq(dates) is not None
            x = pd.to_numeric(group[feature], errors="coerce").reset_index(drop=True)
            y = pd.to_numeric(group.y, errors="coerce").reset_index(drop=True)
            xr = _residual_signal(x, season_length)
            yr = _residual_signal(y, season_length)
            for lag in lags:
                level, pairs = _signed_correlation(x.shift(lag), y)
                change, _ = _signed_correlation(x.shift(lag).diff(), y.diff())
                residual, residual_pairs = _signed_correlation(xr.shift(lag), yr)
                paired = pd.DataFrame({"x": xr.shift(lag), "y": yr}).replace([np.inf, -np.inf], np.nan).dropna()
                midpoint = len(paired) // 2
                first, _ = _signed_correlation(paired.x.iloc[:midpoint], paired.y.iloc[:midpoint])
                last, _ = _signed_correlation(paired.x.iloc[midpoint:], paired.y.iloc[midpoint:])
                stable = (residual is not None and first is not None and last is not None
                          and first * residual > 0 and last * residual > 0
                          and abs(first) >= 0.2 and abs(last) >= 0.2)
                p, effective_pairs = _autocorrelation_adjusted_p(paired, residual)
                adjusted_p = min(1.0, p * search_count) if p is not None else None
                proxy = level is not None and pairs >= 8 and abs(level) >= 1 - 1e-10
                eligible = bool(
                    regular and lag >= horizon and residual_pairs >= minimum_pairs
                    and residual is not None and abs(residual) >= 0.35
                    and change is not None and change * residual > 0 and abs(change) >= 0.15
                    and stable and adjusted_p is not None and adjusted_p <= 0.05 and not proxy
                )
                feature_tests.append({
                    "feature": feature, "unique_id": str(uid), "lag": lag,
                    "level_correlation": level, "change_correlation": change,
                    "residual_correlation": residual, "pairs": pairs,
                    "residual_pairs": residual_pairs, "effective_pairs_approximate": effective_pairs,
                    "first_half_correlation": first, "last_half_correlation": last,
                    "stable_direction": bool(stable), "bonferroni_p_approximate": adjusted_p,
                    "full_horizon_safe_lag": lag >= horizon, "regular_calendar": regular,
                    "target_affine_or_deterministic_proxy": bool(proxy),
                    "eligible_for_predictive_testing": eligible,
                })
        tests.extend(feature_tests)
        common_lags = []
        for lag in range(horizon, max_lag + 1):
            rows = [row for row in feature_tests if row["lag"] == lag]
            if rows and all(row["eligible_for_predictive_testing"] for row in rows):
                signs = {np.sign(row["residual_correlation"]) for row in rows}
                if len(signs) == 1:
                    common_lags.append((lag, float(np.median([abs(row["residual_correlation"]) for row in rows]))))
        proxy = any(row["target_affine_or_deterministic_proxy"] for row in feature_tests)
        best = max(common_lags, key=lambda pair: (pair[1], -pair[0])) if common_lags else None
        enough = any(row["residual_pairs"] >= minimum_pairs for row in feature_tests)
        status = ("rejected" if proxy else "eligible_for_predictive_testing" if best
                  else "insufficient_evidence" if not enough else "context_only")
        reason = (
            "Exact affine/shifted target-like signal requires independent lineage review; not automatically admissible."
            if proxy else "Signed residual and change relationships survive the bounded search and temporal stability checks."
            if best else "Too few paired observations after timing and seasonal/trend controls."
            if not enough else "No common safe lag survives trend/seasonality, temporal stability and multiple-search controls."
        )
        cards.append({
            "feature": feature, "disposition": status, "best_safe_lag": best[0] if best and not proxy else None,
            "signed_direction": (int(np.sign(next(row["residual_correlation"] for row in feature_tests
                                                if row["lag"] == best[0]))) if best and not proxy else None),
            "reason": reason, "tests_searched": len(feature_tests),
            "training_admitted": False,
        })
    collinearity = []
    for _, group in tuning.sort_values("ds").groupby("unique_id", sort=True):
        residuals = {name: _residual_signal(pd.to_numeric(group[name], errors="coerce").reset_index(drop=True),
                                           season_length) for name in selected}
        for index, left in enumerate(selected):
            for right in selected[index + 1:]:
                correlation, pairs = _signed_correlation(residuals[left], residuals[right])
                if correlation is not None and pairs >= minimum_pairs and abs(correlation) >= 0.95:
                    collinearity.append({"unique_id": str(group.unique_id.iloc[0]), "left": left,
                                         "right": right, "residual_correlation": correlation, "pairs": pairs})
    return {
        "schema_version": "nixtla_scaffold.driver_screen.v1",
        "scope": "caller_supplied_tuning_partition_only",
        "features_searched": list(selected), "features_omitted_by_bound": list(omitted),
        "tests_searched": search_count, "lag_range": [-max_lag, max_lag],
        "minimum_pairs": minimum_pairs, "cards": cards, "tests": tests,
        "collinearity": collinearity,
        "limitations": [
            "Association is not causality or incremental forecast value.",
            "Autocorrelation-adjusted p-values are approximate; Bonferroni covers every searched feature/lag/series slot.",
            "Source/vintage admission and later held-out predictive confirmation remain mandatory.",
            "Trend/seasonal controls and feature ranking see only the supplied tuning partition; callers must reserve the tail first.",
        ],
    }


def _residual_signal(values: pd.Series, season: int) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = np.isfinite(values)
    t = np.arange(len(values), dtype=float)
    design = np.column_stack([np.ones(len(t)), t])
    if season > 1:
        if valid.sum() < 2 * season:
            return out
        design = np.column_stack([design, np.eye(season)[np.arange(len(t)) % season, 1:]])
    if valid.sum() <= design.shape[1] + 3:
        return out
    coefficients = np.linalg.lstsq(design[valid], values.loc[valid].to_numpy(), rcond=None)[0]
    residuals = values.loc[valid].to_numpy() - design[valid] @ coefficients
    if np.std(residuals) <= 1e-10 * max(1.0, float(values.loc[valid].std())):
        return out
    out.loc[valid] = residuals
    return out


def _signed_correlation(left: pd.Series, right: pd.Series) -> tuple[float | None, int]:
    pairs = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 4 or pairs.left.std() <= 1e-12 or pairs.right.std() <= 1e-12:
        return None, len(pairs)
    result = float(pairs.left.corr(pairs.right))
    return (result if np.isfinite(result) else None), len(pairs)


def _autocorrelation_adjusted_p(paired: pd.DataFrame, correlation: float | None) -> tuple[float | None, float]:
    from scipy.stats import t
    n = len(paired)
    if correlation is None or n < 8:
        return None, float(n)
    paired = paired.reindex(range(int(paired.index.min()), int(paired.index.max()) + 1))
    rx, _ = _signed_correlation(paired.x, paired.x.shift(1))
    ry, _ = _signed_correlation(paired.y, paired.y.shift(1))
    dependence = min(0.999, abs((rx or 0.0) * (ry or 0.0)))
    effective = min(float(n), n * (1 - dependence) / (1 + dependence))
    if effective <= 3:
        return None, effective
    statistic = abs(correlation) * np.sqrt((effective - 2) / max(1e-15, 1 - correlation ** 2))
    return float(2 * t.sf(statistic, effective - 2)), effective
