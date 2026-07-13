from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

    status = "open" if source_discovery_enabled else "opted_out"
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
