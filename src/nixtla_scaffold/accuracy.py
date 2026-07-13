from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from nixtla_scaffold.signals import signal_discovery_summary

if TYPE_CHECKING:
    from nixtla_scaffold.schema import ForecastRun


ACCURACY_GATE_SCHEMA_VERSION = 1
CONTEXT_RECEIPT_SCHEMA_VERSION = 1
RESEARCH_BUDGET_SCHEMA_VERSION = 1

_FINAL_SOURCE_STATUSES = {"attempted", "available", "unavailable", "irrelevant", "opted_out"}
_INTERVAL_PASS_STATUSES = {"calibrated"}
_HIERARCHY_PASS_STATUSES = {"coherent", "leaf_or_unchecked"}


def build_context_receipt(run: ForecastRun) -> dict[str, Any]:
    context = run.spec.context
    if context is None:
        raise ValueError("context receipt requires ForecastSpec.context")

    required_fields = {
        "decision": context.decision,
        "audience": context.audience,
        "target_semantics": context.target_semantics,
        "grain": context.grain,
        "refresh_cadence": context.refresh_cadence,
    }
    missing_fields = [name for name, value in required_fields.items() if not str(value).strip()]
    source_status_counts: dict[str, int] = {}
    for source in context.sources:
        source_status_counts[source.status] = source_status_counts.get(source.status, 0) + 1

    if not context.source_discovery_enabled:
        discovery_status = "opted_out"
        discovery_complete = True
    elif not context.sources:
        discovery_status = "not_recorded"
        discovery_complete = False
    elif all(source.status in _FINAL_SOURCE_STATUSES for source in context.sources):
        discovery_status = "complete"
        discovery_complete = True
    else:
        discovery_status = "incomplete"
        discovery_complete = False

    signal_discovery = signal_discovery_summary(context)
    context_complete = not missing_fields and discovery_complete and signal_discovery["complete"]
    return {
        "schema_version": CONTEXT_RECEIPT_SCHEMA_VERSION,
        "accuracy_first": True,
        "context_complete": context_complete,
        "missing_required_fields": missing_fields,
        "decision": context.decision,
        "audience": context.audience,
        "target": {
            "semantics": context.target_semantics,
            "units": context.units or run.spec.unit_label or "",
            "grain": context.grain,
            "constraints": list(context.constraints),
            "requested_horizon": context.requested_horizon or run.spec.horizon,
            "refresh_cadence": context.refresh_cadence,
        },
        "known_changes": {
            "breaks": list(context.known_breaks),
            "exclusions": list(context.exclusions),
            "adjustments": list(context.adjustments),
            "events": list(context.known_events),
        },
        "hierarchy_required": context.hierarchy_required,
        "reference_availability": {
            "plan": context.plan_status,
            "budget": context.budget_status,
            "prior_year": context.prior_year_status,
            "benchmark": context.benchmark_status,
        },
        "source_discovery": {
            "enabled": context.source_discovery_enabled,
            "status": discovery_status,
            "complete": discovery_complete,
            "status_counts": source_status_counts,
            "query_count": context.source_query_count(),
        },
        "signal_discovery": signal_discovery,
        "sources": [source.to_dict() for source in context.sources],
        "candidate_drivers": [driver.to_dict() for driver in context.candidate_drivers],
        "signal_needs": [need.to_dict() for need in context.signal_needs],
        "signal_probes": [probe.to_dict() for probe in context.signal_probes],
        "signal_contracts": [contract.to_dict() for contract in context.signal_contracts],
        "research_budget": context.research_budget.to_dict(),
        "accuracy_policy": context.accuracy_policy.to_dict(),
        "promotion_policy": context.promotion_policy.to_dict(),
    }


def format_context_receipt_markdown(receipt: dict[str, Any]) -> str:
    target = receipt["target"]
    discovery = receipt["source_discovery"]
    signal_discovery = receipt["signal_discovery"]
    lines = [
        "# Forecast context receipt",
        "",
        f"- **Context status:** `{'complete' if receipt['context_complete'] else 'incomplete'}`",
        f"- **Decision:** {receipt['decision'] or 'Not recorded'}",
        f"- **Audience:** {receipt['audience'] or 'Not recorded'}",
        f"- **Target semantics:** {target['semantics'] or 'Not recorded'}",
        f"- **Units / grain:** {target['units'] or 'Not recorded'} / {target['grain'] or 'Not recorded'}",
        f"- **Requested horizon:** {target['requested_horizon']}",
        f"- **Refresh cadence:** {target['refresh_cadence'] or 'Not recorded'}",
        f"- **Source discovery:** `{discovery['status']}` ({discovery['query_count']} bounded queries recorded)",
        f"- **Signal discovery:** `{signal_discovery['status']}` "
        f"({signal_discovery['need_count']} needs, {signal_discovery['probe_count']} probes, "
        f"{signal_discovery['contract_count']} contracts)",
        "",
        "## Connected sources",
        "",
        "| Source | Kind | Status | Queries | Provenance |",
        "| --- | --- | --- | ---: | --- |",
    ]
    sources = receipt["sources"]
    if sources:
        for source in sources:
            provenance = source["provenance"] or source["query_ref"] or ""
            lines.append(
                f"| {source['source_id']} | {source['kind']} | {source['status']} | "
                f"{source['query_count']} | {provenance} |"
            )
    else:
        lines.append("| _None recorded_ |  |  | 0 |  |")
    lines.extend(["", "## Candidate drivers", "", "| Driver | Source | Status | Future availability | Leakage verdict |", "| --- | --- | --- | --- | --- |"])
    drivers = receipt["candidate_drivers"]
    if drivers:
        for driver in drivers:
            lines.append(
                f"| {driver['name']} | {driver['source_id']} | {driver['status']} | "
                f"{driver['future_availability']} | {driver['leakage_verdict']} |"
            )
    else:
        lines.append("| _None recorded_ |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Signal discovery",
            "",
            "| Need | Family | Priority | Status | Next probe |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    needs = receipt["signal_needs"]
    if needs:
        for need in needs:
            lines.append(
                f"| {need['need_id']} | {need['signal_family']} | {need['priority']} | "
                f"{need['status']} | {need['next_probe']} |"
            )
    else:
        lines.append("| _Legacy context: no typed signal needs recorded_ |  |  |  |  |")
    if signal_discovery["unresolved_need_ids"]:
        lines.extend(
            [
                "",
                "Unresolved signal needs: "
                + ", ".join(signal_discovery["unresolved_need_ids"])
                + ". Resolve, exhaust, mark unavailable, or explicitly opt out before generic experiments.",
            ]
        )
    if receipt["missing_required_fields"]:
        lines.extend(
            [
                "",
                "## Required remediation",
                "",
                f"Record: {', '.join(receipt['missing_required_fields'])}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_research_budget_receipt(run: ForecastRun) -> dict[str, Any]:
    context = run.spec.context
    if context is None:
        raise ValueError("research budget receipt requires ForecastSpec.context")
    budget = context.research_budget
    source_queries = context.source_query_count()
    candidate_models = 0
    if not run.backtest_metrics.empty and "model" in run.backtest_metrics.columns:
        candidate_models = int(run.backtest_metrics["model"].dropna().astype(str).nunique())
    allocated = budget.to_dict()
    consumed = {
        "baseline_runs": 1,
        "research_iterations": 0,
        "experiment_variants": 0,
        "source_queries": source_queries,
        "candidate_models_evaluated_in_baseline": candidate_models,
        "wall_clock_minutes": None,
        "compute_units": None,
    }
    remaining = {
        "iterations": _remaining(budget.max_iterations, 0),
        "variants_in_next_iteration": budget.max_variants_per_iteration,
        "source_queries": _remaining(budget.max_source_queries, source_queries),
        "wall_clock_minutes": None,
        "compute_units": None,
    }
    overages = []
    if budget.max_source_queries is not None and source_queries > budget.max_source_queries:
        overages.append(
            {
                "bound": "max_source_queries",
                "limit": budget.max_source_queries,
                "consumed": source_queries,
                "exceeded_by": source_queries - budget.max_source_queries,
            }
        )
    return {
        "schema_version": RESEARCH_BUDGET_SCHEMA_VERSION,
        "profile": budget.profile,
        "allocated": allocated,
        "consumed": consumed,
        "remaining": remaining,
        "within_known_bounds": not overages,
        "overages": overages,
        "consumption_basis": (
            "The core forecast is the baseline run, not a research iteration. "
            "Wall-clock and compute consumption remain unrecorded until the Phase 3 experiment orchestrator executes them."
        ),
        "hard_bound_present": any(
            value is not None
            for value in (
                budget.max_iterations,
                budget.max_variants_per_iteration,
                budget.max_wall_clock_minutes,
                budget.max_source_queries,
                budget.max_compute_units,
            )
        ),
    }


def build_accuracy_gate(run: ForecastRun, trust_summary: pd.DataFrame) -> dict[str, Any]:
    context = run.spec.context
    if context is None:
        raise ValueError("accuracy gate requires ForecastSpec.context")
    policy = context.accuracy_policy
    context_receipt = build_context_receipt(run)
    context_gate = _gate(
        passed=context_receipt["context_complete"] or not policy.require_context_discovery,
        required=policy.require_context_discovery,
        evidence=context_receipt["source_discovery"]["status"],
        remediation="Complete intake and record a final disposition for each relevant connected source.",
    )
    driver_gate = _driver_gate(run, required=policy.require_driver_clearance)
    series_rows: list[dict[str, Any]] = []
    full_horizon_flags: list[bool] = []
    forecast_ids = set(run.forecast.get("unique_id", pd.Series(dtype=str)).dropna().astype(str))

    for row in trust_summary.to_dict(orient="records"):
        uid = str(row.get("unique_id", ""))
        forecast_gate = _gate(
            passed=uid in forecast_ids,
            required=True,
            evidence="selected future rows present" if uid in forecast_ids else "no selected future rows",
            remediation="Resolve the modeling failure and rerun the forecast.",
        )
        horizon_pass = _as_bool(row.get("full_horizon_claim_allowed"))
        full_horizon_flags.append(horizon_pass)
        horizon_gate = _gate(
            passed=horizon_pass,
            required=policy.require_full_horizon_validation,
            evidence=str(row.get("horizon_gate_result") or "unavailable"),
            remediation=(
                f"Add enough chronological history for at least two full {run.spec.horizon}-step "
                "cutoff windows, or shorten the decision horizon."
            ),
        )
        trust_score = _as_int(row.get("trust_score_0_100"))
        trust_gate = _gate(
            passed=trust_score is not None and trust_score >= policy.minimum_trust_score,
            required=True,
            evidence=f"score={trust_score if trust_score is not None else 'unavailable'}; minimum={policy.minimum_trust_score}",
            remediation="Address the trust_summary caveats and rerun chronological validation.",
        )
        interval_status = str(row.get("interval_status") or "unavailable")
        interval_gate = _gate(
            passed=interval_status in _INTERVAL_PASS_STATUSES,
            required=policy.require_interval_evidence,
            evidence=interval_status,
            remediation="Collect enough backtest interval observations and recalibrate until interval status is calibrated.",
        )
        hierarchy_required = policy.require_hierarchy_coherence and (
            context.hierarchy_required
            or bool(run.spec.hierarchy)
            or run.spec.hierarchy_reconciliation != "none"
        )
        hierarchy_status = str(row.get("hierarchy_status") or "unchecked")
        hierarchy_gate = _gate(
            passed=hierarchy_status in _HIERARCHY_PASS_STATUSES,
            required=hierarchy_required,
            evidence=hierarchy_status,
            remediation="Provide the required hierarchy and reconcile until parent/child forecasts are coherent.",
        )
        gates = {
            "forecast": forecast_gate,
            "context": context_gate,
            "full_horizon_validation": horizon_gate,
            "trust": trust_gate,
            "intervals": interval_gate,
            "hierarchy": hierarchy_gate,
            "driver_clearance": driver_gate,
        }
        failed = [name for name, gate in gates.items() if gate["required"] and not gate["passed"]]
        critical = [name for name in failed if name in {"forecast", "hierarchy", "driver_clearance"}]
        if critical or (failed and not policy.allow_directional_baseline):
            status = "blocked"
        elif failed:
            status = "directional_only"
        else:
            status = "planning_ready"
        remediations = [gate["remediation"] for gate in gates.values() if gate["required"] and not gate["passed"]]
        series_rows.append(
            {
                "unique_id": uid,
                "status": status,
                "planning_ready_claim_allowed": status == "planning_ready",
                "gates": gates,
                "failed_gates": failed,
                "remediation": list(dict.fromkeys(remediations)),
            }
        )

    if not series_rows:
        series_rows.append(
            {
                "unique_id": "",
                "status": "blocked",
                "planning_ready_claim_allowed": False,
                "gates": {
                    "forecast": _gate(
                        passed=False,
                        required=True,
                        evidence="no trust rows or selected future rows",
                        remediation="Resolve the modeling failure and rerun the forecast.",
                    )
                },
                "failed_gates": ["forecast"],
                "remediation": ["Resolve the modeling failure and rerun the forecast."],
            }
        )

    statuses = [row["status"] for row in series_rows]
    if "blocked" in statuses:
        overall_status = "blocked"
    elif "directional_only" in statuses:
        overall_status = "directional_only"
    else:
        overall_status = "planning_ready"
    return {
        "schema_version": ACCURACY_GATE_SCHEMA_VERSION,
        "accuracy_first": True,
        "status": overall_status,
        "forecast_produced": bool(forecast_ids),
        "full_requested_horizon_validated": bool(full_horizon_flags) and all(full_horizon_flags),
        "planning_ready_claim_allowed": overall_status == "planning_ready",
        "challenger_promotion_allowed": False,
        "challenger_promotion_reason": "No challenger has passed exact shared-cutoff and untouched-confirmation review in this native run.",
        "policy": policy.to_dict(),
        "promotion_policy": context.promotion_policy.to_dict(),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "series": series_rows,
    }


def format_accuracy_gate_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# Accuracy claim gate",
        "",
        f"- **Overall status:** `{gate['status']}`",
        f"- **Forecast produced:** `{str(gate['forecast_produced']).lower()}`",
        f"- **Full requested horizon validated:** `{str(gate['full_requested_horizon_validated']).lower()}`",
        f"- **Planning-ready claim allowed:** `{str(gate['planning_ready_claim_allowed']).lower()}`",
        f"- **Challenger promotion allowed:** `{str(gate['challenger_promotion_allowed']).lower()}`",
        "",
        "A `directional_only` forecast is a statistical baseline, not planning-ready evidence.",
        "",
        "## Series decisions",
        "",
        "| Series | Status | Failed gates | Required remediation |",
        "| --- | --- | --- | --- |",
    ]
    for row in gate["series"]:
        failed = ", ".join(row["failed_gates"]) or "none"
        remediation = " ".join(row["remediation"]) or "none"
        lines.append(f"| {row['unique_id'] or '_run_'} | {row['status']} | {failed} | {remediation} |")
    lines.append("")
    return "\n".join(lines)


def _driver_gate(run: ForecastRun, *, required: bool) -> dict[str, Any]:
    if not required or not run.spec.train_known_future_regressors:
        return _gate(
            passed=True,
            required=False,
            evidence="no external driver was admitted to training",
            remediation="",
        )
    audit = run.driver_availability_audit
    model_candidates = audit
    if not audit.empty and "mode" in audit.columns:
        model_candidates = audit[audit["mode"].astype(str) == "model_candidate"]
    if model_candidates.empty:
        return _gate(
            passed=False,
            required=True,
            evidence="regressor training enabled but no model-candidate audit rows were produced",
            remediation="Declare candidate drivers and rerun the future-availability and leakage audit.",
        )
    passed = "audit_status" in model_candidates.columns and model_candidates["audit_status"].astype(str).eq("passed").all()
    failed_names = []
    if "name" in model_candidates.columns and "audit_status" in model_candidates.columns:
        failed_names = model_candidates.loc[~model_candidates["audit_status"].astype(str).eq("passed"), "name"].astype(str).tolist()
    evidence = "all trained candidates passed" if passed else f"failed candidates: {', '.join(failed_names) or 'unknown'}"
    return _gate(
        passed=bool(passed),
        required=True,
        evidence=evidence,
        remediation="Remove failed drivers from training or resolve leakage, future-value coverage, and latency audit failures.",
    )


def _gate(*, passed: bool, required: bool, evidence: str, remediation: str) -> dict[str, Any]:
    return {
        "required": required,
        "passed": bool(passed) if required else True,
        "evidence": evidence,
        "remediation": remediation if required and not passed else "",
    }


def _remaining(limit: int | None, consumed: int) -> int | None:
    if limit is None:
        return None
    return max(limit - consumed, 0)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
