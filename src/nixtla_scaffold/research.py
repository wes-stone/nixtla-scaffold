from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import pandas as pd

from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.schema import ForecastRun, ForecastSpec, PromotionPolicy, ResearchBudget


RESEARCH_SCHEMA_VERSION = "nixtla_scaffold.research.v1"
_PRIMARY_METRICS = ("avg_rmsse", "avg_mase", "avg_rmse")
_SECONDARY_METRICS = {
    "avg_rmsse": ("avg_mase", "avg_mae"),
    "avg_mase": ("avg_rmsse", "avg_mae"),
    "avg_rmse": ("avg_mae",),
}
_METRIC_COLUMNS = {
    "avg_rmsse": "rmsse",
    "avg_mase": "mase",
    "avg_rmse": "rmse",
    "avg_mae": "mae",
}


@dataclass(frozen=True)
class ResearchResult:
    output_dir: Path
    manifest: dict[str, Any]
    iteration_summary: pd.DataFrame
    decisions: tuple[dict[str, Any], ...]
    next_iteration_questions_markdown: str


@dataclass(frozen=True)
class ReviewerResult:
    reviewer: str
    verdict: str
    blocking_gaps: tuple[str, ...]
    caveats: tuple[str, ...]
    requested_next_experiment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "verdict": self.verdict,
            "blocking_gaps": list(self.blocking_gaps),
            "caveats": list(self.caveats),
            "requested_next_experiment": self.requested_next_experiment,
        }


@dataclass(frozen=True)
class ResearchIteration:
    iteration: int
    hypothesis_id: str
    variant: str
    status: str
    primary_metric: str | None
    baseline_primary: float | None
    candidate_primary: float | None
    primary_improvement: float | None
    secondary_metric: str | None
    secondary_regression: float | None
    tuning_cutoff_coverage: float
    stable_window_fraction: float
    research_decision: str
    review_blockers: str
    elapsed_minutes: float
    compute_units_consumed: float
    evidence_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PromotionDecision:
    promotion_recommended: bool
    candidate_variant: str | None
    official_forecast_mutated: bool
    human_approval_required: bool
    primary_metric: str | None
    secondary_metric: str | None
    tuning_assessment: dict[str, Any] | None
    confirmation: dict[str, Any]
    blockers: tuple[str, ...]
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "promotion_recommended": self.promotion_recommended,
            "candidate_variant": self.candidate_variant,
            "official_forecast_mutated": self.official_forecast_mutated,
            "human_approval_required": self.human_approval_required,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "tuning_assessment": self.tuning_assessment,
            "confirmation": self.confirmation,
            "blockers": list(self.blockers),
            "decision": self.decision,
        }


@dataclass(frozen=True)
class StopReceipt:
    stop_reason: str
    promotion_recommended: bool
    best_candidate: str | None
    budget: dict[str, Any]
    remaining_feasible_hypotheses: int
    largest_remaining_unknown: str
    explicit_user_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "stop_reason": self.stop_reason,
            "promotion_recommended": self.promotion_recommended,
            "best_candidate": self.best_candidate,
            "budget": self.budget,
            "remaining_feasible_hypotheses": self.remaining_feasible_hypotheses,
            "largest_remaining_unknown": self.largest_remaining_unknown,
            "explicit_user_stop": self.explicit_user_stop,
        }


@dataclass(frozen=True)
class ChronologicalSplit:
    tuning_data: pd.DataFrame
    receipt: pd.DataFrame
    confirmation_windows: tuple[tuple[str, pd.DataFrame], ...]
    all_series_eligible: bool
    reasons: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "all_series_eligible": self.all_series_eligible,
            "tuning_rows": int(len(self.tuning_data)),
            "confirmation_rows": int(
                sum(len(frame) for _, frame in self.confirmation_windows)
            ),
            "confirmation_windows": [name for name, _ in self.confirmation_windows],
            "reasons": list(self.reasons),
        }


def run_research_optimizer(
    data: str | Path | pd.DataFrame,
    spec: ForecastSpec,
    *,
    sheet: str | int | None,
    output_dir: str | Path,
    variants: Sequence[str] | None,
    max_iterations: int | None,
    max_variants: int,
    patience: int,
) -> ResearchResult:
    from nixtla_scaffold.experiments import (
        OPTIMIZER_SCHEMA_VERSION,
        _optimizer_decisions,
        _variant_catalog,
        run_experiment,
    )

    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if max_variants < 1:
        raise ValueError("max_variants must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_data = load_forecast_dataset(data, sheet=sheet, spec=spec)
    canonical_spec = replace(
        spec,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    promotion_policy = (
        canonical_spec.context.promotion_policy
        if canonical_spec.context is not None
        else PromotionPolicy()
    )
    enabled_challengers = [
        challenger.engine
        for challenger in canonical_spec.challengers
        if challenger.enabled
    ]
    requested_variants = _normalize_variants(variants) if variants is not None else []
    if (
        enabled_challengers
        and variants is not None
        and not {"finn", "all"}.intersection(requested_variants)
    ):
        engines = ", ".join(sorted(set(enabled_challengers)))
        raise ValueError(
            f"optimize received enabled external challengers ({engines}) but no `finn` hypothesis; "
            "include `finn` so the settings are executed rather than silently persisted"
        )
    budget, limits = _resolve_budget(
        canonical_spec,
        max_iterations=max_iterations,
        max_variants=max_variants,
    )
    signal_gate = _signal_discovery_gate(canonical_spec, limits)
    signal_dispositions = _signal_experiment_dispositions(canonical_spec)
    _write_json(
        out / "signal_experiment_dispositions.json",
        {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "signal_discovery_gate": signal_gate,
            "dispositions": signal_dispositions,
        },
    )
    split = _build_chronological_split(
        full_data,
        horizon=canonical_spec.horizon,
        confirmation_cutoffs=promotion_policy.minimum_confirmation_cutoffs,
    )
    split.receipt.to_csv(out / "chronological_split.csv", index=False)

    hypotheses = (
        _generate_hypotheses(
            split.tuning_data,
            canonical_spec,
            variants=variants,
            signal_dispositions=signal_dispositions,
        )
        if signal_gate["generic_experiments_allowed"]
        else ()
    )
    required_variants = (
        {"finn"}
        if (
            signal_gate["generic_experiments_allowed"]
            and any(challenger.enabled for challenger in canonical_spec.challengers)
        )
        else set()
    )
    hypotheses = _prioritize_required_hypotheses(
        hypotheses,
        max_iterations=limits["max_iterations"],
        required_variants=required_variants,
    )
    required_compute_units = 1.0 + sum(
        float(hypothesis.estimated_cost.get("compute_units", 1))
        for hypothesis in hypotheses
        if hypothesis.variant in required_variants
    )
    _require_compute_budget(
        limits,
        required=required_compute_units,
        purpose=(
            "baseline and required challenger hypotheses"
            if required_variants
            else "baseline"
        ),
    )
    _write_jsonl(out / "hypotheses.jsonl", [hypothesis.to_dict() for hypothesis in hypotheses])
    research_plan = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "purpose": "Evidence-led bounded forecast research with untouched chronological promotion confirmation.",
        "source_hash_sha256": _frame_hash(full_data),
        "tuning_hash_sha256": _frame_hash(split.tuning_data),
        "base_spec": spec.to_dict(),
        "canonical_child_spec": canonical_spec.to_dict(),
        "research_budget": budget,
        "effective_limits": limits,
        "promotion_policy": promotion_policy.to_dict(),
        "signal_discovery_gate": signal_gate,
        "signal_experiment_dispositions": signal_dispositions,
        "chronological_split": split.summary(),
        "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
        "invariants": [
            "Baseline execution does not consume a research iteration.",
            "Baseline and treatment must share a resolved candidate fingerprint unless the candidate-set change is the named treatment.",
            "Only tuning rows may seed hypotheses or candidate selection.",
            "Only one selected native candidate is exposed to untouched confirmation rows.",
            "Confirmation uses a fixed candidate specification with walk-forward retraining; later origins may use only actuals that landed before that origin.",
            "External challengers remain advisory and never mutate forecast.csv.",
        ],
    }
    _write_json(out / "research_plan.json", research_plan)

    started = time.monotonic()
    compute_units = 0.0
    baseline_spec = replace(
        _variant_catalog()["baseline"].build_spec(canonical_spec),
        challengers=(),
    )
    baseline_run = run_forecast(split.tuning_data, baseline_spec)
    baseline_dir = out / "baseline"
    baseline_run.to_directory(baseline_dir)
    compute_units += 1.0
    baseline_evidence = _run_evidence(baseline_run, baseline_dir)
    primary_metric = _select_primary_metric(baseline_evidence)
    secondary_metric = _select_secondary_metric(baseline_evidence, primary_metric)

    summary_frames: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    knowledge_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    no_improvement_count = 0
    deterministic_failures = 0
    executed_variants: set[str] = set()
    stopped_reason: str | None = None

    for hypothesis in hypotheses:
        if len(ledger_rows) >= limits["max_iterations"]:
            stopped_reason = "budget_exhausted"
            break
        if _wall_clock_exhausted(started, limits):
            stopped_reason = "budget_exhausted"
            break
        estimated_compute = float(hypothesis.estimated_cost.get("compute_units", 1))
        if _compute_exhausted(compute_units, estimated_compute, limits):
            stopped_reason = "budget_exhausted"
            break

        iteration = len(ledger_rows) + 1
        iteration_dir = out / f"iteration_{iteration:03d}"
        base_experiment_spec = (
            canonical_spec
            if hypothesis.variant == "finn"
            else replace(canonical_spec, challengers=())
        )
        experiment_spec = _spec_for_signal_hypothesis(
            base_experiment_spec,
            hypothesis,
        )
        experiment = run_experiment(
            split.tuning_data,
            experiment_spec,
            output_dir=iteration_dir,
            variants=(hypothesis.variant,),
            max_variants=min(1, limits["max_variants_per_iteration"]),
            hypothesis=hypothesis,
            matched_control=False,
        )
        executed_variants.add(hypothesis.variant)
        compute_units += estimated_compute
        iteration_summary = experiment.summary.copy()
        if not iteration_summary.empty:
            iteration_summary.insert(0, "iteration", iteration)
            iteration_summary.insert(1, "hypothesis_id", hypothesis.hypothesis_id)
            summary_frames.append(iteration_summary)
        candidate_row = (
            iteration_summary.iloc[0].to_dict()
            if not iteration_summary.empty
            else {"status": "failed", "reason": "experiment produced no summary row"}
        )
        candidate_dir = (
            _optional_path(candidate_row.get("run_path"))
            or iteration_dir / "missing_candidate"
        )
        candidate_evidence = _row_evidence(candidate_row)
        paired = _paired_tuning_evidence(
            baseline_dir,
            candidate_dir,
            primary_metric=primary_metric,
            candidate_backtest_path=_optional_path(
                candidate_evidence.get("backtest_path")
            ),
            require_explicit_candidate_backtest=hypothesis.variant == "finn",
        )
        paired_frame = paired.pop("frame")
        paired_frame.to_csv(iteration_dir / "tuning_cutoff_comparison.csv", index=False)
        assessment = _assess_tuning_candidate(
            baseline_evidence,
            candidate_evidence,
            paired,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            policy=promotion_policy,
        )
        reviews = _review_iteration(
            canonical_spec,
            split,
            hypothesis,
            candidate_evidence,
            paired,
            assessment,
        )
        _write_json(iteration_dir / "reviews.json", {"reviewers": list(reviews)})
        _write_json(iteration_dir / "iteration_decision.json", assessment)

        iteration_record = ResearchIteration(
            iteration=iteration,
            hypothesis_id=hypothesis.hypothesis_id,
            variant=hypothesis.variant,
            status=str(candidate_evidence.get("status", "unknown")),
            primary_metric=primary_metric,
            baseline_primary=_number(baseline_evidence.get(primary_metric)) if primary_metric else None,
            candidate_primary=_number(candidate_evidence.get(primary_metric)) if primary_metric else None,
            primary_improvement=_number(assessment.get("primary_improvement")),
            secondary_metric=secondary_metric,
            secondary_regression=_number(assessment.get("secondary_regression")),
            tuning_cutoff_coverage=float(paired.get("exact_coverage", 0.0)),
            stable_window_fraction=float(paired.get("stable_window_fraction", 0.0)),
            research_decision=str(assessment["decision"]),
            review_blockers=" | ".join(
                blocker
                for review in reviews
                for blocker in review.get("blocking_gaps", [])
            ),
            elapsed_minutes=(time.monotonic() - started) / 60.0,
            compute_units_consumed=compute_units,
            evidence_path=str(iteration_dir),
        )
        ledger_row = iteration_record.to_dict()
        ledger_rows.append(ledger_row)
        decision = {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            **ledger_row,
            "reviewers": list(reviews),
            "advisory_only": True,
        }
        decisions.append(decision)
        knowledge_rows.append(
            _knowledge_entry(
                hypothesis,
                assessment,
                reviews,
                iteration=iteration,
                evidence_path=iteration_dir,
            )
        )

        if assessment["decision"] == "candidate_for_confirmation":
            no_improvement_count = 0
            if best_candidate is None or _better_candidate(assessment, best_candidate["assessment"]):
                best_candidate = {
                    "hypothesis": hypothesis,
                    "assessment": assessment,
                    "candidate_dir": candidate_dir,
                }
        elif candidate_evidence.get("status") == "success":
            no_improvement_count += 1
        if candidate_evidence.get("status") == "failed":
            deterministic_failures += 1
        else:
            deterministic_failures = 0
        required_hypotheses_pending = bool(
            required_variants.difference(executed_variants)
        )
        if deterministic_failures >= 2 and not required_hypotheses_pending:
            stopped_reason = "repeated_deterministic_failure"
            break
        if no_improvement_count >= patience and not required_hypotheses_pending:
            stopped_reason = "patience_exhausted"
            break

    iteration_summary = (
        pd.concat(summary_frames, ignore_index=True, sort=False)
        if summary_frames
        else pd.DataFrame()
    )
    legacy_decisions = tuple(_optimizer_decisions(iteration_summary))
    confirmation = _run_confirmation(
        full_data,
        canonical_spec,
        split,
        best_candidate,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        policy=promotion_policy,
        output_dir=out / "confirmation",
    )
    promotion_decision = _promotion_decision(
        best_candidate,
        confirmation,
        split,
        policy=promotion_policy,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
    )
    _write_json(out / "promotion_decision.json", promotion_decision)

    if promotion_decision["promotion_recommended"]:
        stopped_reason = "target_achieved"
    elif not signal_gate["generic_experiments_allowed"]:
        stopped_reason = "source_discovery_incomplete"
    elif confirmation.get("status") == "insufficient_data":
        stopped_reason = "insufficient_data"
    elif stopped_reason is None and primary_metric is None:
        stopped_reason = "no_comparable_evidence"
    elif stopped_reason is None:
        stopped_reason = "hypothesis_queue_exhausted"

    elapsed_minutes = (time.monotonic() - started) / 60.0
    budget_receipt = _budget_receipt(
        budget,
        limits,
        iterations=len(ledger_rows),
        compute_units=compute_units,
        elapsed_minutes=elapsed_minutes,
        source_queries=(
            canonical_spec.context.source_query_count()
            if canonical_spec.context is not None
            else 0
        ),
    )
    stop_receipt = StopReceipt(
        stop_reason=stopped_reason,
        promotion_recommended=bool(promotion_decision["promotion_recommended"]),
        best_candidate=promotion_decision.get("candidate_variant"),
        budget=budget_receipt,
        remaining_feasible_hypotheses=max(0, len(hypotheses) - len(ledger_rows)),
        largest_remaining_unknown=(
            "Signal discovery remains incomplete for: "
            + ", ".join(signal_gate["unresolved_need_ids"])
            if not signal_gate["generic_experiments_allowed"]
            else _largest_remaining_unknown(
                split,
                best_candidate,
                promotion_decision,
            )
        ),
    ).to_dict()
    _write_json(out / "stop_receipt.json", stop_receipt)

    ledger = pd.DataFrame(ledger_rows)
    ledger.to_csv(out / "iteration_ledger.csv", index=False)
    iteration_summary.to_csv(out / "iteration_summary.csv", index=False)
    _write_jsonl(out / "iteration_decisions.jsonl", decisions)
    _write_jsonl(out / "knowledge_ledger.jsonl", knowledge_rows)
    final_signal_dispositions = _finalize_signal_experiment_dispositions(
        signal_dispositions,
        decisions,
    )
    _write_json(
        out / "signal_experiment_dispositions.json",
        {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "signal_discovery_gate": signal_gate,
            "dispositions": final_signal_dispositions,
        },
    )
    next_questions = _next_questions_markdown(
        stop_receipt,
        promotion_decision,
        decisions,
    )
    (out / "next_iteration_questions.md").write_text(next_questions, encoding="utf-8")

    manifest = {
        "schema_version": OPTIMIZER_SCHEMA_VERSION,
        "research_schema_version": RESEARCH_SCHEMA_VERSION,
        "purpose": "Bounded forecast research with paired tuning evidence, reviewer gates, and untouched chronological confirmation.",
        "max_iterations": limits["max_iterations"],
        "executed_iterations": len(ledger_rows),
        "stopped_reason": stopped_reason,
        "best_variant": promotion_decision.get("candidate_variant") or "baseline",
        "promotion_recommended": promotion_decision["promotion_recommended"],
        "official_forecast_mutated": False,
        "base_spec": spec.to_dict(),
        "research_budget": budget_receipt,
        "chronological_split": split.summary(),
        "legacy_advisory_decisions": list(legacy_decisions),
        "outputs": {
            "research_plan": "research_plan.json",
            "hypotheses": "hypotheses.jsonl",
            "chronological_split": "chronological_split.csv",
            "baseline": "baseline",
            "iteration_summary": "iteration_summary.csv",
            "iteration_ledger": "iteration_ledger.csv",
            "iteration_decisions": "iteration_decisions.jsonl",
            "knowledge_ledger": "knowledge_ledger.jsonl",
            "promotion_decision": "promotion_decision.json",
            "stop_receipt": "stop_receipt.json",
            "signal_experiment_dispositions": "signal_experiment_dispositions.json",
            "next_iteration_questions": "next_iteration_questions.md",
            "confirmation": "confirmation",
            "manifest": "iteration_manifest.json",
        },
    }
    _write_json(out / "iteration_manifest.json", manifest)
    return ResearchResult(
        output_dir=out,
        manifest=manifest,
        iteration_summary=iteration_summary,
        decisions=tuple(decisions),
        next_iteration_questions_markdown=next_questions,
    )


def _resolve_budget(
    spec: ForecastSpec,
    *,
    max_iterations: int | None,
    max_variants: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.context is not None:
        source = spec.context.research_budget
    else:
        source = ResearchBudget(
            profile="custom",
            max_iterations=max_iterations or 1,
            max_variants_per_iteration=max_variants,
        )
    effective_iterations = int(source.max_iterations or max_iterations or 1)
    if max_iterations is not None:
        effective_iterations = min(effective_iterations, max_iterations)
    effective_variants = min(
        max_variants,
        int(source.max_variants_per_iteration or max_variants),
    )
    return source.to_dict(), {
        "max_iterations": effective_iterations,
        "max_variants_per_iteration": effective_variants,
        "max_wall_clock_minutes": source.max_wall_clock_minutes,
        "max_source_queries": source.max_source_queries,
        "max_compute_units": source.max_compute_units,
    }


def _build_chronological_split(
    frame: pd.DataFrame,
    *,
    horizon: int,
    confirmation_cutoffs: int,
) -> ChronologicalSplit:
    holdout_rows = horizon * confirmation_cutoffs
    minimum_tuning_rows = max(8, horizon * 2 + 1)
    tuning_parts: list[pd.DataFrame] = []
    receipt_parts: list[pd.DataFrame] = []
    windows: dict[str, list[pd.DataFrame]] = {
        f"confirmation_{index:03d}": []
        for index in range(1, confirmation_cutoffs + 1)
    }
    reasons: list[str] = []
    all_series_eligible = True

    for uid, group in frame.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds").reset_index(drop=True)
        actual_rows = ordered[ordered["y"].notna()].copy()
        eligible = len(actual_rows) - holdout_rows >= minimum_tuning_rows
        if not eligible:
            all_series_eligible = False
            reasons.append(
                f"{uid}: needs at least {minimum_tuning_rows + holdout_rows} rows for "
                f"{confirmation_cutoffs} untouched cutoff(s), found {len(actual_rows)} actual rows"
            )
            tuning_parts.append(ordered)
            receipt = ordered[["unique_id", "ds"]].copy()
            receipt["role"] = "tuning"
            receipt["confirmation_window"] = ""
            receipt["series_confirmation_eligible"] = False
            receipt_parts.append(receipt)
            continue

        holdout = actual_rows.iloc[-holdout_rows:].copy()
        holdout_start = pd.to_datetime(holdout["ds"]).min()
        tuning = ordered[pd.to_datetime(ordered["ds"]).lt(holdout_start)].copy()
        tuning_parts.append(tuning)
        tuning_receipt = tuning[["unique_id", "ds"]].copy()
        tuning_receipt["role"] = "tuning"
        tuning_receipt["confirmation_window"] = ""
        tuning_receipt["series_confirmation_eligible"] = True
        receipt_parts.append(tuning_receipt)
        for index in range(confirmation_cutoffs):
            name = f"confirmation_{index + 1:03d}"
            block = holdout.iloc[index * horizon : (index + 1) * horizon].copy()
            windows[name].append(block)
            block_receipt = block[["unique_id", "ds"]].copy()
            block_receipt["role"] = "confirmation"
            block_receipt["confirmation_window"] = name
            block_receipt["series_confirmation_eligible"] = True
            receipt_parts.append(block_receipt)
        masked_confirmation_context = ordered[
            pd.to_datetime(ordered["ds"]).ge(holdout_start)
            & pd.to_datetime(ordered["ds"]).le(pd.to_datetime(holdout["ds"]).max())
            & ordered["y"].isna()
        ]
        if not masked_confirmation_context.empty:
            context_receipt = masked_confirmation_context[["unique_id", "ds"]].copy()
            context_receipt["role"] = "excluded_confirmation_context"
            context_receipt["confirmation_window"] = ""
            context_receipt["series_confirmation_eligible"] = True
            receipt_parts.append(context_receipt)
        excluded = ordered[pd.to_datetime(ordered["ds"]).gt(pd.to_datetime(holdout["ds"]).max())]
        if not excluded.empty:
            excluded_receipt = excluded[["unique_id", "ds"]].copy()
            excluded_receipt["role"] = "excluded_future_context"
            excluded_receipt["confirmation_window"] = ""
            excluded_receipt["series_confirmation_eligible"] = True
            receipt_parts.append(excluded_receipt)

    return ChronologicalSplit(
        tuning_data=pd.concat(tuning_parts, ignore_index=True, sort=False),
        receipt=pd.concat(receipt_parts, ignore_index=True, sort=False).sort_values(
            ["unique_id", "ds"]
        ),
        confirmation_windows=tuple(
            (
                name,
                pd.concat(parts, ignore_index=True, sort=False)
                if parts
                else pd.DataFrame(columns=frame.columns),
            )
            for name, parts in windows.items()
        ),
        all_series_eligible=all_series_eligible,
        reasons=tuple(reasons),
    )


def _signal_discovery_gate(
    spec: ForecastSpec,
    limits: dict[str, Any],
) -> dict[str, Any]:
    from nixtla_scaffold.signals import signal_discovery_summary

    if spec.context is None:
        return {
            "status": "legacy_not_recorded",
            "complete": True,
            "budget_exhausted": False,
            "generic_experiments_allowed": True,
            "source_queries_consumed": 0,
            "source_queries_limit": limits.get("max_source_queries"),
            "unresolved_need_ids": [],
            "reason": "No typed forecast context was supplied; legacy behavior remains available.",
        }
    summary = signal_discovery_summary(spec.context)
    consumed = spec.context.source_query_count()
    limit = limits.get("max_source_queries")
    budget_exhausted = bool(
        not summary["complete"]
        and limit is not None
        and consumed >= int(limit)
    )
    allowed = bool(summary["complete"] or budget_exhausted)
    status = (
        "complete"
        if summary["complete"]
        else "budget_exhausted"
        if budget_exhausted
        else "incomplete"
    )
    reason = {
        "complete": "Every typed signal need has a final disposition.",
        "budget_exhausted": "The connected-source query bound is exhausted; generic experiments may proceed with this limitation recorded.",
        "incomplete": "Complete, exhaust, mark unavailable, or opt out of each open signal need before generic experiments.",
    }[status]
    return {
        "status": status,
        "complete": bool(summary["complete"]),
        "budget_exhausted": budget_exhausted,
        "generic_experiments_allowed": allowed,
        "source_queries_consumed": consumed,
        "source_queries_limit": limit,
        "unresolved_need_ids": list(summary["unresolved_need_ids"]),
        "reason": reason,
    }


def _signal_experiment_dispositions(spec: ForecastSpec) -> list[dict[str, Any]]:
    if spec.context is None:
        return []
    rows: list[dict[str, Any]] = []
    for contract in spec.context.signal_contracts:
        variant = ""
        executable_names: list[str] = []
        status = contract.experiment_status
        reason = contract.experiment_reason
        if contract.disposition == "regressor_candidate":
            variant = "known_future_regressors"
            matches = _matching_regressors(spec, contract)
            executable_names = [regressor.name for regressor in matches]
            if status not in {"tested", "blocked", "not_applicable"}:
                if matches:
                    status = "queued"
                    reason = reason or "Validated contract matches an executable regressor declaration."
                else:
                    status = "blocked"
                    reason = (
                        reason
                        or "No matching executable KnownFutureRegressor declaration was supplied."
                    )
        elif contract.disposition == "scenario":
            variant = "events"
            matches = _matching_events(spec, contract)
            executable_names = [event.name for event in matches]
            if status not in {"tested", "blocked", "not_applicable"}:
                if matches:
                    status = "queued"
                    reason = reason or "Scenario contract matches an executable DriverEvent declaration."
                else:
                    status = "blocked"
                    reason = reason or "No matching executable DriverEvent declaration was supplied."
        else:
            status = "not_applicable"
            reason = reason or (
                "Context-only signals do not enter model training."
                if contract.disposition == "context"
                else "Rejected signals cannot seed experiments."
            )
        slug = "".join(
            character.lower() if character.isalnum() else "-"
            for character in contract.signal_id
        ).strip("-")
        rows.append(
            {
                "signal_id": contract.signal_id,
                "signal_name": contract.name,
                "need_id": contract.need_id,
                "probe_id": contract.probe_id,
                "source_id": contract.source_id,
                "disposition": contract.disposition,
                "business_mechanism": contract.business_mechanism,
                "variant": variant,
                "hypothesis_id": f"signal-{slug or 'candidate'}",
                "experiment_status": status,
                "experiment_reason": reason,
                "executable_declarations": executable_names,
                "evidence_refs": [
                    ref
                    for ref in (contract.query_ref, contract.provenance)
                    if str(ref).strip()
                ],
            }
        )
    return rows


def _matching_regressors(spec: ForecastSpec, contract: Any) -> tuple[Any, ...]:
    contract_names = {
        str(contract.name).strip().casefold(),
        str(contract.value_col).strip().casefold(),
    }
    contract_names.discard("")
    return tuple(
        regressor
        for regressor in spec.regressors
        if regressor.mode == "model_candidate"
        and {
            str(regressor.name).strip().casefold(),
            str(regressor.value_col or "").strip().casefold(),
        }.intersection(contract_names)
    )


def _matching_events(spec: ForecastSpec, contract: Any) -> tuple[Any, ...]:
    contract_name = str(contract.name).strip().casefold()
    return tuple(
        event
        for event in spec.events
        if str(event.name).strip().casefold() == contract_name
    )


def _spec_for_signal_hypothesis(
    spec: ForecastSpec,
    hypothesis: Any,
) -> ForecastSpec:
    signal_id = str(getattr(hypothesis, "signal_id", "") or "")
    if not signal_id or spec.context is None:
        return spec
    contract = next(
        (
            item
            for item in spec.context.signal_contracts
            if item.signal_id == signal_id
        ),
        None,
    )
    if contract is None:
        return spec
    if hypothesis.variant == "known_future_regressors":
        return replace(spec, regressors=_matching_regressors(spec, contract))
    if hypothesis.variant == "events":
        return replace(spec, events=_matching_events(spec, contract))
    return spec


def _finalize_signal_experiment_dispositions(
    dispositions: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions_by_hypothesis = {
        str(decision.get("hypothesis_id", "")): decision
        for decision in decisions
    }
    finalized: list[dict[str, Any]] = []
    for disposition in dispositions:
        row = dict(disposition)
        if row.get("experiment_status") == "queued":
            decision = decisions_by_hypothesis.get(str(row.get("hypothesis_id", "")))
            if decision is None:
                row["experiment_status"] = "blocked"
                row["experiment_reason"] = (
                    "The admitted signal was not executed before the discovery, iteration, "
                    "compute, wall-clock, or patience stop."
                )
            else:
                row["experiment_status"] = "tested"
                row["experiment_reason"] = (
                    f"Executed with research decision {decision.get('research_decision', 'unknown')}."
                )
                row["evidence_path"] = str(decision.get("evidence_path", ""))
        finalized.append(row)
    return finalized


def _generate_hypotheses(
    tuning_data: pd.DataFrame,
    spec: ForecastSpec,
    *,
    variants: Sequence[str] | None,
    signal_dispositions: Sequence[dict[str, Any]] = (),
) -> tuple[Any, ...]:
    from nixtla_scaffold.experiments import ExperimentHypothesis

    if variants is not None:
        requested = _normalize_variants(variants)
        if "all" in requested:
            from nixtla_scaffold.experiments import EXPERIMENT_VARIANTS

            requested = list(EXPERIMENT_VARIANTS)
    else:
        requested = []
        if spec.model_policy not in {"standard", "all"}:
            requested.append("all_models")
        if (
            spec.transform.target == "none"
            and pd.to_numeric(tuning_data["y"], errors="coerce").dropna().ge(0).all()
        ):
            requested.append("log1p_transform")
        requested.append("rolling_features")
        if spec.regressors:
            requested.append("known_future_regressors")
        if spec.events:
            requested.append("events")
        if {"hierarchy_level", "parent_id"}.intersection(tuning_data.columns):
            requested.append("hierarchy_methods")
        if any(challenger.enabled for challenger in spec.challengers):
            requested.append("finn")
    requested = [variant for variant in dict.fromkeys(requested) if variant != "baseline"]
    evidence = _hypothesis_evidence(spec, tuning_data)
    templates = {
        "all_models": {
            "dimension": "native model family breadth",
            "mechanism": "A broader serious tournament may capture trend or seasonal structure missed by the current policy.",
            "effect": "At least policy-threshold improvement in equal-weight mean RMSSE or MASE.",
            "required": ("tuning target history", "rolling-origin cutoffs"),
            "falsify": "No meaningful paired-cutoff improvement or weaker secondary/trust evidence.",
            "leakage": "Low; all candidates use target history available at each cutoff.",
            "horizon": "A flexible family may win short steps but fail the full requested horizon.",
        },
        "rolling_features": {
            "dimension": "curated lag and rolling target features",
            "mechanism": "Stable recent-level and local-trend features may reduce residual persistence.",
            "effect": "Lower scale-free error across a majority of tuning cutoffs.",
            "required": ("sufficient target history for lag construction",),
            "falsify": "Window instability, no primary improvement, or a material secondary regression.",
            "leakage": "Feature generation must remain cutoff-local and use lagged target values only.",
            "horizon": "Recursive feature decay can weaken later horizon steps.",
        },
        "log1p_transform": {
            "dimension": "log1p target transformation",
            "mechanism": "A multiplicative growth process may be more stable on the log scale and less dominated by recent level.",
            "effect": "Lower scale-free error with unbiased forecasts after inverse transformation.",
            "required": ("non-negative target history", "output-scale backtest metrics"),
            "falsify": "No paired improvement, inverse-scale bias, or weaker interval/trust evidence.",
            "leakage": "Low; transformation parameters must use training rows only.",
            "horizon": "Inverse-transformed error and bias must hold across the full requested horizon.",
        },
        "known_future_regressors": {
            "dimension": "one declared known-future driver family",
            "mechanism": "A future-known operational driver may explain target variation not present in target lags.",
            "effect": "Paired-cutoff improvement without failed availability or leakage audits.",
            "required": ("declared regressor history", "future values at every simulated origin"),
            "falsify": "Coverage/leakage audit fails or backtest skill does not improve.",
            "leakage": "High unless timing and future availability are proven at every cutoff.",
            "horizon": "Driver values must cover every requested future step.",
        },
        "events": {
            "dimension": "explicit event or scenario overlay",
            "mechanism": "A known event may require a decision scenario that historical models cannot infer.",
            "effect": "More decision-relevant future scenario without claiming unsupported backtest accuracy.",
            "required": ("event timing", "business magnitude assumption"),
            "falsify": "Event timing/magnitude is not supportable or is incorrectly described as trained accuracy.",
            "leakage": "Scenario assumptions must not be injected into historical accuracy evidence.",
            "horizon": "The event must intersect the requested forecast horizon.",
        },
        "hierarchy_methods": {
            "dimension": "hierarchy reconciliation method",
            "mechanism": "Coherent parent-child forecasts may improve decision consistency and aggregate error.",
            "effect": "Coherent forecasts with non-worse comparable node-level evidence.",
            "required": ("complete hierarchy metadata", "node-level target history"),
            "falsify": "Reconciliation is unsafe, incomplete, or worsens aggregate evidence beyond tolerance.",
            "leakage": "Low; hierarchy metadata must be structural rather than outcome-derived.",
            "horizon": "Every hierarchy node must cover the full requested horizon.",
        },
        "finn": {
            "dimension": "FINN/finnts model and settings challenger",
            "mechanism": "An independently configured external engine may capture structure missed by native families.",
            "effect": "Exact shared-cutoff improvement; external evidence remains advisory.",
            "required": ("FINN runtime", "immutable cutoff contract", "scored cutoff forecasts"),
            "falsify": "Execution fails, cutoff coverage is partial, or exact comparable metrics do not improve.",
            "leakage": "External training must honor the native cutoff contract and exclude future actuals.",
            "horizon": "External outputs must match every requested cutoff and horizon step exactly.",
        },
    }
    signal_items = [
        (str(row["variant"]), row)
        for row in signal_dispositions
        if row.get("experiment_status") == "queued"
        and (
            variants is None
            or str(row.get("variant", "")) in requested
            or "all" in _normalize_variants(variants)
        )
    ]
    signal_variants = {
        str(row.get("variant", ""))
        for row in signal_dispositions
        if str(row.get("variant", ""))
    }
    work_items: list[tuple[str, dict[str, Any] | None]] = [
        *signal_items,
        *((variant, None) for variant in requested if variant not in signal_variants),
    ]
    hypotheses: list[ExperimentHypothesis] = []
    for index, (variant, signal_row) in enumerate(work_items, start=1):
        template = templates.get(
            variant,
            {
                "dimension": variant,
                "mechanism": "The requested bounded variant may capture signal missed by the baseline.",
                "effect": "Lower primary error without weaker secondary evidence.",
                "required": ("target history",),
                "falsify": "No meaningful paired improvement.",
                "leakage": "Inputs must be available at every cutoff.",
                "horizon": "Evidence must cover the requested horizon.",
            },
        )
        hypotheses.append(
            ExperimentHypothesis(
                hypothesis_id=(
                    str(signal_row["hypothesis_id"])
                    if signal_row is not None
                    else f"h{index:03d}-{variant.replace('_', '-')}"
                ),
                statement=(
                    (
                        f"If {signal_row['signal_name']} adds real forecast signal through "
                        f"{signal_row['business_mechanism']}, a bounded {variant} experiment "
                        "should improve paired chronological evidence."
                    )
                    if signal_row is not None
                    else (
                        f"If {template['dimension']} adds real forecast signal, a bounded {variant} "
                        "experiment should improve paired chronological evidence."
                    )
                ),
                changed_dimension=(
                    f"signal contract {signal_row['signal_id']}"
                    if signal_row is not None
                    else str(template["dimension"])
                ),
                expected_mechanism=(
                    str(signal_row["business_mechanism"])
                    if signal_row is not None
                    else str(template["mechanism"])
                ),
                predicted_effect=str(template["effect"]),
                required_data=tuple(template["required"])
                + (
                    (
                        f"signal contract {signal_row['signal_id']}",
                        f"probe {signal_row['probe_id']}",
                    )
                    if signal_row is not None
                    else ()
                ),
                falsifying_outcome=str(template["falsify"]),
                leakage_risk=str(template["leakage"]),
                horizon_risk=str(template["horizon"]),
                estimated_cost={
                    "iterations": 1,
                    "variants": 1,
                    "compute_units": 2 if variant == "finn" else 1,
                },
                variant=variant,
                evidence_refs=tuple(evidence)
                + (
                    tuple(str(ref) for ref in signal_row.get("evidence_refs", ()))
                    if signal_row is not None
                    else ()
                ),
                seeded_by=(
                    "signal_contract"
                    if signal_row is not None
                    else "context_and_tuning_evidence"
                ),
                signal_id=str(signal_row.get("signal_id", "")) if signal_row else "",
                probe_id=str(signal_row.get("probe_id", "")) if signal_row else "",
            )
        )
    return tuple(hypotheses)


def _prioritize_required_hypotheses(
    hypotheses: tuple[Any, ...],
    *,
    max_iterations: int,
    required_variants: set[str],
) -> tuple[Any, ...]:
    if not required_variants:
        return hypotheses
    required = [
        hypothesis
        for variant in sorted(required_variants)
        for hypothesis in hypotheses
        if hypothesis.variant == variant
    ]
    if len(required) > max_iterations:
        raise ValueError(
            f"research budget max_iterations={max_iterations} cannot execute "
            f"{len(required)} required challenger hypotheses"
        )
    optional = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.variant not in required_variants
    ]
    return tuple(required + optional)


def _hypothesis_evidence(spec: ForecastSpec, tuning_data: pd.DataFrame) -> list[str]:
    refs = [
        f"tuning rows={len(tuning_data)}",
        f"series={tuning_data['unique_id'].nunique()}",
        f"requested horizon={spec.horizon}",
    ]
    if spec.context is not None:
        refs.append(f"context sources inventoried={len(spec.context.sources)}")
        refs.append(f"candidate drivers inventoried={len(spec.context.candidate_drivers)}")
        if spec.context.known_breaks:
            refs.append(f"known breaks declared={len(spec.context.known_breaks)}")
    return refs


def _run_evidence(run: ForecastRun, path: Path) -> dict[str, Any]:
    selection = run.model_selection
    return {
        "status": "success",
        "run_path": str(path),
        "avg_rmse": _mean(selection, "rmse"),
        "avg_mae": _mean(selection, "mae"),
        "avg_mase": _mean(selection, "mase"),
        "avg_rmsse": _mean(selection, "rmsse"),
        "avg_bias": _mean(selection, "bias"),
        "gate_status": _accuracy_gate_status(path),
        "promotion_evidence_eligible": not selection.empty,
        "evidence_class": "native_chronological_backtest",
        "resolved_candidate_fingerprint": _resolved_candidate_fingerprint(run),
    }


def _resolved_candidate_fingerprint(run: Any) -> str:
    resolution = getattr(run, "model_policy_resolution", {})
    if not isinstance(resolution, dict):
        return ""
    return str(resolution.get("resolved_candidate_fingerprint", "") or "")


def _row_evidence(row: dict[str, Any]) -> dict[str, Any]:
    path_value = str(row.get("run_path", "") or "")
    path = Path(path_value) if path_value else None
    return {
        "status": str(row.get("status", "failed")),
        "reason": str(row.get("reason", "")),
        "run_path": path_value,
        "avg_rmse": _number(row.get("avg_rmse")),
        "avg_mae": _number(row.get("avg_mae")),
        "avg_mase": _number(row.get("avg_mase")),
        "avg_rmsse": _number(row.get("avg_rmsse")),
        "avg_bias": _number(row.get("avg_bias")),
        "gate_status": _accuracy_gate_status(path) if path is not None else "unavailable",
        "promotion_evidence_eligible": _bool(row.get("promotion_evidence_eligible")),
        "evidence_class": str(row.get("evidence_class", "none")),
        "exact_comparability_coverage": _number(row.get("exact_comparability_coverage")),
        "backtest_path": str(row.get("backtest_path", "") or ""),
        "resolved_candidate_fingerprint": str(
            row.get("resolved_candidate_fingerprint", "") or ""
        ),
    }


def _select_primary_metric(evidence: dict[str, Any]) -> str | None:
    return next(
        (
            metric
            for metric in _PRIMARY_METRICS
            if _number(evidence.get(metric)) is not None
        ),
        None,
    )


def _select_secondary_metric(
    evidence: dict[str, Any],
    primary_metric: str | None,
) -> str | None:
    if primary_metric is None:
        return None
    return next(
        (
            metric
            for metric in _SECONDARY_METRICS[primary_metric]
            if _number(evidence.get(metric)) is not None
        ),
        None,
    )


def _paired_tuning_evidence(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    primary_metric: str | None,
    candidate_backtest_path: Path | None = None,
    require_explicit_candidate_backtest: bool = False,
) -> dict[str, Any]:
    empty = {
        "frame": pd.DataFrame(),
        "exact_coverage": 0.0,
        "stable_window_fraction": 0.0,
        "window_count": 0,
        "worst_window_regression": None,
        "missing_rows": 0,
        "extra_rows": 0,
        "actual_mismatch_rows": 0,
        "exact_match": False,
        "reason": "paired evidence unavailable",
    }
    if (
        primary_metric is None
        or not candidate_dir.exists()
        or (
            require_explicit_candidate_backtest
            and candidate_backtest_path is None
        )
    ):
        return empty
    baseline = _selected_backtest_rows(baseline_dir)
    candidate = _selected_backtest_rows(
        candidate_dir,
        backtest_path=candidate_backtest_path,
    )
    if baseline.empty or candidate.empty:
        return empty
    keys = ["unique_id", "cutoff", "ds", "horizon_step"]
    expected = baseline[keys + ["y_actual"]].copy()
    observed = candidate[keys + ["y_actual", "yhat", "mase_scale", "rmsse_scale"]].copy()
    paired = expected.merge(
        observed,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
    )
    matched = paired["_merge"].eq("both")
    actual_equal = (
        pd.to_numeric(paired["y_actual_baseline"], errors="coerce")
        .eq(pd.to_numeric(paired["y_actual_candidate"], errors="coerce"))
        .fillna(False)
    )
    exact_rows = matched & actual_equal
    missing_rows = int(paired["_merge"].eq("left_only").sum())
    extra_rows = int(paired["_merge"].eq("right_only").sum())
    actual_mismatch_rows = int((matched & ~actual_equal).sum())
    exact_coverage = float(exact_rows.sum() / len(expected)) if len(expected) else 0.0

    exact_keys = paired.loc[exact_rows, keys].drop_duplicates()
    if exact_keys.empty:
        empty.update(
            {
                "exact_coverage": exact_coverage,
                "missing_rows": missing_rows,
                "extra_rows": extra_rows,
                "actual_mismatch_rows": actual_mismatch_rows,
                "reason": "no exact paired tuning rows",
            }
        )
        return empty
    baseline_scored = baseline.merge(exact_keys, on=keys, how="inner")
    baseline_scored["variant"] = "baseline"
    candidate_scored = candidate.merge(exact_keys, on=keys, how="inner")
    candidate_scored["variant"] = "candidate"
    metrics = pd.concat(
        [
            _cutoff_metrics(baseline_scored),
            _cutoff_metrics(candidate_scored),
        ],
        ignore_index=True,
    )
    metric_column = _METRIC_COLUMNS[primary_metric]
    pivot = metrics.pivot_table(
        index="cutoff",
        columns="variant",
        values=metric_column,
        aggfunc="mean",
    ).reset_index()
    if {"baseline", "candidate"}.issubset(pivot.columns):
        pivot["primary_improvement"] = pivot.apply(
            lambda row: _relative_improvement(row["baseline"], row["candidate"]),
            axis=1,
        )
        valid = pd.to_numeric(pivot["primary_improvement"], errors="coerce").dropna()
        stable_fraction = float(valid.ge(0).mean()) if not valid.empty else 0.0
        worst_regression = float((-valid).max()) if not valid.empty else None
    else:
        pivot["primary_improvement"] = None
        stable_fraction = 0.0
        worst_regression = None
    return {
        "frame": pivot,
        "exact_coverage": exact_coverage,
        "stable_window_fraction": stable_fraction,
        "window_count": int(len(pivot)),
        "worst_window_regression": worst_regression,
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "actual_mismatch_rows": actual_mismatch_rows,
        "exact_match": bool(
            exact_coverage == 1.0
            and missing_rows == 0
            and extra_rows == 0
            and actual_mismatch_rows == 0
        ),
        "reason": "",
    }


def _selected_backtest_rows(
    run_dir: Path,
    *,
    backtest_path: Path | None = None,
) -> pd.DataFrame:
    path = backtest_path or (run_dir / "appendix" / "backtest_long.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "is_selected_model" in frame.columns:
        selected = frame[frame["is_selected_model"].map(_bool)].copy()
    elif backtest_path is not None:
        selected = frame.copy()
    else:
        return pd.DataFrame()
    selected["cutoff"] = pd.to_datetime(selected["cutoff"])
    selected["ds"] = pd.to_datetime(selected["ds"])
    return selected.drop_duplicates(
        ["unique_id", "cutoff", "ds", "horizon_step"],
        keep="first",
    )


def _cutoff_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, cutoff, uid), group in frame.groupby(
        ["variant", "cutoff", "unique_id"],
        sort=False,
    ):
        actual = pd.to_numeric(group["y_actual"], errors="coerce")
        forecast = pd.to_numeric(group["yhat"], errors="coerce")
        valid = actual.notna() & forecast.notna()
        if not valid.any():
            continue
        errors = actual[valid] - forecast[valid]
        mae = float(errors.abs().mean())
        rmse = float(math.sqrt((errors**2).mean()))
        mase_scale = _first_positive(group.get("mase_scale"))
        rmsse_scale = _first_positive(group.get("rmsse_scale"))
        rows.append(
            {
                "variant": variant,
                "cutoff": cutoff,
                "unique_id": uid,
                "mae": mae,
                "rmse": rmse,
                "mase": mae / mase_scale if mase_scale else None,
                "rmsse": rmse / rmsse_scale if rmsse_scale else None,
            }
        )
    return pd.DataFrame(rows)


def _assess_tuning_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    *,
    hypothesis: Any,
    primary_metric: str | None,
    secondary_metric: str | None,
    policy: PromotionPolicy,
) -> dict[str, Any]:
    blockers: list[str] = []
    primary_improvement = _relative_improvement(
        baseline.get(primary_metric) if primary_metric else None,
        candidate.get(primary_metric) if primary_metric else None,
    )
    secondary_regression = _relative_regression(
        baseline.get(secondary_metric) if secondary_metric else None,
        candidate.get(secondary_metric) if secondary_metric else None,
    )
    if candidate.get("status") != "success":
        blockers.append(
            candidate.get("reason")
            or f"candidate status was {candidate.get('status')}"
        )
    if primary_metric is None or primary_improvement is None:
        blockers.append("no comparable primary tuning metric")
    elif primary_improvement <= 0:
        blockers.append("candidate tied or underperformed; retain the simpler baseline")
    elif primary_improvement < policy.minimum_primary_metric_improvement:
        blockers.append(
            f"primary improvement {primary_improvement:.4f} is below "
            f"{policy.minimum_primary_metric_improvement:.4f}"
        )
    if (
        secondary_regression is not None
        and secondary_regression > policy.maximum_secondary_metric_regression
    ):
        blockers.append(
            f"secondary regression {secondary_regression:.4f} exceeds "
            f"{policy.maximum_secondary_metric_regression:.4f}"
        )
    if paired.get("exact_coverage", 0.0) < policy.exact_cutoff_coverage:
        blockers.append(
            f"paired tuning cutoff coverage {paired.get('exact_coverage', 0.0):.4f} "
            f"is below {policy.exact_cutoff_coverage:.4f}"
        )
    if paired.get("extra_rows", 0):
        blockers.append(
            f"candidate produced {paired['extra_rows']} extra tuning cutoff rows"
        )
    if paired.get("actual_mismatch_rows", 0):
        blockers.append(
            f"candidate produced {paired['actual_mismatch_rows']} tuning actual mismatches"
        )
    if paired.get("window_count", 0) < 1:
        blockers.append("no paired tuning windows")
    elif paired.get("stable_window_fraction", 0.0) < 0.6:
        blockers.append("candidate improves fewer than 60% of paired tuning cutoffs")
    if (
        policy.require_no_new_gate_failures
        and _new_gate_failure(
            baseline.get("gate_status"),
            candidate.get("gate_status"),
        )
    ):
        blockers.append("candidate introduced a stronger accuracy-gate failure")
    if not candidate.get("promotion_evidence_eligible", False):
        blockers.append("candidate evidence is directional or non-comparable")
    baseline_fingerprint = str(
        baseline.get("resolved_candidate_fingerprint", "") or ""
    )
    candidate_fingerprint = str(
        candidate.get("resolved_candidate_fingerprint", "") or ""
    )
    candidate_set_changed = bool(
        baseline_fingerprint
        and candidate_fingerprint
        and baseline_fingerprint != candidate_fingerprint
    )
    intentional_candidate_set_treatment = hypothesis.variant == "all_models"
    fingerprint_evidence_available = bool(
        baseline_fingerprint or candidate_fingerprint
    )
    candidate_set_compatible = bool(
        not fingerprint_evidence_available
        or (
            baseline_fingerprint
            and candidate_fingerprint
            and (not candidate_set_changed or intentional_candidate_set_treatment)
        )
    )
    if not candidate_set_compatible:
        blockers.append(
            "resolved candidate fingerprint changed without a named candidate-set treatment"
        )
    if hypothesis.variant == "finn":
        blockers.append("external challenger promotion remains advisory and requires a separate approval path")

    if not blockers:
        decision = "candidate_for_confirmation"
    elif candidate.get("status") != "success":
        decision = "failed"
    elif primary_improvement is not None and primary_improvement > 0:
        decision = "keep_for_learning"
    else:
        decision = "reject"
    return {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "hypothesis_id": hypothesis.hypothesis_id,
        "variant": hypothesis.variant,
        "signal_id": str(getattr(hypothesis, "signal_id", "") or ""),
        "probe_id": str(getattr(hypothesis, "probe_id", "") or ""),
        "decision": decision,
        "primary_metric": primary_metric,
        "primary_improvement": primary_improvement,
        "minimum_primary_improvement": policy.minimum_primary_metric_improvement,
        "secondary_metric": secondary_metric,
        "secondary_regression": secondary_regression,
        "maximum_secondary_regression": policy.maximum_secondary_metric_regression,
        "paired_tuning_cutoff_coverage": paired.get("exact_coverage"),
        "stable_window_fraction": paired.get("stable_window_fraction"),
        "baseline_resolved_candidate_fingerprint": baseline_fingerprint,
        "candidate_resolved_candidate_fingerprint": candidate_fingerprint,
        "candidate_set_changed": candidate_set_changed,
        "candidate_set_change_is_treatment": intentional_candidate_set_treatment,
        "candidate_set_compatible": candidate_set_compatible,
        "candidate_fingerprint_evidence_available": fingerprint_evidence_available,
        "blockers": blockers,
        "advisory_only": True,
    }


def _review_iteration(
    spec: ForecastSpec,
    split: ChronologicalSplit,
    hypothesis: Any,
    candidate: dict[str, Any],
    paired: dict[str, Any],
    assessment: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    data_gaps: list[str] = []
    if paired.get("exact_coverage", 0.0) < 1.0:
        data_gaps.append("baseline and candidate do not share every tuning cutoff row")
    if candidate.get("status") != "success":
        data_gaps.append(
            f"candidate execution failed or was skipped: "
            f"{candidate.get('reason') or candidate.get('status')}"
        )
    forecast_gaps = [
        blocker
        for blocker in assessment["blockers"]
        if "external challenger" not in blocker
    ]
    business_gaps: list[str] = []
    next_source_probe: dict[str, Any] = {}
    if spec.context is None:
        business_gaps.append("decision, audience, units, and source context were not supplied")
    else:
        missing = [
            name
            for name, value in {
                "decision": spec.context.decision,
                "audience": spec.context.audience,
                "target_semantics": spec.context.target_semantics,
                "units": spec.context.units,
                "grain": spec.context.grain,
            }.items()
            if not str(value).strip()
        ]
        if missing:
            business_gaps.append(f"context fields remain incomplete: {missing}")
        unresolved_needs = [
            need
            for need in spec.context.signal_needs
            if need.status not in {"satisfied", "exhausted", "unavailable", "opted_out"}
        ]
        if unresolved_needs:
            need = sorted(
                unresolved_needs,
                key=lambda item: (item.priority, item.need_id),
            )[0]
            business_gaps.append(
                f"signal discovery need remains unresolved: {need.need_id}"
            )
            next_source_probe = {
                "need_id": need.need_id,
                "signal_family": need.signal_family,
                "required_capabilities": list(need.route_capabilities),
                "probe": need.next_probe,
            }
    claim_gaps = []
    if assessment["decision"] == "candidate_for_confirmation":
        claim_gaps.append("tuning evidence cannot support promotion until untouched confirmation passes")
    if hypothesis.variant == "finn":
        claim_gaps.append("FINN evidence is advisory and cannot silently replace the official champion")
    if not split.all_series_eligible:
        claim_gaps.append("not every series has untouched confirmation history")

    return (
        _review(
            "data_skeptic",
            data_gaps,
            "Inspect the smallest mismatched cutoff set or failed child-run receipt.",
        ),
        _review(
            "forecast_skeptic",
            forecast_gaps,
            "Test the next evidence-led hypothesis; expose only one selected candidate to confirmation.",
            blocking_gaps=forecast_gaps,
        ),
        _review(
            "business_context_reviewer",
            business_gaps,
            "Complete the missing decision context before making a planning claim.",
            next_source_probe=next_source_probe,
        ),
        _review(
            "claim_reviewer",
            claim_gaps,
            "Keep the result directional until every promotion and claim gate passes.",
        ),
    )


def _review(
    reviewer: str,
    gaps: list[str],
    next_experiment: str,
    *,
    blocking_gaps: list[str] | None = None,
    next_source_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocking = (
        list(blocking_gaps)
        if blocking_gaps is not None
        else [
            gap
            for gap in gaps
            if any(
                token in gap
                for token in (
                    "failed",
                    "do not share",
                    "cannot support",
                    "not every series",
                    "do not",
                )
            )
        ]
    )
    verdict = "block" if blocking else ("pass_with_caveats" if gaps else "pass")
    payload = ReviewerResult(
        reviewer=reviewer,
        verdict=verdict,
        blocking_gaps=tuple(blocking),
        caveats=tuple(gaps),
        requested_next_experiment=next_experiment if gaps else "",
    ).to_dict()
    payload["next_source_probe"] = dict(next_source_probe or {})
    payload["documented_stop_fallback"] = "stop_receipt.json" if gaps else ""
    return payload


def _run_confirmation(
    full_data: pd.DataFrame,
    spec: ForecastSpec,
    split: ChronologicalSplit,
    best_candidate: dict[str, Any] | None,
    *,
    primary_metric: str | None,
    secondary_metric: str | None,
    policy: PromotionPolicy,
    output_dir: Path,
) -> dict[str, Any]:
    from nixtla_scaffold.experiments import _variant_catalog

    if best_candidate is None:
        return {
            "status": "not_run",
            "reason": "no tuning candidate passed the promotion threshold",
        }
    if not policy.require_untouched_confirmation:
        return {
            "status": "not_required",
            "reason": "promotion policy disabled untouched confirmation",
            "passed": True,
        }
    if (
        not split.all_series_eligible
        or primary_metric is None
        or not split.confirmation_windows
    ):
        return {
            "status": "insufficient_data",
            "reason": "not every series has enough later history for untouched confirmation",
            "passed": False,
        }
    hypothesis = best_candidate["hypothesis"]
    if hypothesis.variant == "finn":
        return {
            "status": "advisory_external",
            "reason": "external challengers require explicit approval and a dedicated untouched-cutoff execution",
            "passed": False,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_spec = replace(
        _variant_catalog()["baseline"].build_spec(spec),
        challengers=(),
    )
    candidate_spec = replace(
        _variant_catalog()[hypothesis.variant].build_spec(spec),
        challengers=(),
    )
    metric_rows: list[dict[str, Any]] = []
    paired_rows: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    coverage_expected = 0
    coverage_matched = 0
    coverage_extra = 0
    baseline_extra_rows = 0
    candidate_extra_rows = 0
    gate_failures: list[str] = []

    for name, actuals in split.confirmation_windows:
        if actuals.empty:
            continue
        training = _training_before_window(full_data, actuals)
        runs: dict[str, ForecastRun] = {}
        scored_frames: dict[str, pd.DataFrame] = {}
        coverages: dict[str, dict[str, int]] = {}
        for variant, run_spec in (
            ("baseline", baseline_spec),
            ("candidate", candidate_spec),
        ):
            model_input = training
            if variant == "candidate" and hypothesis.variant == "known_future_regressors":
                future = actuals.copy()
                future["y"] = pd.NA
                model_input = pd.concat([training, future], ignore_index=True, sort=False)
            run = run_forecast(model_input, run_spec)
            run_dir = output_dir / name / variant
            run.to_directory(run_dir)
            runs[variant] = run
            scored, coverage = _score_confirmation_run(
                run,
                actuals,
                variant=variant,
                window=name,
            )
            scored_frames[variant] = scored
            coverages[variant] = coverage
        baseline_fingerprint = _resolved_candidate_fingerprint(runs["baseline"])
        candidate_fingerprint = _resolved_candidate_fingerprint(runs["candidate"])
        if (
            baseline_fingerprint != candidate_fingerprint
            and hypothesis.variant != "all_models"
        ):
            gate_failures.append(f"{name}:resolved_candidate_fingerprint_changed")
        expected = actuals[["unique_id", "ds", "y"]].copy()
        expected["ds"] = pd.to_datetime(expected["ds"])
        expected = expected.rename(columns={"y": "y_actual"})
        paired = expected.merge(
            scored_frames["baseline"][["unique_id", "ds", "yhat"]].rename(
                columns={"yhat": "baseline_yhat"}
            ),
            on=["unique_id", "ds"],
            how="left",
            validate="one_to_one",
        ).merge(
            scored_frames["candidate"][["unique_id", "ds", "yhat"]].rename(
                columns={"yhat": "candidate_yhat"}
            ),
            on=["unique_id", "ds"],
            how="left",
            validate="one_to_one",
        )
        matched = paired["baseline_yhat"].notna() & paired["candidate_yhat"].notna()
        exact = paired[matched].copy()
        exact.insert(0, "window", name)
        paired_rows.append(exact)
        for variant in ("baseline", "candidate"):
            variant_rows = exact[
                ["unique_id", "ds", "y_actual", f"{variant}_yhat"]
            ].rename(columns={f"{variant}_yhat": "yhat"})
            metric_rows.extend(
                _confirmation_metrics(
                    variant_rows,
                    training,
                    season_length=runs[variant].profile.season_length,
                    variant=variant,
                    window=name,
                )
            )
        window_expected = int(len(expected))
        window_matched = int(matched.sum())
        window_baseline_extra = coverages["baseline"]["extra"]
        window_candidate_extra = coverages["candidate"]["extra"]
        coverage_expected += window_expected
        coverage_matched += window_matched
        baseline_extra_rows += window_baseline_extra
        candidate_extra_rows += window_candidate_extra
        coverage_extra += window_baseline_extra + window_candidate_extra
        if _new_gate_failure(
            _accuracy_gate_status(output_dir / name / "baseline"),
            _accuracy_gate_status(output_dir / name / "candidate"),
        ):
            gate_failures.append(name)
        window_comparison = _confirmation_window_comparison(
            metric_rows,
            window=name,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
        )
        window_comparison.update(
            {
                "expected_rows": window_expected,
                "matched_rows": window_matched,
                "missing_rows": window_expected - window_matched,
                "baseline_extra_rows": window_baseline_extra,
                "candidate_extra_rows": window_candidate_extra,
            }
        )
        comparison_rows.append(window_comparison)

    metrics = pd.DataFrame(metric_rows)
    paired_detail = (
        pd.concat(paired_rows, ignore_index=True, sort=False)
        if paired_rows
        else pd.DataFrame()
    )
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(output_dir / "confirmation_metrics.csv", index=False)
    paired_detail.to_csv(output_dir / "confirmation_paired_rows.csv", index=False)
    comparisons.to_csv(output_dir / "confirmation_comparison.csv", index=False)
    exact_coverage = (
        coverage_matched / coverage_expected
        if coverage_expected
        else 0.0
    )
    valid_comparisons = comparisons[
        comparisons["primary_improvement"].notna()
    ] if not comparisons.empty else comparisons
    primary_pass = bool(
        not valid_comparisons.empty
        and (
            valid_comparisons["primary_improvement"].gt(0)
            & valid_comparisons["primary_improvement"].ge(
                policy.minimum_primary_metric_improvement
            )
        ).all()
    )
    secondary_pass = bool(
        valid_comparisons.empty
        or valid_comparisons["secondary_regression"]
        .fillna(0.0)
        .le(policy.maximum_secondary_metric_regression)
        .all()
    )
    coverage_pass = (
        exact_coverage >= policy.exact_cutoff_coverage
        and coverage_extra == 0
    )
    cutoff_pass = len(valid_comparisons) >= policy.minimum_confirmation_cutoffs
    gate_pass = (
        not policy.require_no_new_gate_failures
        or not gate_failures
    )
    passed = primary_pass and secondary_pass and coverage_pass and cutoff_pass and gate_pass
    receipt = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "status": "completed",
        "passed": passed,
        "training_protocol": "walk_forward_fixed_candidate",
        "candidate_configuration_fixed_before_confirmation": True,
        "candidate_variant": hypothesis.variant,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "confirmation_cutoffs": int(len(valid_comparisons)),
        "required_confirmation_cutoffs": policy.minimum_confirmation_cutoffs,
        "expected_rows": coverage_expected,
        "matched_rows": coverage_matched,
        "extra_rows": coverage_extra,
        "baseline_extra_rows": baseline_extra_rows,
        "candidate_extra_rows": candidate_extra_rows,
        "exact_coverage": exact_coverage,
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "coverage_pass": coverage_pass,
        "cutoff_count_pass": cutoff_pass,
        "no_new_gate_failures": not gate_failures,
        "require_no_new_gate_failures": policy.require_no_new_gate_failures,
        "gate_policy_pass": gate_pass,
        "gate_failure_windows": gate_failures,
        "outputs": {
            "metrics": "confirmation_metrics.csv",
            "paired_rows": "confirmation_paired_rows.csv",
            "comparison": "confirmation_comparison.csv",
        },
    }
    _write_json(output_dir / "confirmation_receipt.json", receipt)
    return receipt


def _training_before_window(
    full_data: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for uid, expected in actuals.groupby("unique_id", sort=False):
        cutoff = pd.to_datetime(expected["ds"]).min()
        history = full_data[
            full_data["unique_id"].eq(uid)
            & pd.to_datetime(full_data["ds"]).lt(cutoff)
        ]
        parts.append(history)
    return pd.concat(parts, ignore_index=True, sort=False)


def _score_confirmation_run(
    run: ForecastRun,
    actuals: pd.DataFrame,
    *,
    variant: str,
    window: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    expected = actuals[["unique_id", "ds", "y"]].copy()
    expected["ds"] = pd.to_datetime(expected["ds"])
    forecast = run.forecast[["unique_id", "ds", "yhat"]].copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast = forecast.drop_duplicates(["unique_id", "ds"], keep="first")
    extra = forecast.merge(
        expected[["unique_id", "ds"]],
        on=["unique_id", "ds"],
        how="left",
        indicator=True,
    )["_merge"].eq("left_only")
    merged = expected.merge(
        forecast,
        on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )
    matched = merged["yhat"].notna()
    scored = merged.loc[
        matched,
        ["unique_id", "ds", "y", "yhat"],
    ].rename(columns={"y": "y_actual"})
    scored.insert(0, "window", window)
    scored.insert(1, "variant", variant)
    return scored.reset_index(drop=True), {
        "expected": int(len(expected)),
        "matched": int(matched.sum()),
        "extra": int(extra.sum()),
    }


def _confirmation_metrics(
    scored: pd.DataFrame,
    training: pd.DataFrame,
    *,
    season_length: int,
    variant: str,
    window: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid, group in scored.groupby("unique_id", sort=False):
        actual = pd.to_numeric(group["y_actual"], errors="coerce")
        prediction = pd.to_numeric(group["yhat"], errors="coerce")
        errors = actual - prediction
        mae = float(errors.abs().mean())
        rmse = float(math.sqrt((errors**2).mean()))
        history = pd.to_numeric(
            training.loc[training["unique_id"].eq(uid), "y"],
            errors="coerce",
        ).dropna()
        mase_scale, rmsse_scale = _naive_scales(
            history,
            season_length,
        )
        rows.append(
            {
                "window": window,
                "variant": variant,
                "unique_id": uid,
                "rows": int(len(group)),
                "mae": mae,
                "rmse": rmse,
                "mase": mae / mase_scale if mase_scale else None,
                "rmsse": rmse / rmsse_scale if rmsse_scale else None,
                "bias": float(errors.mean()),
            }
        )
    return rows


def _confirmation_window_comparison(
    rows: list[dict[str, Any]],
    *,
    window: str,
    primary_metric: str,
    secondary_metric: str | None,
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty or "window" not in frame.columns:
        return {
            "window": window,
            "baseline_primary": None,
            "candidate_primary": None,
            "primary_improvement": None,
            "baseline_secondary": None,
            "candidate_secondary": None,
            "secondary_regression": None,
        }
    frame = frame[frame["window"].eq(window)]
    primary_column = _METRIC_COLUMNS[primary_metric]
    baseline = frame[frame["variant"].eq("baseline")]
    candidate = frame[frame["variant"].eq("candidate")]
    baseline_primary = _mean(baseline, primary_column)
    candidate_primary = _mean(candidate, primary_column)
    secondary_column = (
        _METRIC_COLUMNS[secondary_metric]
        if secondary_metric is not None
        else None
    )
    baseline_secondary = (
        _mean(baseline, secondary_column)
        if secondary_column is not None
        else None
    )
    candidate_secondary = (
        _mean(candidate, secondary_column)
        if secondary_column is not None
        else None
    )
    return {
        "window": window,
        "baseline_primary": baseline_primary,
        "candidate_primary": candidate_primary,
        "primary_improvement": _relative_improvement(
            baseline_primary,
            candidate_primary,
        ),
        "baseline_secondary": baseline_secondary,
        "candidate_secondary": candidate_secondary,
        "secondary_regression": _relative_regression(
            baseline_secondary,
            candidate_secondary,
        ),
    }


def _promotion_decision(
    best_candidate: dict[str, Any] | None,
    confirmation: dict[str, Any],
    split: ChronologicalSplit,
    *,
    policy: PromotionPolicy,
    primary_metric: str | None,
    secondary_metric: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    candidate_variant = None
    tuning_assessment = None
    if best_candidate is None:
        blockers.append("no candidate passed tuning thresholds and reviewer gates")
    else:
        candidate_variant = best_candidate["hypothesis"].variant
        tuning_assessment = best_candidate["assessment"]
    if policy.require_untouched_confirmation and not confirmation.get("passed", False):
        blockers.append(
            confirmation.get("reason")
            or "untouched chronological confirmation did not pass"
        )
    if not split.all_series_eligible:
        blockers.append("not every series was eligible for untouched confirmation")
    if candidate_variant == "finn":
        blockers.append("external challenger remains advisory and requires explicit approval")
    recommended = bool(best_candidate is not None and not blockers)
    return PromotionDecision(
        promotion_recommended=recommended,
        candidate_variant=candidate_variant,
        official_forecast_mutated=False,
        human_approval_required=policy.human_approval_required,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        tuning_assessment=tuning_assessment,
        confirmation=confirmation,
        blockers=tuple(dict.fromkeys(blockers)),
        decision=(
            "recommend_candidate_for_human_approval"
            if recommended and policy.human_approval_required
            else "recommend_candidate"
            if recommended
            else "retain_baseline"
        ),
    ).to_dict()


def _knowledge_entry(
    hypothesis: Any,
    assessment: dict[str, Any],
    reviews: tuple[dict[str, Any], ...],
    *,
    iteration: int,
    evidence_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "iteration": iteration,
        "hypothesis_id": hypothesis.hypothesis_id,
        "variant": hypothesis.variant,
        "claim": (
            f"{hypothesis.variant} produced "
            f"{assessment.get('primary_improvement')} relative primary-metric improvement on tuning evidence."
        ),
        "decision": assessment["decision"],
        "evidence_refs": [
            str(evidence_path / "tuning_cutoff_comparison.csv"),
            str(evidence_path / "reviews.json"),
            str(evidence_path / "iteration_decision.json"),
        ],
        "confidence": (
            "medium"
            if assessment["decision"] == "candidate_for_confirmation"
            else "low"
        ),
        "caveat": "Tuning evidence is not promotion evidence; confirmation is stored separately.",
        "blockers": assessment["blockers"],
        "review_verdicts": {
            review["reviewer"]: review["verdict"]
            for review in reviews
        },
    }


def _better_candidate(
    candidate: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    candidate_value = _number(candidate.get("primary_improvement"))
    current_value = _number(current.get("primary_improvement"))
    if candidate_value is None:
        return False
    if current_value is None:
        return True
    if candidate_value != current_value:
        return candidate_value > current_value
    candidate_secondary = _number(candidate.get("secondary_regression")) or 0.0
    current_secondary = _number(current.get("secondary_regression")) or 0.0
    return candidate_secondary < current_secondary


def _budget_receipt(
    budget: dict[str, Any],
    limits: dict[str, Any],
    *,
    iterations: int,
    compute_units: float,
    elapsed_minutes: float,
    source_queries: int = 0,
) -> dict[str, Any]:
    max_compute = limits.get("max_compute_units")
    max_wall = limits.get("max_wall_clock_minutes")
    return {
        "selected": budget,
        "effective_limits": limits,
        "consumed": {
            "iterations": iterations,
            "compute_units": compute_units,
            "wall_clock_minutes": elapsed_minutes,
            "source_queries": source_queries,
        },
        "remaining": {
            "iterations": max(0, limits["max_iterations"] - iterations),
            "compute_units": (
                max(0.0, float(max_compute) - compute_units)
                if max_compute is not None
                else None
            ),
            "wall_clock_minutes": (
                max(0.0, float(max_wall) - elapsed_minutes)
                if max_wall is not None
                else None
            ),
            "source_queries": (
                max(0, int(limits["max_source_queries"]) - source_queries)
                if limits.get("max_source_queries") is not None
                else None
            ),
        },
    }


def _largest_remaining_unknown(
    split: ChronologicalSplit,
    best_candidate: dict[str, Any] | None,
    promotion: dict[str, Any],
) -> str:
    if not split.all_series_eligible:
        return "Untouched confirmation evidence is missing for at least one series."
    if best_candidate is None:
        return "No tested hypothesis produced stable, practically meaningful tuning improvement."
    if promotion.get("blockers"):
        return str(promotion["blockers"][0])
    return "No material evidence gap remains for the tested candidate; human approval is the remaining control."


def _next_questions_markdown(
    stop_receipt: dict[str, Any],
    promotion: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> str:
    lines = [
        "# Next iteration questions",
        "",
        f"Stop reason: `{stop_receipt['stop_reason']}`.",
        "",
    ]
    if promotion["promotion_recommended"]:
        lines.extend(
            [
                "The selected candidate passed tuning and untouched confirmation. Review the promotion receipt before changing the official champion.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The official baseline remains unchanged.",
                "",
                "## Blocking evidence",
                "",
            ]
        )
        blockers = promotion.get("blockers", [])
        lines.extend(f"- {blocker}" for blocker in blockers)
        if not blockers:
            lines.append("- No candidate reached the configured promotion threshold.")
        lines.append("")
    unresolved = [
        gap
        for decision in decisions
        for review in decision.get("reviewers", [])
        for gap in review.get("blocking_gaps", [])
    ]
    if unresolved:
        lines.extend(["## Reviewer-requested follow-up", ""])
        lines.extend(f"- {gap}" for gap in list(dict.fromkeys(unresolved)))
        lines.append("")
    lines.extend(
        [
            "## Largest remaining unknown",
            "",
            stop_receipt["largest_remaining_unknown"],
            "",
        ]
    )
    return "\n".join(lines)


def _accuracy_gate_status(run_dir: Path | None) -> str:
    if run_dir is None:
        return "unavailable"
    path = run_dir / "appendix" / "accuracy_gate.json"
    if not path.exists():
        return "not_evaluated"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    return str(payload.get("claim_status", payload.get("status", "unknown")))


def _new_gate_failure(baseline: Any, candidate: Any) -> bool:
    order = {
        "planning_ready": 0,
        "not_evaluated": 1,
        "directional_only": 2,
        "unavailable": 3,
        "unknown": 3,
        "unreadable": 4,
        "blocked": 5,
    }
    return order.get(str(candidate), 4) > order.get(str(baseline), 4)


def _naive_scales(
    history: pd.Series,
    season_length: int,
) -> tuple[float | None, float | None]:
    if len(history) < 2:
        return None, None
    lag = season_length if season_length > 1 and len(history) > season_length else 1
    differences = history.diff(lag).dropna()
    if differences.empty:
        return None, None
    mase = float(differences.abs().mean())
    rmsse = float(math.sqrt((differences**2).mean()))
    return (
        mase if math.isfinite(mase) and mase > 0 else None,
        rmsse if math.isfinite(rmsse) and rmsse > 0 else None,
    )


def _relative_improvement(baseline: Any, candidate: Any) -> float | None:
    base = _number(baseline)
    challenger = _number(candidate)
    if base is None or challenger is None:
        return None
    if base == 0:
        return 0.0 if challenger == 0 else None
    return float((base - challenger) / abs(base))


def _relative_regression(baseline: Any, candidate: Any) -> float | None:
    improvement = _relative_improvement(baseline, candidate)
    return None if improvement is None else -improvement


def _first_positive(values: Any) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[numeric.gt(0)]
    return float(valid.iloc[0]) if not valid.empty else None


def _mean(frame: pd.DataFrame, column: str | None) -> float | None:
    if column is None or frame.empty or column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_path(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_variants(variants: Sequence[str]) -> list[str]:
    names: list[str] = []
    for value in variants:
        names.extend(
            part.strip()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        )
    return list(dict.fromkeys(names))


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return sha256(
        pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    ).hexdigest()


def _require_compute_budget(
    limits: dict[str, Any],
    *,
    required: float,
    purpose: str,
) -> None:
    maximum = limits.get("max_compute_units")
    if maximum is not None and float(maximum) < required:
        raise ValueError(
            f"research budget max_compute_units={maximum} cannot fund the "
            f"{required:g}-unit {purpose}"
        )


def _compute_exhausted(
    consumed: float,
    required: float,
    limits: dict[str, Any],
) -> bool:
    maximum = limits.get("max_compute_units")
    return maximum is not None and consumed + required > float(maximum)


def _wall_clock_exhausted(
    started: float,
    limits: dict[str, Any],
) -> bool:
    maximum = limits.get("max_wall_clock_minutes")
    if maximum is None:
        return False
    return (time.monotonic() - started) / 60.0 >= float(maximum)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
