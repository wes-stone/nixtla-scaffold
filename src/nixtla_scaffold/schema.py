from __future__ import annotations

import hashlib
import json
import math
import os
from importlib import metadata
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
import yaml

from nixtla_scaffold.model_families import MODEL_ALLOWLIST_CANDIDATES, canonicalize_model_allowlist

FillMethod = Literal["ffill", "zero", "interpolate", "drop"]
ModelPolicy = Literal["standard", "light", "auto", "baseline", "statsforecast", "mlforecast", "all"]
RunMode = Literal["accuracy-first", "quick", "legacy"]
EventEffect = Literal["additive", "multiplicative"]
TargetTransform = Literal["none", "log", "log1p"]
HierarchyReconciliationMethod = Literal["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"]
RegressorAvailability = Literal["calendar", "contracted", "scheduled", "plan", "forecasted", "historical_only"]
RegressorMode = Literal["audit_only", "model_candidate"]
MLForecastFeaturePolicy = Literal["basic", "rolling"]
EnsemblePolicy = Literal["legacy_weighted", "top_k_average", "family_diverse_average"]
EnsembleScoringMode = Literal["prior_only"]
EnsembleDeploymentMode = Literal["full_backtest", "last_cutoff"]
ParallelProcessingMode = Literal["none", "local_machine", "spark"]
ChallengerOnError = Literal["skip", "fail"]
ResearchBudgetProfile = Literal["time-boxed", "balanced", "deep", "custom"]
ContextSourceStatus = Literal["planned", "attempted", "available", "unavailable", "irrelevant", "opted_out"]
ReferenceAvailability = Literal["unknown", "available", "unavailable", "not_applicable"]
CandidateDriverStatus = Literal["discovered", "audit_only", "eligible_for_experiment", "rejected"]
DriverFutureAvailability = Literal["unknown", "known", "planned", "scenario", "forecasted", "historical_only", "unavailable"]
LeakageVerdict = Literal["unreviewed", "pass", "warning", "fail"]
SignalNeedStatus = Literal["open", "probing", "satisfied", "exhausted", "unavailable", "opted_out"]
SignalProbeStage = Literal["schema", "count", "sample", "aggregate"]
SignalProbeStatus = Literal["planned", "completed", "failed", "skipped"]
SignalDisposition = Literal["context", "scenario", "regressor_candidate", "reject"]
SignalExperimentStatus = Literal["not_requested", "queued", "tested", "blocked", "not_applicable"]


@dataclass(frozen=True)
class DriverEvent:
    """Auditable event or driver assumption applied after the statistical baseline."""

    name: str
    start: str
    end: str | None = None
    effect: EventEffect = "multiplicative"
    magnitude: float = 0.0
    affected_unique_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("driver event name is required")
        if self.effect not in {"additive", "multiplicative"}:
            raise ValueError("driver event effect must be 'additive' or 'multiplicative'")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("driver event confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["affected_unique_ids"] = list(self.affected_unique_ids)
        return data


@dataclass(frozen=True)
class DriverSnapshotGrant:
    """Caller-authorized local snapshot, bound to content and series scope."""

    source_id: str
    path: str
    sha256: str
    series_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        from pathlib import Path
        if not self.source_id.strip() or not Path(self.path).is_absolute():
            raise ValueError("driver snapshot grants require a source ID and an absolute local path")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("driver snapshot grant requires a lowercase SHA256")
        if not self.series_ids or any(not isinstance(uid, str) or not uid for uid in self.series_ids):
            raise ValueError("driver snapshot grant requires explicit nonblank text series IDs")
        if len(set(self.series_ids)) != len(self.series_ids):
            raise ValueError("driver snapshot grant contains duplicate series IDs")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "series_ids": list(self.series_ids)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriverSnapshotGrant:
        return cls(**{**data, "series_ids": tuple(data.get("series_ids", ()))})


@dataclass(frozen=True)
class DriverSourceContract:
    """Declared semantics; validated against a separately granted snapshot."""

    source_id: str
    grain: str
    units: str
    measure_kind: Literal["stock", "flow", "rate"]
    query_ref: str
    business_mechanism: str
    entity_col: str = "unique_id"
    time_col: str = "ds"
    value_col: str = "value"
    release_col: str = "available_at"
    vintage_col: str = "vintage"
    period_end_col: str = "period_end"
    units_col: str = "units"
    aggregation: Literal["none"] = "none"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"driver source contract requires nonblank {name}")
        if self.measure_kind not in {"stock", "flow", "rate"} or self.aggregation != "none":
            raise ValueError("driver snapshots must already have the declared stock/flow/rate grain; implicit aggregation is forbidden")
        columns = (self.entity_col, self.time_col, self.value_col, self.release_col,
                   self.vintage_col, self.period_end_col, self.units_col)
        if len(set(columns)) != len(columns):
            raise ValueError("driver source contract column roles must be distinct")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnownFutureRegressor:
    """Declared candidate model input with future-value and leakage audit metadata.

    Scaffold releases audit this contract by default. MLForecast can use passing
    model candidates only when opt-in regressor training is enabled.
    """

    name: str
    value_col: str | None = None
    availability: RegressorAvailability = "historical_only"
    mode: RegressorMode = "audit_only"
    future_file: str | None = None
    known_as_of_col: str = "known_as_of"
    source_system: str = ""
    source_query_file: str = ""
    owner: str = ""
    refresh_latency_days: int | None = None
    notes: str = ""
    source: DriverSourceContract | None = None
    historical_lags: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("known-future regressor name is required")
        if self.value_col is not None and not str(self.value_col).strip():
            raise ValueError("known-future regressor value_col cannot be blank")
        if self.future_file is not None and not str(self.future_file).strip():
            raise ValueError("known-future regressor future_file cannot be blank")
        if not str(self.known_as_of_col).strip():
            raise ValueError("known-future regressor known_as_of_col cannot be blank")
        if self.availability not in {"calendar", "contracted", "scheduled", "plan", "forecasted", "historical_only"}:
            raise ValueError("known-future regressor availability must be one of: calendar, contracted, scheduled, plan, forecasted, historical_only")
        if self.mode not in {"audit_only", "model_candidate"}:
            raise ValueError("known-future regressor mode must be 'audit_only' or 'model_candidate'")
        if self.refresh_latency_days is not None and self.refresh_latency_days < 0:
            raise ValueError("known-future regressor refresh_latency_days must be >= 0")
        if any(type(lag) is not int or lag < 1 for lag in self.historical_lags):
            raise ValueError("historical_lags must contain positive integer periods")
        if len(set(self.historical_lags)) != len(self.historical_lags):
            raise ValueError("historical_lags must not contain duplicates")
        if self.historical_lags and self.availability != "historical_only":
            raise ValueError("historical_lags apply only to historical-only signals")
        if self.source is not None:
            if not isinstance(self.source, DriverSourceContract):
                raise ValueError("regressor source must be a DriverSourceContract")
            if self.future_file is not None:
                raise ValueError("contracted drivers use only the granted snapshot, not a second future_file")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.source is None:
            data.pop("source")
        if self.historical_lags:
            data["historical_lags"] = list(self.historical_lags)
        else:
            data.pop("historical_lags")
        return data


_RESEARCH_BUDGET_DEFAULTS: dict[str, dict[str, int | float | None]] = {
    "time-boxed": {
        "max_iterations": 2,
        "max_variants_per_iteration": 3,
        "max_wall_clock_minutes": 15,
        "max_source_queries": 4,
        "max_compute_units": None,
    },
    "balanced": {
        "max_iterations": 5,
        "max_variants_per_iteration": 4,
        "max_wall_clock_minutes": 60,
        "max_source_queries": 12,
        "max_compute_units": None,
    },
    "deep": {
        "max_iterations": 10,
        "max_variants_per_iteration": 8,
        "max_wall_clock_minutes": 240,
        "max_source_queries": 30,
        "max_compute_units": None,
    },
}


@dataclass(frozen=True)
class ResearchBudget:
    """Hard bounds for context discovery and forecast improvement research."""

    profile: ResearchBudgetProfile = "balanced"
    max_iterations: int | None = None
    max_variants_per_iteration: int | None = None
    max_wall_clock_minutes: int | None = None
    max_source_queries: int | None = None
    max_compute_units: float | None = None

    def __post_init__(self) -> None:
        if self.profile not in {"time-boxed", "balanced", "deep", "custom"}:
            raise ValueError("research budget profile must be one of: time-boxed, balanced, deep, custom")
        if self.profile != "custom":
            defaults = _RESEARCH_BUDGET_DEFAULTS[self.profile]
            for field_name, default in defaults.items():
                if getattr(self, field_name) is None:
                    object.__setattr__(self, field_name, default)
        bounds = {
            "max_iterations": self.max_iterations,
            "max_variants_per_iteration": self.max_variants_per_iteration,
            "max_wall_clock_minutes": self.max_wall_clock_minutes,
            "max_source_queries": self.max_source_queries,
            "max_compute_units": self.max_compute_units,
        }
        if not any(value is not None for value in bounds.values()):
            raise ValueError("research budget requires at least one hard bound")
        invalid = [name for name, value in bounds.items() if value is not None and value <= 0]
        if invalid:
            raise ValueError(f"research budget bounds must be positive: {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyPolicy:
    """Claim gates for an accuracy-first forecast."""

    minimum_trust_score: int = 70
    require_full_horizon_validation: bool = True
    require_interval_evidence: bool = True
    require_hierarchy_coherence: bool = True
    require_driver_clearance: bool = True
    require_context_discovery: bool = True
    allow_directional_baseline: bool = True

    def __post_init__(self) -> None:
        if self.minimum_trust_score < 0 or self.minimum_trust_score > 100:
            raise ValueError("accuracy policy minimum_trust_score must be between 0 and 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionPolicy:
    """Chronological evidence required before recommending a new champion."""

    exact_cutoff_coverage: float = 1.0
    minimum_primary_metric_improvement: float = 0.02
    maximum_secondary_metric_regression: float = 0.01
    minimum_confirmation_cutoffs: int = 1
    require_untouched_confirmation: bool = True
    require_no_new_gate_failures: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        if self.exact_cutoff_coverage <= 0 or self.exact_cutoff_coverage > 1:
            raise ValueError("promotion policy exact_cutoff_coverage must be in (0, 1]")
        if self.minimum_primary_metric_improvement < 0:
            raise ValueError("promotion policy minimum_primary_metric_improvement must be >= 0")
        if self.maximum_secondary_metric_regression < 0:
            raise ValueError("promotion policy maximum_secondary_metric_regression must be >= 0")
        if self.minimum_confirmation_cutoffs < 1:
            raise ValueError("promotion policy minimum_confirmation_cutoffs must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextSource:
    """Auditable disposition for one potentially relevant connected source."""

    source_id: str
    kind: str
    status: ContextSourceStatus = "planned"
    provenance: str = ""
    query_ref: str = ""
    query_count: int = 0
    row_count: int | None = None
    known_as_of: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("context source_id is required")
        if not str(self.kind).strip():
            raise ValueError("context source kind is required")
        if self.status not in {"planned", "attempted", "available", "unavailable", "irrelevant", "opted_out"}:
            raise ValueError("context source status must be one of: planned, attempted, available, unavailable, irrelevant, opted_out")
        if self.query_count < 0:
            raise ValueError("context source query_count must be >= 0")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("context source row_count must be >= 0")
        if self.status == "available" and not (str(self.provenance).strip() or str(self.query_ref).strip()):
            raise ValueError("available context sources require provenance or query_ref")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateDriver:
    """Driver discovery evidence that remains separate from model admission."""

    name: str
    source_id: str
    status: CandidateDriverStatus = "discovered"
    timing: str = ""
    refresh_latency_days: int | None = None
    future_availability: DriverFutureAvailability = "unknown"
    leakage_verdict: LeakageVerdict = "unreviewed"
    business_rationale: str = ""
    evidence_ref: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("candidate driver name is required")
        if not str(self.source_id).strip():
            raise ValueError("candidate driver source_id is required")
        if self.status not in {"discovered", "audit_only", "eligible_for_experiment", "rejected"}:
            raise ValueError("candidate driver status must be one of: discovered, audit_only, eligible_for_experiment, rejected")
        if self.future_availability not in {"unknown", "known", "planned", "scenario", "forecasted", "historical_only", "unavailable"}:
            raise ValueError("candidate driver future_availability is invalid")
        if self.leakage_verdict not in {"unreviewed", "pass", "warning", "fail"}:
            raise ValueError("candidate driver leakage_verdict must be one of: unreviewed, pass, warning, fail")
        if self.refresh_latency_days is not None and self.refresh_latency_days < 0:
            raise ValueError("candidate driver refresh_latency_days must be >= 0")
        if self.status == "eligible_for_experiment" and self.leakage_verdict != "pass":
            raise ValueError("candidate drivers require leakage_verdict='pass' before becoming eligible_for_experiment")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalNeed:
    """A diagnosis-led information need before choosing a model experiment."""

    need_id: str
    signal_family: str
    question: str
    business_mechanism: str
    route_capabilities: tuple[str, ...] = ()
    priority: int = 3
    status: SignalNeedStatus = "open"
    evidence_refs: tuple[str, ...] = ()
    next_probe: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        required = {
            "need_id": self.need_id,
            "signal_family": self.signal_family,
            "question": self.question,
            "business_mechanism": self.business_mechanism,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"signal need requires non-blank fields: {missing}")
        if self.priority < 1 or self.priority > 5:
            raise ValueError("signal need priority must be between 1 and 5")
        if self.status not in {"open", "probing", "satisfied", "exhausted", "unavailable", "opted_out"}:
            raise ValueError("signal need status is invalid")
        if any(not str(capability).strip() for capability in self.route_capabilities):
            raise ValueError("signal need route_capabilities cannot contain blanks")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["route_capabilities"] = list(self.route_capabilities)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class SignalProbe:
    """One bounded, read-only source probe executed by the agent."""

    probe_id: str
    need_id: str
    source_id: str
    capability: str
    stage: SignalProbeStage
    status: SignalProbeStatus = "planned"
    query_count: int = 0
    query_ref: str = ""
    provenance: str = ""
    row_count: int | None = None
    known_as_of: str = ""
    result_summary: str = ""
    next_blocked_probe: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        required = {
            "probe_id": self.probe_id,
            "need_id": self.need_id,
            "source_id": self.source_id,
            "capability": self.capability,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"signal probe requires non-blank fields: {missing}")
        if self.stage not in {"schema", "count", "sample", "aggregate"}:
            raise ValueError("signal probe stage must be one of: schema, count, sample, aggregate")
        if self.status not in {"planned", "completed", "failed", "skipped"}:
            raise ValueError("signal probe status must be one of: planned, completed, failed, skipped")
        if self.query_count < 0:
            raise ValueError("signal probe query_count must be >= 0")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("signal probe row_count must be >= 0")
        if self.status == "completed" and not (str(self.query_ref).strip() or str(self.provenance).strip()):
            raise ValueError("completed signal probes require query_ref or provenance")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalContract:
    """Validated signal disposition and temporal/grain contract."""

    signal_id: str
    need_id: str
    probe_id: str
    name: str
    source_id: str
    disposition: SignalDisposition
    business_mechanism: str = ""
    entity_keys: tuple[str, ...] = ()
    time_key: str = ""
    grain: str = ""
    value_col: str = ""
    known_as_of_col: str = ""
    refresh_latency_days: int | None = None
    future_value_mode: DriverFutureAvailability = "unknown"
    coverage: float | None = None
    query_ref: str = ""
    provenance: str = ""
    leakage_verdict: LeakageVerdict = "unreviewed"
    target_proxy_verdict: LeakageVerdict = "unreviewed"
    next_blocked_probe: str = ""
    experiment_status: SignalExperimentStatus = "not_requested"
    experiment_reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        required = {
            "signal_id": self.signal_id,
            "need_id": self.need_id,
            "probe_id": self.probe_id,
            "name": self.name,
            "source_id": self.source_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"signal contract requires non-blank fields: {missing}")
        if self.disposition not in {"context", "scenario", "regressor_candidate", "reject"}:
            raise ValueError("signal contract disposition is invalid")
        if self.refresh_latency_days is not None and self.refresh_latency_days < 0:
            raise ValueError("signal contract refresh_latency_days must be >= 0")
        if self.coverage is not None and (self.coverage < 0 or self.coverage > 1):
            raise ValueError("signal contract coverage must be between 0 and 1")
        if self.future_value_mode not in {"unknown", "known", "planned", "scenario", "forecasted", "historical_only", "unavailable"}:
            raise ValueError("signal contract future_value_mode is invalid")
        if self.leakage_verdict not in {"unreviewed", "pass", "warning", "fail"}:
            raise ValueError("signal contract leakage_verdict is invalid")
        if self.target_proxy_verdict not in {"unreviewed", "pass", "warning", "fail"}:
            raise ValueError("signal contract target_proxy_verdict is invalid")
        if self.experiment_status not in {"not_requested", "queued", "tested", "blocked", "not_applicable"}:
            raise ValueError("signal contract experiment_status is invalid")
        if self.disposition == "regressor_candidate":
            regressor_required = {
                "business_mechanism": self.business_mechanism,
                "time_key": self.time_key,
                "grain": self.grain,
                "value_col": self.value_col,
                "known_as_of_col": self.known_as_of_col,
            }
            missing_regressor_fields = [
                name for name, value in regressor_required.items() if not str(value).strip()
            ]
            if not self.entity_keys:
                missing_regressor_fields.append("entity_keys")
            if missing_regressor_fields:
                raise ValueError(
                    "regressor-candidate signal contracts require: "
                    + ", ".join(missing_regressor_fields)
                )
            if self.leakage_verdict != "pass" or self.target_proxy_verdict != "pass":
                raise ValueError(
                    "regressor-candidate signal contracts require leakage_verdict='pass' "
                    "and target_proxy_verdict='pass'"
                )
            if self.future_value_mode in {"unknown", "unavailable"}:
                raise ValueError(
                    "regressor-candidate signal contracts require a usable future_value_mode"
                )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entity_keys"] = list(self.entity_keys)
        return data


@dataclass(frozen=True)
class ForecastContext:
    """Decision context, discovery provenance, and accuracy claim policies."""

    decision: str = ""
    audience: str = ""
    target_semantics: str = ""
    units: str = ""
    grain: str = ""
    constraints: tuple[str, ...] = ()
    requested_horizon: int | None = None
    refresh_cadence: str = ""
    known_breaks: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()
    known_events: tuple[str, ...] = ()
    hierarchy_required: bool = False
    plan_status: ReferenceAvailability = "unknown"
    budget_status: ReferenceAvailability = "unknown"
    prior_year_status: ReferenceAvailability = "unknown"
    benchmark_status: ReferenceAvailability = "unknown"
    source_discovery_enabled: bool = True
    sources: tuple[ContextSource, ...] = ()
    candidate_drivers: tuple[CandidateDriver, ...] = ()
    signal_needs: tuple[SignalNeed, ...] = ()
    signal_probes: tuple[SignalProbe, ...] = ()
    signal_contracts: tuple[SignalContract, ...] = ()
    research_budget: ResearchBudget = field(default_factory=ResearchBudget)
    accuracy_policy: AccuracyPolicy = field(default_factory=AccuracyPolicy)
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)

    def __post_init__(self) -> None:
        if self.requested_horizon is not None and self.requested_horizon < 1:
            raise ValueError("forecast context requested_horizon must be >= 1")
        allowed_reference_statuses = {"unknown", "available", "unavailable", "not_applicable"}
        statuses = {
            self.plan_status,
            self.budget_status,
            self.prior_year_status,
            self.benchmark_status,
        }
        if not statuses.issubset(allowed_reference_statuses):
            raise ValueError("forecast context reference statuses must be unknown, available, unavailable, or not_applicable")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("forecast context source_ids must be unique")
        driver_names = [driver.name for driver in self.candidate_drivers]
        if len(driver_names) != len(set(driver_names)):
            raise ValueError("forecast context candidate driver names must be unique")
        unknown_source_ids = sorted({driver.source_id for driver in self.candidate_drivers} - set(source_ids))
        if unknown_source_ids:
            raise ValueError(f"candidate drivers reference unknown context sources: {unknown_source_ids}")
        need_ids = [need.need_id for need in self.signal_needs]
        if len(need_ids) != len(set(need_ids)):
            raise ValueError("forecast context signal need_ids must be unique")
        probe_ids = [probe.probe_id for probe in self.signal_probes]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("forecast context signal probe_ids must be unique")
        contract_ids = [contract.signal_id for contract in self.signal_contracts]
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("forecast context signal_ids must be unique")
        unknown_probe_needs = sorted({probe.need_id for probe in self.signal_probes} - set(need_ids))
        if unknown_probe_needs:
            raise ValueError(f"signal probes reference unknown signal needs: {unknown_probe_needs}")
        unknown_probe_sources = sorted({probe.source_id for probe in self.signal_probes} - set(source_ids))
        if unknown_probe_sources:
            raise ValueError(f"signal probes reference unknown context sources: {unknown_probe_sources}")
        unknown_contract_needs = sorted({contract.need_id for contract in self.signal_contracts} - set(need_ids))
        if unknown_contract_needs:
            raise ValueError(f"signal contracts reference unknown signal needs: {unknown_contract_needs}")
        unknown_contract_probes = sorted({contract.probe_id for contract in self.signal_contracts} - set(probe_ids))
        if unknown_contract_probes:
            raise ValueError(f"signal contracts reference unknown signal probes: {unknown_contract_probes}")
        unknown_contract_sources = sorted({contract.source_id for contract in self.signal_contracts} - set(source_ids))
        if unknown_contract_sources:
            raise ValueError(f"signal contracts reference unknown context sources: {unknown_contract_sources}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field_name in ("constraints", "known_breaks", "exclusions", "adjustments", "known_events"):
            data[field_name] = list(getattr(self, field_name))
        data["sources"] = [source.to_dict() for source in self.sources]
        data["candidate_drivers"] = [driver.to_dict() for driver in self.candidate_drivers]
        data["signal_needs"] = [need.to_dict() for need in self.signal_needs]
        data["signal_probes"] = [probe.to_dict() for probe in self.signal_probes]
        data["signal_contracts"] = [contract.to_dict() for contract in self.signal_contracts]
        data["research_budget"] = self.research_budget.to_dict()
        data["accuracy_policy"] = self.accuracy_policy.to_dict()
        data["promotion_policy"] = self.promotion_policy.to_dict()
        return data

    def source_query_count(self) -> int:
        """Return one conservative count when source and probe receipts overlap."""

        source_total = sum(source.query_count for source in self.sources)
        probe_total = sum(probe.query_count for probe in self.signal_probes)
        return max(source_total, probe_total)


@dataclass(frozen=True)
class TransformSpec:
    """Finance target preparation before modeling."""

    target: TargetTransform = "none"
    normalization_factor_col: str | None = None
    normalization_label: str = ""

    def __post_init__(self) -> None:
        if self.target not in {"none", "log", "log1p"}:
            raise ValueError("target transform must be one of: none, log, log1p")
        if self.normalization_factor_col is not None and not str(self.normalization_factor_col).strip():
            raise ValueError("normalization_factor_col cannot be blank")

    @property
    def enabled(self) -> bool:
        return self.target != "none" or self.normalization_factor_col is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureRecipeSpec:
    """FINN-inspired feature recipe metadata for audited native experiments."""

    fiscal_year_start: int = 1
    fourier_periods: tuple[int, ...] = ()
    lag_periods: tuple[int, ...] = ()
    rolling_window_periods: tuple[int, ...] = ()
    recipes_to_run: tuple[str, ...] = ()
    pca: bool | None = None
    feature_selection: bool = False
    weekly_to_daily: bool = True
    date_features_as_dummies: bool = False
    mlforecast_strategy: Literal["recursive", "direct"] = "recursive"
    horizon_features: tuple[tuple[int, tuple[str, ...]], ...] = ()
    lookup_lag_columns: tuple[str, ...] = ()
    pooled_rolling_window: int | None = None
    pooled_partition_columns: tuple[str, ...] = ()
    pooled_series_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.date_features_as_dummies, bool):
            raise ValueError("feature_recipe.date_features_as_dummies must be a boolean")
        if self.mlforecast_strategy not in {"recursive", "direct"}:
            raise ValueError("feature_recipe.mlforecast_strategy must be recursive or direct")
        features = self.horizon_features.items() if isinstance(self.horizon_features, dict) else self.horizon_features
        normalized = []
        for horizon, columns in features:
            if isinstance(horizon, str) and re.fullmatch(r"[1-9][0-9]*", horizon):
                horizon = int(horizon)
            if type(horizon) is not int or horizon < 1:
                raise ValueError("horizon_features keys must be positive integers")
            if not isinstance(columns, (list, tuple)) or not columns:
                raise ValueError("horizon_features values must be nonempty column arrays")
            normalized.append((horizon, _feature_columns(columns)))
        if len({h for h, _ in normalized}) != len(normalized):
            raise ValueError("horizon_features cannot repeat a horizon")
        object.__setattr__(self, "horizon_features", tuple(sorted(normalized)))
        if self.horizon_features and self.mlforecast_strategy != "direct":
            raise ValueError("horizon_features require the explicit direct forecasting strategy")
        for name in ("lookup_lag_columns", "pooled_partition_columns", "pooled_series_ids"):
            object.__setattr__(self, name, _feature_columns(getattr(self, name)))
        if self.lookup_lag_columns and self.mlforecast_strategy != "direct":
            raise ValueError("LookupLag requires direct strategy so lookups never use generated future targets")
        if self.pooled_rolling_window is not None:
            if type(self.pooled_rolling_window) is not int or self.pooled_rolling_window < 1:
                raise ValueError("pooled_rolling_window must be a positive integer")
            if not self.pooled_partition_columns and len(self.pooled_series_ids) < 2:
                raise ValueError("Pooled rolling features require a declared multi-series cohort or partition columns")
        elif self.pooled_partition_columns or self.pooled_series_ids:
            raise ValueError("Declare pooled_rolling_window when enabling pooled or partitioned transforms")
        if self.fiscal_year_start < 1 or self.fiscal_year_start > 12:
            raise ValueError("feature_recipe.fiscal_year_start must be between 1 and 12")
        object.__setattr__(self, "fourier_periods", _positive_int_tuple(self.fourier_periods, "feature_recipe.fourier_periods"))
        object.__setattr__(self, "lag_periods", _positive_int_tuple(self.lag_periods, "feature_recipe.lag_periods"))
        object.__setattr__(
            self,
            "rolling_window_periods",
            _positive_int_tuple(self.rolling_window_periods, "feature_recipe.rolling_window_periods"),
        )
        object.__setattr__(self, "recipes_to_run", tuple(str(item).strip() for item in self.recipes_to_run if str(item).strip()))

    @property
    def enabled(self) -> bool:
        return bool(self.fourier_periods or self.lag_periods or self.rolling_window_periods or self.recipes_to_run or self.pca or self.feature_selection or self.date_features_as_dummies or self.mlforecast_strategy == "direct" or self.lookup_lag_columns or self.pooled_rolling_window)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fourier_periods"] = list(self.fourier_periods)
        data["lag_periods"] = list(self.lag_periods)
        data["rolling_window_periods"] = list(self.rolling_window_periods)
        data["recipes_to_run"] = list(self.recipes_to_run)
        data["horizon_features"] = {str(h): list(columns) for h, columns in self.horizon_features}
        for name in ("lookup_lag_columns", "pooled_partition_columns", "pooled_series_ids"):
            data[name] = list(getattr(self, name))
        return data


def _feature_columns(values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Feature columns and series IDs must be arrays of nonblank strings")
    if len(set(values)) != len(values):
        raise ValueError("Feature columns and series IDs cannot contain duplicates")
    return tuple(values)


@dataclass(frozen=True)
class CleaningSpec:
    """FINN-inspired data cleaning and forecast-bound controls."""

    clean_missing_values: bool = True
    clean_outliers: bool = False
    negative_forecast: bool = True
    combo_cleanup_date: str | None = None

    def __post_init__(self) -> None:
        if self.combo_cleanup_date is not None and not str(self.combo_cleanup_date).strip():
            raise ValueError("cleaning.combo_cleanup_date cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleSpec:
    """Audited ensemble policies inspired by FINN's model averaging workflow."""

    policies: tuple[EnsemblePolicy, ...] = ("legacy_weighted",)
    max_models: int = 3
    scoring: EnsembleScoringMode = "prior_only"
    deployment: EnsembleDeploymentMode = "full_backtest"

    def __post_init__(self) -> None:
        allowed = {"legacy_weighted", "top_k_average", "family_diverse_average"}
        policies = tuple(dict.fromkeys(str(policy) for policy in self.policies))
        invalid = sorted(set(policies) - allowed)
        if invalid:
            raise ValueError(f"ensemble.policies contains unknown value(s): {invalid}; expected one of {sorted(allowed)}")
        if self.max_models < 1:
            raise ValueError("ensemble.max_models must be >= 1")
        if self.scoring != "prior_only":
            raise ValueError("ensemble.scoring must be 'prior_only'")
        if self.deployment not in {"full_backtest", "last_cutoff"}:
            raise ValueError("ensemble.deployment must be 'full_backtest' or 'last_cutoff'")
        object.__setattr__(self, "policies", policies or ("legacy_weighted",))

    @property
    def advisory_policies(self) -> tuple[EnsemblePolicy, ...]:
        return tuple(policy for policy in self.policies if policy != "legacy_weighted")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policies"] = list(self.policies)
        return data


@dataclass(frozen=True)
class ParallelSpec:
    """Local/remote execution intent metadata; no cloud orchestration is implied."""

    processing: ParallelProcessingMode = "none"
    inner_parallel: bool = False
    num_cores: int | None = None

    def __post_init__(self) -> None:
        if self.processing not in {"none", "local_machine", "spark"}:
            raise ValueError("parallel.processing must be one of: none, local_machine, spark")
        if self.num_cores is not None and self.num_cores < 1:
            raise ValueError("parallel.num_cores must be >= 1 when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChallengerSpec:
    """External challenger engine run beside the canonical pipeline in an advisory lane.

    Challengers never mutate the canonical ``forecast.csv``; they contribute
    compare/score artifacts under ``<run>/<source_id>`` with explicit cutoff-contract evidence classes.
    """

    engine: str = "finn"
    enabled: bool = True
    on_error: ChallengerOnError = "skip"
    models: tuple[str, ...] = ()
    back_test_scenarios: int | None = None
    back_test_spacing: int | None = None
    forecast_approach: str = "bottoms_up"
    run_ensemble_models: bool = False
    feature_selection: bool = False
    rscript: str = "Rscript"
    timeout_seconds: int = 3600
    seed: int = 123
    source_id: str = ""
    model_name: str = ""
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        engine = str(self.engine).strip().lower()
        if not engine:
            raise ValueError("challenger engine is required")
        object.__setattr__(self, "engine", engine)
        if self.on_error not in {"skip", "fail"}:
            raise ValueError("challenger on_error must be 'skip' or 'fail'")
        models = tuple(dict.fromkeys(str(model).strip() for model in self.models if str(model).strip()))
        object.__setattr__(self, "models", models)
        if self.back_test_scenarios is not None and self.back_test_scenarios < 1:
            raise ValueError("challenger back_test_scenarios must be >= 1 when provided")
        if self.back_test_spacing is not None and self.back_test_spacing < 1:
            raise ValueError("challenger back_test_spacing must be >= 1 when provided")
        if self.forecast_approach not in {"bottoms_up", "standard_hierarchy", "grouped_hierarchy"}:
            raise ValueError("challenger forecast_approach must be one of: bottoms_up, standard_hierarchy, grouped_hierarchy")
        if self.timeout_seconds < 1:
            raise ValueError("challenger timeout_seconds must be >= 1")
        if not str(self.rscript).strip():
            raise ValueError("challenger rscript cannot be blank")
        source_id = str(self.source_id).strip() or engine
        object.__setattr__(self, "source_id", source_id)
        model_name = str(self.model_name).strip() or engine.upper()
        object.__setattr__(self, "model_name", model_name)
        extra: list[tuple[str, str]] = []
        for item in self.extra:
            key, value = item
            key = str(key).strip()
            if not key:
                raise ValueError("challenger extra keys cannot be blank")
            extra.append((key, str(value)))
        object.__setattr__(self, "extra", tuple(extra))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "enabled": self.enabled,
            "on_error": self.on_error,
            "models": list(self.models),
            "back_test_scenarios": self.back_test_scenarios,
            "back_test_spacing": self.back_test_spacing,
            "forecast_approach": self.forecast_approach,
            "run_ensemble_models": self.run_ensemble_models,
            "feature_selection": self.feature_selection,
            "rscript": self.rscript,
            "timeout_seconds": self.timeout_seconds,
            "seed": self.seed,
            "source_id": self.source_id,
            "model_name": self.model_name,
            "extra": {key: value for key, value in self.extra},
        }


@dataclass(frozen=True)
class CustomModelSpec:
    """Opt-in executable model challenger with a strict forecast output contract."""

    name: str
    callable_path: str | None = None
    callable: Callable[..., pd.DataFrame] | None = field(default=None, repr=False, compare=False)
    script_path: str | None = None
    timeout_seconds: int = 120
    extra_args: tuple[str, ...] = ()
    source_id: str = "custom"
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("custom model name is required")
        invocation_count = sum(
            [
                self.callable_path is not None,
                self.callable is not None,
                self.script_path is not None,
            ]
        )
        if invocation_count != 1:
            raise ValueError("custom model requires exactly one of callable_path, callable, or script_path")
        if self.callable_path is not None and not str(self.callable_path).strip():
            raise ValueError("custom model callable_path cannot be blank")
        if self.script_path is not None and not str(self.script_path).strip():
            raise ValueError("custom model script_path cannot be blank")
        if self.timeout_seconds < 1:
            raise ValueError("custom model timeout_seconds must be >= 1")
        if not str(self.source_id).strip():
            raise ValueError("custom model source_id cannot be blank")
        if self.model_name == "WeightedEnsemble" or "-lo-" in self.model_name or "-hi-" in self.model_name:
            raise ValueError("custom model name conflicts with reserved forecast column naming")
        blank_args = [arg for arg in self.extra_args if not str(arg).strip()]
        if blank_args:
            raise ValueError("custom model extra_args cannot contain blank values")

    @property
    def invocation_type(self) -> str:
        if self.script_path is not None:
            return "script"
        if self.callable_path is not None:
            return "callable_path"
        return "callable_object"

    @property
    def model_name(self) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(self.name).strip()).strip("_")
        if not cleaned:
            cleaned = "model"
        return cleaned if cleaned.startswith("Custom_") else f"Custom_{cleaned}"

    def to_dict(self) -> dict[str, Any]:
        callable_ref = ""
        if self.callable is not None:
            callable_ref = f"{getattr(self.callable, '__module__', '')}.{getattr(self.callable, '__qualname__', getattr(self.callable, '__name__', 'callable'))}".strip(".")
        return {
            "name": self.name,
            "model_name": self.model_name,
            "invocation_type": self.invocation_type,
            "callable_path": self.callable_path,
            "callable_object_ref": callable_ref,
            "script_path": self.script_path,
            "timeout_seconds": self.timeout_seconds,
            "extra_args": list(self.extra_args),
            "source_id": self.source_id,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelConfiguration:
    """Explicit native settings; empty values retain the original model recipes."""

    seasonal_periods: tuple[int, ...] = ()
    mfles_seasonality: Literal["auto", "additive", "multiplicative"] = "auto"
    mfles_fourier_order: int | None = None
    automfles_metric: Literal["smape", "mse", "mae"] = "smape"
    automfles_test_size: int | None = None
    automfles_windows: int = 2
    automfles_step_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seasonal_periods, (tuple, list)):
            raise ValueError("seasonal_periods must be an array of integer periods")
        periods = tuple(self.seasonal_periods)
        if len(periods) > 3 or any(type(p) is not int or p < 2 for p in periods):
            raise ValueError("seasonal_periods requires at most three integer periods >= 2")
        if len(set(periods)) != len(periods):
            raise ValueError("seasonal_periods must be unique, in order of modeling importance")
        object.__setattr__(self, "seasonal_periods", periods)
        if self.mfles_seasonality not in {"auto", "additive", "multiplicative"}:
            raise ValueError("mfles_seasonality must be auto, additive, or multiplicative")
        if self.mfles_fourier_order is not None and (
            type(self.mfles_fourier_order) is not int or self.mfles_fourier_order < 1
        ):
            raise ValueError("mfles_fourier_order must be a positive integer")
        if self.automfles_metric not in {"smape", "mse", "mae"}:
            raise ValueError("automfles_metric must be smape, mse, or mae; monthly landing is an outer objective")
        for name in ("automfles_test_size", "automfles_windows", "automfles_step_size"):
            value = getattr(self, name)
            if value is None and name != "automfles_windows":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "seasonal_periods": list(self.seasonal_periods)}


@dataclass(frozen=True)
class EvaluationPolicy:
    strategy: Literal["rolling-origin", "legacy", "none"] = "rolling-origin"
    max_windows: int = 24
    max_calibration_windows: int = 20
    step_size: int | None = 1
    minimum_training_observations: int = 8
    minimum_seasonal_cycles: int = 2
    reference_minimum_training_observations: int = 4
    training_window: int | None = None
    evidence_directory: str | None = None
    known_target_exposure: Literal["unrecorded", "exposed"] = "unrecorded"
    cache_directory: str | None = None
    cache_max_bytes: int = 268435456

    def __post_init__(self) -> None:
        if self.strategy not in {"rolling-origin", "legacy", "none"}:
            raise ValueError("evaluation strategy must be rolling-origin, legacy, or none")
        if self.known_target_exposure not in {"unrecorded", "exposed"}:
            raise ValueError("known_target_exposure must be unrecorded or exposed; callers cannot certify an unexposed history")
        if self.cache_directory is not None and (
            not isinstance(self.cache_directory, str) or not self.cache_directory.strip()
        ):
            raise ValueError("cache_directory must be a nonblank local path")
        if type(self.cache_max_bytes) is not int or not 1024 <= self.cache_max_bytes <= 4 * 1024**3:
            raise ValueError("cache_max_bytes must be between 1024 and 4294967296")
        if self.evidence_directory is not None and (
            not isinstance(self.evidence_directory, str) or not self.evidence_directory.strip()
        ):
            raise ValueError("evidence_directory must be a nonblank local path")
        for name in ("max_windows", "max_calibration_windows", "minimum_training_observations", "minimum_seasonal_cycles",
                     "reference_minimum_training_observations", "step_size", "training_window"):
            value = getattr(self, name)
            if value is None and name in {"step_size", "training_window"}:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"evaluation {name} must be a positive integer")
        if self.reference_minimum_training_observations > self.minimum_training_observations:
            raise ValueError("reference minimum cannot exceed the ordinary training minimum")
        if self.training_window is not None and self.training_window < self.reference_minimum_training_observations:
            raise ValueError("training_window must support the reference training minimum")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionPolicy:
    """Within-run guard; cross-run admission remains governed by PromotionPolicy."""

    naive_guard_min_improvement: float = 0.05
    objective: Literal["daily_rmse", "monthly_landing"] = "daily_rmse"
    clip_negative: bool = False
    configuration_guard: Literal["exclude", "warn"] = "exclude"

    def __post_init__(self) -> None:
        if self.objective not in {"daily_rmse", "monthly_landing"}:
            raise ValueError("selection objective must be daily_rmse or monthly_landing")
        if type(self.clip_negative) is not bool:
            raise ValueError("selection clip_negative must be a boolean")
        if not isinstance(self.configuration_guard, str) or self.configuration_guard not in {"exclude", "warn"}:
            raise ValueError("configuration_guard must be exclude or warn")
        if not math.isfinite(self.naive_guard_min_improvement) or not 0 <= self.naive_guard_min_improvement < 1:
            raise ValueError("naive_guard_min_improvement must be finite and in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contract(self) -> dict[str, Any]:
        if self.objective == "monthly_landing":
            return {
                "version": 2, "objective": "monthly_landing", "scale": "original_target_units",
                "primary": "landing_mape_pct", "secondary": ["next28_mape_pct", "daily_wmape_pct"],
                "calendar": "daily flows; current and next calendar month closing within observed history and forecast horizon",
                "aggregation": "actuals through origin plus remaining daily forecasts; equal window mean absolute percentage error",
                "daily_aggregation": "clip each origin prediction before per-date averaging",
                "clip_negative": self.clip_negative, "missing_window_policy": "whole_candidate_unavailable",
                "zero_denominator": "undefined_not_zero",
                "naive_guard_min_improvement": self.naive_guard_min_improvement,
                "configuration_guard": self.configuration_guard,
                "exact_ties": "reference_guard_then_stable_model_name_not_superiority",
            }
        return {
            "version": 1,
            "objective": "statistical_squared_error_not_business_loss",
            "scale": "original_target_units",
            "primary": "rmse",
            "secondary": ["mae", "absolute_signed_bias"],
            "horizon_weighting": "equal_observation",
            "references": ["Naive", "SeasonalNaive"],
            "naive_guard_min_improvement": self.naive_guard_min_improvement,
            "configuration_guard": self.configuration_guard,
            "exact_ties": "reference_guard_then_stable_model_name_not_superiority",
            "undefined_primary": "mae_when_no_finite_rmse_in_tournament_otherwise_directional_fallback",
            "scaled_error_reference": "training_through_earliest_supplied_cutoff",
            "undefined_scaled_errors": "unavailable_not_zero",
            "panel_primary_order": ["equal_series_rmsse", "equal_series_mase", "rmse_single_series_only"],
            "panel_missing_metric": "whole_declared_cohort_unavailable_no_silent_series_exclusion",
        }


@dataclass(frozen=True)
class IntervalCalibrationSpec:
    """Opt-in residual calibration; never changes the point-model recipe."""

    method: Literal["native", "split_conformal", "nexcp"] = "native"
    rho: float = 0.99
    max_windows: int = 64
    aggregate_totals: bool = False

    def __post_init__(self) -> None:
        if type(self.aggregate_totals) is not bool:
            raise ValueError("aggregate_totals must be a boolean")
        if self.method not in {"native", "split_conformal", "nexcp"}:
            raise ValueError("interval calibration method must be native, split_conformal, or nexcp")
        if isinstance(self.rho, bool) or not isinstance(self.rho, (int, float)) or not 0 < self.rho <= 1:
            raise ValueError("interval calibration rho must be in (0, 1]")
        if type(self.max_windows) is not int or not 20 <= self.max_windows <= 256:
            raise ValueError("interval calibration max_windows must be an integer between 20 and 256")


@dataclass(frozen=True)
class CalendarSpec:
    """Deterministic daily features, not judgment multipliers or inferred geography."""

    enabled: bool = False
    holiday_calendar: Literal["none", "us_federal", "custom"] = "none"
    custom_dates: tuple[str, ...] = ()
    known_as_of: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or self.holiday_calendar not in {"none", "us_federal", "custom"}:
            raise ValueError("calendar requires a boolean enabled and none, us_federal, or custom holiday_calendar")
        if not isinstance(self.custom_dates, (tuple, list)):
            raise ValueError("calendar custom_dates must be a list of ISO dates")
        for value in self.custom_dates:
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("calendar custom_dates must contain YYYY-MM-DD dates")
            pd.Timestamp(value)
        object.__setattr__(self, "custom_dates", tuple(sorted(set(self.custom_dates))))
        if bool(self.custom_dates) != (self.holiday_calendar == "custom"):
            raise ValueError("custom dates require the custom calendar, with at least one date")
        if self.known_as_of is not None:
            if not isinstance(self.known_as_of, str):
                raise ValueError("calendar known_as_of must be an ISO timestamp string")
            stamp = pd.Timestamp(self.known_as_of)
            if pd.isna(stamp):
                raise ValueError("calendar known_as_of must be a valid timestamp")
        if self.holiday_calendar == "custom" and self.known_as_of is None:
            raise ValueError("custom calendar dates require known_as_of for historical availability")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnomalySpec:
    """Versioned, training-only robust seasonal treatment with conservative guards."""

    mode: Literal["off", "detect", "adjust"] = "detect"
    version: int = 1
    lookback: int = 182
    threshold: float = 6.0
    minimum_relative_deviation: float = 0.25
    maximum_adjusted_fraction: float = 0.03

    def __post_init__(self) -> None:
        if self.mode not in {"off", "detect", "adjust"} or type(self.version) is not int or self.version != 1:
            raise ValueError("anomaly requires mode off, detect, or adjust and version 1")
        if type(self.lookback) is not int or not 56 <= self.lookback <= 1096:
            raise ValueError("anomaly lookback must be an integer between 56 and 1096 observations")
        for name, lower, upper in (("threshold", 3.5, 20), ("minimum_relative_deviation", 0.05, 2),
                                   ("maximum_adjusted_fraction", 0.001, 0.05)):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"anomaly {name} must be between {lower} and {upper}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductionPolicy:
    """Versioned statistical release contract, not business approval or a guarantee."""

    version: int = 1
    max_error_pct: float = 10.0
    max_abs_bias_pct: float = 5.0
    coverage_tolerance: float = 0.05
    minimum_selection_windows: int = 3
    calibration_windows: int = 20
    minimum_assessment_windows: int = 5
    ensemble_members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("production version must be 1")
        for name, lower, upper in (("max_error_pct", 0, 100), ("max_abs_bias_pct", 0, 100),
                                   ("coverage_tolerance", 0, 0.1)):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"production {name} must be finite and in [{lower}, {upper}]")
        for name, lower in (("minimum_selection_windows", 3), ("calibration_windows", 20),
                            ("minimum_assessment_windows", 5)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= 64:
                raise ValueError(f"production {name} must be an integer in [{lower}, 64]")
        if not isinstance(self.ensemble_members, (tuple, list)) or any(
            not isinstance(member, str) for member in self.ensemble_members
        ):
            raise ValueError("production ensemble_members must be a list or tuple of canonical model names")
        members = tuple(self.ensemble_members)
        if (members and len(members) < 2) or len(members) != len(set(members)):
            raise ValueError("production ensemble_members must contain at least two distinct native models")
        if any(member not in MODEL_ALLOWLIST_CANDIDATES for member in members):
            raise ValueError("production ensemble_members must use canonical native model names")
        object.__setattr__(self, "ensemble_members", members)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastSpec:
    """User-facing forecast settings with safe defaults."""

    horizon: int = 6
    freq: str | None = None
    season_length: int | None = None
    levels: tuple[int, ...] = (80, 95)
    model_policy: ModelPolicy = "standard"
    fill_method: FillMethod = "ffill"
    id_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"
    unit_label: str | None = None
    model_allowlist: tuple[str, ...] = ()
    hierarchy: tuple[str, ...] = ()
    events: tuple[DriverEvent, ...] = field(default_factory=tuple)
    regressors: tuple[KnownFutureRegressor, ...] = field(default_factory=tuple)
    driver_snapshots: tuple[DriverSnapshotGrant, ...] = ()
    custom_models: tuple[CustomModelSpec, ...] = field(default_factory=tuple)
    challengers: tuple[ChallengerSpec, ...] = field(default_factory=tuple)
    transform: TransformSpec = field(default_factory=TransformSpec)
    feature_recipe: FeatureRecipeSpec = field(default_factory=FeatureRecipeSpec)
    cleaning: CleaningSpec = field(default_factory=CleaningSpec)
    ensemble: EnsembleSpec = field(default_factory=EnsembleSpec)
    parallel: ParallelSpec = field(default_factory=ParallelSpec)
    hierarchy_reconciliation: HierarchyReconciliationMethod = "none"
    train_known_future_regressors: bool = False
    mlforecast_feature_policy: MLForecastFeaturePolicy = "basic"
    require_backtest: bool = False
    strict_cv_horizon: bool = True
    weighted_ensemble: bool = True
    paired_validation: bool = True
    verbose: bool = True
    context: ForecastContext | None = None
    run_mode: RunMode = "accuracy-first"
    policy_version: int = 1
    selection_policy: SelectionPolicy = field(default_factory=SelectionPolicy)
    model_configuration: ModelConfiguration = field(default_factory=ModelConfiguration)
    calendar: CalendarSpec = field(default_factory=CalendarSpec)
    anomaly: AnomalySpec = field(default_factory=AnomalySpec)
    evaluation: EvaluationPolicy = field(default_factory=EvaluationPolicy)
    interval_calibration: IntervalCalibrationSpec = field(default_factory=IntervalCalibrationSpec)
    production: ProductionPolicy | None = None

    def __post_init__(self) -> None:
        if self.production is not None and not isinstance(self.production, ProductionPolicy):
            raise ValueError("production must be a ProductionPolicy or None")
        if self.cleaning.clean_outliers and self.anomaly.mode == "detect":
            object.__setattr__(self, "anomaly", replace(self.anomaly, mode="adjust"))
        if self.anomaly.mode == "adjust" and self.interval_calibration.method == "native":
            object.__setattr__(self, "interval_calibration", replace(self.interval_calibration, method="split_conformal"))
        source_ids = [grant.source_id for grant in self.driver_snapshots]
        if len(source_ids) != len(set(source_ids)) or len(source_ids) > 8:
            raise ValueError("driver snapshots require unique source IDs and at most eight grants")
        if self.run_mode not in {"accuracy-first", "quick", "legacy"}:
            raise ValueError("run_mode must be accuracy-first, quick, or legacy")
        expected_version = 0 if self.run_mode == "legacy" else 1
        if type(self.policy_version) is not int or self.policy_version != expected_version:
            raise ValueError(f"run_mode {self.run_mode!r} requires policy_version {expected_version}")
        if self.run_mode == "legacy" and self.evaluation == EvaluationPolicy():
            object.__setattr__(self, "evaluation", replace(self.evaluation, strategy="legacy"))
        elif self.run_mode == "quick" and self.evaluation == EvaluationPolicy():
            object.__setattr__(self, "evaluation", EvaluationPolicy(
                max_windows=2, step_size=None, minimum_training_observations=4, minimum_seasonal_cycles=1,
            ))
        if self.context is None and self.run_mode != "legacy":
            object.__setattr__(self, "context", ForecastContext(research_budget=ResearchBudget(
                profile="time-boxed", max_iterations=2, max_variants_per_iteration=1,
            )))
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if any(h > self.horizon for h, _ in self.feature_recipe.horizon_features):
            raise ValueError("horizon_features cannot exceed the forecast horizon")
        if self.season_length is not None and (type(self.season_length) is not int or self.season_length < 1):
            raise ValueError("season_length must be an integer >= 1")
        invalid_levels = [level for level in self.levels if level <= 0 or level >= 100]
        if invalid_levels:
            raise ValueError(f"interval levels must be between 0 and 100, got {invalid_levels}")
        if self.unit_label is not None and not str(self.unit_label).strip():
            raise ValueError("unit_label cannot be blank")
        model_policy = "light" if self.model_policy == "auto" else self.model_policy
        if model_policy not in {"standard", "light", "baseline", "statsforecast", "mlforecast", "all"}:
            raise ValueError("model_policy must be one of: standard, light, baseline, statsforecast, mlforecast, all")
        object.__setattr__(self, "model_policy", model_policy)
        canonical_models, unknown_models = canonicalize_model_allowlist(tuple(self.model_allowlist))
        if unknown_models:
            unknown = ", ".join(repr(model) for model in unknown_models)
            valid = ", ".join(sorted(MODEL_ALLOWLIST_CANDIDATES))
            raise ValueError(f"unknown model_allowlist entr{'y' if len(unknown_models) == 1 else 'ies'}: {unknown}. Known models: {valid}")
        object.__setattr__(self, "model_allowlist", canonical_models)
        if self.hierarchy_reconciliation not in {"none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"}:
            raise ValueError("hierarchy_reconciliation must be one of: none, bottom_up, top_down, both, mint_ols, mint_wls_struct")
        if self.mlforecast_feature_policy not in {"basic", "rolling"}:
            raise ValueError("mlforecast_feature_policy must be one of: basic, rolling")
        if len(self.custom_models) > 1:
            raise ValueError("custom model v1 supports at most one custom model per run")
        custom_names = [custom.model_name for custom in self.custom_models]
        if len(custom_names) != len(set(custom_names)):
            raise ValueError("custom model names must be unique")
        challenger_engines = [challenger.engine for challenger in self.challengers]
        if len(challenger_engines) != len(set(challenger_engines)):
            raise ValueError("challenger engines must be unique")
        challenger_source_ids = [challenger.source_id for challenger in self.challengers]
        if len(challenger_source_ids) != len(set(challenger_source_ids)):
            raise ValueError("challenger source_ids must be unique")
        if self.context is not None and self.context.requested_horizon not in {None, self.horizon}:
            raise ValueError("forecast context requested_horizon must match ForecastSpec.horizon")

    @property
    def accuracy_first(self) -> bool:
        return self.run_mode == "accuracy-first"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["levels"] = list(self.levels)
        data["model_allowlist"] = list(self.model_allowlist)
        data["hierarchy"] = list(self.hierarchy)
        data["events"] = [event.to_dict() for event in self.events]
        data["regressors"] = [regressor.to_dict() for regressor in self.regressors]
        if self.driver_snapshots:
            data["driver_snapshots"] = [grant.to_dict() for grant in self.driver_snapshots]
        else:
            data.pop("driver_snapshots", None)
        if self.custom_models:
            data["custom_models"] = [custom.to_dict() for custom in self.custom_models]
        else:
            data.pop("custom_models", None)
        if self.challengers:
            data["challengers"] = [challenger.to_dict() for challenger in self.challengers]
        else:
            data.pop("challengers", None)
        data["transform"] = self.transform.to_dict()
        data["feature_recipe"] = self.feature_recipe.to_dict()
        data["cleaning"] = self.cleaning.to_dict()
        data["ensemble"] = self.ensemble.to_dict()
        data["parallel"] = self.parallel.to_dict()
        data["model_configuration"] = self.model_configuration.to_dict()
        if self.context is None:
            data.pop("context", None)
        else:
            data["context"] = self.context.to_dict()
        return data


def _positive_int_tuple(values: Sequence[Any], field_name: str) -> tuple[int, ...]:
    out = tuple(int(value) for value in values)
    invalid = [value for value in out if value < 1]
    if invalid:
        raise ValueError(f"{field_name} values must be positive integers, got {invalid}")
    return out


def forecast_spec_from_dict(data: dict[str, Any]) -> ForecastSpec:
    """Rehydrate a persisted manifest spec into a ForecastSpec."""

    transform_data = data.get("transform") if isinstance(data.get("transform"), dict) else {}
    feature_recipe_data = data.get("feature_recipe") if isinstance(data.get("feature_recipe"), dict) else {}
    cleaning_data = data.get("cleaning") if isinstance(data.get("cleaning"), dict) else {}
    ensemble_data = data.get("ensemble") if isinstance(data.get("ensemble"), dict) else {}
    parallel_data = data.get("parallel") if isinstance(data.get("parallel"), dict) else {}
    events = tuple(_driver_event_from_dict(item) for item in data.get("events", []) if isinstance(item, dict))
    regressors = tuple(_known_future_regressor_from_dict(item) for item in data.get("regressors", []) if isinstance(item, dict))
    custom_models = tuple(_custom_model_from_dict(item) for item in data.get("custom_models", []) if isinstance(item, dict))
    challengers = tuple(_challenger_from_dict(item) for item in data.get("challengers", []) if isinstance(item, dict))
    context_data = data.get("context") if isinstance(data.get("context"), dict) else None
    run_mode = data.get("run_mode", "legacy")
    legacy = run_mode == "legacy"
    if not legacy and "policy_version" not in data:
        raise ValueError("A saved non-legacy spec requires policy_version; use resolve_forecast_spec for a new request")
    selection_data = data.get("selection_policy", {})
    if not isinstance(selection_data, dict):
        raise ValueError("selection_policy must be an object")
    selection_data = {"configuration_guard": "warn", **selection_data}
    evaluation_data = data.get("evaluation", {"strategy": "legacy"} if legacy else {})
    if not isinstance(evaluation_data, dict):
        raise ValueError("evaluation must be an object")
    interval_data = data.get("interval_calibration", {})
    if not isinstance(interval_data, dict):
        raise ValueError("interval_calibration must be an object")
    production_data = data.get("production")
    if production_data is not None and (
        not isinstance(production_data, dict) or set(production_data) - set(ProductionPolicy.__dataclass_fields__)
    ):
        raise ValueError("production must be an object containing only declared production settings")
    configuration_data = data.get("model_configuration", {})
    if not isinstance(configuration_data, dict):
        raise ValueError("model_configuration must be an object")
    unknown_configuration = set(configuration_data) - set(ModelConfiguration.__dataclass_fields__)
    if unknown_configuration:
        raise ValueError(f"Unknown model_configuration settings: {sorted(unknown_configuration)}")
    policies = {}
    for name, kind, default in (("calendar", CalendarSpec, {}), ("anomaly", AnomalySpec, {"mode": "off"})):
        payload = data.get(name, default)
        if not isinstance(payload, dict) or set(payload) - set(kind.__dataclass_fields__):
            raise ValueError(f"{name} must be an object containing only declared {name} settings")
        policies[name] = kind(**payload)
    return ForecastSpec(
        horizon=int(data.get("horizon", 12 if legacy else 6)),
        freq=data.get("freq"),
        season_length=data.get("season_length"),
        levels=tuple(int(level) for level in data.get("levels", (80, 95))),
        model_policy=data.get("model_policy", "light" if legacy else "standard"),
        fill_method=data.get("fill_method", "ffill"),
        id_col=data.get("id_col", "unique_id"),
        time_col=data.get("time_col", "ds"),
        target_col=data.get("target_col", "y"),
        unit_label=data.get("unit_label"),
        model_allowlist=tuple(data.get("model_allowlist", ())),
        hierarchy=tuple(data.get("hierarchy", ())),
        events=events,
        regressors=regressors,
        driver_snapshots=tuple(DriverSnapshotGrant.from_dict(item) for item in data.get("driver_snapshots", ())),
        custom_models=custom_models,
        challengers=challengers,
        transform=TransformSpec(
            target=transform_data.get("target", "none"),
            normalization_factor_col=transform_data.get("normalization_factor_col"),
            normalization_label=transform_data.get("normalization_label", ""),
        ),
        feature_recipe=FeatureRecipeSpec(
            fiscal_year_start=int(feature_recipe_data.get("fiscal_year_start", 1)),
            fourier_periods=tuple(feature_recipe_data.get("fourier_periods", ())),
            lag_periods=tuple(feature_recipe_data.get("lag_periods", ())),
            rolling_window_periods=tuple(feature_recipe_data.get("rolling_window_periods", ())),
            recipes_to_run=tuple(feature_recipe_data.get("recipes_to_run", ())),
            pca=feature_recipe_data.get("pca"),
            feature_selection=bool(feature_recipe_data.get("feature_selection", False)),
            weekly_to_daily=bool(feature_recipe_data.get("weekly_to_daily", True)),
            date_features_as_dummies=feature_recipe_data.get("date_features_as_dummies", False),
            mlforecast_strategy=feature_recipe_data.get("mlforecast_strategy", "recursive"),
            horizon_features=feature_recipe_data.get("horizon_features", ()),
            lookup_lag_columns=feature_recipe_data.get("lookup_lag_columns", ()),
            pooled_rolling_window=feature_recipe_data.get("pooled_rolling_window"),
            pooled_partition_columns=feature_recipe_data.get("pooled_partition_columns", ()),
            pooled_series_ids=feature_recipe_data.get("pooled_series_ids", ()),
        ),
        cleaning=CleaningSpec(
            clean_missing_values=bool(cleaning_data.get("clean_missing_values", True)),
            clean_outliers=bool(cleaning_data.get("clean_outliers", False)),
            negative_forecast=bool(cleaning_data.get("negative_forecast", True)),
            combo_cleanup_date=str(cleaning_data["combo_cleanup_date"]) if cleaning_data.get("combo_cleanup_date") not in (None, "") else None,
        ),
        ensemble=EnsembleSpec(
            policies=tuple(ensemble_data.get("policies", ("legacy_weighted",))),
            max_models=int(ensemble_data.get("max_models", 3)),
            scoring=ensemble_data.get("scoring", "prior_only"),
            deployment=ensemble_data.get("deployment", "full_backtest"),
        ),
        parallel=ParallelSpec(
            processing=parallel_data.get("processing", "none"),
            inner_parallel=bool(parallel_data.get("inner_parallel", False)),
            num_cores=_optional_int(parallel_data.get("num_cores")),
        ),
        hierarchy_reconciliation=data.get("hierarchy_reconciliation", "none"),
        train_known_future_regressors=bool(data.get("train_known_future_regressors", False)),
        mlforecast_feature_policy=data.get("mlforecast_feature_policy", "basic"),
        require_backtest=bool(data.get("require_backtest", False)),
        strict_cv_horizon=bool(data.get("strict_cv_horizon", not legacy)),
        weighted_ensemble=bool(data.get("weighted_ensemble", True)),
        paired_validation=bool(data.get("paired_validation", not legacy)),
        verbose=bool(data.get("verbose", True)),
        context=forecast_context_from_dict(context_data) if context_data is not None else None,
        run_mode=run_mode,
        policy_version=data.get("policy_version", 0 if legacy else 1),
        selection_policy=SelectionPolicy(**selection_data),
        model_configuration=ModelConfiguration(**configuration_data),
        calendar=policies["calendar"],
        anomaly=policies["anomaly"],
        evaluation=EvaluationPolicy(**evaluation_data),
        interval_calibration=IntervalCalibrationSpec(**interval_data),
        production=ProductionPolicy(**production_data) if production_data is not None else None,
    )


def forecast_context_from_dict(data: dict[str, Any]) -> ForecastContext:
    """Rehydrate a persisted accuracy-first context contract."""

    budget_data = data.get("research_budget") if isinstance(data.get("research_budget"), dict) else {}
    accuracy_data = data.get("accuracy_policy") if isinstance(data.get("accuracy_policy"), dict) else {}
    promotion_data = data.get("promotion_policy") if isinstance(data.get("promotion_policy"), dict) else {}
    sources = tuple(_context_source_from_dict(item) for item in data.get("sources", ()) if isinstance(item, dict))
    drivers = tuple(_candidate_driver_from_dict(item) for item in data.get("candidate_drivers", ()) if isinstance(item, dict))
    signal_needs = tuple(_signal_need_from_dict(item) for item in data.get("signal_needs", ()) if isinstance(item, dict))
    signal_probes = tuple(_signal_probe_from_dict(item) for item in data.get("signal_probes", ()) if isinstance(item, dict))
    signal_contracts = tuple(
        _signal_contract_from_dict(item)
        for item in data.get("signal_contracts", ())
        if isinstance(item, dict)
    )
    return ForecastContext(
        decision=str(data.get("decision", "")),
        audience=str(data.get("audience", "")),
        target_semantics=str(data.get("target_semantics", "")),
        units=str(data.get("units", "")),
        grain=str(data.get("grain", "")),
        constraints=_string_tuple(data.get("constraints", ())),
        requested_horizon=_optional_int(data.get("requested_horizon")),
        refresh_cadence=str(data.get("refresh_cadence", "")),
        known_breaks=_string_tuple(data.get("known_breaks", ())),
        exclusions=_string_tuple(data.get("exclusions", ())),
        adjustments=_string_tuple(data.get("adjustments", ())),
        known_events=_string_tuple(data.get("known_events", ())),
        hierarchy_required=_coerce_bool(data.get("hierarchy_required"), default=False),
        plan_status=data.get("plan_status", "unknown"),
        budget_status=data.get("budget_status", "unknown"),
        prior_year_status=data.get("prior_year_status", "unknown"),
        benchmark_status=data.get("benchmark_status", "unknown"),
        source_discovery_enabled=_coerce_bool(data.get("source_discovery_enabled"), default=True),
        sources=sources,
        candidate_drivers=drivers,
        signal_needs=signal_needs,
        signal_probes=signal_probes,
        signal_contracts=signal_contracts,
        research_budget=ResearchBudget(
            profile=budget_data.get("profile", "balanced"),
            max_iterations=_optional_int(budget_data.get("max_iterations")),
            max_variants_per_iteration=_optional_int(budget_data.get("max_variants_per_iteration")),
            max_wall_clock_minutes=_optional_int(budget_data.get("max_wall_clock_minutes")),
            max_source_queries=_optional_int(budget_data.get("max_source_queries")),
            max_compute_units=_optional_float(budget_data.get("max_compute_units")),
        ),
        accuracy_policy=AccuracyPolicy(
            minimum_trust_score=int(accuracy_data.get("minimum_trust_score", 70)),
            require_full_horizon_validation=_coerce_bool(accuracy_data.get("require_full_horizon_validation"), default=True),
            require_interval_evidence=_coerce_bool(accuracy_data.get("require_interval_evidence"), default=True),
            require_hierarchy_coherence=_coerce_bool(accuracy_data.get("require_hierarchy_coherence"), default=True),
            require_driver_clearance=_coerce_bool(accuracy_data.get("require_driver_clearance"), default=True),
            require_context_discovery=_coerce_bool(accuracy_data.get("require_context_discovery"), default=True),
            allow_directional_baseline=_coerce_bool(accuracy_data.get("allow_directional_baseline"), default=True),
        ),
        promotion_policy=PromotionPolicy(
            exact_cutoff_coverage=float(promotion_data.get("exact_cutoff_coverage", 1.0)),
            minimum_primary_metric_improvement=float(promotion_data.get("minimum_primary_metric_improvement", 0.02)),
            maximum_secondary_metric_regression=float(promotion_data.get("maximum_secondary_metric_regression", 0.01)),
            minimum_confirmation_cutoffs=int(promotion_data.get("minimum_confirmation_cutoffs", 1)),
            require_untouched_confirmation=_coerce_bool(promotion_data.get("require_untouched_confirmation"), default=True),
            require_no_new_gate_failures=_coerce_bool(promotion_data.get("require_no_new_gate_failures"), default=True),
            human_approval_required=_coerce_bool(promotion_data.get("human_approval_required"), default=True),
        ),
    )


def load_forecast_context(path: str | Path) -> ForecastContext:
    """Load a ForecastContext from JSON or YAML."""

    context_path = Path(path)
    if context_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(context_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("forecast context file must contain an object")
    nested = payload.get("forecast_context")
    if isinstance(nested, dict):
        payload = nested
    return forecast_context_from_dict(payload)


def _context_source_from_dict(data: dict[str, Any]) -> ContextSource:
    return ContextSource(
        source_id=str(data.get("source_id", "")),
        kind=str(data.get("kind", "")),
        status=data.get("status", "planned"),
        provenance=str(data.get("provenance", "")),
        query_ref=str(data.get("query_ref", "")),
        query_count=_optional_int(data.get("query_count")) or 0,
        row_count=_optional_int(data.get("row_count")),
        known_as_of=str(data.get("known_as_of", "")),
        notes=str(data.get("notes", "")),
    )


def _candidate_driver_from_dict(data: dict[str, Any]) -> CandidateDriver:
    return CandidateDriver(
        name=str(data.get("name", "")),
        source_id=str(data.get("source_id", "")),
        status=data.get("status", "discovered"),
        timing=str(data.get("timing", "")),
        refresh_latency_days=_optional_int(data.get("refresh_latency_days")),
        future_availability=data.get("future_availability", "unknown"),
        leakage_verdict=data.get("leakage_verdict", "unreviewed"),
        business_rationale=str(data.get("business_rationale", "")),
        evidence_ref=str(data.get("evidence_ref", "")),
        notes=str(data.get("notes", "")),
    )


def _signal_need_from_dict(data: dict[str, Any]) -> SignalNeed:
    return SignalNeed(
        need_id=str(data.get("need_id", "")),
        signal_family=str(data.get("signal_family", "")),
        question=str(data.get("question", "")),
        business_mechanism=str(data.get("business_mechanism", "")),
        route_capabilities=_string_tuple(data.get("route_capabilities", ())),
        priority=int(data.get("priority", 3)),
        status=data.get("status", "open"),
        evidence_refs=_string_tuple(data.get("evidence_refs", ())),
        next_probe=str(data.get("next_probe", "")),
        notes=str(data.get("notes", "")),
    )


def _signal_probe_from_dict(data: dict[str, Any]) -> SignalProbe:
    return SignalProbe(
        probe_id=str(data.get("probe_id", "")),
        need_id=str(data.get("need_id", "")),
        source_id=str(data.get("source_id", "")),
        capability=str(data.get("capability", "")),
        stage=data.get("stage", "schema"),
        status=data.get("status", "planned"),
        query_count=_optional_int(data.get("query_count")) or 0,
        query_ref=str(data.get("query_ref", "")),
        provenance=str(data.get("provenance", "")),
        row_count=_optional_int(data.get("row_count")),
        known_as_of=str(data.get("known_as_of", "")),
        result_summary=str(data.get("result_summary", "")),
        next_blocked_probe=str(data.get("next_blocked_probe", "")),
        notes=str(data.get("notes", "")),
    )


def _signal_contract_from_dict(data: dict[str, Any]) -> SignalContract:
    return SignalContract(
        signal_id=str(data.get("signal_id", "")),
        need_id=str(data.get("need_id", "")),
        probe_id=str(data.get("probe_id", "")),
        name=str(data.get("name", "")),
        source_id=str(data.get("source_id", "")),
        disposition=data.get("disposition", "reject"),
        business_mechanism=str(data.get("business_mechanism", "")),
        entity_keys=_string_tuple(data.get("entity_keys", ())),
        time_key=str(data.get("time_key", "")),
        grain=str(data.get("grain", "")),
        value_col=str(data.get("value_col", "")),
        known_as_of_col=str(data.get("known_as_of_col", "")),
        refresh_latency_days=_optional_int(data.get("refresh_latency_days")),
        future_value_mode=data.get("future_value_mode", "unknown"),
        coverage=_optional_float(data.get("coverage")),
        query_ref=str(data.get("query_ref", "")),
        provenance=str(data.get("provenance", "")),
        leakage_verdict=data.get("leakage_verdict", "unreviewed"),
        target_proxy_verdict=data.get("target_proxy_verdict", "unreviewed"),
        next_blocked_probe=str(data.get("next_blocked_probe", "")),
        experiment_status=data.get("experiment_status", "not_requested"),
        experiment_reason=str(data.get("experiment_reason", "")),
        notes=str(data.get("notes", "")),
    )


def _driver_event_from_dict(data: dict[str, Any]) -> DriverEvent:
    affected = data.get("affected_unique_ids", ())
    if isinstance(affected, str):
        affected_tuple = tuple(part.strip() for part in affected.replace(";", ",").split(",") if part.strip())
    else:
        affected_tuple = tuple(str(item) for item in affected or ())
    return DriverEvent(
        name=str(data.get("name", "")),
        start=str(data.get("start", "")),
        end=str(data["end"]) if data.get("end") not in (None, "") else None,
        effect=data.get("effect", "multiplicative"),
        magnitude=float(data.get("magnitude", 0.0)),
        affected_unique_ids=affected_tuple,
        confidence=float(data.get("confidence", 1.0)),
        notes=str(data.get("notes", "")),
    )


def _known_future_regressor_from_dict(data: dict[str, Any]) -> KnownFutureRegressor:
    return KnownFutureRegressor(
        name=str(data.get("name", "")),
        value_col=str(data["value_col"]) if data.get("value_col") not in (None, "") else None,
        availability=data.get("availability", "historical_only"),
        mode=data.get("mode", "audit_only"),
        future_file=str(data["future_file"]) if data.get("future_file") not in (None, "") else None,
        known_as_of_col=str(data.get("known_as_of_col", "known_as_of")),
        source_system=str(data.get("source_system", "")),
        source_query_file=str(data.get("source_query_file", "")),
        owner=str(data.get("owner", "")),
        refresh_latency_days=_optional_int(data.get("refresh_latency_days")),
        notes=str(data.get("notes", "")),
        source=DriverSourceContract(**data["source"]) if data.get("source") is not None else None,
        historical_lags=tuple(data.get("historical_lags", ())),
    )


def _challenger_from_dict(data: dict[str, Any]) -> ChallengerSpec:
    extra_data = data.get("extra")
    if isinstance(extra_data, dict):
        extra = tuple((str(key), str(value)) for key, value in extra_data.items())
    else:
        extra = tuple((str(key), str(value)) for key, value in (extra_data or ()))
    return ChallengerSpec(
        engine=str(data.get("engine", "finn")),
        enabled=bool(data.get("enabled", True)),
        on_error=data.get("on_error", "skip"),
        models=tuple(str(model) for model in data.get("models", ())),
        back_test_scenarios=_optional_int(data.get("back_test_scenarios")),
        back_test_spacing=_optional_int(data.get("back_test_spacing")),
        forecast_approach=str(data.get("forecast_approach", "bottoms_up")),
        run_ensemble_models=bool(data.get("run_ensemble_models", False)),
        feature_selection=bool(data.get("feature_selection", False)),
        rscript=str(data.get("rscript", "Rscript")),
        timeout_seconds=int(data.get("timeout_seconds", 3600)),
        seed=int(data.get("seed", 123)),
        source_id=str(data.get("source_id", "")),
        model_name=str(data.get("model_name", "")),
        extra=extra,
    )


def _custom_model_from_dict(data: dict[str, Any]) -> CustomModelSpec:
    callable_path = data.get("callable_path")
    script_path = data.get("script_path")
    if not callable_path and not script_path:
        raise ValueError("cannot refresh a manifest custom model without callable_path or script_path")
    return CustomModelSpec(
        name=str(data.get("name") or data.get("model_name") or "custom_model"),
        callable_path=str(callable_path) if callable_path else None,
        script_path=str(script_path) if script_path else None,
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        extra_args=tuple(str(arg) for arg in data.get("extra_args", ())),
        source_id=str(data.get("source_id", "custom")),
        notes=str(data.get("notes", "")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"expected a boolean value, got {value!r}")
    return bool(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace(";", ",").split(",") if part.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class SeriesProfile:
    unique_id: str
    rows: int
    start: str | None
    end: str | None
    missing_timestamps: int
    null_y: int
    zero_y: int
    negative_y: int
    readiness: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataProfile:
    rows: int
    series_count: int
    freq: str
    season_length: int
    start: str | None
    end: str | None
    min_obs_per_series: int
    max_obs_per_series: int
    duplicate_rows: int
    missing_timestamps: int
    null_y: int
    zero_y: int
    negative_y: int
    data_freshness: str | None
    warnings: tuple[str, ...] = ()
    series: tuple[SeriesProfile, ...] = ()
    configuration_diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["series"] = [profile.to_dict() for profile in self.series]
        return data


@dataclass
class ForecastRun:
    """Forecast result plus audit artifacts."""

    history: pd.DataFrame
    forecast: pd.DataFrame
    all_models: pd.DataFrame
    model_selection: pd.DataFrame
    backtest_metrics: pd.DataFrame
    profile: DataProfile
    spec: ForecastSpec
    backtest_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_explainability: pd.DataFrame = field(default_factory=pd.DataFrame)
    transformation_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_availability_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_model_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_model_cv_delta: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_invocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    unreconciled_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy_reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy_reconciliation_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    engine: str = "baseline"
    model_policy_resolution: dict[str, Any] = field(default_factory=dict)
    observed_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    def manifest(self) -> dict[str, Any]:
        outputs = {
            **({
                "production_readiness": "production_readiness.json",
                "production_backtest": "audit/production_backtest.csv",
                "production_decisions": "audit/production_decisions.json",
                "production_calibration": "audit/production_calibration.json",
                **({"production_release": "release/manifest.json"} if
                   self.model_policy_resolution.get("production", {}).get("readiness", {}).get("release_allowed") else {}),
            } if self.spec.production is not None else {}),
            "history": "appendix/history.csv",
            "forecast": "forecast.csv",
            "forecast_long": "appendix/forecast_long.csv",
            "backtest_long": "appendix/backtest_long.csv",
            "cutoff_contract": "appendix/cutoff_contract.csv",
            "series_summary": "appendix/series_summary.csv",
            "series_features": "appendix/series_features.csv",
            "borrowed_strength_advisor": "appendix/borrowed_strength_advisor.csv",
            "model_audit": "appendix/model_audit.csv",
            "model_win_rates": "appendix/model_win_rates.csv",
            "model_tradeoff_scores": "appendix/model_tradeoff_scores.csv",
            "model_pareto_frontier": "appendix/model_pareto_frontier.csv",
            "feature_selection_receipts": "appendix/feature_selection_receipts.csv",
            "ensemble_policy_receipts": "appendix/ensemble_policy_receipts.csv",
            "ensemble_comparisons": "appendix/ensemble_comparisons.json",
            "candidate_evidence": "audit/candidate_evidence.json",
            "candidate_fits": "audit/candidate_fits.csv",
            "model_window_metrics": "appendix/model_window_metrics.csv",
            "residual_diagnostics": "appendix/residual_diagnostics.csv",
            "residual_tests": "appendix/residual_tests.csv",
            "interval_diagnostics": "appendix/interval_diagnostics.csv",
            "trust_summary": "appendix/trust_summary.csv",
            "model_explainability": "appendix/model_explainability.csv",
            "all_models": "audit/all_models.csv",
            "model_selection": "audit/model_selection.csv",
            "backtest_metrics": "audit/backtest_metrics.csv",
            "backtest_predictions": "audit/backtest_predictions.csv",
            "backtest_windows": "audit/backtest_windows.csv",
            "model_weights": "audit/model_weights.csv",
            "target_transform_audit": "audit/target_transform_audit.csv",
            "seasonality_profile": "audit/seasonality_profile.csv",
            "seasonality_summary": "audit/seasonality_summary.csv",
            "seasonality_diagnostics": "audit/seasonality_diagnostics.csv",
            "seasonality_decomposition": "audit/seasonality_decomposition.csv",
            "interpretation": "audit/interpretation.json",
            "interpretation_markdown": "interpretation.md",
            "output_open_first": "OPEN_ME_FIRST.html",
            "output_index": "output/index.html",
            "output_workbook": "output/forecast_review.xlsx",
            "output_forecast": "output/forecast_for_review.csv",
            "output_decision_summary": "output/decision_summary.csv",
            "output_model_leaderboard": "output/appendix/model_leaderboard.csv",
            "output_forecast_brief": "output/appendix/forecast_brief.csv",
            "output_artifact_guide": "output/appendix/artifact_guide.csv",
            "html_report": "report.html",
            "html_report_base64": "report_base64.txt",
            "streamlit_app": "streamlit_app.py",
            "streamlit_requirements": "streamlit_requirements.txt",
            "streamlit_launcher_ps1": "run_streamlit.ps1",
            "streamlit_launcher_cmd": "run_streamlit.cmd",
            "control_pane_state": "control_pane_state.json",
            "training_progress": "audit/training_progress.jsonl",
            "best_practice_receipts": "appendix/best_practice_receipts.csv",
            "run_receipt": "appendix/run_receipt.json",
            "run_receipt_markdown": "appendix/run_receipt.md",
            "validation_receipt": "appendix/validation_receipt.csv",
            "validation_receipt_json": "appendix/validation_receipt.json",
            "diagnostics": "diagnostics.json",
            "diagnostics_markdown": "diagnostics.md",
            "llm_context": "llm_context.json",
            "model_card": "model_card.md",
            "workbook": "forecast.xlsx",
        }
        if "hierarchy_depth" in self.forecast.columns:
            outputs["hierarchy_rollup"] = "appendix/hierarchy_rollup.csv"
            outputs["hierarchy_coherence"] = "appendix/hierarchy_coherence.csv"
            outputs["hierarchy_contribution"] = "appendix/hierarchy_contribution.csv"
            outputs["hierarchy_backtest_comparison"] = "audit/hierarchy_backtest_comparison.csv"
            if not self.unreconciled_forecast.empty:
                outputs["hierarchy_unreconciled_forecast"] = "audit/hierarchy_unreconciled_forecast.csv"
                outputs["hierarchy_coherence_pre"] = "audit/hierarchy_coherence_pre.csv"
                outputs["hierarchy_coherence_post"] = "audit/hierarchy_coherence_post.csv"
            if not self.hierarchy_reconciliation.empty:
                outputs["hierarchy_reconciliation"] = "appendix/hierarchy_reconciliation.csv"
            if not self.hierarchy_reconciliation_comparison.empty:
                outputs["hierarchy_reconciliation_comparison"] = "appendix/hierarchy_reconciliation_comparison.csv"
        if not self.observed_history.empty:
            outputs["observed_history"] = "appendix/observed_history.csv"
        if self.model_policy_resolution.get("configuration"):
            outputs["configuration"] = "audit/configuration.json"
            for name in ("series", "seasonality", "phases", "models"):
                outputs[f"configuration_{name}"] = f"audit/configuration_{name}.csv"
        if self.model_policy_resolution.get("anomalies"):
            outputs["anomalies"] = "audit/anomalies.json"
            outputs["anomaly_points"] = "audit/anomaly_points.csv"
            outputs["anomaly_series"] = "audit/anomaly_series.csv"
        if any("native_fitted" in fit for fit in self.model_policy_resolution.get("candidate_evidence", {}).get("fits", [])):
            outputs["native_fitted_points"] = "audit/native_fitted_points.csv"
            outputs["native_fitted_summary"] = "audit/native_fitted_summary.csv"
        if self.spec.interval_calibration.method != "native":
            outputs["interval_comparison"] = "appendix/interval_comparison.csv"
            outputs["interval_calibration"] = "audit/interval_calibration.json"
            outputs["interval_support"] = "appendix/interval_support.csv"
        if self.spec.selection_policy.objective == "monthly_landing":
            outputs["finance_scorecard"] = "appendix/finance_scorecard.csv"
            outputs["finance_windows"] = "appendix/finance_windows.csv"
        if self.spec.interval_calibration.aggregate_totals:
            outputs["aggregate_intervals"] = "appendix/aggregate_intervals.csv"
            outputs["aggregate_interval_support"] = "audit/aggregate_interval_support.json"
        if self.spec.events:
            outputs["scenario_assumptions"] = "appendix/scenario_assumptions.csv"
            outputs["scenario_forecast"] = "appendix/scenario_forecast.csv"
            outputs["driver_experiment_summary"] = "appendix/driver_experiment_summary.csv"
        if self.spec.context is not None:
            outputs["context_receipt"] = "appendix/context_receipt.json"
            outputs["context_receipt_markdown"] = "appendix/context_receipt.md"
            outputs["research_budget"] = "appendix/research_budget.json"
            outputs["accuracy_gate"] = "appendix/accuracy_gate.json"
            outputs["accuracy_gate_markdown"] = "appendix/accuracy_gate.md"
            if (
                self.spec.context.signal_needs
                or self.spec.context.signal_probes
                or self.spec.context.signal_contracts
            ):
                outputs["signal_needs"] = "appendix/signal_needs.json"
                outputs["signal_probe_ledger"] = "appendix/signal_probe_ledger.jsonl"
                outputs["signal_contracts"] = "appendix/signal_contracts.json"
        if self.spec.regressors or not self.driver_availability_audit.empty:
            outputs["known_future_regressors"] = "appendix/known_future_regressors.csv"
            outputs["driver_availability_audit"] = "appendix/driver_availability_audit.csv"
            outputs["driver_experiment_summary"] = "appendix/driver_experiment_summary.csv"
        if self.spec.ensemble.advisory_policies:
            outputs["ensemble_backtest"] = "appendix/ensemble_backtest.csv"
            outputs["ensemble_selection"] = "appendix/ensemble_selection.csv"
            outputs["ensemble_forecast"] = "appendix/ensemble_forecast.csv"
        if not self.driver_model_features.empty:
            outputs["driver_model_features"] = "appendix/driver_model_features.csv"
        if not self.driver_model_cv_delta.empty:
            outputs["driver_model_cv_delta"] = "appendix/driver_model_cv_delta.csv"
        if self.spec.custom_models or not self.custom_model_contracts.empty or not self.custom_model_invocations.empty:
            outputs["custom_model_contracts"] = "appendix/custom_model_contracts.csv"
            outputs["custom_model_invocations"] = "audit/custom_model_invocations.csv"
        return {
            "engine": self.engine,
            "spec": self.spec.to_dict(),
            "scoring_contract": self.spec.selection_policy.contract(),
            "promotion_policy": (self.spec.context.promotion_policy if self.spec.context else PromotionPolicy()).to_dict(),
            "effective_levels": self.effective_levels(),
            "profile": self.profile.to_dict(),
            "reproducibility": _reproducibility_metadata(self),
            "warnings": self.warnings,
            "model_policy_resolution": self.model_policy_resolution,
            "best_practice_receipts": self.best_practice_receipts(),
            "outputs": outputs,
        }

    def effective_levels(self) -> list[int]:
        levels = []
        for column in self.forecast.columns:
            if column.startswith("yhat_lo_"):
                levels.append(int(column.rsplit("_", 1)[-1]))
        return sorted(set(levels))

    def best_practice_receipts(self) -> list[dict[str, str]]:
        from nixtla_scaffold.best_practices import best_practice_receipts

        return best_practice_receipts(self)

    def diagnostics(self) -> dict[str, Any]:
        from nixtla_scaffold.diagnostics import build_run_diagnostics
        from nixtla_scaffold.outputs import cached_output_tables

        with cached_output_tables(self):
            return build_run_diagnostics(self)

    def interpretation(self) -> dict[str, Any]:
        from nixtla_scaffold.interpretation import build_interpretation_payload

        return build_interpretation_payload(self)

    def explanation(self) -> str:
        from nixtla_scaffold.explain import build_model_card

        return build_model_card(self)

    def to_directory(self, output_dir: str | Path, *, release: bool = True) -> Path:
        from nixtla_scaffold.outputs import write_run

        return write_run(self, output_dir, release=release)

    def to_excel(self, output_path: str | Path) -> Path:
        from nixtla_scaffold.outputs import write_workbook

        return write_workbook(self, output_path)


def _reproducibility_metadata(run: ForecastRun) -> dict[str, Any]:
    return {
        "data_hash_sha256": _frame_hash(run.history, columns=["unique_id", "ds", "y"]),
        "observed_data_hash_sha256": (
            _frame_hash(run.observed_history, columns=["unique_id", "ds", "y"])
            if not run.observed_history.empty else None
        ),
        "forecast_origin": run.profile.end,
        "frequency": run.profile.freq,
        "season_length": run.profile.season_length,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(
            [
                "nixtla-scaffold",
                "pandas",
                "numpy",
                "scipy",
                "scikit-learn",
                "coreforecast",
                "statsforecast",
                "utilsforecast",
                "mlforecast",
                "lightgbm",
                "hierarchicalforecast",
                "streamlit",
                "plotly",
            ]
        ),
        "git_sha": _git_sha(),
        "runtime_source": _runtime_source_identity(),
        "thread_environment": {
            key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def _runtime_source_identity(package_root: Path | None = None) -> dict[str, Any]:
    root = package_root or Path(__file__).resolve().parent
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.py"))
    }
    lock = root / "runtime.lock"
    if not lock.is_file():
        lock = root.parent.parent / "uv.lock"
    return {
        "kind": "observed_source_snapshot_not_execution_certification",
        "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "files": hashes,
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
    }


def _frame_hash(frame: pd.DataFrame, *, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return ""
    canonical = frame[available].copy()
    if "ds" in canonical.columns:
        canonical["ds"] = pd.to_datetime(canonical["ds"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    canonical = canonical.sort_values(available).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip()
    return sha or None
