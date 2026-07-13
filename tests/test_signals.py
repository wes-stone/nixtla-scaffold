from __future__ import annotations

import json

import pandas as pd
import pytest

from nixtla_scaffold import (
    ContextSource,
    ForecastContext,
    ForecastSpec,
    KnownFutureRegressor,
    ResearchBudget,
    SignalContract,
    SignalNeed,
    SignalProbe,
    forecast_context_from_dict,
    run_optimizer,
    signal_discovery_summary,
)
from nixtla_scaffold.mcp_contracts import signal_routes_for_family
from nixtla_scaffold.research import (
    _generate_hypotheses,
    _signal_experiment_dispositions,
)


def _source(*, query_count: int = 1) -> ContextSource:
    return ContextSource(
        source_id="target",
        kind="csv",
        status="available",
        provenance="input.csv",
        query_count=query_count,
    )


def _need(*, status: str = "satisfied") -> SignalNeed:
    return SignalNeed(
        need_id="driver-need",
        signal_family="operational_driver",
        question="Could planned seats lead revenue?",
        business_mechanism="Seats create future billable capacity.",
        route_capabilities=("bounded_time_series_aggregate",),
        status=status,
    )


def _probe(*, query_count: int = 1) -> SignalProbe:
    return SignalProbe(
        probe_id="driver-probe",
        need_id="driver-need",
        source_id="target",
        capability="bounded_time_series_aggregate",
        stage="aggregate",
        status="completed",
        query_count=query_count,
        query_ref="queries/seats.sql",
    )


def _contract() -> SignalContract:
    return SignalContract(
        signal_id="seat-plan",
        need_id="driver-need",
        probe_id="driver-probe",
        name="Seats plan",
        source_id="target",
        disposition="regressor_candidate",
        business_mechanism="Committed seats create billable revenue.",
        entity_keys=("unique_id",),
        time_key="ds",
        grain="monthly",
        value_col="seats_plan",
        known_as_of_col="known_as_of",
        future_value_mode="planned",
        coverage=1.0,
        query_ref="queries/seats.sql",
        leakage_verdict="pass",
        target_proxy_verdict="pass",
    )


def test_signal_contract_round_trip_and_capability_routing() -> None:
    context = ForecastContext(
        sources=(_source(query_count=2),),
        signal_needs=(_need(),),
        signal_probes=(_probe(query_count=2),),
        signal_contracts=(_contract(),),
    )

    restored = forecast_context_from_dict(context.to_dict())
    summary = signal_discovery_summary(restored)
    routes = signal_routes_for_family("operational_driver")

    assert restored == context
    assert summary["status"] == "complete"
    assert summary["recorded_source_queries"] == 2
    assert routes[0]["route_id"] == "operational-driver"
    assert "telemetry_store" in routes[0]["source_families"]


def test_regressor_candidate_requires_temporal_and_leakage_contract() -> None:
    with pytest.raises(ValueError, match="known_as_of_col"):
        SignalContract(
            signal_id="bad",
            need_id="driver-need",
            probe_id="driver-probe",
            name="Bad signal",
            source_id="target",
            disposition="regressor_candidate",
            business_mechanism="Plausible",
            entity_keys=("unique_id",),
            time_key="ds",
            grain="monthly",
            value_col="bad",
            future_value_mode="planned",
            leakage_verdict="pass",
            target_proxy_verdict="pass",
        )


def test_admitted_signal_queues_attributable_hypothesis_or_records_blocker() -> None:
    context = ForecastContext(
        sources=(_source(),),
        signal_needs=(_need(),),
        signal_probes=(_probe(),),
        signal_contracts=(_contract(),),
    )
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": range(8),
            "seats_plan": range(10, 18),
        }
    )
    spec = ForecastSpec(
        horizon=2,
        model_policy="baseline",
        context=context,
        regressors=(
            KnownFutureRegressor(
                name="Seats plan",
                value_col="seats_plan",
                availability="plan",
                mode="model_candidate",
                future_file="future_seats.csv",
            ),
        ),
    )

    dispositions = _signal_experiment_dispositions(spec)
    hypotheses = _generate_hypotheses(
        frame,
        spec,
        variants=("known_future_regressors",),
        signal_dispositions=dispositions,
    )

    assert dispositions[0]["experiment_status"] == "queued"
    assert hypotheses[0].signal_id == "seat-plan"
    assert hypotheses[0].probe_id == "driver-probe"
    assert hypotheses[0].seeded_by == "signal_contract"
    blocked = _signal_experiment_dispositions(
        ForecastSpec(horizon=2, model_policy="baseline", context=context)
    )
    assert blocked[0]["experiment_status"] == "blocked"
    assert "No matching executable" in blocked[0]["experiment_reason"]


def test_optimizer_stops_before_generic_experiments_and_inherits_source_budget(
    tmp_path,
) -> None:
    context = ForecastContext(
        sources=(_source(query_count=2),),
        signal_needs=(_need(status="open"),),
        research_budget=ResearchBudget(
            profile="custom",
            max_iterations=1,
            max_source_queries=5,
            max_compute_units=1,
        ),
    )
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
            "y": range(100, 112),
        }
    )

    result = run_optimizer(
        frame,
        ForecastSpec(horizon=2, model_policy="baseline", context=context),
        output_dir=tmp_path / "research",
        max_iterations=1,
    )
    stop = json.loads(
        (result.output_dir / "stop_receipt.json").read_text(encoding="utf-8")
    )
    dispositions = json.loads(
        (result.output_dir / "signal_experiment_dispositions.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.manifest["stopped_reason"] == "source_discovery_incomplete"
    assert not (result.output_dir / "iteration_001").exists()
    assert stop["budget"]["consumed"]["source_queries"] == 2
    assert stop["budget"]["remaining"]["source_queries"] == 3
    assert dispositions["signal_discovery_gate"]["unresolved_need_ids"] == [
        "driver-need"
    ]
