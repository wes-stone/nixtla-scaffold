from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from nixtla_scaffold import CandidateDriver, ContextSource, ForecastContext, ForecastSpec, run_forecast
from nixtla_scaffold.accuracy import build_accuracy_gate


def _complete_context(*, candidate_drivers: tuple[CandidateDriver, ...] = ()) -> ForecastContext:
    return ForecastContext(
        decision="Set the operating plan",
        audience="Finance",
        target_semantics="Monthly revenue",
        units="USD",
        grain="monthly",
        requested_horizon=3,
        refresh_cadence="monthly",
        source_discovery_enabled=False,
        sources=(ContextSource(source_id="target", kind="csv", status="opted_out"),),
        candidate_drivers=candidate_drivers,
    )


def _passing_trust_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "unique_id": "Revenue",
                "trust_score_0_100": 82,
                "full_horizon_claim_allowed": True,
                "horizon_gate_result": "passed",
                "interval_status": "calibrated",
                "hierarchy_status": "no_hierarchy",
            }
        ]
    )


def test_accuracy_gate_allows_planning_ready_only_when_every_required_gate_passes() -> None:
    run = SimpleNamespace(
        spec=ForecastSpec(horizon=3, context=_complete_context()),
        forecast=pd.DataFrame({"unique_id": ["Revenue"] * 3, "ds": pd.date_range("2026-01-31", periods=3, freq="ME"), "yhat": [100.0] * 3}),
        driver_availability_audit=pd.DataFrame(),
    )

    passing = build_accuracy_gate(run, _passing_trust_row())
    directional_trust = _passing_trust_row()
    directional_trust.loc[0, "interval_status"] = "future_only"
    directional = build_accuracy_gate(run, directional_trust)

    assert passing["status"] == "planning_ready"
    assert passing["planning_ready_claim_allowed"] is True
    assert passing["challenger_promotion_allowed"] is False
    assert directional["status"] == "directional_only"
    assert directional["series"][0]["failed_gates"] == ["intervals"]


def test_failed_trained_driver_audit_blocks_claims() -> None:
    run = SimpleNamespace(
        spec=ForecastSpec(horizon=3, context=_complete_context(), train_known_future_regressors=True),
        forecast=pd.DataFrame({"unique_id": ["Revenue"] * 3, "ds": pd.date_range("2026-01-31", periods=3, freq="ME"), "yhat": [100.0] * 3}),
        driver_availability_audit=pd.DataFrame(
            {
                "name": ["Future actual proxy"],
                "mode": ["model_candidate"],
                "audit_status": ["failed"],
            }
        ),
    )

    gate = build_accuracy_gate(run, _passing_trust_row())

    assert gate["status"] == "blocked"
    assert gate["series"][0]["failed_gates"] == ["driver_clearance"]


def test_configuration_warning_limits_claim_without_rewriting_forecast() -> None:
    forecast = pd.DataFrame({"unique_id": ["Revenue"] * 3,
                            "ds": pd.date_range("2026-01-31", periods=3, freq="ME"), "yhat": [100.] * 3})
    run = SimpleNamespace(
        spec=ForecastSpec(horizon=3, context=_complete_context()), forecast=forecast.copy(),
        driver_availability_audit=pd.DataFrame(),
        model_policy_resolution={"configuration": {"review_series": ["Revenue"]}},
    )
    gate = build_accuracy_gate(run, _passing_trust_row())
    assert gate["status"] == "directional_only"
    assert gate["series"][0]["failed_gates"] == ["configuration_review"]
    pd.testing.assert_frame_equal(run.forecast, forecast)


def test_unknown_business_context_allows_baseline_but_never_planning_claim(monkeypatch) -> None:
    def no_interview(*args):
        raise AssertionError("Unknown business context must not force an intake interview.")

    monkeypatch.setattr("builtins.input", no_interview)
    frame = pd.DataFrame({
        "unique_id": ["Revenue"] * 36,
        "ds": pd.date_range("2020-01-01", periods=36, freq="MS"),
        "y": [100 + i for i in range(36)],
    })
    context = ForecastContext(requested_horizon=3)
    run = run_forecast(frame, ForecastSpec(horizon=3, freq="MS", model_policy="baseline", context=context))
    assert len(run.forecast) == 3
    assert run.forecast["yhat"].notna().all()
    assert run.spec.context.decision == run.spec.context.audience == run.spec.context.units == ""
    gate = build_accuracy_gate(run, _passing_trust_row())
    assert gate["status"] == "directional_only"
    assert not gate["planning_ready_claim_allowed"]


def test_discovered_context_driver_never_enters_training_implicitly() -> None:
    context = _complete_context(
        candidate_drivers=(
            CandidateDriver(
                name="Seat plan",
                source_id="target",
                status="eligible_for_experiment",
                future_availability="planned",
                leakage_verdict="pass",
                business_rationale="Seats plausibly drive revenue.",
            ),
        )
    )
    spec = ForecastSpec(horizon=3, freq="ME", model_policy="baseline", context=context)
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 10,
            "ds": pd.date_range("2025-01-31", periods=10, freq="ME"),
            "y": [100, 103, 106, 108, 111, 115, 118, 121, 125, 129],
            "seat_plan": [10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
        }
    )

    run = run_forecast(frame, spec)

    assert run.spec.regressors == ()
    assert run.driver_availability_audit.empty
    assert run.driver_model_features.empty
