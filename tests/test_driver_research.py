from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import DriverSnapshotGrant, DriverSourceContract, ForecastSpec, KnownFutureRegressor, EvaluationPolicy
from nixtla_scaffold.driver_research import prepare_driver_research, matched_driver_specs, driver_pair_contract
from nixtla_scaffold.experiments import run_optimizer


def source_case(tmp_path, n=150):
    rng = np.random.default_rng(83)
    values = rng.normal(0, 10, n)
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    source = pd.DataFrame({"unique_id": "A", "ds": dates, "value": values, "units": "seats",
                           "period_end": dates, "available_at": dates, "vintage": "first"})
    path = tmp_path / "driver.csv"
    source.to_csv(path, index=False)
    grant = DriverSnapshotGrant("capacity", str(path), sha256(path.read_bytes()).hexdigest(), ("A",))
    contract = DriverSourceContract("capacity", "MS", "seats", "stock", "mock:capacity",
                                    "Capacity leads demand by five periods")
    regressor = KnownFutureRegressor("capacity", availability="historical_only", mode="model_candidate",
                                    source=contract, historical_lags=(5,))
    spec = ForecastSpec(horizon=3, freq="MS", season_length=1, levels=(), model_policy="mlforecast",
                        model_allowlist=("Ridge",), weighted_ensemble=False,
                        evaluation=EvaluationPolicy(max_windows=3),
                        regressors=(regressor,), driver_snapshots=(grant,))
    target = pd.DataFrame({"unique_id": "A", "ds": dates,
                           "y": 200 + 3 * np.roll(values, 5) + rng.normal(0, 0.1, n)})
    return target, source, spec


def test_screen_selects_only_safe_frozen_lag_without_changing_base_learner(tmp_path):
    target, _, spec = source_case(tmp_path)
    screened, receipt = prepare_driver_research(target.iloc[:-3], spec)
    assert screened.regressors[0].historical_lags == (5,)
    assert receipt["screening"]["cards"][0]["disposition"] == "eligible_for_predictive_testing"
    control, candidate = matched_driver_specs(screened, "capacity", "Ridge")
    assert control == replace(candidate, regressors=(), train_known_future_regressors=False)
    assert candidate.model_policy == spec.model_policy
    assert not receipt["automatic_adoption"]
    with pytest.raises(ValueError, match="silently upgrade"):
        matched_driver_specs(replace(screened, model_policy="baseline"), "capacity", "Ridge")


def test_source_authorization_and_unsupported_future_drivers_leave_a_baseline(tmp_path):
    target, _, spec = source_case(tmp_path)
    screened, receipt = prepare_driver_research(target, replace(spec, driver_snapshots=()))
    assert not screened.regressors
    assert "no explicit snapshot grant" in receipt["sources"][0]["reason"]
    screened, receipt = prepare_driver_research(target, replace(spec, regressors=(
        replace(spec.regressors[0], availability="forecasted", historical_lags=()),)))
    assert not screened.regressors and receipt["sources"][0]["status"] == "blocked"


def test_spurious_and_underpowered_inputs_never_queue_feature_training(tmp_path):
    target, _, spec = source_case(tmp_path)
    target["y"] = np.random.default_rng(99).normal(size=len(target)) + np.arange(len(target))
    screened, receipt = prepare_driver_research(target, spec)
    assert not screened.regressors
    assert receipt["screening"]["cards"][0]["disposition"] != "eligible_for_predictive_testing"
    screened, receipt = prepare_driver_research(target.iloc[:15], spec)
    assert not screened.regressors
    assert receipt["screening"]["cards"][0]["disposition"] == "insufficient_evidence"


def test_real_optimizer_runs_matched_ablation_and_frozen_confirmation(tmp_path):
    target, _, spec = source_case(tmp_path)
    result = run_optimizer(target, spec, variants=("known_future_regressors",), max_iterations=1,
                           max_variants=1, output_dir=tmp_path / "research")
    root = result.output_dir
    ablation = json.loads((root / "iteration_001" / "driver_ablation.json").read_text())
    assert ablation["contract"]["matched"], ablation
    assert ablation["passed"], ablation
    confirmation = json.loads((root / "confirmation" / "confirmation_receipt.json").read_text())
    assert confirmation["driver_confirmations"][0]["passed"], confirmation
    assert confirmation["passed"], confirmation
    assert confirmation["frozen_candidate_spec"]["regressors"][0]["historical_lags"] == [5]
    promotion = json.loads((root / "promotion_decision.json").read_text())
    assert promotion["human_approval_required"] and not promotion["official_forecast_mutated"]
    from nixtla_scaffold.research import archive_research_evidence
    from nixtla_scaffold.workbench_store import WorkbenchStore
    archived = archive_research_evidence(root)
    (tmp_path / "workbench").mkdir()
    store = WorkbenchStore(tmp_path / "workbench")
    store.initialize()
    imported = store.import_run(archived, metadata={"adopted": False, "purpose": "research_tuning_baseline"})
    index = json.loads(store.read_artifact(imported["id"], "research/evidence_index.json"))
    assert "driver_research.json" in index["sha256"]
    assert "iteration_001/driver_ablation.json" in index["sha256"]
    assert "confirmation/confirmation_receipt.json" in index["sha256"]
    candidate_dir = root / "iteration_001" / "variants" / "known_future_regressors"
    control_dir = root / "iteration_001" / "variants" / "baseline"
    evidence_path = candidate_dir / "audit" / "candidate_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    for recipe in evidence["recipes"].values():
        if recipe["model"] == "Ridge":
            recipe["execution"]["actual_feature_names"].append("unrelated_feature")
    evidence_path.write_text(json.dumps(evidence))
    assert not driver_pair_contract(control_dir, candidate_dir, "Ridge")["matched"]
    for recipe in evidence["recipes"].values():
        if recipe["model"] == "Ridge":
            recipe["execution"]["actual_feature_names"].remove("unrelated_feature")
    for recipe in evidence["recipes"].values():
        if recipe["model"] == "Ridge":
            recipe["execution"]["lags"] = [99]
    evidence_path.write_text(json.dumps(evidence))
    assert not driver_pair_contract(control_dir, candidate_dir, "Ridge")["matched"]


def test_canvas_research_archives_baseline_without_inventing_confirmation_consumption(tmp_path):
    from nixtla_scaffold.workbench import dispatch, execute_job
    from nixtla_scaffold.workbench_store import WorkbenchStore
    target, _, _ = source_case(tmp_path, n=48)
    source = tmp_path / "target.csv"
    target.to_csv(source, index=False)
    dispatch(tmp_path, "initialize", {})
    profiled = dispatch(tmp_path, "profile", {"path": str(source), "horizon": 3})
    job = dispatch(tmp_path, "start_job", {
        "kind": "research", "sourceId": profiled["sourceId"], "requestKey": "research",
        "spec": {"horizon": 3, "model_policy": "baseline", "context": {
            "research_budget": {"profile": "custom", "max_iterations": 1, "max_compute_units": 1}}},
    })
    store = WorkbenchStore(tmp_path)
    finished = execute_job(store, job["jobId"])
    assert finished["status"] == "completed", finished
    result = finished["result"]
    assert result["runId"] and not result["confirmationConsumed"]
    assert result["stopReason"] == "budget_exhausted"
    metadata = json.loads(store.run(result["runId"])[1]["metadata"])
    assert not metadata["adopted"] and metadata["purpose"] == "research_tuning_baseline"
    assert store.read_artifact(result["runId"], "research/stop_receipt.json")


@pytest.mark.parametrize("kind", ["spurious", "insufficient"])
def test_complete_optimizer_retains_baseline_for_unhelpful_sources(tmp_path, kind):
    target, _, spec = source_case(tmp_path, n=150 if kind == "spurious" else 18)
    if kind == "spurious":
        target["y"] = np.arange(len(target)) * 10 + np.random.default_rng(151).normal(size=len(target))
    result = run_optimizer(target, spec, variants=("known_future_regressors",), max_iterations=1,
                           max_variants=1, output_dir=tmp_path / "research")
    assert (result.output_dir / "baseline" / "forecast.csv").exists()
    assert not (result.output_dir / "iteration_001").exists()
    promotion = json.loads((result.output_dir / "promotion_decision.json").read_text())
    assert not promotion["promotion_recommended"] and not promotion["official_forecast_mutated"]
    screen = json.loads((result.output_dir / "driver_research.json").read_text())
    assert screen["sources"][0]["status"] != "eligible_for_predictive_testing"


def test_confirmation_budget_does_not_consume_protected_outcomes(tmp_path):
    from nixtla_scaffold.schema import ForecastContext, ResearchBudget
    target, _, spec = source_case(tmp_path)
    spec = replace(spec, context=ForecastContext(research_budget=ResearchBudget(
        profile="custom", max_iterations=1, max_compute_units=3)))
    result = run_optimizer(target, spec, variants=("known_future_regressors",), max_iterations=1,
                           max_variants=1, output_dir=tmp_path / "research")
    promotion = json.loads((result.output_dir / "promotion_decision.json").read_text())
    assert promotion["tuning_assessment"]["driver_ablation"]["passed"]
    assert promotion["confirmation"]["status"] == "budget_exhausted"
    assert not (result.output_dir / "confirmation").exists()
    assert result.manifest["research_budget"]["consumed"]["compute_units"] == 3


def test_only_tuning_targets_reach_screening_and_lag_selection(tmp_path, monkeypatch):
    import nixtla_scaffold.driver_research as driver_research
    target, _, spec = source_case(tmp_path)
    original = driver_research.prepare_driver_research
    observed = []
    def inspect_tuning(frame, settings):
        observed.append(frame.ds.max())
        assert len(frame) == len(target) - spec.horizon
        return original(frame, settings)
    monkeypatch.setattr(driver_research, "prepare_driver_research", inspect_tuning)
    from nixtla_scaffold.schema import ForecastContext, ResearchBudget
    spec = replace(spec, context=ForecastContext(research_budget=ResearchBudget(
        profile="custom", max_iterations=1, max_compute_units=1)))
    run_optimizer(target, spec, variants=("known_future_regressors",), max_iterations=1,
                  max_variants=1, output_dir=tmp_path / "research")
    assert observed == [target.ds.iloc[-4]]


def test_research_archive_supports_deep_windows_paths(tmp_path):
    from nixtla_scaffold.research import archive_research_evidence
    from nixtla_scaffold.workbench_store import _filesystem_path
    raw = tmp_path / ("long-" + "x" * 140)
    root = _filesystem_path(raw)
    (root / "baseline").mkdir(parents=True)
    (root / "baseline" / "manifest.json").write_text('{"spec":{},"outputs":{}}')
    evidence = root / "iteration_001" / "variants" / "known_future_regressors" / "driver_sources"
    evidence.mkdir(parents=True)
    (evidence / ("a" * 64 + ".csv")).write_text("unique_id,ds,value\n")
    archived = archive_research_evidence(raw)
    index = json.loads((archived / "research" / "evidence_index.json").read_text())
    name = next(iter(index["sha256"]))
    assert (archived / "research" / name).is_file()
