import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest

from nixtla_scaffold import workbench, workbench_jobs
from nixtla_scaffold.workbench import dispatch, execute_job
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore, canonical_json


@pytest.fixture
def job_request(tmp_path):
    dispatch(tmp_path, "initialize", {})
    path = tmp_path / "source.csv"
    pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2020-01-01", periods=32, freq="MS"),
                  "y": range(100, 132)}).to_csv(path, index=False)
    source = dispatch(tmp_path, "profile", {"path": str(path)})["sourceId"]
    return {"kind": "forecast", "requestKey": "original", "sourceId": source, "maxWallClockSeconds": 90,
            "spec": {"horizon": 2, "model_policy": "baseline", "evaluation": {"max_windows": 2}}}


def remove_worker(store, job_id):
    with store.connect() as conn:
        conn.execute("UPDATE wb_jobs SET input=json_remove(input,'$.workerOwner') WHERE id=?", (job_id,))


def stopped(tmp_path, request):
    job_id = dispatch(tmp_path, "start_job", request)["jobId"]
    store = WorkbenchStore(tmp_path)
    assert workbench_jobs.claim_worker(store, job_id)
    store.finish_job(job_id, "interrupted", error="Synthetic worker interruption.")
    remove_worker(store, job_id)
    return store, job_id


def test_resume_preserves_remaining_budget_and_has_only_one_child(tmp_path, job_request, monkeypatch):
    request = job_request
    clock = [100.]
    monkeypatch.setattr(workbench_jobs, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    job_id = dispatch(tmp_path, "start_job", request)["jobId"]
    store = WorkbenchStore(tmp_path)
    workbench_jobs.claim_worker(store, job_id)
    clock[0] += 10
    store.finish_job(job_id, "interrupted")
    remove_worker(store, job_id)
    resume = {"jobId": job_id, "requestKey": "continuation"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda _: dispatch(tmp_path, "resume_job", resume), range(3)))
    assert sum(item["created"] for item in responses) == 1
    child_id = responses[0]["jobId"]
    budget = store.job(child_id)["input"]["executionBudget"]
    assert {key: budget[key] for key in ("limitSeconds", "remainingSeconds", "consumedSeconds")} == {
        "limitSeconds": 90, "remainingSeconds": 80., "consumedSeconds": 10.}
    with pytest.raises(WorkbenchError, match="already has a continuation"):
        dispatch(tmp_path, "resume_job", {**resume, "requestKey": "duplicate"})
    workbench_jobs.claim_worker(store, child_id)
    clock[0] += 12
    store.finish_job(child_id, "cancelled")
    remove_worker(store, child_id)
    grandchild = dispatch(tmp_path, "resume_job", {"jobId": child_id, "requestKey": "continue-again"})
    assert grandchild["budgetSeconds"] == 68.
    with pytest.raises(WorkbenchError, match="Additional"):
        dispatch(tmp_path, "resume_job", {**resume, "maxWallClockSeconds": 900})


@pytest.mark.parametrize("mutation, message", [
    ("legacy", "no verifiable"), ("expired", "No time remains"),
    ("runtime", "runtime changed"), ("source", "changed"),
    ("research", "Only stopped native"), ("confirm_blend", "Only stopped native"),
])
def test_resume_refuses_unverifiable_or_changed_execution(tmp_path, job_request, mutation, message):
    request = job_request
    store, job_id = stopped(tmp_path, request)
    data = store.job(job_id)["input"]
    if mutation == "legacy":
        data.pop("executionBudget")
    elif mutation == "expired":
        data["executionBudget"]["remainingSeconds"] = 0.
    elif mutation == "runtime":
        data["runtimeFingerprint"] = "different-build"
    elif mutation == "source":
        path, _ = store.source(data["sourceId"])
        path.write_text("changed", encoding="utf-8")
    with store.connect() as conn:
        conn.execute("UPDATE wb_jobs SET input=?,kind=? WHERE id=?",
                     (canonical_json(data), mutation if mutation in {"research", "confirm_blend"} else "forecast", job_id))
    with pytest.raises(WorkbenchError, match=message):
        dispatch(tmp_path, "resume_job", {"jobId": job_id, "requestKey": "rejected"})
    with store.connect(readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM wb_jobs").fetchone()[0] == 1


def test_resume_reuses_immutable_baseline_and_frozen_hypotheses(tmp_path, job_request, monkeypatch):
    request = job_request
    store = WorkbenchStore(tmp_path)
    job_id = dispatch(tmp_path, "start_job", request)["jobId"]
    original = workbench._evaluate
    calls = []
    def interrupted(store, run_id, recipe):
        calls.append(recipe.policy)
        if len(calls) == 2:
            raise RuntimeError("Synthetic interruption in the second hypothesis.")
        return original(store, run_id, recipe)
    monkeypatch.setattr(workbench, "_evaluate", interrupted)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        execute_job(store, job_id)
    partial = store.job(job_id)["result"]
    assert partial["runId"] and len(partial["ensembles"]) == 1
    remove_worker(store, job_id)
    child = dispatch(tmp_path, "resume_job", {"jobId": job_id, "requestKey": "resume"})
    def no_refit(*args, **kwargs):
        pytest.fail("A saved immutable baseline must not be refitted on resume.")
    monkeypatch.setattr(workbench, "run_forecast", no_refit)
    result = execute_job(store, child["jobId"])
    assert result["status"] == "completed"
    assert result["result"]["runId"] == partial["runId"]
    assert len(result["result"]["ensembles"]) == 2
    assert calls[1] == calls[2]
    assert store.job(job_id)["result"] == partial
    with store.connect(readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM wb_runs").fetchone()[0] == 1


def test_expired_budget_does_not_start_a_model(tmp_path, job_request, monkeypatch):
    request = job_request
    store = WorkbenchStore(tmp_path)
    job_id = dispatch(tmp_path, "start_job", request)["jobId"]
    data = store.job(job_id)["input"]
    data["executionBudget"]["remainingSeconds"] = 0.
    with store.connect() as conn:
        conn.execute("UPDATE wb_jobs SET input=? WHERE id=?", (json.dumps(data), job_id))
    monkeypatch.setattr(workbench, "run_forecast", lambda *args: pytest.fail("Expired budgets cannot fit models."))
    with pytest.raises(WorkbenchError, match="budget expired"):
        execute_job(store, job_id)
    assert store.job(job_id)["status"] == "failed"


def test_smaller_context_budget_is_enforced_by_service_receipt(tmp_path, job_request):
    request = job_request
    request["spec"]["context"] = {"research_budget": {"max_wall_clock_minutes": 1}}
    result = dispatch(tmp_path, "start_job", request)
    assert 59 < result["budgetSeconds"] <= 60


def test_direct_worker_runtime_drift_is_terminal_and_settles_budget(tmp_path, job_request, monkeypatch):
    store = WorkbenchStore(tmp_path)
    job_id = dispatch(tmp_path, "start_job", job_request)["jobId"]
    monkeypatch.setattr(workbench_jobs, "runtime_fingerprint", lambda: "changed-runtime")
    with pytest.raises(WorkbenchError, match="runtime changed"):
        execute_job(store, job_id)
    job = store.job(job_id)
    assert job["status"] == "failed"
    assert "runtime changed" in job["error"]
    assert "startedMonotonic" not in job["input"]["executionBudget"]


def test_owner_ending_before_claim_settles_interrupted_budget(tmp_path, job_request, monkeypatch):
    store = WorkbenchStore(tmp_path)
    job_id = dispatch(tmp_path, "start_job", job_request)["jobId"]
    monkeypatch.setattr(workbench_jobs, "alive", lambda owner: False)
    assert not workbench_jobs.claim_worker(store, job_id)
    job = store.job(job_id)
    assert job["status"] == "interrupted"
    assert "startedMonotonic" not in job["input"]["executionBudget"]
