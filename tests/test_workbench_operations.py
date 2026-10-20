import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import psutil
import pytest

from nixtla_scaffold.workbench import dispatch, execute_job
from nixtla_scaffold.workbench_jobs import identity, alive
from nixtla_scaffold.workbench_store import WorkbenchStore, WorkbenchError, canonical_json


def source(tmp_path, periods=48):
    path = tmp_path / f"history-{periods}.csv"
    pd.DataFrame({"unique_id": "x", "ds": pd.date_range("2020-01-01", periods=periods, freq="MS"),
                  "y": [100 + i + i % 12 for i in range(periods)]}).to_csv(path, index=False)
    return dispatch(tmp_path, "profile", {"path": str(path), "horizon": 3})["sourceId"]


def run_job(tmp_path, **kwargs):
    job = dispatch(tmp_path, "start_job", kwargs)
    return execute_job(WorkbenchStore(tmp_path), job["jobId"])["result"]


@pytest.fixture
def native(tmp_path):
    dispatch(tmp_path, "initialize", {})
    sid = source(tmp_path)
    result = run_job(tmp_path, kind="forecast", sourceId=sid, requestKey="native",
                     spec={"horizon": 3, "model_policy": "baseline"}, name="Revenue")
    return tmp_path, sid, result["runId"]


def test_actuals_are_separate_idempotent_vintages_and_refresh_preserves_parent(native):
    root, sid, parent = native
    store = WorkbenchStore(root)
    original = store.read_artifact(parent, "forecast.csv")
    actuals = root / "actuals.csv"
    pd.DataFrame({"unique_id": ["x"] * 3, "ds": pd.date_range("2024-01-01", periods=3, freq="MS"),
                  "y": [151, 153, 155]}).to_csv(actuals, index=False)
    actual_id = dispatch(root, "profile", {"path": str(actuals), "mapping": {"freq": "MS"}})["sourceId"]
    request = dict(kind="actuals", sourceId=actual_id, runId=parent, approved=True, knownAsOf="2024-04-01T00:00:00Z")
    first = run_job(root, requestKey="actuals-1", **request)
    second = run_job(root, requestKey="actuals-2", **request)
    assert first["actual_revision_id"] == second["actual_revision_id"]
    assert second["reused"]
    assert dispatch(root, "ledger_view", {"runId": parent, "kind": "actuals"})["total"] == 3
    before = dispatch(root, "catalog", {})
    for kind in ["versions", "snapshot", "metrics", "performance_summary", "adjustments",
                 "corrected_actuals", "regimes", "deltas", "source_refreshes"]:
        result = dispatch(root, "ledger_view", {"runId": parent, "kind": kind, "series": "x", "limit": 1})
        assert "total" in result and len(result["rows"]) <= 1
    versions = dispatch(root, "ledger_view", {"runId": parent, "kind": "versions"})
    version_id = versions["rows"][0]["forecast_version_id"]
    filtered = dispatch(root, "ledger_view", {"runId": parent, "kind": "snapshot", "versions": [version_id]})
    assert filtered["rows"] and all(row["forecast_version_id"] == version_id for row in filtered["rows"])
    assert dispatch(root, "ledger_view", {"runId": parent, "kind": "snapshot", "versions": ["' OR 1=1 --"]})["total"] == 0
    assert dispatch(root, "catalog", {})["jobs"] == before["jobs"]
    assert store.read_artifact(parent, "forecast.csv") == original
    unchanged = run_job(root, kind="refresh", sourceId=sid, runId=parent, approved=True, requestKey="same")
    assert unchanged["runId"] == parent and unchanged["reused"]
    new_id = source(root, 51)
    refreshed = run_job(root, kind="refresh", sourceId=new_id, runId=parent, approved=True, requestKey="refresh")
    assert refreshed["runId"] != parent
    assert json.loads(store.run(refreshed["runId"])[1]["metadata"])["parentRunId"] == parent
    assert store.read_artifact(parent, "forecast.csv") == original
    with pytest.raises(WorkbenchError, match="official planning lock"):
        dispatch(root, "start_job", {"kind": "lock", "runId": parent, "lockLabel": "Plan", "requestKey": "lock", "approved": True})


def test_frozen_blend_refresh_reuses_recipe_not_native_champion(native):
    root, _, parent = native
    blend = dispatch(root, "view", {"runId": parent})["blends"][0]
    adopted = run_job(root, kind="adopt", runId=parent, blendId=blend["id"], expectedReceipt=blend["receiptHash"],
                      requestKey="adopt", approved=True)["runId"]
    original = WorkbenchStore(root).read_artifact(adopted, "forecast.csv")
    refreshed = run_job(root, kind="refresh", sourceId=source(root, 51), runId=adopted, approved=True, requestKey="refresh-blend")
    view = dispatch(root, "view", {"runId": refreshed["runId"]})
    assert {row["model"] for row in view["rows"]} == {"Blend"}
    assert refreshed["recipeFingerprint"] == blend["receipt"]["recipe_fingerprint"]
    assert view["run"]["metadata"]["parentRunId"] == adopted
    assert not refreshed["planningReady"]
    assert WorkbenchStore(root).read_artifact(adopted, "forecast.csv") == original


def test_recovery_stops_only_verified_orphan_and_never_replays(tmp_path):
    dispatch(tmp_path, "initialize", {})
    sid = source(tmp_path)
    request = {"kind": "forecast", "sourceId": sid, "requestKey": "orphan", "spec": {"horizon": 3}}
    job = dispatch(tmp_path, "start_job", request)
    store = WorkbenchStore(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        data = store.job(job["jobId"])["input"]
        data["serviceOwner"] = {**identity(os.getpid()), "created": 0}
        data["workerOwner"] = identity(child.pid)
        with store.connect() as conn:
            conn.execute("UPDATE wb_jobs SET input=?,status='running' WHERE id=?", (canonical_json(data), job["jobId"]))
        result = dispatch(tmp_path, "recover_jobs", {"approved": True})
        assert result["recovered"] == [job["jobId"]] and not result["replayed"]
        assert not alive(data["workerOwner"])
        assert store.job(job["jobId"])["status"] == "interrupted"
        assert dispatch(tmp_path, "catalog", {})["runs"] == []
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_recovery_does_not_stop_live_owners_or_reused_pids(tmp_path):
    dispatch(tmp_path, "initialize", {})
    job = dispatch(tmp_path, "start_job", {"kind": "forecast", "sourceId": source(tmp_path),
                                         "requestKey": "live", "spec": {"horizon": 3}})
    result = dispatch(tmp_path, "recover_jobs", {"approved": True})
    assert result["active"] == [job["jobId"]]
    assert not alive({**identity(os.getpid()), "created": 0})
    assert psutil.Process(os.getpid()).is_running()


def test_concurrent_imports_publish_one_immutable_version(tmp_path):
    dispatch(tmp_path, "initialize", {})
    out = tmp_path / "import"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 1}}), encoding="utf-8")
    (out / "forecast.csv").write_text("unique_id,ds,yhat\nx,2025-01-01,12\n", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: WorkbenchStore(tmp_path).import_run(out), range(3)))
    assert len({item["id"] for item in results}) == 1
    assert sum(not item["reused"] for item in results) == 1
