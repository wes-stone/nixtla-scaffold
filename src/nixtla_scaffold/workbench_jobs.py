"""Process-identity-bound recovery; a reused PID is never an owned worker."""
from __future__ import annotations

import os
import json
import time
import uuid

import psutil

from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest, timestamp


def runtime_fingerprint() -> str:
    from nixtla_scaffold.models import _with_resolved_candidate_identity
    return _with_resolved_candidate_identity({}, resolved_candidates=[])["resolved_candidate_fingerprint"]


def start_budget(budget: dict) -> dict:
    return {**budget, "startedMonotonic": time.monotonic(), "bootTime": psutil.boot_time()}


def remaining_seconds(data: dict) -> float:
    budget = data.get("executionBudget")
    if not budget:
        raise WorkbenchError("budget_unverifiable", "This legacy job has no verifiable execution budget; it cannot be resumed.")
    remaining = float(budget["remainingSeconds"])
    if "startedMonotonic" in budget:
        elapsed = time.monotonic() - budget["startedMonotonic"]
        if elapsed < 0 or abs(psutil.boot_time() - budget["bootTime"]) > 2:
            return 0.
        remaining -= elapsed
    return max(0., remaining)


def settle_budget(data: dict) -> dict:
    if "executionBudget" not in data:
        return data
    budget = data["executionBudget"]
    remaining = remaining_seconds(data)
    return {**data, "executionBudget": {
        "limitSeconds": budget["limitSeconds"], "remainingSeconds": remaining,
        "consumedSeconds": budget["limitSeconds"] - remaining,
    }}


def check_budget(store, job_id: str) -> None:
    job = store.job(job_id)
    if job["status"] == "cancel_requested":
        raise WorkbenchError("cancelled", "Stopped before the next analysis stage.")
    if "executionBudget" in job["input"] and remaining_seconds(job["input"]) <= 0:
        raise WorkbenchError("budget_exhausted", "The original analysis budget expired. Partial evidence is retained; no fresh budget was started.")


def resume_job(store, request: dict) -> dict:
    fingerprint = digest(canonical_json({"resume": request}).encode())
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT * FROM wb_jobs WHERE request_key=?", (request["requestKey"],)).fetchone()
        if existing:
            if json.loads(existing["input"]).get("requestFingerprint") != fingerprint:
                raise WorkbenchError("request_conflict", "This request key already refers to different inputs.")
            return {"jobId": existing["id"], "created": False, "status": existing["status"]}
        row = conn.execute("SELECT * FROM wb_jobs WHERE id=?", (request["jobId"],)).fetchone()
        if row is None:
            raise WorkbenchError("not_found", "The original job no longer exists.")
        if row["kind"] != "forecast" or row["status"] not in {"failed", "cancelled", "interrupted"}:
            raise WorkbenchError("resume_unavailable", "Only stopped native forecasts can resume. Protected research, confirmation and reviewed side effects are never replayed.")
        data = json.loads(row["input"])
        remaining = remaining_seconds(data)
        if remaining <= 0:
            raise WorkbenchError("budget_exhausted", "No time remains in the original approved budget.")
        if data.get("resumeChildId"):
            raise WorkbenchError("already_resumed", "This job already has a continuation. Review that job rather than creating another search.")
        if alive(data.get("workerOwner")):
            raise WorkbenchError("worker_busy", "The original worker is still alive. Reconcile it before resuming.")
        if data.get("runtimeFingerprint") != runtime_fingerprint():
            raise WorkbenchError("runtime_changed", "The original runtime changed; resume cannot reinterpret the frozen request.")
        source, _ = store.source(data["sourceId"])
        if digest(source.read_bytes()) != data["sourceFingerprint"]:
            raise WorkbenchError("source_changed", "The original source snapshot changed; resume is not permitted.")
        if conn.execute("SELECT COUNT(*) FROM wb_jobs WHERE status IN ('queued','running','cancel_requested')").fetchone()[0] >= 2:
            raise WorkbenchError("busy", "Two jobs are already active. Stop or finish one first.")
        child_id = str(uuid.uuid4())
        child = {**settle_budget(data), "requestKey": request["requestKey"], "requestFingerprint": fingerprint,
                 "resumeParentId": row["id"], "serviceOwner": identity(int(os.environ.get("NIXTLA_WORKBENCH_OWNER_PID", os.getpid())))}
        child["executionBudget"] = start_budget(child["executionBudget"])
        child.pop("workerOwner", None)
        conn.execute("INSERT INTO wb_jobs VALUES(?,?,?,?,?,?,?,?,?)", (
            child_id, request["requestKey"], "forecast", "queued", canonical_json(child), row["result"], None, timestamp(), timestamp(),
        ))
        data["resumeChildId"] = child_id
        conn.execute("UPDATE wb_jobs SET input=? WHERE id=?", (canonical_json(data), row["id"]))
    store.event(child_id, "queued", "Explicit continuation under the remaining original budget; source, runtime and hypotheses remain frozen.")
    return {"jobId": child_id, "created": True, "status": "queued", "budgetSeconds": remaining_seconds(child)}


def identity(pid: int) -> dict:
    process = psutil.Process(pid)
    return {"pid": pid, "created": process.create_time()}


def alive(owner: dict | None) -> bool:
    if not owner:
        return False
    try:
        process = psutil.Process(owner["pid"])
        return process.create_time() == owner["created"] and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (KeyError, TypeError, psutil.AccessDenied) as exc:
        raise WorkbenchError("owner_unverifiable", "The recorded process owner cannot be verified; no process was stopped.") from exc


def claim_worker(store, job_id: str) -> bool:
    import json
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT input,status FROM wb_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise WorkbenchError("not_found", "Job no longer exists.")
        if row["status"] != "queued":
            return False
        data = json.loads(row["input"])
        parent = data.get("serviceOwner")
        if parent and not alive(parent):
            conn.execute("UPDATE wb_jobs SET status='interrupted',input=?,error=?,updated_at=? WHERE id=?",
                         (canonical_json(settle_budget(data)), "The owning session ended before execution.", timestamp(), job_id))
            return False
        data["workerOwner"] = identity(os.getpid())
        if data.get("runtimeFingerprint") and data["runtimeFingerprint"] != runtime_fingerprint():
            raise WorkbenchError("runtime_changed", "The runtime changed after admission; no model was executed.")
        if "executionBudget" in data:
            if "startedMonotonic" not in data["executionBudget"]:
                data["executionBudget"] = start_budget(data["executionBudget"])
        conn.execute("UPDATE wb_jobs SET input=?,status='running',updated_at=? WHERE id=?",
                     (canonical_json(data), timestamp(), job_id))
    return True


def recover_jobs(store) -> dict:
    recovered, active, unresolved = [], [], []
    with store.connect(readonly=True) as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM wb_jobs WHERE status IN ('queued','running','cancel_requested')")]
    for job_id in ids:
        job = store.job(job_id)
        parent = job["input"].get("serviceOwner")
        if not parent:
            unresolved.append({"jobId": job_id, "reason": "Legacy job has no verifiable owner. Stop and reconcile it in the original session."})
            continue
        if alive(parent):
            active.append(job_id)
            continue
        worker = job["input"].get("workerOwner")
        if alive(worker):
            process = psutil.Process(worker["pid"])
            if process.create_time() != worker["created"]:
                raise WorkbenchError("owner_changed", "Worker identity changed during recovery; no replacement process was stopped.")
            children = process.children(recursive=True)
            owned = [*reversed(children), process]
            for child in owned:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, pending = psutil.wait_procs(owned, timeout=2)
            for child in pending:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, pending = psutil.wait_procs(pending, timeout=3)
            if pending:
                raise WorkbenchError("worker_busy", "An owned worker did not stop; its job was not marked recovered.")
        status = "cancelled" if job["status"] == "cancel_requested" else "interrupted"
        store.finish_job(job_id, status, error="Recovered after the owning session ended. Partial artifacts retained; no automatic replay.")
        store.event(job_id, status, "Verified dead owner; any surviving owned worker was stopped. Review retained evidence before a new request.")
        recovered.append(job_id)
    return {"recovered": recovered, "active": active, "unresolved": unresolved, "replayed": False}
