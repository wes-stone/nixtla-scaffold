import json
import os
from pathlib import Path

import pandas as pd
import pytest

from nixtla_scaffold.workbench import dispatch, execute_job, run_frame
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore


def setup_source(tmp_path):
    dispatch(tmp_path, "initialize", {})
    source = tmp_path / "source.csv"
    pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=48, freq="MS"),
        "y": [100 + i + (i % 12) * 3 for i in range(48)],
    }).to_csv(source, index=False)
    result = dispatch(tmp_path, "profile", {"path": str(source)})
    return result["sourceId"]


def test_catalog_open_creates_no_storage(tmp_path):
    assert dispatch(tmp_path, "catalog", {})["initialized"] is False
    assert not (tmp_path / ".nixtla-workbench").exists()


def test_profile_analyzes_unambiguous_source_without_questions(tmp_path):
    source_id = setup_source(tmp_path)
    assert source_id
    source, metadata = WorkbenchStore(tmp_path).source(source_id)
    assert source.exists()
    assert metadata["profile"]["series_count"] == 1
    assert metadata["requiresRepair"] is False


def test_profile_does_not_guess_between_targets(tmp_path):
    dispatch(tmp_path, "initialize", {})
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"month": ["2025-01-01"], "revenue": [20], "cost": [5]}).to_csv(path, index=False)
    result = dispatch(tmp_path, "profile", {"path": str(path)})
    assert result["sourceId"] is None
    assert len(result["questions"]) == 1


def test_workspace_revision_conflict_preserves_saved_state(tmp_path):
    dispatch(tmp_path, "initialize", {})
    assert dispatch(tmp_path, "save_state", {"expectedRevision": 0, "state": {"tabs": []}})["revision"] == 1
    with pytest.raises(WorkbenchError, match="another view"):
        dispatch(tmp_path, "save_state", {"expectedRevision": 0, "state": {}})
    assert dispatch(tmp_path, "catalog", {})["state"] == {"tabs": []}


def test_jobs_are_idempotent_and_reject_changed_request(tmp_path):
    source = setup_source(tmp_path)
    request = {"kind": "forecast", "sourceId": source, "requestKey": "request", "spec": {"horizon": 3, "model_policy": "baseline"}}
    first = dispatch(tmp_path, "start_job", request)
    second = dispatch(tmp_path, "start_job", request)
    assert first["jobId"] == second["jobId"]
    assert first["created"] and not second["created"]
    with pytest.raises(WorkbenchError, match="different inputs"):
        dispatch(tmp_path, "start_job", {**request, "spec": {"horizon": 6}})


def test_worker_produces_readable_immutable_evidence(tmp_path):
    source = setup_source(tmp_path)
    job = dispatch(tmp_path, "start_job", {
        "kind": "forecast", "sourceId": source, "requestKey": "run",
        "spec": {"horizon": 3, "model_policy": "baseline"},
    })
    result = execute_job(WorkbenchStore(tmp_path), job["jobId"])
    assert result["status"] == "completed"
    run_id = result["result"]["runId"]
    view = dispatch(tmp_path, "view", {"runId": run_id})
    assert view["total"] == 3
    assert len(view["blends"]) == 2
    assert view["gate"]["status"] == "directional_only"
    assert dispatch(tmp_path, "context", {"runId": run_id})["runId"] == run_id
    store = WorkbenchStore(tmp_path)
    root, _ = store.run(run_id)
    (root / "forecast.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="hash changed"):
        dispatch(tmp_path, "view", {"runId": run_id})


def test_arbitrary_operation_and_fields_rejected(tmp_path):
    with pytest.raises(WorkbenchError, match="Unknown"):
        dispatch(tmp_path, "shell", {})
    with pytest.raises(WorkbenchError, match="Additional"):
        dispatch(tmp_path, "catalog", {"path": "arbitrary"})


def test_owned_paths_cannot_escape(tmp_path):
    store = WorkbenchStore(tmp_path)
    with pytest.raises(WorkbenchError, match="outside"):
        store.owned("../outside")


def test_new_document_context_does_not_require_a_run(tmp_path):
    dispatch(tmp_path, "initialize", {})
    context = dispatch(tmp_path, "context", {"document": "Data"})
    assert "No execution" in context["authorization"]


def test_ensemble_replay_and_hash_verification(tmp_path):
    source = setup_source(tmp_path)
    job = dispatch(tmp_path, "start_job", {
        "kind": "forecast", "sourceId": source, "requestKey": "blend-test",
        "spec": {"horizon": 3, "model_policy": "baseline"},
    })
    store = WorkbenchStore(tmp_path)
    result = execute_job(store, job["jobId"])["result"]
    blend_id = result["ensembles"][0]["blendId"]
    replay = dispatch(tmp_path, "view", {"runId": result["runId"], "blendId": blend_id, "kind": "ensemble_backtest"})
    assert "weights" in replay["columns"]
    assert replay["total"] > 0
    (store.owned(f"blends/{blend_id}") / "forecast.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="hash changed"):
        dispatch(tmp_path, "view", {"runId": result["runId"], "blendId": blend_id, "kind": "ensemble"})


def test_export_preserves_numeric_text_and_all_scoped_rows(tmp_path):
    dispatch(tmp_path, "initialize", {})
    run = tmp_path / "import"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 1}}), encoding="utf-8")
    (run / "forecast.csv").write_text(
        "unique_id,ds,yhat\n" + "001,2025-01-01,12345678901234.56789\n" * 2001 + "002,2025-01-01,999\n", encoding="utf-8",
    )
    imported = dispatch(tmp_path, "import_run", {"path": str(run)})
    result = dispatch(tmp_path, "export_values", {"runId": imported["id"], "kind": "forecast", "series": "001"})
    exported = Path(result["path"]).read_text(encoding="utf-8")
    assert result["rows"] == 2001
    assert exported.count("12345678901234.56789") == 2001
    assert "002" not in exported


def test_concurrent_request_key_creates_one_job(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    source = setup_source(tmp_path)
    request = {"kind": "forecast", "sourceId": source, "requestKey": "same", "spec": {"horizon": 3}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: dispatch(tmp_path, "start_job", request), range(3)))
    assert len({row["jobId"] for row in results}) == 1
    assert sum(row["created"] for row in results) == 1


def test_directional_adoption_creates_new_candidate_outputs(tmp_path):
    source = setup_source(tmp_path)
    store = WorkbenchStore(tmp_path)
    job = dispatch(tmp_path, "start_job", {
        "kind": "forecast", "sourceId": source, "requestKey": "original",
        "spec": {"horizon": 3, "model_policy": "baseline"},
    })
    parent_id = execute_job(store, job["jobId"])["result"]["runId"]
    original = store.read_artifact(parent_id, "forecast.csv")
    blend = dispatch(tmp_path, "view", {"runId": parent_id})["blends"][0]
    request = {"kind": "adopt", "requestKey": "adoption", "runId": parent_id,
               "blendId": blend["id"], "expectedReceipt": blend["receiptHash"]}
    with pytest.raises(WorkbenchError, match="Explicitly approve"):
        dispatch(tmp_path, "start_job", request)
    stale = dispatch(tmp_path, "start_job", {
        **request, "requestKey": "stale-adoption", "expectedReceipt": "stale", "approved": True,
    })
    with pytest.raises(WorkbenchError, match="receipt changed"):
        execute_job(store, stale["jobId"])
    assert store.job(stale["jobId"])["status"] == "failed"
    assert len(dispatch(tmp_path, "catalog", {})["runs"]) == 1
    approved = dispatch(tmp_path, "start_job", {**request, "approved": True})
    result = execute_job(store, approved["jobId"])["result"]
    assert result["runId"] != parent_id
    assert result["planningReady"] is False
    adopted = dispatch(tmp_path, "view", {"runId": result["runId"]})
    assert set(row["model"] for row in adopted["rows"]) == {"Blend"}
    actual_windows = run_frame(store, result["runId"], "backtest")["cutoff"].nunique()
    assert actual_windows > 2
    assert all(row["cv_windows"] == actual_windows and row["selection_horizon"] == 3 for row in adopted["rows"])
    assert all(row["horizon_trust_state"] != "no_rolling_origin_evidence" for row in adopted["rows"])
    assert adopted["gate"]["status"] != "planning_ready"
    assert store.read_artifact(parent_id, "forecast.csv") == original
    assert store.read_artifact(result["runId"], "appendix/observed_history.csv") == store.read_artifact(
        parent_id, "appendix/observed_history.csv")
    assert "ensemble_adoption.json" in adopted["artifacts"]
    comparison = dispatch(tmp_path, "compare_versions", {"runId": result["runId"], "referenceRunId": parent_id, "series": "series_1"})
    assert comparison["total"] == 3
    assert all(row["paired"] for row in comparison["rows"])


def test_profile_reserves_confirmation_and_research_cannot_reuse_full_history(tmp_path):
    source = setup_source(tmp_path)
    store = WorkbenchStore(tmp_path)
    _, metadata = store.source(source)
    assert metadata["confirmationEligible"]
    assert metadata["profile"]["rows"] == 42
    dispatch(tmp_path, "start_job", {"kind": "forecast", "sourceId": source, "requestKey": "consume", "spec": {"horizon": 6}})
    with pytest.raises(WorkbenchError, match="already entered"):
        dispatch(tmp_path, "start_job", {"kind": "research", "sourceId": source, "requestKey": "research", "spec": {"horizon": 6}})


@pytest.mark.parametrize("reserved", [1, 2])
def test_research_reservation_covers_requested_confirmation_cutoffs(tmp_path, reserved):
    dispatch(tmp_path, "initialize", {})
    path = tmp_path / "research.csv"
    pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=48, freq="MS"), "y": range(48)}).to_csv(path, index=False)
    profile = dispatch(tmp_path, "profile", {"path": str(path), "horizon": 3, "holdoutWindows": reserved})
    request = {"kind": "research", "sourceId": profile["sourceId"], "requestKey": "research",
               "spec": {"horizon": 3, "context": {"promotion_policy": {"minimum_confirmation_cutoffs": 2}}}}
    if reserved == 1:
        with pytest.raises(WorkbenchError, match="Re-profile"):
            dispatch(tmp_path, "start_job", request)
    else:
        assert dispatch(tmp_path, "start_job", request)["created"]


def test_revised_source_cannot_hide_prior_target_profile_exposure(tmp_path):
    dispatch(tmp_path, "initialize", {})
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=48, freq="MS"), "y": range(48)})
    path = tmp_path / "original.csv"
    frame.to_csv(path, index=False)
    dispatch(tmp_path, "profile", {"path": str(path), "horizon": 3})
    revised = tmp_path / "revised.csv"
    frame.assign(y=frame.y + 1).to_csv(revised, index=False)
    source = dispatch(tmp_path, "profile", {"path": str(revised), "horizon": 3, "holdoutWindows": 2})
    with pytest.raises(WorkbenchError, match="already reached target profiling"):
        dispatch(tmp_path, "start_job", {"kind": "research", "sourceId": source["sourceId"], "requestKey": "revised",
                                       "spec": {"horizon": 3, "context": {"promotion_policy": {"minimum_confirmation_cutoffs": 2}}}})


def test_snapshot_publication_retries_only_bounded_windows_lock_errors(tmp_path, monkeypatch, caplog):
    from nixtla_scaffold.workbench_store import _publish_snapshot

    stage, destination = tmp_path / "stage", tmp_path / "version"
    stage.mkdir()
    (stage / "evidence.txt").write_text("preserved", encoding="utf-8")
    rename = Path.rename
    attempts = []
    def temporarily_locked(path, target):
        attempts.append(path)
        if len(attempts) == 1:
            error = PermissionError("Windows sharing lock")
            error.winerror = 32
            raise error
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", temporarily_locked)
    monkeypatch.setattr("nixtla_scaffold.workbench_store.time.sleep", lambda _: None)
    _publish_snapshot(stage, destination)
    assert len(attempts) == 2
    assert (destination / "evidence.txt").read_text(encoding="utf-8") == "preserved"
    assert "Windows blocked snapshot publication" in caplog.text

    def always_locked(path, target):
        attempts.append(path)
        error = PermissionError("Still locked")
        error.winerror = 32
        raise error
    attempts.clear()
    monkeypatch.setattr(Path, "rename", always_locked)
    with pytest.raises(PermissionError, match="Still locked"):
        _publish_snapshot(stage, destination)
    assert len(attempts) == 5

    def not_a_windows_lock(path, target):
        attempts.append(path)
        raise PermissionError("Permission denied")
    attempts.clear()
    monkeypatch.setattr(Path, "rename", not_a_windows_lock)
    with pytest.raises(PermissionError, match="Permission denied"):
        _publish_snapshot(stage, destination)
    assert len(attempts) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-path regression")
def test_deep_workspace_snapshots_preserve_all_backtest_artifacts(tmp_path):
    from nixtla_scaffold.workbench_store import _filesystem_path

    workspace = _filesystem_path(tmp_path / ("w" * max(1, 190 - len(str(tmp_path)))))
    workspace.mkdir()
    store = WorkbenchStore(workspace)
    store.initialize()
    source = workspace / "original"
    (source / "audit").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 1}}), encoding="utf-8")
    (source / "forecast.csv").write_text("unique_id,ds,yhat\nx,2025-03-01,12\n", encoding="utf-8")
    evidence = b"unique_id,ds,cutoff,y,Naive\nx,2025-02-01,2025-01-01,12,10\n"
    (source / "audit" / "backtest_predictions.csv").write_bytes(evidence)
    imported = store.import_run(source)
    snapshot, _ = store.run(imported["id"])
    assert len(str(snapshot / "audit" / "backtest_predictions.csv")) > 260
    assert store.read_artifact(imported["id"], "audit/backtest_predictions.csv") == evidence
    assert store.import_run(source)["reused"]


def test_adoption_horizon_metadata_uses_actual_paired_windows():
    from nixtla_scaffold.workbench import _blend_cv_metadata

    backtest = pd.DataFrame({
        "unique_id": ["x"] * 5,
        "cutoff": pd.to_datetime(["2025-01-01"] * 3 + ["2025-04-01"] * 2),
        "ds": pd.to_datetime(["2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01"]),
    })
    row = _blend_cv_metadata(backtest, horizon=3, freq="MS").iloc[0]
    assert row["selection_horizon"] == 2
    assert row["cv_windows"] == 2
    assert row["cv_step_size"] == 3
    assert not row["cv_horizon_matches_requested"]
    with pytest.raises(WorkbenchError, match="contiguous evaluation"):
        _blend_cv_metadata(backtest.drop(index=1), horizon=3, freq="MS")
