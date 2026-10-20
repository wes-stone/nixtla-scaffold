import json
import io
from pathlib import Path

import pandas as pd
import pytest

from nixtla_scaffold.workbench import dispatch, execute_job
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore


@pytest.fixture
def pipeline(tmp_path):
    dispatch(tmp_path, "initialize", {})
    script = tmp_path / "extract.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nimport pandas as pd\n"
        "out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)\n"
        "pd.DataFrame({'unique_id': 'x', 'ds': pd.date_range('2020-01-01', periods=48, freq='MS'),"
        " 'y': range(100,148)}).to_csv(out, index=False)\n", encoding="utf-8")
    definition = {
        "name": "Reviewed synthetic pipeline",
        "extracts": [{"name": "extract", "script": str(script), "args": ["{output:extract}"],
                      "outputs": {"extract": "raw/extract.csv"}}],
        "forecast": {"freq": "MS", "horizon": 3, "model_policy": "baseline", "paired_validation": True},
    }
    return tmp_path, definition


def execute(root, proposal, key):
    job = dispatch(root, "start_job", {"kind": "pipeline", "requestKey": key, "proposalId": proposal["proposalId"],
                                      "expectedReceipt": proposal["expectedReceipt"], "approved": True})
    return execute_job(WorkbenchStore(root), job["jobId"])["result"]


def test_prepare_is_read_only_execution_requires_exact_review_and_frozen_bytes(pipeline):
    root, definition = pipeline
    proposal = dispatch(root, "prepare_pipeline", {"definition": definition, "dataOnly": True})
    assert not (root / "raw").exists()
    assert not dispatch(root, "catalog", {})["jobs"]
    request = {"kind": "pipeline", "proposalId": proposal["proposalId"],
               "expectedReceipt": proposal["expectedReceipt"], "requestKey": "no-approval"}
    with pytest.raises(WorkbenchError, match="approve"):
        dispatch(root, "start_job", request)
    with pytest.raises(WorkbenchError, match="receipt"):
        dispatch(root, "start_job", {**request, "approved": True, "expectedReceipt": "stale"})
    Path(definition["extracts"][0]["script"]).write_text("raise RuntimeError('original changed')", encoding="utf-8")
    result = execute(root, proposal, "copy")
    assert result["sourceId"] and result["pipeline"] == "succeeded"
    assert not dispatch(root, "catalog", {})["runs"]
    copied = WorkbenchStore(root).owned("proposals/" + proposal["proposalId"]) / "stage-0.py"
    copied.write_text("raise RuntimeError('tampered')", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="changed"):
        execute(root, proposal, "tamper")


def test_pipeline_forecast_and_refresh_preserve_saved_spec_and_parent(pipeline):
    root, definition = pipeline
    first = execute(root, dispatch(root, "prepare_pipeline", {"definition": definition}), "first")
    parent = first["runId"]
    store = WorkbenchStore(root)
    original = store.read_artifact(parent, "forecast.csv")
    definition["forecast"]["horizon"] = 12
    next_proposal = dispatch(root, "prepare_pipeline", {"definition": definition, "runId": parent})
    assert next_proposal["receipt"]["spec"]["horizon"] == 3
    second = execute(root, next_proposal, "refresh")
    view = dispatch(root, "view", {"runId": second["runId"]})
    assert view["spec"]["horizon"] == 3
    assert view["run"]["metadata"]["parentRunId"] == parent
    assert store.read_artifact(parent, "forecast.csv") == original
    assert "appendix/source_pipeline_manifest.json" in view["artifacts"]


def test_extract_only_can_discover_calendar_without_an_existing_forecast(pipeline):
    root, definition = pipeline
    definition["forecast"]["freq"] = None
    with pytest.raises(WorkbenchError, match="calendar"):
        dispatch(root, "prepare_pipeline", {"definition": definition})
    result = execute(root, dispatch(root, "prepare_pipeline", {"definition": definition, "dataOnly": True}), "inspect-first")
    _, metadata = WorkbenchStore(root).source(result["sourceId"])
    assert metadata["mapping"]["freq"] == "MS"
    assert not dispatch(root, "catalog", {})["runs"]


@pytest.mark.parametrize("output", ["../outside.csv", "C:\\outside.csv"])
def test_pipeline_rejects_declared_outputs_outside_job(pipeline, output):
    root, definition = pipeline
    definition["extracts"][0]["outputs"]["extract"] = output
    with pytest.raises(WorkbenchError, match="owned job"):
        dispatch(root, "prepare_pipeline", {"definition": definition})


def test_pipeline_failure_has_durable_manifest_and_failed_job(pipeline):
    root, definition = pipeline
    Path(definition["extracts"][0]["script"]).write_text("raise RuntimeError('provider unavailable')", encoding="utf-8")
    proposal = dispatch(root, "prepare_pipeline", {"definition": definition})
    with pytest.raises(RuntimeError, match="failed"):
        execute(root, proposal, "fail")
    store = WorkbenchStore(root)
    job = dispatch(root, "catalog", {})["jobs"][0]
    assert job["status"] == "failed"
    manifest = json.loads((store.owned(f"jobs/{job['id']}/pipeline") / "pipeline_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "provider unavailable" in manifest["steps"][0]["stderr_tail"]


def test_reviewed_custom_model_uses_copied_entrypoint_and_paired_evidence(pipeline):
    root, definition = pipeline
    data = execute(root, dispatch(root, "prepare_pipeline", {"definition": definition, "dataOnly": True}), "source")
    script = root / "custom.py"
    script.write_text(
        "import argparse\nimport pandas as pd\n"
        "p=argparse.ArgumentParser()\n"
        "for key in ['history','future-grid','context','output']: p.add_argument('--'+key)\n"
        "a=p.parse_args(); h=pd.read_csv(a.history); f=pd.read_csv(a.future_grid)\n"
        "f['yhat']=f['unique_id'].map(h.groupby('unique_id')['y'].last())\n"
        "f.to_csv(a.output,index=False)\n", encoding="utf-8")
    spec = {**definition["forecast"], "custom_models": [{"name": "Reviewed", "script_path": str(script)}]}
    with pytest.raises(WorkbenchError, match="review"):
        dispatch(root, "start_job", {"kind": "forecast", "sourceId": data["sourceId"], "requestKey": "unsafe", "spec": spec})
    proposal = dispatch(root, "prepare_forecast", {"sourceId": data["sourceId"], "spec": spec})
    script.write_text("raise RuntimeError('not the approved entrypoint')", encoding="utf-8")
    result = execute(root, proposal, "custom")
    view = dispatch(root, "view", {"runId": result["runId"], "kind": "metrics"})
    assert "Custom_Reviewed" in view["models"]
    assert view["spec"]["paired_validation"]
    assert any(row["model"] == "Custom_Reviewed" for row in view["rows"])


def test_external_future_is_directional_and_history_keeps_cutoff_contract(pipeline):
    root, definition = pipeline
    original = execute(root, dispatch(root, "prepare_pipeline", {"definition": definition}), "native")["runId"]
    store = WorkbenchStore(root)
    forecast = store.read_artifact(original, "forecast.csv")
    future = pd.read_csv(io.BytesIO(forecast))
    path = root / "external.csv"
    future[["unique_id", "ds", "yhat"]].assign(model="BYO").to_csv(path, index=False)
    request = {"runId": original, "path": str(path), "freq": "MS", "units": ""}
    compared = execute(root, dispatch(root, "prepare_external", request), "future")
    assert not compared["scored"] and not compared["promotionAllowed"]
    assert dispatch(root, "view", {"runId": compared["runId"], "kind": "external"})["total"] > 0
    bt = pd.read_csv(io.BytesIO(store.read_artifact(original, "audit/backtest_predictions.csv")))
    historical = bt[["unique_id", "ds", "cutoff", "Naive"]].rename(columns={"Naive": "yhat"}).assign(model="BYO")
    pd.concat([historical, future[["unique_id", "ds", "yhat"]].assign(model="BYO", cutoff="2023-12-01")], ignore_index=True).to_csv(path, index=False)
    scored = execute(root, dispatch(root, "prepare_external", request), "historical")
    assert scored["scored"] and not scored["promotionAllowed"]
    assert dispatch(root, "view", {"runId": scored["runId"], "kind": "external_metrics"})["total"] > 0
    assert store.read_artifact(original, "forecast.csv") == forecast


def test_stage_timeout_stops_script_descendants(tmp_path):
    import subprocess
    import sys
    import psutil
    from nixtla_scaffold.processes import run_script
    script = tmp_path / "parent.py"
    script.write_text("import subprocess, sys, time\nfrom pathlib import Path\n"
                      "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)'])\n"
                      "Path('child.pid').write_text(str(p.pid))\ntime.sleep(90)\n", encoding="utf-8")
    with pytest.raises(subprocess.TimeoutExpired):
        run_script([sys.executable, str(script)], cwd=tmp_path, timeout=3)
    assert not psutil.pid_exists(int((tmp_path / "child.pid").read_text()))
