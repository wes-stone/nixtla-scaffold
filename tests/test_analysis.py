import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Timer

import pandas as pd
import pytest

from nixtla_scaffold import analyze
from nixtla_scaffold.processes import run_script, ProcessCancelled
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec
from nixtla_scaffold.workbench import run_frame
from nixtla_scaffold.workbench_store import WorkbenchStore, WorkbenchError


def sample():
    return pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2020-01-01", periods=40, freq="MS"),
                         "y": range(100, 140)})


def test_python_cli_and_workbench_share_frozen_plan(tmp_path):
    spec = ForecastSpec(horizon=3, freq="MS", model_policy="baseline", levels=(), verbose=False,
                        evaluation=EvaluationPolicy(max_windows=5, step_size=3))
    job = analyze(sample(), spec, workspace=tmp_path / "python", max_wall_clock_seconds=120,
                  improvement_policy="greedy")
    assert job["status"] == "completed" and len(job["result"]["ensembles"]) == 2
    store = WorkbenchStore(tmp_path / "python")
    plan = json.loads(store.read_artifact(job["result"]["runId"], "analyst_plan.json"))
    assert plan["hypotheses"][-1]["recipe"]["policy"] == "greedy"
    assert not any(item["receipt"]["promotion_recommended"] for item in job["result"]["ensembles"])
    source = tmp_path / "data.csv"
    sample().to_csv(source, index=False)
    recipe = tmp_path / "spec.json"
    recipe.write_text(json.dumps(spec.to_dict()))
    cli = run_script([sys.executable, "-m", "nixtla_scaffold.cli", "analyze",
                      "--input", str(source), "--spec-file", str(recipe), "--workspace", str(tmp_path / "cli"),
                      "--max-seconds", "120", "--improvement-policy", "greedy"], cwd=Path.cwd(), timeout=140)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    other = json.loads(cli.stdout)
    assert other["status"] == "completed"
    pd.testing.assert_frame_equal(run_frame(store, job["result"]["runId"], "forecast"),
                                  run_frame(WorkbenchStore(tmp_path / "cli"), other["result"]["runId"], "forecast"))


def test_quick_does_not_run_improvements(tmp_path):
    result = analyze(sample(), ForecastSpec(horizon=2, freq="MS", model_policy="baseline", run_mode="quick"),
                     workspace=tmp_path, max_wall_clock_seconds=120)
    assert result["result"]["ensembles"] == []


@pytest.mark.parametrize("cancel", [True, False])
def test_owned_process_is_preempted(tmp_path, cancel):
    event = Event()
    timer = Timer(.25, event.set)
    if cancel:
        timer.start()
    with pytest.raises(ProcessCancelled if cancel else subprocess.TimeoutExpired):
        run_script([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                   timeout=1, cancelled=event.is_set)
    timer.cancel()


def test_invalid_budget_fails_before_execution(tmp_path):
    with pytest.raises(ValueError, match="max_wall_clock_seconds"):
        analyze(sample(), workspace=tmp_path, max_wall_clock_seconds=901)


def test_unqualified_horizon_policy_is_rejected_before_profiling(tmp_path):
    with pytest.raises(ValueError, match="unqualified benchmark prototype"):
        analyze(sample(), workspace=tmp_path, improvement_policy="horizon_greedy")
    assert not list(tmp_path.iterdir())


def test_headless_analysis_requires_an_explicit_gap_repair(tmp_path):
    with pytest.raises(WorkbenchError, match="missing|gap|repair"):
        analyze(sample().drop(index=5), {"freq": "MS", "horizon": 3, "model_policy": "baseline"},
                workspace=tmp_path, max_wall_clock_seconds=120)
