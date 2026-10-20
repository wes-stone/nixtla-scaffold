"""One bounded, persistent analyst workflow for Python, CLI and the workbench."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pandas as pd

from nixtla_scaffold.blend import BlendRecipe, component_models
from nixtla_scaffold.schema import ForecastSpec


def run_analysis_job(store, job_id, source, spec, out, *, forecast, evaluate, hypotheses):
    from nixtla_scaffold.workbench_jobs import check_budget
    from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json

    job = store.job(job_id)
    data, result = job["input"], job.get("result")
    if not result or not result.get("runId"):
        with store.connect(readonly=True) as conn:
            saved = conn.execute("SELECT id FROM wb_runs WHERE json_extract(metadata,'$.jobId')=?",
                                 (data.get("resumeParentId", job_id),)).fetchone()
        if saved:
            result = {"id": saved["id"], "runId": saved["id"], "ensembles": []}
    if not result or not result.get("runId"):
        check_budget(store, job_id)
        store.event(job_id, "running", "Fitting eligible native models and evaluating rolling origins.")
        started = time.perf_counter()
        run = forecast(str(source), spec)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        run.to_directory(out / "forecast")
        export_seconds = time.perf_counter() - started
        candidates = component_models(run.all_models)[:12]
        plan = {
            "hypotheses": [] if spec.run_mode == "quick" else _diagnostic_hypotheses(run, hypotheses(run, candidates)),
            "candidates": candidates, "maxImprovements": 2,
            "scope": "Retrospective evidence; not untouched confirmation or automatic adoption.",
            "configuration_review": {
                "artifact": "audit/configuration.json",
                "status": run.model_policy_resolution.get("configuration", {}).get("status", "unavailable"),
                "scope": "Read the recorded data diagnostics and native settings before further configuration experiments. This plan does not execute additional variants.",
            },
        }
        policy = data.get("improvementPolicy", "legacy")
        if policy != "legacy" and plan["hypotheses"] and len(candidates) >= 2:
            if policy != "greedy":
                raise ValueError("Only the global greedy challenger is available for analyst workflows.")
            recipe = BlendRecipe(
                policy, tuple(candidates), objective=spec.selection_policy.objective,
            )
            greedy = {
                "policy": policy, "recipe": recipe.to_dict(), "status": "retrospective_hypothesis",
                "reason": "Explicit opt-in replaces, rather than adds to, the second improvement slot.",
            }
            plan["hypotheses"] = [plan["hypotheses"][0], greedy]
        elif policy != "legacy" and len(candidates) < 2:
            plan["improvement_policy_status"] = "unavailable_needs_two_components"
        if len(plan["hypotheses"]) > 2:
            raise ValueError("Analyst workflow permits at most two frozen improvements.")
        (out / "forecast" / "analyst_plan.json").write_text(canonical_json(plan), encoding="utf-8")
        result = store.import_run(out / "forecast", name=data.get("name") or "Forecast", metadata={
            "sourceId": data["sourceId"], "jobId": job_id, "adopted": False,
        })
        result.update(runId=result["id"], ensembles=[], improvements=[])
        result["timings"] = {"forecast_and_scoring_seconds": fit_seconds, "export_seconds": export_seconds,
                             "improvements_seconds": []}
        store.checkpoint_job(job_id, result)
    try:
        plan = json.loads(store.read_artifact(result["runId"], "analyst_plan.json"))
    except FileNotFoundError as exc:
        raise WorkbenchError("missing_evidence", "The frozen analysis plan is unavailable.") from exc
    result.setdefault("ensembles", [])
    result.setdefault("improvements", list(result["ensembles"]))
    for index, hypothesis in enumerate(plan["hypotheses"][len(result["improvements"]):],
                                       start=len(result["improvements"])):
        check_budget(store, job_id)
        started = time.perf_counter()
        if hypothesis.get("kind") == "treatment":
            improvement = _evaluate_treatment(store, result["runId"], source, hypothesis,
                                              out / f"improvement-{index + 1}", forecast)
        else:
            recipe = BlendRecipe.from_dict(hypothesis["recipe"]) if "recipe" in hypothesis else BlendRecipe(
                hypothesis["policy"], tuple(plan["candidates"]),
            )
            improvement = evaluate(store, result["runId"], recipe)
            result["ensembles"].append(improvement)
        result["improvements"].append(improvement)
        result.setdefault("timings", {}).setdefault("improvements_seconds", []).append(time.perf_counter() - started)
        store.checkpoint_job(job_id, result)
    check_budget(store, job_id)
    result["stopReason"] = "bounded_analysis_complete"
    return result


def _diagnostic_hypotheses(run, blends):
    from nixtla_scaffold.anomalies import anomaly_adjustment_spec, enhanced_calendar_spec
    from nixtla_scaffold.model_families import model_family
    spec = run.spec
    if spec.events or spec.hierarchy or spec.hierarchy_reconciliation != "none" or run.backtest_predictions.empty:
        return blends
    choices = []
    report = run.model_policy_resolution.get("anomalies", {})
    eligible = sum(row["eligible"] for prefix in report.get("prefixes", [])
                   for row in prefix.get("series", []) if row["stage"] == "future")
    if spec.anomaly.mode == "detect" and eligible:
        choices.append({"kind": "treatment", "variant": "anomaly_adjustment",
                        "reason": f"{eligible} isolated training transients passed conservative guards; compare on unchanged actuals.",
                        "spec": anomaly_adjustment_spec(spec).to_dict()})
    ml_ran = any(model_family(model).lower() == "mlforecast" for model in component_models(run.all_models))
    if run.profile.freq in {"D", "B"} and not spec.calendar.enabled and ml_ran:
        choices.append({"kind": "treatment", "variant": "enhanced_calendar",
                        "reason": "Test categorical calendar effects in the existing ML family; no inferred holiday geography.",
                        "spec": enhanced_calendar_spec(spec).to_dict()})
    return [*choices, *blends][:2]


def _evaluate_treatment(store, reference_id, source, hypothesis, out, forecast):
    from nixtla_scaffold.research import _paired_tuning_evidence
    from nixtla_scaffold.schema import forecast_spec_from_dict
    from nixtla_scaffold.workbench_store import canonical_json

    spec = forecast_spec_from_dict(hypothesis["spec"])
    run = forecast(str(source), spec)
    run.to_directory(out)
    reference, _ = store.run(reference_id)
    primary = "avg_landing_mape_pct" if spec.selection_policy.objective == "monthly_landing" else "avg_rmse"
    paired = _paired_tuning_evidence(reference, out, primary_metric=primary)
    frame = paired.pop("frame")
    frame.to_csv(out / "analysis_pair.csv", index=False)
    improvement = None
    if paired["exact_match"] and {"baseline", "candidate"}.issubset(frame.columns):
        base, candidate = frame.baseline.mean(), frame.candidate.mean()
        improvement = float(1 - candidate / base) if base > 0 else None
    receipt = {
        "kind": "treatment", "variant": hypothesis["variant"], "referenceRunId": reference_id,
        "primary_metric": primary, "primary_improvement": improvement, **paired,
        "scope": "Retrospective paired selected-path comparison, not untouched confirmation.",
        "promotion_recommended": False, "adopted": False,
        "reason": "Review primary error, stable windows, bias and interval evidence before any separate adoption.",
    }
    (out / "analysis_pair.json").write_text(canonical_json(receipt), encoding="utf-8")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"].update(analysis_pair="analysis_pair.json", analysis_pair_windows="analysis_pair.csv")
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    saved = store.import_run(out, name=f"Diagnostic trial: {hypothesis['variant']}",
                             metadata={"adopted": False, "referenceRunId": reference_id,
                                       "diagnosticTreatment": hypothesis["variant"]})
    return {"runId": saved["id"], "receipt": receipt}


def _execute(workspace, admitted, cancel_event=None):
    from nixtla_scaffold.processes import run_script, ProcessCancelled
    from nixtla_scaffold.workbench_jobs import remaining_seconds
    from nixtla_scaffold.workbench_store import WorkbenchStore

    store = WorkbenchStore(workspace)
    job_id = admitted["jobId"]
    remaining = remaining_seconds(store.job(job_id)["input"])
    if remaining <= 0:
        store.finish_job(job_id, "interrupted", error="Original analysis budget exhausted.")
        raise TimeoutError(f"Analysis {job_id}: original budget exhausted")
    try:
        completed = run_script(
            [sys.executable, "-m", "nixtla_scaffold.cli", "workbench", "worker",
             "--workspace", str(store.workspace), "--job-id", job_id],
            cwd=store.workspace, timeout=remaining,
            cancelled=(lambda: cancel_event.is_set()) if cancel_event is not None else None,
        )
    except (subprocess.TimeoutExpired, ProcessCancelled, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, subprocess.TimeoutExpired) else "cancelled"
        store.finish_job(job_id, status, error=f"Analysis {status}; owned worker stopped.")
        raise
    if completed.returncode:
        raise RuntimeError(f"Analysis {job_id} failed: {completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
    job = store.job(job_id)
    if job["status"] != "completed":
        raise RuntimeError(f"Analysis {job_id} did not complete: {job['status']}")
    return {**job, "workspace": str(store.workspace)}


def analyze(data: str | Path | pd.DataFrame, spec: ForecastSpec | dict | None = None, *,
            workspace: str | Path, max_wall_clock_seconds: float = 900, cancel_event=None,
            improvement_policy: str = "legacy") -> dict:
    """Profile, save a baseline and replay <=2 improvements; never adopt.

    Returns the persistent job with runId, improvements and separate ensemble receipts.
    ``cancel_event`` may be a threading.Event. Timeouts/cancellation stop the owned
    process tree. Resume explicitly with ``resume_analysis`` under the original bound.
    """
    from nixtla_scaffold.workbench import dispatch
    from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore
    from nixtla_scaffold.processes import run_script

    if isinstance(max_wall_clock_seconds, bool) or not math.isfinite(max_wall_clock_seconds) or not 0 < max_wall_clock_seconds <= 900:
        raise ValueError("max_wall_clock_seconds must be in (0, 900]")
    if improvement_policy not in {"legacy", "greedy"}:
        raise ValueError("improvement_policy must be legacy or greedy; horizon grouping is an unqualified benchmark prototype")
    started = time.monotonic()
    explicit_spec = isinstance(spec, ForecastSpec) or isinstance(spec, dict) and "fill_method" in spec
    if not isinstance(spec, ForecastSpec):
        from nixtla_scaffold.presets import resolve_forecast_spec
        spec = resolve_forecast_spec(spec or {})
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    store = WorkbenchStore(workspace)
    store.initialize()
    if isinstance(data, pd.DataFrame):
        source = store.owned(f"inputs/{uuid.uuid4()}.csv")
        source.parent.mkdir(exist_ok=True)
        data.to_csv(source, index=False)
    else:
        source = Path(data).resolve(strict=True)
    mapping = {"id": spec.id_col, "time": spec.time_col, "target": spec.target_col}
    if spec.freq:
        mapping["freq"] = spec.freq
    remaining = max_wall_clock_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Analysis budget exhausted while saving the input.")
    profiled = run_script(
        [sys.executable, "-m", "nixtla_scaffold.cli", "workbench", "worker", "--workspace", str(workspace)],
        cwd=workspace, timeout=remaining,
        cancelled=(lambda: cancel_event.is_set()) if cancel_event is not None else None,
        input_text=json.dumps({"operation": "profile", "input": {
            "path": str(source), "horizon": spec.horizon, "mapping": mapping,
        }}),
        output_limit=8 * 1024 * 1024 + 1,
    )
    if profiled.returncode:
        raise WorkbenchError("profile_failed", profiled.stdout[-4000:] + profiled.stderr[-4000:])
    profile = json.loads(profiled.stdout)["result"]
    if not profile["sourceId"]:
        raise WorkbenchError("source_decision_required", "; ".join(profile["questions"]))
    if profile["questions"] and not explicit_spec:
        raise WorkbenchError("repair_required", "; ".join(profile["questions"]))
    remaining = max_wall_clock_seconds - (time.monotonic() - started)
    if remaining <= 0 or (cancel_event is not None and cancel_event.is_set()):
        raise TimeoutError("Analysis stopped during profiling; no model job started.")
    canonical_spec = replace(spec, id_col="unique_id", time_col="ds", target_col="y")
    admitted = dispatch(workspace, "start_job", {
        "kind": "forecast", "requestKey": str(uuid.uuid4()), "sourceId": profile["sourceId"],
        "spec": canonical_spec.to_dict(), "maxWallClockSeconds": remaining,
        "improvementPolicy": improvement_policy,
    })
    return _execute(workspace, admitted, cancel_event)


def resume_analysis(workspace: str | Path, job_id: str, *, cancel_event=None) -> dict:
    """Reuse the frozen baseline/plan and remaining original budget, never refill it."""
    from nixtla_scaffold.workbench import dispatch
    admitted = dispatch(workspace, "resume_job", {"jobId": job_id, "requestKey": str(uuid.uuid4())})
    return _execute(workspace, admitted, cancel_event)
