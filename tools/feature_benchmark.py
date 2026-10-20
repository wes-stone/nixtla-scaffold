"""Frozen synthetic benchmark. No business sources, downloads or optional engines."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
import sys
import subprocess
import threading

import psutil

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from nixtla_scaffold import run_forecast
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec, SelectionPolicy
from nixtla_scaffold.blend import BlendRecipe, evaluate_blend, _weights, _predict, _score
from nixtla_scaffold.ensemble import observable_backtest_rows


PROTOCOL_PATH = Path(__file__).with_name("forecast_benchmark.json")


def fixture(task, seed):
    frames = []
    for item in range(task["series"]):
        rng = np.random.default_rng(seed + item)
        t = np.arange(task["length"], dtype=float)
        y = 100 + item * 25 + 12 * np.sin(2 * np.pi * t / task["season"])
        if task["kind"] == "trend":
            y += t * .15
        elif task["kind"] == "shift":
            y += 40 * (t >= len(t) * .65)
        y += rng.normal(0, 4, len(t))
        if task["kind"] == "intermittent":
            y = rng.binomial(1, .18, len(t)) * rng.gamma(3, 8, len(t))
        frames.append(pd.DataFrame({
            "unique_id": f"{task['name']}_{item}",
            "ds": pd.date_range("2015-01-01", periods=len(t), freq=task["freq"]), "y": y,
        }))
    return pd.concat(frames, ignore_index=True)


def specification(task, protocol, out):
    return ForecastSpec(
        horizon=task["horizon"], freq=task["freq"], season_length=task["season"],
        levels=tuple(protocol["interval_levels"]), model_policy="all",
        model_allowlist=tuple(protocol["models"]), run_mode="legacy", policy_version=0,
        weighted_ensemble=False, verbose=False,
        evaluation=EvaluationPolicy(max_windows=protocol["windows"], step_size=task["step"],
                                    evidence_directory=str(out / "exposure")),
        selection_policy=SelectionPolicy(objective=task["objective"], clip_negative=True),
    )


def native_predictions(data, task, protocol):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive, SeasonalNaive
    from mlforecast import MLForecast
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    h, season = task["horizon"], task["season"]
    sf = StatsForecast(models=[Naive(), SeasonalNaive(season_length=season),
                               AutoETS(season_length=season)], freq=task["freq"], n_jobs=1)
    lags = sorted(set([*range(1, 14), season, 2 * season]))
    mlf = MLForecast(models={"Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.))},
                     freq=task["freq"], lags=lags,
                     date_features=["month", "dayofweek"] if task["freq"] == "D" else ["month"])
    kwargs = dict(h=h, n_windows=protocol["windows"], step_size=task["step"], refit=True)
    sf_future = sf.forecast(df=data, h=h)
    sf_cv = sf.cross_validation(df=data, **kwargs)
    mlf.fit(data)
    future = sf_future.merge(mlf.predict(h), on=["unique_id", "ds"], validate="one_to_one")
    cv = sf_cv.merge(mlf.cross_validation(data, **kwargs).drop(columns="y"),
                     on=["unique_id", "cutoff", "ds"], validate="one_to_one")
    for frame in (future, cv):
        frame[protocol["models"]] = frame[protocol["models"]].clip(lower=0)
    return future, cv


def assert_parity(actual, expected, keys, models):
    actual, expected = actual.copy(), expected.copy()
    for column in ("ds", "cutoff"):
        if column in keys:
            actual[column] = pd.to_datetime(actual[column])
            expected[column] = pd.to_datetime(expected[column])
    columns = [*keys, *models]
    pd.testing.assert_frame_equal(
        actual[columns].sort_values(keys).reset_index(drop=True),
        expected[columns].sort_values(keys).reset_index(drop=True),
        check_dtype=False, rtol=1e-7, atol=1e-8,
    )


def native_selection(cv, future, history, recipe, horizon):
    """Reasonable native control: earlier-objective best single versus equal."""
    from nixtla_scaffold.greedy import objective_function
    output = cv.copy()
    output["NativeSelected"] = np.nan
    for _, group in output.groupby("unique_id"):
        for cutoff, current in group.groupby("cutoff"):
            prior = observable_backtest_rows(group, cutoff)
            prior = prior.loc[prior.groupby("cutoff").ds.transform("size").eq(horizon)]
            visible = history.loc[history.ds.le(cutoff)]
            best = _weights(replace(recipe, max_steps=1, max_models=1), prior, history=visible, horizon=horizon)
            if not best:
                continue
            equal = {name: 1 / len(recipe.members) for name in recipe.members}
            score = objective_function(prior, visible, objective=recipe.objective, horizon=horizon)
            chosen = best if score(_predict(prior, best).to_numpy()) <= score(_predict(prior, equal).to_numpy()) else equal
            output.loc[current.index, "NativeSelected"] = _predict(current, chosen)
    return output


def full_comparison(data, run, cv, future, task, protocol, target):
    from nixtla_scaffold import analyze
    from nixtla_scaffold.workbench_store import WorkbenchStore
    from nixtla_scaffold.workbench import run_frame
    members = tuple(protocol["models"])
    spec = replace(run.spec, evaluation=replace(run.spec.evaluation, cache_directory=str(target / "cache")))
    uncached_timings = []
    for _ in range(2):
        start = time.perf_counter()
        repeated = run_forecast(data, run.spec)
        uncached_timings.append(time.perf_counter() - start)
        assert_parity(repeated.backtest_predictions, run.backtest_predictions, ["unique_id", "cutoff", "ds"], members)
    cached_runs, timings = [], []
    for _ in range(3):
        start = time.perf_counter()
        cached_runs.append(run_forecast(data, spec))
        timings.append(time.perf_counter() - start)
    for cached in cached_runs:
        assert_parity(cached.all_models, run.all_models, ["unique_id", "ds"], members)
        assert_parity(cached.backtest_predictions, run.backtest_predictions, ["unique_id", "cutoff", "ds"], members)
    if cached_runs[-1].model_policy_resolution["prediction_cache"]["native_fits"]:
        raise AssertionError("Identical warm run performed redundant native fits")
    workspace = target / "workflow"
    start = time.perf_counter()
    # Separate cold workspace/cache; its allowance includes profiling, export and both improvements.
    workflow_spec = replace(spec, evaluation=replace(spec.evaluation, cache_directory=str(target / "workflow-cache")))
    result = analyze(data, workflow_spec, workspace=workspace, max_wall_clock_seconds=protocol["max_seconds_per_lane"],
                     improvement_policy="greedy")
    workflow_seconds = time.perf_counter() - start
    store = WorkbenchStore(workspace)
    workflow_cv = run_frame(store, result["result"]["runId"], "backtest")
    assert_parity(workflow_cv, cv, ["unique_id", "cutoff", "ds"], members)
    ablations = ensemble_comparison(data, cv, future, task, protocol, target)
    return {
        "cold_cache_seconds": timings[0], "warm_cache_seconds": timings[1:],
        "repeat_uncached_seconds": uncached_timings,
        "cache_receipts": [cached.model_policy_resolution["prediction_cache"] for cached in cached_runs],
        "workflow_seconds": workflow_seconds, "workflow_job": result["id"],
        "workflow_improvements": len(result["result"]["ensembles"]),
        "workflow_stage_timings": result["result"]["timings"], **ablations,
    }


def ensemble_comparison(data, cv, future, task, protocol, target):
    recipe = BlendRecipe("greedy", tuple(protocol["models"]), objective=task["objective"])
    start = time.perf_counter()
    greedy, _, _, greedy_receipt = evaluate_blend(cv, future, recipe, history=data, horizon=task["horizon"], clip_negative=True)
    horizon_recipe = replace(recipe, policy="horizon_greedy",
                             horizon_breaks=(task["horizon"] // 3, 2 * task["horizon"] // 3))
    grouped, _, _, grouped_receipt = evaluate_blend(cv, future, horizon_recipe, history=data,
                                                  horizon=task["horizon"], clip_negative=True, _allow_horizon_lab=True)
    native = native_selection(cv, future, data, recipe, task["horizon"])
    keys = ["unique_id", "cutoff", "ds"]
    paired = greedy.merge(grouped[keys + ["Blend"]].rename(columns={"Blend": "HorizonBlend"}),
                          on=keys, validate="one_to_one")
    paired = paired.merge(native[keys + ["NativeSelected"]], on=keys, validate="one_to_one")
    if not len(paired) or not np.isfinite(paired[["Blend", "HorizonBlend", "NativeSelected"]]).all().all():
        raise AssertionError("Incomplete matched ensemble comparison")
    scores = pd.DataFrame([{"unique_id": uid, **_score(group, model)}
                           for uid, group in paired.groupby("unique_id")
                           for model in ["NativeSelected", "EqualBlend", "PriorBest", "Blend", "HorizonBlend"]])
    if task["objective"] == "monthly_landing":
        from nixtla_scaffold.finance import finance_scorecard
        finance, _ = finance_scorecard(data, paired, list(scores.model.unique()), horizon=task["horizon"])
        scores = scores.merge(finance, on=["unique_id", "model"], validate="one_to_one")
    paired.to_csv(target / "paired_ablations.csv", index=False)
    scores.to_csv(target / "scores.csv", index=False)
    return {
        "ensemble_ablation_seconds": time.perf_counter() - start,
        "greedy": greedy_receipt, "horizon": grouped_receipt,
        "matched_rows": len(paired), "matched_cutoffs": int(paired.cutoff.nunique()),
        "scores": scores.replace({np.nan: None}).to_dict("records"),
        "intervals": {"status": "not_requested", "coverage": None, "width": None},
    }


def replay_development(source, out, protocol, tasks):
    from nixtla_scaffold.workbench import run_frame
    from nixtla_scaffold.workbench_store import WorkbenchStore
    from nixtla_scaffold.schema import _runtime_source_identity
    receipts = json.loads((source / "results.json").read_text())
    results = []
    for task in protocol["tasks"]:
        if tasks and task["name"] not in tasks:
            continue
        receipt = next(row for row in receipts if row["task"] == task["name"])
        if receipt["partition"] != "development":
            raise ValueError("Final assessment cannot be repurposed as development by this command.")
        if receipt["protocol_sha256"] != sha256(PROTOCOL_PATH.read_bytes()).hexdigest():
            raise ValueError("Replay requires the identical frozen task protocol.")
        old = source / task["name"] / task["name"]
        data = pd.read_csv(old / "input.csv", dtype={"unique_id": str}, parse_dates=["ds"], float_precision="round_trip")
        cv = pd.read_csv(old / "native_backtest.csv", dtype={"unique_id": str}, parse_dates=["ds", "cutoff"], float_precision="round_trip")
        if sha256((old / "input.csv").read_bytes()).hexdigest() != receipt["data_sha256"]:
            raise ValueError("Frozen benchmark source changed.")
        store = WorkbenchStore(old / "workflow")
        job = store.job(receipt["workflow_job"])
        future = run_frame(store, job["result"]["runId"], "audit/all_models.csv")
        target = out / task["name"]
        target.mkdir()
        results.append({
            "task": task["name"], "partition": "development", "frozen_control": str(old),
            "current_source": _runtime_source_identity(), "protocol_sha256": receipt["protocol_sha256"],
            **ensemble_comparison(data, cv, future, task, protocol, target),
        })
        print(f"{task['name']}: replay completed; no native refits", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=["development", "confirmation"], default="development")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--replay-from", help="Development-only ensemble ablation on immutable prior benchmark predictions")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_bytes(PROTOCOL_PATH.read_bytes())
    if args.replay_from:
        if args.partition != "development":
            raise ValueError("Replay is development-only; reserve a fresh final assessment.")
        replay_development(Path(args.replay_from).resolve(), out, protocol, args.tasks)
        return
    if not args.task_worker:
        from nixtla_scaffold.processes import run_script
        if args.partition == "confirmation":
            marker = out.parent / f"confirmation-{sha256(PROTOCOL_PATH.read_bytes()).hexdigest()[:16]}.consumed"
            with marker.open("x", encoding="utf-8") as handle:
                handle.write(str(out))
        results = []
        for task in protocol["tasks"]:
            if args.tasks and task["name"] not in args.tasks:
                continue
            destination = out / task["name"]
            command = [sys.executable, "-W", "ignore::FutureWarning", str(Path(__file__).resolve()),
                       "--task-worker", "--tasks", task["name"], "--partition", args.partition,
                       "--output", str(destination)]
            if args.baseline_only:
                command.append("--baseline-only")
            started = time.perf_counter()
            try:
                completed = run_script(command, cwd=Path.cwd(), timeout=protocol["max_seconds_per_lane"])
                if completed.returncode:
                    raise RuntimeError(completed.stderr[-4000:] + completed.stdout[-4000:])
                results.extend(json.loads((destination / "results.json").read_text()))
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                results.append({"task": task["name"], "status": "failed", "error": str(exc)})
            (out / "results.json").write_text(json.dumps(results, indent=2, default=str, allow_nan=False), encoding="utf-8")
            print(f"{task['name']}: {results[-1].get('status', 'completed')} ({time.perf_counter() - started:.1f}s)", flush=True)
        if any(row.get("status") == "failed" for row in results):
            raise SystemExit(1)
        return
    receipts = []
    for index, task in enumerate(protocol["tasks"]):
        if args.tasks and task["name"] not in args.tasks:
            continue
        target = out / task["name"]
        target.mkdir()
        data = fixture(task, protocol["seed"][args.partition] + index * 100)
        data.to_csv(target / "input.csv", index=False)
        spec = specification(task, protocol, target)
        peak = [psutil.Process().memory_info().rss]
        stopped = threading.Event()
        def measure():
            process = psutil.Process()
            while not stopped.wait(.1):
                try:
                    processes = [process, *process.children(recursive=True)]
                    peak[0] = max(peak[0], sum(p.memory_info().rss for p in processes if p.is_running()))
                except psutil.NoSuchProcess:
                    continue
        sampler = threading.Thread(target=measure, daemon=True)
        sampler.start()
        started = time.perf_counter()
        future, cv = native_predictions(data, task, protocol)
        native_seconds = time.perf_counter() - started
        started = time.perf_counter()
        run = run_forecast(data, spec)
        wrapper_seconds = time.perf_counter() - started
        assert_parity(run.all_models, future, ["unique_id", "ds"], protocol["models"])
        assert_parity(run.backtest_predictions, cv, ["unique_id", "cutoff", "ds"], protocol["models"])
        cv.to_csv(target / "native_backtest.csv", index=False)
        run.backtest_metrics.to_csv(target / "matched_scores.csv", index=False)
        receipt = {
            "task": task["name"], "partition": args.partition, "parity": True,
            "native_seconds": native_seconds, "wrapper_seconds": wrapper_seconds,
            "data_sha256": sha256((target / "input.csv").read_bytes()).hexdigest(),
            "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "runtime": run.model_policy_resolution.get("resolved_candidate_identity", {}),
            "native_model_fits": (3 * task["series"] + 1) * (protocol["windows"] + 1),
            "fit_count_definition": "Estimator-origin calls; SF per series, Ridge shared across the panel. Excludes internal AutoETS search subfits.",
            "wrapper_origin_group_fits": protocol["windows"] + 1,
            "spec": spec.to_dict(),
            "thread_limit": 1,
        }
        if not args.baseline_only:
            receipt.update(full_comparison(data, run, cv, future, task, protocol, target))
        stopped.set()
        sampler.join()
        receipt["peak_process_tree_rss_bytes"] = peak[0]
        receipts.append(receipt)
        (out / "results.json").write_text(json.dumps(receipts, indent=2, default=str, allow_nan=False), encoding="utf-8")
        print(f"{task['name']}: native={native_seconds:.2f}s matched={wrapper_seconds:.2f}s parity=pass", flush=True)


if __name__ == "__main__":
    main()
