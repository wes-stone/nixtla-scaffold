"""Typed local workbench operations. Opening the catalog is read-only."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import fields, replace
import csv
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import uuid

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from nixtla_scaffold.blend import LEARNED, BlendRecipe, component_models, evaluate_blend
from nixtla_scaffold.data import canonicalize_forecast_frame, read_tabular_source
from nixtla_scaffold.evaluation import disjoint_window_count
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.schema import ForecastContext, ForecastRun, ForecastSpec, forecast_spec_from_dict
from nixtla_scaffold.workbench_store import (
    MAX_ARTIFACT_BYTES, WorkbenchError, WorkbenchStore, _filesystem_path, canonical_json, digest, timestamp,
)

CONTRACT = json.loads(Path(__file__).with_name("workbench_contract.json").read_text(encoding="utf-8"))
ARTIFACTS = {
    "forecast": "forecast.csv",
    "history": "appendix/history.csv",
    "backtest": "audit/backtest_predictions.csv",
    "metrics": "audit/backtest_metrics.csv",
    "trust": "appendix/trust_summary.csv",
    "external": "external/comparison.csv",
    "external_metrics": "external/metrics.csv",
    "external_forecasts": "external/forecasts.csv",
}


def records(frame: pd.DataFrame) -> list[dict]:
    def value(item):
        if isinstance(item, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(item).isoformat() if pd.notna(item) else None
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, (int, np.integer)) and abs(item) > 2**53 - 1:
            return str(item)
        return item.item() if isinstance(item, np.generic) else item
    return [{key: value(item) for key, item in row.items()}
            for row in frame.astype(object).where(pd.notna(frame), None).to_dict("records")]


def _json_artifact(store: WorkbenchStore, run_id: str, name: str, default=None):
    _, run = store.run(run_id)
    if name not in json.loads(run["hashes"]):
        return default
    return json.loads(store.read_artifact(run_id, name))


def _frame(store: WorkbenchStore, run_id: str, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(store.read_artifact(run_id, name)), dtype={"unique_id": str},
                           keep_default_na=False, na_values=[""], float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _artifact_name(store: WorkbenchStore, run_id: str, name: str) -> str:
    _, run = store.run(run_id)
    names = json.loads(run["hashes"])
    if name in names:
        return name
    legacy = name.split("/")[-1]
    if legacy in names:
        return legacy
    raise WorkbenchError("missing_evidence", f"This version has no {name}. Recreate evidence rather than inferring it.")


def run_frame(store: WorkbenchStore, run_id: str, kind: str) -> pd.DataFrame:
    name = ARTIFACTS[kind] if kind in ARTIFACTS else kind
    return _frame(store, run_id, _artifact_name(store, run_id, name))


def _catalog(store: WorkbenchStore, data: dict | None = None) -> dict:
    offset = (data or {}).get("offset", 0)
    source_offset = (data or {}).get("sourceOffset", 0)
    if not store.initialized:
        return {"initialized": False, "runs": [], "jobs": [], "sources": [], "state": {}, "revision": 0}
    with store.connect(readonly=True) as conn:
        runs = [dict(row) for row in conn.execute(
            "SELECT id,name,created_at,metadata FROM wb_runs ORDER BY created_at DESC LIMIT 200 OFFSET ?", (offset,)
        )]
        sources = [dict(row) for row in conn.execute(
            "SELECT id,name,metadata FROM wb_sources ORDER BY rowid DESC LIMIT 200 OFFSET ?", (source_offset,)
        )]
        jobs = [dict(row) for row in conn.execute(
            "SELECT id,kind,status,created_at,updated_at,error,result,input FROM wb_jobs ORDER BY created_at DESC LIMIT 50"
        )]
        state = conn.execute("SELECT * FROM wb_state WHERE id='workspace'").fetchone()
        count = conn.execute("SELECT COUNT(*) FROM wb_runs").fetchone()[0]
        family_count = conn.execute("SELECT COUNT(DISTINCT name) FROM wb_runs").fetchone()[0]
        source_count = conn.execute("SELECT COUNT(*) FROM wb_sources").fetchone()[0]
        note_count = conn.execute("SELECT COUNT(*) FROM wb_notes").fetchone()[0]
        job_count = conn.execute("SELECT COUNT(*) FROM wb_jobs").fetchone()[0]
    for row in runs + sources:
        row["metadata"] = json.loads(row["metadata"])
    for row in sources:
        row["metadata"]["profile"] = {key: value for key, value in row["metadata"].get("profile", {}).items() if key != "series"}
    for row in jobs:
        data = json.loads(row.pop("input"))
        row["remainingSeconds"] = data.get("executionBudget", {}).get("remainingSeconds")
        row["resumeAvailable"] = (row["kind"] == "forecast" and row["status"] in {"failed", "interrupted", "cancelled"}
                                  and (row["remainingSeconds"] or 0) > 0 and not data.get("resumeChildId"))
        row["result"] = json.loads(row["result"]) if row["result"] else None
        if row["result"]:
            row["result"] = {key: value for key, value in row["result"].items() if key not in {"ensembles", "receipt", "challengers"}}
    return {
        "initialized": True, "runs": runs, "sources": sources, "jobs": jobs,
        "state": json.loads(state["body"]) if state else {},
        "revision": state["revision"] if state else 0,
        "catalogTruncated": count > offset + len(runs), "offset": offset, "totalRuns": count,
        "nextOffset": offset + len(runs) if count > offset + len(runs) else None,
        "sourceOffset": source_offset, "totalSources": source_count,
        "totalNotes": note_count, "totalJobs": job_count, "workspace": str(store.workspace),
        "totalForecasts": family_count,
        "nextSourceOffset": source_offset + len(sources) if source_count > source_offset + len(sources) else None,
    }


def _profile(store: WorkbenchStore, data: dict) -> dict:
    path = _filesystem_path(data["path"]).resolve(strict=True)
    if path.suffix.lower() not in {".csv", ".xlsx", ".xlsm"} or not path.is_file():
        raise WorkbenchError("unsupported_source", "Select a CSV or supported saved-value workbook.")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise WorkbenchError("too_large", "Source exceeds the local 128 MiB limit; provide an aggregate extract.")
    mapping = data.get("mapping", {})
    text_columns = ("unique_id", "series_id", "series", "product", mapping.get("id", "unique_id"))
    raw = read_tabular_source(path, sheet=mapping.get("sheet"), text_columns=text_columns)
    if raw.empty:
        raise WorkbenchError("empty_source", "No rows were found in the selected source.")
    columns = list(map(str, raw.columns))
    dates = [c for c in columns if c.lower() in {"ds", "date", "month", "period", "time", "timestamp"}]
    ids = [c for c in columns if c.lower() in {"unique_id", "series_id", "series", "product"}]
    time = mapping.get("time") or (dates[0] if len(dates) == 1 else "")
    series = mapping.get("id") or (ids[0] if len(ids) == 1 else "")
    numeric = [c for c in columns if c not in {time, series} and pd.api.types.is_numeric_dtype(raw[c])]
    target = mapping.get("target") or ("y" if "y" in columns else numeric[0] if len(numeric) == 1 else "")
    result = {
        "columns": columns, "mapping": {"time": time, "target": target, "id": series},
        "questions": [], "sourceId": None, "rows": len(raw), "preview": [],
    }
    if not time or not target:
        result["questions"] = ["Which column is the time axis?" if not time else "Which numeric measure should be forecast?"]
        return result
    if series and series not in text_columns:
        raw = read_tabular_source(path, sheet=mapping.get("sheet"), text_columns=(series,))
    frame = canonicalize_forecast_frame(raw, id_col=series or "unique_id", time_col=time, target_col=target)
    frequency = mapping.get("freq") or None
    dates = pd.to_datetime(frame["ds"]).drop_duplicates().sort_values()
    if frequency is None and len(dates) >= 3:
        frequency = pd.infer_freq(dates)
        if frequency is None:
            if dates.dt.is_month_end.all():
                frequency = "ME"
            elif dates.dt.is_month_start.all():
                frequency = "MS"
    if not frequency:
        result["questions"] = ["The calendar is irregular or too short to infer. What frequency should this forecast use?"]
        return result
    from nixtla_scaffold.research import _build_chronological_split
    horizon = data.get("horizon", 6)
    spec = ForecastSpec(freq=frequency, horizon=horizon, verbose=False)
    reserved = data.get("holdoutWindows", 1)
    split = _build_chronological_split(frame, horizon=horizon, confirmation_cutoffs=reserved)
    profile = profile_dataset(split.tuning_data, spec).to_dict()
    structural = profile_dataset(frame[["unique_id", "ds"]].assign(y=np.nan), spec).to_dict()
    # Structural checks can see keys and missingness, never held-out targets.
    profile["duplicate_rows"] = structural["duplicate_rows"]
    profile["missing_timestamps"] = structural["missing_timestamps"]
    profile["null_y"] = int(frame["y"].isna().sum())
    result.update({
        "profile": profile, "preview": records(split.tuning_data.head(8)),
        "mapping": {**result["mapping"], "freq": frequency},
        "findings": [
            f"{profile['series_count']} series across {profile['rows']} observations.",
            f"Observed calendar: {frequency}; tuning history ends {profile['end']}.",
            f"Latest {horizon * reserved} periods per eligible series reserved before target diagnostics; full-history forecasting consumes this holdout.",
            *profile["warnings"],
        ],
        "assumptions": [
            "Target units and stock/flow semantics are not inferred.",
            "This profile describes the supplied snapshot, not its source-system freshness.",
            "Saved workbook values are read without recalculation or external refresh." if path.suffix.lower() != ".csv"
            else "CSV values are a snapshot; no external connection was contacted.",
        ],
    })
    if profile["duplicate_rows"]:
        result["questions"] = ["Duplicate series/period keys need an explicit aggregation or deduplication decision."]
        return result
    metadata = {
        "mapping": result["mapping"], "profile": profile, "originalName": path.name,
        "originalHash": digest(path.read_bytes()),
        "requiresRepair": bool(profile["null_y"] or profile["missing_timestamps"]),
        "confirmationHorizon": horizon, "confirmationEligible": split.all_series_eligible,
        "reservedWindows": reserved,
    }
    result["sourceId"] = store.save_source(path.stem, frame.to_csv(index=False).encode("utf-8"), metadata)
    if metadata["requiresRepair"]:
        result["questions"] = ["Missing observations need an explicit repair policy before modeling."]
    return result


def _blend_artifact(store: WorkbenchStore, run_id: str, blend_id: str, name: str) -> bytes:
    with store.connect(readonly=True) as conn:
        row = conn.execute("SELECT * FROM wb_blends WHERE id=? AND run_id=?", (blend_id, run_id)).fetchone()
    if row is None:
        raise WorkbenchError("not_found", "Select a completed ensemble experiment.")
    receipt = json.loads(row["receipt"])
    if name not in receipt["artifact_hashes"]:
        raise WorkbenchError("missing_evidence", "This ensemble has no requested evidence artifact.")
    content = store.owned(str(Path(row["path"]) / name)).read_bytes()
    if digest(content) != receipt["artifact_hashes"][name]:
        raise WorkbenchError("evidence_changed", "Ensemble artifact hash changed; reimport its evidence.")
    return content


def _view(store: WorkbenchStore, data: dict) -> dict:
    run_id = data["runId"]
    _, run = store.run(run_id)
    kind = data.get("kind", "forecast")
    if kind.startswith("ensemble"):
        name = {"ensemble": "forecast.csv", "ensemble_backtest": "backtest.csv", "ensemble_metrics": "metrics.csv"}[kind]
        try:
            frame = pd.read_csv(io.BytesIO(_blend_artifact(store, run_id, data.get("blendId", ""), name)),
                                dtype={"unique_id": str}, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
    else:
        frame = run_frame(store, run_id, kind)
    if data.get("series") and "unique_id" in frame:
        frame = frame[frame["unique_id"].astype(str) == data["series"]]
    cutoffs = sorted(frame["cutoff"].dropna().astype(str).unique().tolist()) if "cutoff" in frame else []
    if data.get("cutoff") and "cutoff" in frame:
        frame = frame[pd.to_datetime(frame["cutoff"]) == pd.Timestamp(data["cutoff"])]
    offset, limit = data.get("offset", 0), data.get("limit", 500)
    if data.get("tail"):
        order = [column for column in ("cutoff", "ds") if column in frame]
        if order:
            frame = frame.sort_values(order)
        offset = max(0, len(frame) - limit)
    page = frame.iloc[offset:offset + limit]
    manifest = _json_artifact(store, run_id, "manifest.json", {})
    profile = _json_artifact(store, run_id, "profile.json", {})
    forecast = run_frame(store, run_id, "forecast")
    series = sorted(forecast["unique_id"].astype(str).unique().tolist())
    with store.connect(readonly=True) as conn:
        blends = [dict(r) for r in conn.execute("SELECT id,recipe,receipt FROM wb_blends WHERE run_id=?", (run_id,))]
        notes = [dict(r) for r in conn.execute("SELECT * FROM wb_notes WHERE run_id=? ORDER BY id DESC LIMIT 50", (run_id,))]
    for row in blends:
        row["recipe"] = json.loads(row["recipe"])
        row["receipt"] = json.loads(row["receipt"])
        row["receiptHash"] = digest(canonical_json(row["receipt"]).encode())
    artifact_names = json.loads(run["hashes"])
    all_models = run_frame(store, run_id, "audit/all_models.csv") if any(
        name in artifact_names for name in ("audit/all_models.csv", "all_models.csv")
    ) else pd.DataFrame()
    return {
        "rows": records(page), "columns": list(frame.columns), "total": len(frame),
        "nextOffset": offset + len(page) if offset + len(page) < len(frame) else None,
        "offset": offset, "cutoffs": cutoffs,
        "run": {"id": run_id, "name": run["name"], "metadata": json.loads(run["metadata"])},
        "spec": manifest.get("spec", {}), "profile": profile,
        "series": series[:1000], "seriesTruncated": len(series) > 1000,
        "models": component_models(all_models), "blends": blends, "notes": notes,
        "hypotheses": _json_artifact(store, run_id, "analyst_plan.json", {}).get("hypotheses", []),
        "artifacts": sorted(json.loads(run["hashes"])),
        "gate": _json_artifact(store, run_id, "appendix/accuracy_gate.json", {
            "status": "directional_only", "reason": "This version has no recorded accuracy gate; planning readiness is not established.",
        }),
    }


def _safe_spec(value: dict) -> ForecastSpec:
    unknown = set(value) - {field.name for field in fields(ForecastSpec)}
    if unknown:
        raise WorkbenchError("invalid_spec", f"Unknown forecast settings: {sorted(unknown)}")
    from nixtla_scaffold.presets import resolve_forecast_spec
    spec = resolve_forecast_spec(value)
    if spec.custom_models or spec.challengers:
        raise WorkbenchError("external_review_required", "Custom executables and FINN require the reviewed execution proposal, not automatic native analysis.")
    if (spec.id_col, spec.time_col, spec.target_col) != ("unique_id", "ds", "y"):
        raise WorkbenchError("canonical_columns_required", "Map source columns during inspection; saved snapshots always use unique_id, ds and y.")
    return replace(spec, verbose=False)


def _blend_hypotheses(run, candidates):
    from nixtla_scaffold.model_families import model_family
    if len(candidates) < 2 or run.backtest_predictions.empty or run.spec.events or run.spec.hierarchy or run.spec.hierarchy_reconciliation != "none" or run.spec.transform.normalization_factor_col:
        return []
    errors = run.backtest_metrics.loc[run.backtest_metrics["model"].isin(candidates), "rmse"].dropna()
    spread = float(errors.std(ddof=0) / errors.mean()) if len(errors) and errors.mean() > 0 else 0.
    families = sorted({model_family(model) for model in candidates})
    return [
        {"policy": "inverse_error" if spread > .2 else "equal",
         "reason": f"Observed relative RMSE dispersion is {spread:.2f}. " + ("Test whether prior-error weighting avoids consistently weak members." if spread > .2 else "Similar observed errors favor a simple equal-weight control."),
         "mechanism": "Reduce component-specific error without fitting to the evaluation origin.", "status": "retrospective_hypothesis"},
        {"policy": "family_diverse" if len(families) > 1 else "top_k",
         "reason": f"{len(families)} eligible families: {', '.join(families)}. " + ("Test whether independent family representation beats redundant models." if len(families) > 1 else "Test a capped subset chosen only from earlier observable outcomes."),
         "mechanism": "Reduce redundant or weak components; compare on matched outer windows.", "status": "retrospective_hypothesis"},
    ]


def _compare_versions(store: WorkbenchStore, data: dict) -> dict:
    run_id, reference_id = data["runId"], data["referenceRunId"]
    current_spec = _json_artifact(store, run_id, "manifest.json", {})["spec"]
    previous_spec = _json_artifact(store, reference_id, "manifest.json", {})["spec"]
    for field in ("freq", "unit_label"):
        if current_spec.get(field) != previous_spec.get(field):
            raise WorkbenchError("incompatible_versions", f"Versions have different {field}; no comparable delta was computed.")
    frames = []
    for selected, label in ((run_id, "current"), (reference_id, "previous")):
        frame = run_frame(store, selected, "forecast")[["unique_id", "ds", "yhat"]]
        if data.get("series"):
            frame = frame[frame["unique_id"].astype(str) == data["series"]]
        if frame.duplicated(["unique_id", "ds"]).any():
            raise WorkbenchError("duplicate_keys", "Version has duplicate series/period keys.")
        frame["ds"] = pd.to_datetime(frame["ds"])
        frames.append(frame.rename(columns={"yhat": label}))
    compared = frames[0].merge(frames[1], on=["unique_id", "ds"], how="outer", validate="one_to_one").sort_values(["unique_id", "ds"])
    compared["delta"] = compared["current"] - compared["previous"]
    compared["paired"] = np.isfinite(compared[["current", "previous"]]).all(axis=1)
    offset, limit = data.get("offset", 0), data.get("limit", 500)
    page = compared.iloc[offset:offset + limit]
    return {
        "rows": records(page), "columns": list(compared.columns), "total": len(compared),
        "nextOffset": offset + len(page) if offset + len(page) < len(compared) else None,
        "runId": run_id, "referenceRunId": reference_id,
        "limitations": "Forecast-to-forecast differences, not an accuracy ranking. Origins may differ; missing periods remain unpaired. Matching labels do not certify business semantics.",
    }


def _start_job(store: WorkbenchStore, data: dict) -> dict:
    from nixtla_scaffold.workbench_jobs import identity, runtime_fingerprint, start_budget, remaining_seconds
    request_fingerprint = digest(canonical_json(data).encode())
    with store.connect() as conn:
        existing = conn.execute("SELECT * FROM wb_jobs WHERE request_key=?", (data["requestKey"],)).fetchone()
        if existing:
            if json.loads(existing["input"]).get("requestFingerprint") != request_fingerprint:
                raise WorkbenchError("request_conflict", "This request key already refers to different inputs.")
            return {"jobId": existing["id"], "created": False, "status": existing["status"]}
        if conn.execute("SELECT COUNT(*) FROM wb_jobs WHERE status IN ('queued','running','cancel_requested')").fetchone()[0] >= 2:
            raise WorkbenchError("busy", "Two jobs are already active. Stop or finish one before starting another.")
    kind = data["kind"]
    if kind in {"forecast", "research"}:
        path, metadata = store.source(data.get("sourceId", ""))
        settings = {**data.get("spec", {})}
        if "horizon" not in settings:
            raise WorkbenchError("horizon_required", "Set a forecast horizon; the worker will not invent the decision horizon.")
        settings.setdefault("freq", metadata["mapping"]["freq"])
        if metadata["requiresRepair"] and "fill_method" not in settings:
            raise WorkbenchError("repair_required", "Choose a missing-observation policy explicitly before modeling.")
        spec = _safe_spec(settings)
        if spec.horizon > 120:
            raise WorkbenchError("budget_review_required", "Horizons beyond 120 periods require a reviewed external workflow.")
        data = {**data, "spec": spec.to_dict(), "sourceFingerprint": digest(path.read_bytes())}
    elif kind == "blend":
        store.run(data.get("runId", ""))
        BlendRecipe.from_dict(data.get("recipe", {}))
        spec = _json_artifact(store, data["runId"], "manifest.json", {}).get("spec", {})
        if spec.get("events") or spec.get("hierarchy_reconciliation", "none") != "none" or spec.get("transform", {}).get("normalization_factor_col"):
            raise WorkbenchError("postprocessing_evidence_required", "This run has scenario, normalization or reconciliation postprocessing. Evaluate the complete deployed pipeline in research before constructing a blend.")
    elif kind == "adopt":
        if data.get("approved") is not True or not all(data.get(key) for key in ("runId", "blendId", "expectedReceipt")):
            raise WorkbenchError("approval_required", "Explicitly approve directional adoption of the exact reviewed ensemble receipt.")
        store.run(data["runId"])
    elif kind in {"refresh", "actuals", "lock"}:
        from nixtla_scaffold.workbench_operations import prepare
        data = prepare(store, data)
    elif kind == "confirm_blend":
        from nixtla_scaffold.workbench_confirmation import prepare
        data = prepare(store, data)
    elif kind == "pipeline":
        from nixtla_scaffold.workbench_execution import reviewed
        _, receipt = reviewed(store, data)
        if receipt.get("sourceId"):
            data = {**data, "sourceId": receipt["sourceId"], "sourceFingerprint": receipt["sourceFingerprint"]}
    # Windows virtualenv launchers can be short-lived parents of the real worker.
    # The service declares its stable identity instead of relying on that parent.
    owner_pid = int(os.environ.get("NIXTLA_WORKBENCH_OWNER_PID", os.getpid()))
    budget_seconds = min(900, data.get("maxWallClockSeconds", 900))
    context_budget = data.get("spec", {}).get("context", {}).get("research_budget", {}) if data.get("spec", {}).get("context") else {}
    if context_budget.get("max_wall_clock_minutes") is not None:
        budget_seconds = min(budget_seconds, context_budget["max_wall_clock_minutes"] * 60)
    data = {**data, "requestFingerprint": request_fingerprint, "serviceOwner": identity(owner_pid),
            "runtimeFingerprint": runtime_fingerprint(),
            "executionBudget": start_budget({"limitSeconds": budget_seconds, "remainingSeconds": budget_seconds, "consumedSeconds": 0.})}
    job_id = str(uuid.uuid4())
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT * FROM wb_jobs WHERE request_key=?", (data["requestKey"],)).fetchone()
        if existing:
            if json.loads(existing["input"]).get("requestFingerprint") != request_fingerprint:
                raise WorkbenchError("request_conflict", "This request key already refers to different inputs.")
            return {"jobId": existing["id"], "created": False, "status": existing["status"]}
        if conn.execute("SELECT COUNT(*) FROM wb_jobs WHERE status IN ('queued','running','cancel_requested')").fetchone()[0] >= 2:
            raise WorkbenchError("busy", "Two jobs are already active. Stop or finish one before starting another.")
        if kind == "research":
            windows = spec.context.promotion_policy.minimum_confirmation_cutoffs if spec.context else 1
            if not metadata.get("confirmationEligible") or metadata.get("confirmationHorizon") != spec.horizon or metadata.get("reservedWindows", 1) < windows:
                raise WorkbenchError("confirmation_unavailable", "Re-profile this source for the research horizon. Every series needs enough untouched confirmation history.")
            if conn.execute("SELECT 1 FROM wb_jobs WHERE json_extract(input,'$.sourceFingerprint')=? LIMIT 1",
                            (data["sourceFingerprint"],)).fetchone():
                raise WorkbenchError("confirmation_consumed", "This snapshot has already entered a forecast or research job. Its confirmation cannot be reused as untouched. Bring a new unexposed snapshot for confirmatory research.")
            from nixtla_scaffold.workbench_confirmation import assert_unexposed
            assert_unexposed(store, path, spec.horizon, windows)
        if kind == "confirm_blend":
            from nixtla_scaffold.workbench_confirmation import prepare
            prepare(store, data)
        conn.execute("INSERT INTO wb_jobs VALUES(?,?,?,?,?,?,?,?,?)", (
            job_id, data["requestKey"], kind, "queued", canonical_json(data), None,
            None, timestamp(), timestamp(),
        ))
    store.event(job_id, "queued", f"{kind.capitalize()} queued within the local work budget.")
    return {"jobId": job_id, "created": True, "status": "queued", "budgetSeconds": remaining_seconds(data)}


def _evaluate(store: WorkbenchStore, run_id: str, recipe: BlendRecipe) -> dict:
    with store.snapshot_lock():
        return _evaluate_locked(store, run_id, recipe)


def _evaluate_locked(store: WorkbenchStore, run_id: str, recipe: BlendRecipe) -> dict:
    manifest = _json_artifact(store, run_id, "manifest.json", {})
    spec = forecast_spec_from_dict(manifest["spec"])
    if recipe.policy in LEARNED and recipe.objective != spec.selection_policy.objective:
        raise ValueError("The learned recipe must optimize the run's declared objective.")
    history = run_frame(store, run_id, manifest["outputs"].get("observed_history", "history"))
    bt, future, metrics, receipt = evaluate_blend(
        run_frame(store, run_id, "backtest"),
        run_frame(store, run_id, "audit/all_models.csv"), recipe,
        history=history, horizon=spec.horizon, clip_negative=spec.selection_policy.clip_negative,
    )
    blend_id = digest((run_id + recipe.fingerprint).encode())
    out = store.owned(f"blends/{blend_id}")
    with store.connect(readonly=True) as conn:
        existing = conn.execute("SELECT receipt FROM wb_blends WHERE id=?", (blend_id,)).fetchone()
    if existing:
        for name in json.loads(existing[0])["artifact_hashes"]:
            _blend_artifact(store, run_id, blend_id, name)
        return {"blendId": blend_id, "receipt": json.loads(existing[0]), "runId": run_id}
    out.mkdir(parents=True, exist_ok=True)
    manifest = _json_artifact(store, run_id, "manifest.json", {})
    if spec.selection_policy.objective == "monthly_landing" and not bt.empty:
        from nixtla_scaffold.finance import finance_scorecard
        spec = forecast_spec_from_dict(manifest["spec"])
        history = run_frame(store, run_id, manifest["outputs"].get("observed_history", "history"))
        scored, windows = finance_scorecard(history, bt, list(metrics.model.unique()), horizon=spec.horizon,
                                            clip_negative=spec.selection_policy.clip_negative)
        metrics = metrics.merge(scored, on=["unique_id", "model"], how="left", validate="one_to_one")
        receipt["finance_scorecard"] = {"contract": spec.selection_policy.contract(), "scores": records(scored),
                                        "windows": records(windows), "cohort": "exact_paired_blend_replay"}
    bt.to_csv(out / "backtest.csv", index=False)
    future.to_csv(out / "forecast.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    receipt["artifact_hashes"] = {name: digest((out / name).read_bytes()) for name in ("backtest.csv", "forecast.csv", "metrics.csv")}
    (out / "receipt.json").write_text(canonical_json(receipt), encoding="utf-8")
    with store.connect() as conn:
        conn.execute("INSERT INTO wb_blends VALUES(?,?,?,?,?)", (
            blend_id, run_id, canonical_json(recipe.to_dict()),
            str(out.relative_to(store.root)), canonical_json(receipt),
        ))
    return {"blendId": blend_id, "receipt": receipt, "runId": run_id}


def _blend_cv_metadata(backtest: pd.DataFrame, *, horizon: int, freq: str) -> pd.DataFrame:
    rows = []
    for uid, group in backtest.groupby("unique_id", sort=True):
        lengths = []
        cutoffs = []
        for cutoff, window in group.groupby("cutoff", sort=True):
            expected = pd.date_range(cutoff, periods=len(window) + 1, freq=freq)[1:]
            observed = pd.DatetimeIndex(window["ds"].sort_values())
            if not observed.equals(expected):
                raise WorkbenchError("coverage_gap", "Adoption needs contiguous evaluation periods from each origin. Review the incomplete paired cohort without adopting it.")
            lengths.append(len(window))
            cutoffs.append(cutoff)
        steps = {len(pd.date_range(left, right, freq=freq)) - 1 for left, right in zip(cutoffs, cutoffs[1:])}
        selected_horizon = min(lengths)
        rows.append({
            "unique_id": uid, "requested_horizon": horizon, "selection_horizon": selected_horizon,
            "cv_windows": len(cutoffs), "cv_step_size": next(iter(steps)) if len(steps) == 1 else None,
            "cv_disjoint_windows": disjoint_window_count(group, horizon=horizon),
            "cv_horizon_matches_requested": selected_horizon == horizon,
        })
    return pd.DataFrame(rows)


def _adopt_blend(store: WorkbenchStore, data: dict, out: Path) -> dict:
    run_id, blend_id = data["runId"], data["blendId"]
    _, parent = store.run(run_id)
    with store.connect(readonly=True) as conn:
        blend = conn.execute("SELECT * FROM wb_blends WHERE id=? AND run_id=?", (blend_id, run_id)).fetchone()
        prior = conn.execute("SELECT id,metadata FROM wb_runs WHERE json_extract(metadata,'$.adoptedBlendId')=?", (blend_id,)).fetchone()
    if blend is None:
        raise WorkbenchError("not_found", "The reviewed ensemble experiment no longer exists.")
    receipt = json.loads(blend["receipt"])
    if receipt.get("recipe", {}).get("policy") == "horizon_greedy":
        raise WorkbenchError("unqualified_policy", "The horizon-group prototype did not qualify for adoption.")
    if digest(canonical_json(receipt).encode()) != data["expectedReceipt"]:
        raise WorkbenchError("stale_evidence", "The reviewed ensemble receipt changed. Review it again before adoption.")
    for name in receipt["artifact_hashes"]:
        _blend_artifact(store, run_id, blend_id, name)
    if prior:
        return {"runId": prior["id"], "reused": True, "planningReady": json.loads(prior["metadata"]).get("planningReady", False)}
    if not receipt["paired_rows"]:
        raise WorkbenchError("insufficient_evidence", "A blend without paired backtest evidence cannot be adopted.")
    source_manifest = _json_artifact(store, run_id, "manifest.json", {})
    source_spec = source_manifest["spec"]
    spec = forecast_spec_from_dict(source_spec)
    if spec.events or spec.hierarchy or spec.hierarchy_reconciliation != "none" or spec.transform.normalization_factor_col:
        raise WorkbenchError("postprocessing_evidence_required", "Adoption requires full-pipeline evidence for scenarios, hierarchy and normalized targets.")
    context = spec.context or ForecastContext(source_discovery_enabled=False, requested_horizon=spec.horizon)
    spec = replace(spec, context=replace(context, accuracy_policy=replace(context.accuracy_policy, require_interval_evidence=True)))
    def frame(name):
        return pd.read_csv(io.BytesIO(_blend_artifact(store, run_id, blend_id, name)), dtype={"unique_id": str}, float_precision="round_trip")
    future, bt, metrics = frame("forecast.csv"), frame("backtest.csv"), frame("metrics.csv")
    original = run_frame(store, run_id, "forecast")
    key = lambda df: set(zip(df["unique_id"].astype(str), pd.to_datetime(df["ds"])))
    if key(future) != key(original) or future.duplicated(["unique_id", "ds"]).any() or not np.isfinite(future["yhat"]).all():
        raise WorkbenchError("coverage_gap", "Every original series and future period must have one finite blend prediction.")
    history = run_frame(store, run_id, "history")
    observed_path = source_manifest.get("outputs", {}).get("observed_history")
    observed_history = run_frame(store, run_id, observed_path) if observed_path else pd.DataFrame()
    history["ds"] = pd.to_datetime(history["ds"])
    future["ds"] = pd.to_datetime(future["ds"])
    bt["ds"], bt["cutoff"] = pd.to_datetime(bt["ds"]), pd.to_datetime(bt["cutoff"])
    freq = spec.freq or _json_artifact(store, run_id, "profile.json", {}).get("freq")
    if not freq:
        raise WorkbenchError("missing_evidence", "The saved evaluation calendar is required for adoption.")
    if spec.interval_calibration.method != "native" and "interval_calibration" not in receipt:
        from nixtla_scaffold.workbench_confirmation import _experimental_blend_intervals
        from nixtla_scaffold.blend import BlendRecipe
        bt, future, evaluation = _experimental_blend_intervals(
            bt.iloc[:0], bt.assign(partition="confirmation"), future, replace(spec, freq=freq),
            BlendRecipe.from_dict(receipt["recipe"]))
        bt = bt.drop(columns=["partition"])
        receipt = {**receipt, "interval_calibration": evaluation, "interval_status": "experimental",
                   "interval_calibration_source": "retrospective_frozen_replay",
                   "calibration": {"method": f"conformalforecast_{spec.interval_calibration.method}"}}
    metrics = metrics.merge(_blend_cv_metadata(bt, horizon=spec.horizon, freq=freq), on="unique_id", validate="many_to_one")
    selected = metrics[metrics["model"] == "Blend"].rename(columns={"model": "selected_model"}).copy()
    if set(selected["unique_id"]) != set(future["unique_id"]):
        raise WorkbenchError("coverage_gap", "Every adopted series needs paired historical blend evidence.")
    selected["selection_reason"] = "Explicit adoption of a frozen ensemble; inspect its confirmation receipt. Not an automatic promotion or planning lock."
    metrics["abs_bias"] = metrics["bias"].abs()
    # Regenerate every portable output from the candidate itself. Never copy the
    # parent's charts, intervals or readiness claims beside new predictions.
    interval_names = {col: col.replace("yhat_lo_", "Blend-lo-").replace("yhat_hi_", "Blend-hi-") for col in future if col.startswith(("yhat_lo_", "yhat_hi_"))}
    confirmed = receipt.get("confirmation_status") == "independently_reserved_within_workspace"
    from nixtla_scaffold.intervals import blend_candidate_evidence
    candidate = ForecastRun(
        history=history, forecast=future.drop(columns=["weights"]),
        observed_history=observed_history,
        all_models=future[["unique_id", "ds", "yhat", *interval_names]].rename(columns={"yhat": "Blend", **interval_names}),
        model_selection=selected, backtest_metrics=metrics[metrics["model"] == "Blend"],
        backtest_predictions=bt[["unique_id", "ds", "cutoff", "y", "Blend", *[col for col in bt if col.startswith(("Blend-lo-", "Blend-hi-"))]]],
        profile=profile_dataset(history, spec), spec=spec, engine="workbench_blend",
        model_policy_resolution={"candidate_evidence": blend_candidate_evidence(
            future.rename(columns={"yhat": "Blend", **interval_names}), bt, receipt, spec.horizon,
        ), **({"interval_calibration": receipt["interval_calibration"]} if "interval_calibration" in receipt else {})},
        warnings=[
            "Explicit ensemble adoption; reserved confirmation and prior-residual calibration are recorded." if confirmed else "Explicit directional ensemble adoption; no independent confirmation or calibrated intervals.",
            "This recipe has no shrinkage, scenario or reconciliation postprocessing.",
            *(["Experimental intervals are recalibrated from this exact frozen replay; prior confirmation is not reused."]
              if receipt.get("interval_calibration_source") == "retrospective_frozen_replay" else []),
        ],
    )
    from nixtla_scaffold.finance import enrich_frozen_run
    candidate = enrich_frozen_run(candidate)
    destination = out / "adopted"
    candidate.to_directory(destination)
    gate_path = destination / "appendix" / "accuracy_gate.json"
    if not gate_path.exists():
        raise WorkbenchError("claim_requires_confirmation", "The adopted version needs its own regenerated claim gate.")
    planning = json.loads(gate_path.read_text(encoding="utf-8")).get("status") == "planning_ready"
    if planning and (not confirmed or not all(row["passes_numeric_policy"] for row in receipt["confirmation"])):
        raise WorkbenchError("claim_requires_confirmation", "An unconfirmed candidate cannot acquire planning-ready status.")
    evidence = {**receipt, "parentRunId": run_id, "blendId": blend_id, "adoption": "explicit_reviewed" if confirmed else "explicit_directional",
                "postprocessing": "none", "planningReady": planning}
    (destination / "ensemble_adoption.json").write_text(canonical_json(evidence), encoding="utf-8")
    if data.get("refreshParentId"):
        from nixtla_scaffold.refresh import write_refresh_artifacts
        write_refresh_artifacts(store.run(data["refreshParentId"])[0], destination)
    result = store.import_run(destination, name=parent["name"], metadata={
        **json.loads(parent["metadata"]), "adoptedBlendId": blend_id,
        "parentRunId": data.get("refreshParentId", run_id), "componentRunId": run_id,
        "adopted": True, "planningReady": planning,
    })
    return {"runId": result["id"], "reused": result["reused"], "planningReady": planning}


def execute_job(store: WorkbenchStore, job_id: str) -> dict:
    from nixtla_scaffold.workbench_jobs import claim_worker, check_budget
    try:
        if not claim_worker(store, job_id):
            return store.job(job_id)
    except WorkbenchError as exc:
        if exc.code == "runtime_changed":
            store.finish_job(job_id, "failed", error=str(exc))
            store.event(job_id, "failed", str(exc))
        raise
    job = store.job(job_id)
    data = job["input"]
    out = store.owned(f"jobs/{job_id}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        check_budget(store, job_id)
        if job["kind"] == "pipeline":
            from nixtla_scaffold.workbench_execution import execute
            store.event(job_id, "running", "Executing the explicitly reviewed script and query snapshots within the local budget.")
            result = execute(store, {**data, "jobId": job_id}, out)
        elif job["kind"] == "confirm_blend":
            from nixtla_scaffold.workbench_confirmation import execute
            result = execute(store, {**data, "jobId": job_id}, out)
        elif job["kind"] in {"refresh", "actuals", "lock"}:
            from nixtla_scaffold.workbench_operations import execute
            store.event(job_id, "running", "Applying the reviewed version maintenance operation; original forecast artifacts stay immutable.")
            result = execute(store, data, out)
        elif job["kind"] == "adopt":
            store.event(job_id, "running", "Saving the explicitly approved directional blend as a new version; previous evidence is unchanged.")
            result = _adopt_blend(store, data, out)
        elif job["kind"] == "blend":
            store.event(job_id, "running", "Evaluating the frozen recipe on paired, observable historical outcomes.")
            result = _evaluate(store, data["runId"], BlendRecipe.from_dict(data["recipe"]))
        else:
            source, _ = store.source(data["sourceId"])
            if digest(source.read_bytes()) != data["sourceFingerprint"]:
                raise WorkbenchError("source_changed", "The queued source changed before execution.")
            spec = _safe_spec(data["spec"])
            if job["kind"] == "research":
                from nixtla_scaffold.experiments import run_optimizer
                store.event(job_id, "running", "Running the existing chronological optimizer: at most two iterations, one variant each.")
                research = run_optimizer(
                    source, spec, output_dir=out / "research",
                    max_iterations=2, max_variants=1,
                )
                from nixtla_scaffold.research import archive_research_evidence
                baseline = archive_research_evidence(research.output_dir)
                imported = store.import_run(baseline, name=data.get("name") or "Research baseline", metadata={
                    "researchSourceId": data["sourceId"], "jobId": job_id,
                    "adopted": False, "purpose": "research_tuning_baseline",
                })
                promotion = json.loads((research.output_dir / "promotion_decision.json").read_text(encoding="utf-8"))
                result = {"research": True, "jobId": job_id, "runId": imported["id"],
                          "stopReason": research.manifest["stopped_reason"],
                          "confirmationConsumed": promotion["confirmation"].get("exposure", {}).get("status") == "consumed_for_frozen_confirmation"}
            else:
                from nixtla_scaffold.analysis import run_analysis_job
                result = run_analysis_job(store, job_id, source, spec, out,
                                          forecast=run_forecast, evaluate=_evaluate, hypotheses=_blend_hypotheses)
        if store.job(job_id)["status"] == "cancel_requested":
            return store.finish_job(job_id, "cancelled", error="Stopped before completion.")
        store.event(job_id, "completed", "Artifacts saved. Statistical readiness is separate from job completion.")
        return store.finish_job(job_id, "completed", result=result)
    except (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error) as exc:
        status = "cancelled" if isinstance(exc, WorkbenchError) and exc.code == "cancelled" else "failed"
        store.event(job_id, status, str(exc))
        store.finish_job(job_id, status, error=str(exc))
        raise


def dispatch(workspace: str | Path, operation: str, data: dict) -> dict:
    if operation not in CONTRACT["operations"]:
        raise WorkbenchError("unknown_operation", "Unknown workbench operation.")
    errors = list(Draft202012Validator(CONTRACT["operations"][operation]).iter_errors(data))
    if errors:
        raise WorkbenchError("invalid_input", errors[0].message)
    if operation == "runtime_capabilities":
        from nixtla_scaffold.runtime import runtime_capabilities
        return runtime_capabilities()
    store = WorkbenchStore(workspace)
    if operation == "catalog":
        return _catalog(store, data)
    if operation == "initialize":
        return store.initialize()
    if not store.initialized:
        raise WorkbenchError("not_initialized", "Create this local workspace before saving or running analysis.")
    if operation == "profile":
        return _profile(store, data)
    if operation == "import_run":
        return store.import_run(data["path"], name=data.get("name", ""))
    if operation == "view":
        return _view(store, data)
    if operation == "compare_versions":
        return _compare_versions(store, data)
    if operation in {"judgment_source", "judgment_import", "judgment_save", "judgment_export", "judgment_pipeline"}:
        from nixtla_scaffold import workbench_judgment
        return {"judgment_source": workbench_judgment.source, "judgment_import": workbench_judgment.import_rows,
                "judgment_save": workbench_judgment.save, "judgment_export": workbench_judgment.export,
                "judgment_pipeline": workbench_judgment.pipeline}[operation](store, data)
    if operation in {"calendar_preview", "calendar_export"}:
        from nixtla_scaffold import workbench_calendar
        return (workbench_calendar.preview if operation == "calendar_preview" else workbench_calendar.export)(store, data)
    if operation in {"explore_catalog", "explore_schema", "explore_preview", "explore_queries", "explore_save", "explore_export", "explore_download"}:
        from nixtla_scaffold import workbench_explore
        handler = {"explore_catalog": workbench_explore.catalog, "explore_schema": workbench_explore.schema,
                   "explore_preview": workbench_explore.preview, "explore_queries": workbench_explore.saved_queries,
                   "explore_save": workbench_explore.save_query, "explore_export": workbench_explore.export,
                   "explore_download": workbench_explore.download}[operation]
        return handler(store, data)
    if operation in {"library", "source_view", "file_view"}:
        from nixtla_scaffold import workbench_library
        handler = {"library": workbench_library.browse, "source_view": workbench_library.source_view,
                   "file_view": workbench_library.file_view}[operation]
        return handler(store, data)
    if operation == "ledger_view":
        from nixtla_scaffold.workbench_operations import ledger_view
        return ledger_view(store, data)
    if operation == "prepare_pipeline":
        from nixtla_scaffold.workbench_execution import prepare
        return prepare(store, data)
    if operation == "prepare_forecast":
        from nixtla_scaffold.workbench_execution import prepare_forecast
        return prepare_forecast(store, data)
    if operation == "prepare_external":
        from nixtla_scaffold.workbench_execution import prepare_external
        return prepare_external(store, data)
    if operation == "context":
        if not data.get("runId"):
            return {"document": data.get("document", "Data"),
                    "authorization": "Context only. No forecast version selected. No execution or new source access authorized."}
        _, row = store.run(data["runId"])
        gate = _json_artifact(store, data["runId"], "appendix/accuracy_gate.json", None)
        return {
            "runId": row["id"], "name": row["name"], "series": data.get("series"),
            "document": data.get("document"), "gate": gate,
            "spec": _json_artifact(store, data["runId"], "manifest.json", {}).get("spec"),
            "evidence": sorted(json.loads(row["hashes"])),
            "authorization": "Context only. Does not authorize execution, adoption, external access or publication.",
        }
    if operation == "artifact":
        content = store.read_artifact(data["runId"], data["name"])
        if len(content) > 64 * 1024 or Path(data["name"]).suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".txt", ".yaml"}:
            raise WorkbenchError("bounded_preview", "This artifact is binary or exceeds the 64 KiB preview limit. Use the original portable output.")
        return {"name": data["name"], "text": content.decode("utf-8"), "sha256": digest(content)}
    if operation == "start_job":
        return _start_job(store, data)
    if operation == "resume_job":
        from nixtla_scaffold.workbench_jobs import resume_job
        return resume_job(store, data)
    if operation == "recover_jobs":
        from nixtla_scaffold.workbench_jobs import recover_jobs
        return recover_jobs(store)
    if operation == "job":
        result = store.job(data["jobId"])
        with store.connect(readonly=True) as conn:
            result["events"] = [dict(row) for row in conn.execute(
                "SELECT * FROM wb_events WHERE job_id=? ORDER BY sequence DESC LIMIT 100", (data["jobId"],)
            )]
        return result
    if operation == "job_evidence":
        store.job(data["jobId"])
        base = store.owned(f"jobs/{data['jobId']}/research")
        result = {}
        for name in ("promotion_decision.json", "stop_receipt.json", "research_plan.json",
                     "driver_research.json", "signal_experiment_dispositions.json"):
            path = base / name
            if path.exists() and path.stat().st_size <= 64 * 1024:
                result[name] = json.loads(path.read_text(encoding="utf-8"))
            elif path.exists():
                result[name] = {"status": "exceeds_inline_bound", "bytes": path.stat().st_size,
                                "artifact": f"research/{name}", "reason": "Open the saved research version's file explorer; evidence was not sampled."}
        return result
    if operation == "cancel_job":
        with store.connect() as conn:
            conn.execute("UPDATE wb_jobs SET status='cancel_requested',updated_at=? WHERE id=? AND status IN ('queued','running')",
                         (timestamp(), data["jobId"]))
        return store.job(data["jobId"])
    if operation == "save_state":
        body = canonical_json(data["state"])
        if len(body.encode()) > 64 * 1024:
            raise WorkbenchError("too_large", "Workspace state exceeds 64 KiB.")
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT revision FROM wb_state WHERE id='workspace'").fetchone()
            revision = row[0] if row else 0
            if revision != data["expectedRevision"]:
                raise WorkbenchError("revision_conflict", "Workspace changed in another view. Reload saved state before editing.")
            conn.execute("INSERT INTO wb_state VALUES('workspace',?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,body=excluded.body",
                         (revision + 1, body))
        return {"revision": revision + 1}
    if operation == "save_note":
        store.run(data["runId"])
        with store.connect() as conn:
            cursor = conn.execute("INSERT INTO wb_notes(run_id,body,created_at) VALUES(?,?,?)", (data["runId"], data["body"], timestamp()))
        return {"id": cursor.lastrowid}
    if operation == "evidence":
        from nixtla_scaffold.workbench_evidence import evidence
        return evidence(store, data)
    if operation == "evidence_image":
        from nixtla_scaffold.workbench_evidence import image
        return image(store, data)
    if operation in {"export_values", "export_evidence"}:
        kind = data.get("kind", "")
        if operation == "export_evidence":
            from nixtla_scaffold.workbench_evidence import content as evidence_content
            if not data["name"].endswith(".csv"):
                raise WorkbenchError("invalid_evidence", "Choose a saved CSV artifact to export.")
            _, content = evidence_content(store, data)
            if content is None:
                raise WorkbenchError("missing_evidence", "This artifact was not produced in this version.")
        elif kind.startswith("ensemble"):
            name = {"ensemble": "forecast.csv", "ensemble_backtest": "backtest.csv", "ensemble_metrics": "metrics.csv"}[kind]
            content = _blend_artifact(store, data["runId"], data.get("blendId", ""), name)
        else:
            content = store.read_artifact(data["runId"], _artifact_name(store, data["runId"], ARTIFACTS[kind]))
        csv.field_size_limit(128 * 1024 * 1024)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        if not reader.fieldnames:
            raise WorkbenchError("missing_evidence", "The requested CSV has no columns.")
        scope_column = data.get("seriesColumn") or next((key for key in ("unique_id", "parent_unique_id") if key in reader.fieldnames), "unique_id")
        if data.get("seriesColumn") and scope_column not in reader.fieldnames:
            raise WorkbenchError("invalid_scope", "The requested series scope column is absent.")
        if data.get("models") and "model" not in reader.fieldnames:
            raise WorkbenchError("invalid_scope", "This export has no model column.")
        if data.get("cutoff") and "cutoff" not in reader.fieldnames:
            raise WorkbenchError("invalid_scope", "This export has no rolling-origin cutoff.")
        path = store.owned(f"exports/{uuid.uuid4()}.csv")
        path.parent.mkdir(exist_ok=True)
        count, escaped = 0, 0
        with path.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if data.get("series") and scope_column in reader.fieldnames and row.get(scope_column) != data["series"]:
                    continue
                if data.get("models") and row.get("model") not in data["models"]:
                    continue
                if data.get("cutoff") and pd.Timestamp(row.get("cutoff")) != pd.Timestamp(data["cutoff"]):
                    continue
                for key, value in row.items():
                    if value and value.startswith(("=", "+", "-", "@", "\t", "\r")) and not re.fullmatch(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", value):
                        row[key] = "'" + value
                        escaped += 1
                writer.writerow(row)
                count += 1
        return {"path": str(path), "rows": count, "sha256": digest(path.read_bytes()), "formulaEscapedCells": escaped}
    raise WorkbenchError("unknown_operation", "Operation has no implementation.")


def worker_main(workspace: str, *, job_id: str | None = None, terminal_status: str | None = None) -> int:
    try:
        with redirect_stdout(sys.stderr):
            if terminal_status and job_id:
                result = WorkbenchStore(workspace).finish_job(
                    job_id, terminal_status, error="Owned worker stopped before successful completion.",
                )
            elif job_id:
                result = execute_job(WorkbenchStore(workspace), job_id)
            else:
                text = sys.stdin.read(256 * 1024 + 1)
                if len(text.encode("utf-8")) > 256 * 1024:
                    raise WorkbenchError("too_large", "Worker request exceeds 256 KiB.")
                request = json.loads(text)
                if not isinstance(request, dict) or set(request) != {"operation", "input"}:
                    raise WorkbenchError("invalid_input", "Expected operation and input.")
                result = dispatch(workspace, request["operation"], request["input"])
        response = canonical_json({"version": CONTRACT["version"], "result": result})
        if len(response.encode("utf-8")) > 8 * 1024 * 1024:
            raise WorkbenchError("too_large", "Response exceeds 8 MiB. Narrow the evidence scope.")
        print(response)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, TypeError, sqlite3.Error) as exc:
        print(canonical_json({"error": {"code": getattr(exc, "code", "operation_failed"), "message": str(exc)}}))
        return 2
