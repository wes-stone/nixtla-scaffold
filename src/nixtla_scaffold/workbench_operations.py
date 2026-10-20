"""Forecast maintenance through the existing ledger and refresh engine."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nixtla_scaffold.ledger import ingest_actuals, lock_version
from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest


def prepare(store, data: dict) -> dict:
    from nixtla_scaffold.workbench import _json_artifact, _safe_spec
    _, run = store.run(data.get("runId", ""))
    if data.get("approved") is not True:
        raise WorkbenchError("approval_required", "Review and explicitly approve this version's maintenance operation.")
    if data["kind"] == "refresh" and "judgment.json" in json.loads(run["hashes"]):
        raise WorkbenchError("judgment_refresh", "Refresh the underlying model run, then explicitly rebase judgment onto its new periods. A model refresh must not silently discard or replay analyst adjustments.")
    if data["kind"] == "lock":
        if not data.get("lockLabel", "").strip():
            raise WorkbenchError("label_required", "Name the official lock.")
        gate = _json_artifact(store, data["runId"], "appendix/accuracy_gate.json", {})
        if not gate.get("planning_ready_claim_allowed"):
            raise WorkbenchError("not_planning_ready", "The saved claim gate does not permit an official planning lock.")
        return data
    source, metadata = store.source(data.get("sourceId", ""))
    from nixtla_scaffold.feature_sources import bind_saved_driver_snapshots
    manifest = _json_artifact(store, data["runId"], "manifest.json", {})
    spec = bind_saved_driver_snapshots(_safe_spec(manifest["spec"]), store.run(data["runId"])[0], manifest)
    if metadata["mapping"]["freq"] != spec.freq:
        raise WorkbenchError("calendar_changed", "The new snapshot's calendar differs. Start a new analysis rather than silently rebinding this version.")
    if data["kind"] == "refresh" and "spec" in data:
        raise WorkbenchError("frozen_spec", "Refresh preserves the full saved spec. Use a new analysis to edit it.")
    if data["kind"] == "actuals":
        if not data.get("knownAsOf"):
            raise WorkbenchError("vintage_required", "Record when these actuals became known.")
        known = pd.Timestamp(data["knownAsOf"])
        if pd.isna(known) or known.tzinfo is None:
            raise WorkbenchError("vintage_required", "Actuals known-as-of needs a valid timezone-aware timestamp.")
        frame = pd.read_csv(source, dtype={"unique_id": str})
        if frame.duplicated(["unique_id", "ds"]).any() or not np.isfinite(pd.to_numeric(frame["y"], errors="coerce")).all():
            raise WorkbenchError("invalid_actuals", "Actuals must have unique keys and finite values; no fill policy is applied.")
    return {**data, "sourceFingerprint": digest(source.read_bytes()), "spec": spec.to_dict(), "name": run["name"]}


def execute(store, data: dict, out):
    from nixtla_scaffold.workbench import _json_artifact, _safe_spec, _evaluate, _adopt_blend
    from nixtla_scaffold.blend import BlendRecipe
    from nixtla_scaffold.forecast import run_forecast
    from nixtla_scaffold.refresh import write_refresh_artifacts

    parent, row = store.run(data["runId"])
    context = _json_artifact(store, data["runId"], "ledger_context.json", {})
    if data["kind"] == "lock":
        with store.snapshot_lock():
            return lock_version(store.ledger, version_id=context["forecast_version_id"],
                                lock_label=data["lockLabel"], reason=data.get("reason", ""), export=False).to_dict()
    source, _ = store.source(data["sourceId"])
    if digest(source.read_bytes()) != data["sourceFingerprint"]:
        raise WorkbenchError("source_changed", "The reviewed snapshot changed; no maintenance operation was executed.")
    if data["kind"] == "actuals":
        source_key = digest(canonical_json({
            "fingerprint": data["sourceFingerprint"], "knownAsOf": data["knownAsOf"],
            "label": data.get("revisionLabel", ""),
        }).encode())
        with store.snapshot_lock():
            with store.connect(readonly=True) as conn:
                existing = conn.execute("SELECT actual_revision_id,COUNT(*) FROM actual_revisions WHERE forecast_key=? AND source_id=? GROUP BY actual_revision_id",
                                        (context["forecast_key"], source_key)).fetchone()
            if existing:
                return {"actual_revision_id": existing[0], "rows": existing[1], "reused": True}
            return ingest_actuals(
                store.ledger, source, forecast_key=context["forecast_key"], source_id=source_key,
                known_as_of=data["knownAsOf"], revision_label=data.get("revisionLabel", ""),
                source_kind="workbench_snapshot", export=False,
            ).to_dict()
    spec = _safe_spec(data["spec"])
    prior_source = json.loads(row["metadata"]).get("sourceId")
    if prior_source and digest(store.source(prior_source)[0].read_bytes()) == data["sourceFingerprint"]:
        return {"runId": data["runId"], "reused": True, "stopReason": "unchanged_source_and_recipe"}
    run = run_forecast(source, spec)
    destination = run.to_directory(out / "forecast")
    write_refresh_artifacts(parent, destination)
    original_recipe = _json_artifact(store, data["runId"], "ensemble_adoption.json")
    refit = store.import_run(destination, name=row["name"], metadata={
        "sourceId": data["sourceId"], "parentRunId": data["runId"], "adopted": False,
        "purpose": "component_refit" if original_recipe else "refresh",
    })
    if not original_recipe:
        return {"runId": refit["id"], "reused": refit["reused"]}
    recipe = BlendRecipe.from_dict(original_recipe["recipe"])
    blend = _evaluate(store, refit["id"], recipe)
    result = _adopt_blend(store, {
        "runId": refit["id"], "blendId": blend["blendId"], "refreshParentId": data["runId"],
        "expectedReceipt": digest(canonical_json(blend["receipt"]).encode()),
    }, out)
    return {**result, "recipeFingerprint": recipe.fingerprint, "confirmation": "new_snapshot_requires_new_confirmation"}


def ledger_view(store, data: dict) -> dict:
    from nixtla_scaffold.workbench import _json_artifact, records
    context = _json_artifact(store, data["runId"], "ledger_context.json", {})
    if not context.get("forecast_key") or not context.get("forecast_version_id"):
        raise WorkbenchError("missing_evidence", "This saved run has no operating-ledger binding.")
    kind = data.get("kind", "actuals")
    table, field, key = {
        "actuals": ("actual_revisions", "forecast_key", context["forecast_key"]),
        "performance": ("forecast_actuals", "forecast_version_id", context["forecast_version_id"]),
        "locks": ("official_forecast_locks", "forecast_version_id", context["forecast_version_id"]),
        "versions": ("forecast_versions", "forecast_key", context["forecast_key"]),
        "snapshot": ("forecast_snapshot", "forecast_key", context["forecast_key"]),
        "metrics": ("forecast_version_metrics", "forecast_key", context["forecast_key"]),
        "performance_summary": ("forecast_performance", "forecast_key", context["forecast_key"]),
        "adjustments": ("forecast_adjustments", "forecast_key", context["forecast_key"]),
        "corrected_actuals": ("corrected_actuals", "forecast_key", context["forecast_key"]),
        "regimes": ("regime_changes", "forecast_key", context["forecast_key"]),
        "deltas": ("forecast_version_deltas", "forecast_key", context["forecast_key"]),
        "source_refreshes": ("source_refreshes", "forecast_key", context["forecast_key"]),
    }[kind]
    where = f"{field}=?"
    args = [key]
    if data.get("series") and kind not in {"locks", "versions", "source_refreshes"}:
        where += " AND unique_id=?"
        args.append(data["series"])
    if data.get("versions"):
        if kind not in {"snapshot", "metrics", "performance_summary", "performance", "locks", "versions", "source_refreshes"}:
            raise WorkbenchError("invalid_scope", "Version filtering is not applicable to this ledger table.")
        where += f" AND forecast_version_id IN ({','.join('?' for _ in data['versions'])})"
        args.extend(data["versions"])
    offset, limit = data.get("offset", 0), data.get("limit", 500)
    with store.connect(readonly=True) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]
        frame = pd.read_sql_query(f"SELECT * FROM {table} WHERE {where} ORDER BY rowid DESC LIMIT ? OFFSET ?", conn, params=[*args, limit, offset])
    return {"rows": records(frame), "columns": list(frame.columns), "total": total,
            "nextOffset": offset + len(frame) if offset + len(frame) < total else None}
