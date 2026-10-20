"""Explicit review boundary for script-backed pipelines and external challengers."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from dataclasses import fields, replace
from nixtla_scaffold.schema import ForecastSpec, forecast_spec_from_dict
from nixtla_scaffold.presets import resolve_forecast_spec

from nixtla_scaffold.workbench_store import MAX_ARTIFACT_BYTES, WorkbenchError, _filesystem_path, canonical_json, digest


def prepare_external(store, data):
    from nixtla_scaffold.workbench import _json_artifact
    spec = _json_artifact(store, data["runId"], "manifest.json")["spec"]
    if data["freq"] != spec.get("freq") or data["units"] != (spec.get("unit_label") or ""):
        raise WorkbenchError("incompatible_external", "External units and calendar must match the saved forecast. This declaration is not independent verification.")
    path = _filesystem_path(data["path"]).resolve(strict=True)
    if path.suffix.lower() != ".csv" or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise WorkbenchError("unsupported_source", "Select a canonical external forecast CSV of at most 128 MiB.")
    content = path.read_bytes()
    receipt = {"kind": "external_comparison", "runId": data["runId"], "path": str(path),
               "freq": data["freq"], "units": data["units"], "hashes": {"external.csv": digest(content)},
               "authorization": "Import this CSV beside the saved forecast. No model or provider is called. Cutoff-labeled history can be scored on observed actuals; future-only rows remain scenario comparisons. Matching labels are not proof of leakage-free training. Nothing is promoted or blended automatically."}
    proposal_id = digest(canonical_json(receipt).encode())
    with store.snapshot_lock():
        root = store.owned("proposals/" + proposal_id)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "external.csv"
        if target.exists() and target.read_bytes() != content:
            raise WorkbenchError("proposal_changed", "The reviewed external snapshot changed.")
        target.write_bytes(content)
        with store.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO wb_state VALUES(?,0,?)", ("proposal:" + proposal_id, canonical_json(receipt)))
    return {"proposalId": proposal_id, "receipt": receipt, "expectedReceipt": proposal_id}


def prepare_forecast(store, data):
    path, metadata = store.source(data["sourceId"])
    settings = data["spec"]
    if set(settings) - {field.name for field in fields(ForecastSpec)}:
        raise WorkbenchError("invalid_spec", "External review does not accept unknown spec fields.")
    spec = resolve_forecast_spec({**settings, "freq": settings.get("freq") or metadata["mapping"]["freq"],
                                 "verbose": False})
    if (spec.id_col, spec.time_col, spec.target_col) != ("unique_id", "ds", "y"):
        raise WorkbenchError("canonical_columns_required", "Reviewed forecasts use the already mapped unique_id, ds and y snapshot.")
    if "horizon" not in settings or spec.horizon > 120:
        raise WorkbenchError("horizon_required", "Review a horizon of 1-120 periods.")
    if metadata["requiresRepair"] and "fill_method" not in settings:
        raise WorkbenchError("repair_required", "Review a missing-observation policy explicitly.")
    receipt = {"kind": "reviewed_forecast", "spec": spec.to_dict(), "sourceId": data["sourceId"],
               "sourceFingerprint": digest(path.read_bytes()), "hashes": {},
               "authorization": "Approval runs the copied custom Python entrypoint and enabled challengers under your local account. Entrypoints must be self-contained or use already installed packages; sibling files are not copied. This may contact the named external engine. No new credentials, installations, publication or schedules are authorized."}
    files = {}
    for index, custom in enumerate(spec.custom_models):
        if not custom.script_path:
            raise WorkbenchError("reviewable_script_required", "Canvas custom execution requires a reviewable .py entrypoint; callable-only plugins stay in the explicit CLI workflow.")
        script = _filesystem_path(custom.script_path).resolve(strict=True)
        if script.suffix.lower() != ".py" or script.stat().st_size > 256 * 1024:
            raise WorkbenchError("unsupported_script", "Choose a Python entrypoint of at most 256 KiB.")
        files[f"custom-{index}.py"] = script.read_bytes()
    receipt["hashes"] = {name: digest(blob) for name, blob in files.items()}
    proposal_id = digest(canonical_json(receipt).encode())
    with store.snapshot_lock():
        root = store.owned("proposals/" + proposal_id)
        root.mkdir(parents=True, exist_ok=True)
        for name, blob in files.items():
            target = root / name
            if target.exists() and target.read_bytes() != blob:
                raise WorkbenchError("proposal_changed", "A reviewed entrypoint was altered.")
            target.write_bytes(blob)
        with store.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO wb_state VALUES(?,0,?)", ("proposal:" + proposal_id, canonical_json(receipt)))
    return {"proposalId": proposal_id, "receipt": receipt, "expectedReceipt": proposal_id}


def prepare(store, data):
    from nixtla_scaffold.workbench import _safe_spec, _json_artifact
    if "path" in data:
        from nixtla_scaffold.pipelines import _load_pipeline_config
        path = _filesystem_path(data["path"]).resolve(strict=True)
        definition = _load_pipeline_config(path)
        base = path.parent
    else:
        definition = data["definition"]
        base = store.workspace
    if not isinstance(definition.get("extracts"), list) or not definition["extracts"]:
        raise WorkbenchError("invalid_pipeline", "Define at least one extract.")
    payload = json.loads(canonical_json(definition))
    parent = data.get("runId")
    spec_data = (_json_artifact(store, parent, "manifest.json", {})["spec"] if parent
                 else payload.get("forecast", {}))
    from nixtla_scaffold.pipelines import _forecast_spec_from_config
    spec = _safe_spec(spec_data if parent else _forecast_spec_from_config(spec_data).to_dict())
    if parent:
        from nixtla_scaffold.feature_sources import bind_saved_driver_snapshots
        spec = bind_saved_driver_snapshots(spec, store.run(parent)[0], _json_artifact(store, parent, "manifest.json", {}))
    if not spec.freq and not data.get("dataOnly"):
        raise WorkbenchError("calendar_required", "Record the pipeline calendar before approving execution.")
    if parent and not data.get("dataOnly") and _json_artifact(store, parent, "ensemble_adoption.json"):
        raise WorkbenchError("frozen_blend", "Run the extract without modeling, inspect its output, then use Refresh saved forecast to retain the adopted blend recipe.")
    payload["forecast"] = {**spec.to_dict(), "output": "forecast"}
    payload.setdefault("canonical", {})["output"] = "prepared/forecast_input.csv"
    files, steps = {}, []
    for index, step in enumerate([*payload["extracts"], *([payload["transform"]] if payload.get("transform") else [])]):
        if not isinstance(step, dict):
            raise WorkbenchError("invalid_pipeline", "Each stage must be an object.")
        if not isinstance(step.get("script"), str):
            raise WorkbenchError("invalid_pipeline", "Each stage needs a Python script path.")
        script = _filesystem_path(base / step["script"]).resolve(strict=True)
        if script.suffix.lower() != ".py" or script.stat().st_size > 256 * 1024:
            raise WorkbenchError("unsupported_script", "Each reviewed stage must be a Python script of at most 256 KiB.")
        name = f"stage-{index}.py"
        files[name] = script.read_bytes()
        step["script"] = name
        queries = step.pop("query_file", step.get("query_files", []))
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            raise WorkbenchError("invalid_pipeline", "query_files must be an array.")
        query_names = []
        for number, query in enumerate(queries):
            path = _filesystem_path(base / query).resolve(strict=True)
            if path.stat().st_size > 256 * 1024:
                raise WorkbenchError("too_large", "Query file exceeds 256 KiB.")
            relative = f"stage-{index}-query-{number}{path.suffix}"
            files[relative] = path.read_bytes()
            query_names.append(relative)
        step["query_files"] = query_names
        timeout = int(step.get("timeout_seconds", 120))
        if not 1 <= timeout <= 300:
            raise WorkbenchError("budget_review_required", "Each reviewed stage needs a timeout between 1 and 300 seconds.")
        step["timeout_seconds"] = timeout
        outputs = step.get("outputs", {})
        if not isinstance(outputs, dict):
            raise WorkbenchError("invalid_pipeline", "Declared stage outputs must be a mapping.")
        declared = [*outputs.values(), *([step["output"]] if "output" in step else [])]
        for output in declared:
            path = Path(str(output))
            if path.is_absolute() or path.drive or ".." in path.parts:
                raise WorkbenchError("unsafe_output", "Pipeline declared outputs must stay within the new owned job directory.")
        if not isinstance(step.get("args", []), list) or not all(isinstance(arg, str) for arg in step.get("args", [])):
            raise WorkbenchError("invalid_pipeline", "Stage arguments must be a string array.")
        steps.append({"name": step.get("name", name), "script": str(script),
                      "sha256": digest(files[name]), "args": step.get("args", []),
                      "outputs": step.get("outputs", step.get("output")), "queryFiles": query_names})
    payload["name"] = payload.get("name", "Reviewed source pipeline")
    files["pipeline.yaml"] = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    receipt = {"kind": "pipeline", "name": payload["name"], "steps": steps, "dataOnly": bool(data.get("dataOnly")),
               "spec": spec.to_dict(), "runId": parent,
               "hashes": {name: digest(blob) for name, blob in files.items()},
               "authorization": "Approval runs these copied Python scripts with your local account and configured access; this is not a sandbox. Entrypoints must be self-contained or use installed packages. Only declared query files are copied; review any other path arguments. Scripts may read files or contact providers. No credentials, schedules or publication are authorized. End-to-end limit: 15 minutes."}
    proposal_id = digest(canonical_json(receipt).encode())
    with store.snapshot_lock():
        root = store.owned(f"proposals/{proposal_id}")
        root.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            target = root / name
            if target.exists() and target.read_bytes() != content:
                raise WorkbenchError("proposal_changed", "A saved execution proposal was altered; review a new proposal.")
            target.write_bytes(content)
        with store.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO wb_state VALUES(?,0,?)", ("proposal:" + proposal_id, canonical_json(receipt)))
    return {"proposalId": proposal_id, "receipt": receipt, "expectedReceipt": proposal_id}


def reviewed(store, data):
    if data.get("approved") is not True:
        raise WorkbenchError("approval_required", "Explicitly approve the displayed execution receipt first.")
    with store.connect(readonly=True) as conn:
        row = conn.execute("SELECT body FROM wb_state WHERE id=?", ("proposal:" + data.get("proposalId", ""),)).fetchone()
    if not row or digest(row[0].encode()) != data.get("expectedReceipt"):
        raise WorkbenchError("stale_evidence", "The reviewed execution receipt does not match this request.")
    receipt = json.loads(row[0])
    root = store.owned("proposals/" + data["proposalId"])
    for name, expected in receipt["hashes"].items():
        if digest((root / name).read_bytes()) != expected:
            raise WorkbenchError("proposal_changed", "A reviewed script or query file changed. Nothing was executed.")
    return root, receipt


def execute(store, data, out):
    from nixtla_scaffold.pipelines import run_pipeline, refresh_pipeline
    from nixtla_scaffold.workbench import _profile
    root, receipt = reviewed(store, data)
    if receipt["kind"] == "external_comparison":
        from nixtla_scaffold.byo_model import load_byo_model_forecasts
        from nixtla_scaffold.comparisons import write_forecast_comparison
        from nixtla_scaffold.external_scoring import write_external_forecast_scores
        from nixtla_scaffold.workbench import run_frame, _json_artifact
        import pandas as pd
        external = load_byo_model_forecasts(root / "external.csv")
        evidence = out / "comparison"
        _, parent = store.run(receipt["runId"])
        for name in json.loads(parent["hashes"]):
            if name.startswith("external/") or name == "ledger_context.json":
                continue
            target = evidence / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(store.read_artifact(receipt["runId"], name))
        destination = evidence / "external"
        destination.mkdir()
        external.to_csv(destination / "forecasts.csv", index=False)
        history = run_frame(store, receipt["runId"], "history")
        historical = external[external["cutoff"].notna() & (pd.to_datetime(external["ds"]) <= pd.to_datetime(history["ds"]).max())] if "cutoff" in external else external.iloc[:0]
        if not historical.empty:
            spec = _json_artifact(store, receipt["runId"], "manifest.json")["spec"]
            scored = write_external_forecast_scores(historical, history, destination, external_format="long",
                        requested_horizon=spec["horizon"], season_length=spec.get("season_length") or 1,
                        cutoff_contract=run_frame(store, receipt["runId"], "appendix/cutoff_contract.csv"))
            scored.model_metrics.to_csv(destination / "metrics.csv", index=False)
        comparison = write_forecast_comparison(evidence, external, output_dir=destination, external_format="long")
        comparison.comparison.to_csv(destination / "comparison.csv", index=False)
        (destination / "review.json").write_text(canonical_json(receipt), encoding="utf-8")
        imported = store.import_run(evidence, name=parent["name"], metadata={
            **json.loads(parent["metadata"]), "parentRunId": receipt["runId"], "externalComparison": True,
            "externalProposalId": data["proposalId"], "jobId": data["jobId"],
        })
        return {"runId": imported["id"], "scored": not historical.empty, "promotionAllowed": False}
    if receipt["kind"] == "reviewed_forecast":
        from nixtla_scaffold.forecast import run_forecast
        from nixtla_scaffold.challengers import run_challengers
        source, _ = store.source(receipt["sourceId"])
        if digest(source.read_bytes()) != receipt["sourceFingerprint"]:
            raise WorkbenchError("source_changed", "The approved source snapshot changed.")
        spec = forecast_spec_from_dict(receipt["spec"])
        custom = tuple(replace(model, script_path=str(root / f"custom-{index}.py")) for index, model in enumerate(spec.custom_models))
        spec = replace(spec, custom_models=custom)
        forecast = run_forecast(source, spec).to_directory(out / "forecast")
        challengers = run_challengers(forecast, challengers=spec.challengers) if any(item.enabled for item in spec.challengers) else None
        imported = store.import_run(forecast, name=data.get("name") or "Reviewed forecast", metadata={
            "sourceId": receipt["sourceId"], "proposalId": data["proposalId"], "jobId": data["jobId"], "externalReviewed": True,
        })
        return {"runId": imported["id"], "challengers": challengers, "proposalId": data["proposalId"]}
    if receipt.get("dataOnly"):
        result = run_pipeline(root / "pipeline.yaml", out / "pipeline", forecast=False)
    elif receipt.get("runId"):
        result = refresh_pipeline(root / "pipeline.yaml", store.run(receipt["runId"])[0], out / "pipeline")
    else:
        result = run_pipeline(root / "pipeline.yaml", out / "pipeline")
    canonical = result["outputs"]["canonical_input"]
    profile = _profile(store, {"path": canonical, "horizon": receipt["spec"]["horizon"],
                              "mapping": {"freq": receipt["spec"]["freq"]} if receipt["spec"]["freq"] else {}})
    if receipt.get("dataOnly"):
        return {"sourceId": profile["sourceId"], "pipeline": result["status"], "proposalId": data["proposalId"], "questions": profile["questions"]}
    imported = store.import_run(result["outputs"]["forecast_run"], name=receipt["name"], metadata={
        "sourceId": profile["sourceId"], "pipelineProposalId": data["proposalId"],
        "parentRunId": receipt.get("runId"), "jobId": data["jobId"],
    })
    return {"runId": imported["id"], "pipeline": result["status"], "proposalId": data["proposalId"]}
