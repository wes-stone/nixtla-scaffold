"""Immutable, point-only analyst judgments over one saved model and series."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, localcontext
import io
import json
import re
from pathlib import Path
import tempfile
import uuid

from jsonschema import Draft202012Validator
import pandas as pd

from nixtla_scaffold.workbench_evidence import content
from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest, timestamp
from nixtla_scaffold.workbench_calendar_layer import SETTINGS as CALENDAR_SETTINGS, apply_layer

NUMBER = {"type": "string", "maxLength": 80}
SETTINGS = {
    "type": "object", "additionalProperties": False,
    "required": ["model", "percent", "amount", "start", "end", "edits", "label", "reason"],
    "properties": {
        "model": {"type": "string", "maxLength": 200},
        "percent": NUMBER, "amount": NUMBER,
        "start": {"type": "string", "maxLength": 40}, "end": {"type": "string", "maxLength": 40},
        "label": {"type": "string", "maxLength": 120}, "reason": {"type": "string", "maxLength": 2000},
        "calendar": CALENDAR_SETTINGS,
        "edits": {"type": "array", "maxItems": 500, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["ds", "percent", "amount", "override", "note"],
            "properties": {"ds": {"type": "string", "maxLength": 40}, "percent": NUMBER,
                           "amount": NUMBER, "override": NUMBER, "note": {"type": "string", "maxLength": 400}},
        }},
    },
}
DEFAULTS = {"model": "", "percent": "0", "amount": "0", "start": "", "end": "", "edits": [], "label": "", "reason": ""}
METHOD = (
    "Adjusted = (calendar-layer value when enabled, otherwise chosen model) * (1 + percent / 100) + absolute change; a final-value override replaces this calculation. "
    "Negative percentages/amounts are haircuts; positive values are uplifts. Absolute changes are per period in original source units, not display-scaled units. "
    "Bulk changes apply only inside the selected inclusive period range. Nonblank row inputs replace, not compound with, the bulk input. "
    "Blank row inputs inherit; an override of zero is a real zero. No implicit zero floor. Totals are sums of the selected series and horizon, "
    "not a final-period balance or a sum of different series. Original model, observations and statistical evidence remain unchanged. "
    "Judgment has no backtest or calibrated interval evidence and is not an official planning lock."
)


def numeric(value, *, blank=False):
    if blank and value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?", value.strip(), re.I):
        raise WorkbenchError("invalid_judgment", "Enter a number; blank bulk inputs are not zero.")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise WorkbenchError("invalid_judgment", f"Invalid numeric input: {value[:80]}") from exc
    if not number.is_finite() or abs(number.adjusted()) > 100:
        raise WorkbenchError("invalid_judgment", "Values must be finite with magnitude between 1e-100 and 1e100, or zero.")
    return number


def period(value):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is not None:
            raise ValueError("Missing or timezone-aware timestamp")
    except (ValueError, TypeError) as exc:
        raise WorkbenchError("invalid_judgment", f"Invalid period: {value}") from exc
    return stamp.date().isoformat() if stamp == stamp.normalize() else stamp.isoformat()


def table(store, run_id, name, series, *, required=True):
    resolved, raw = content(store, {"runId": run_id, "name": name})
    if raw is None:
        if required:
            raise WorkbenchError("missing_evidence", f"This version has no {name}.")
        return [], None
    if len(raw) > 32 * 1024 * 1024:
        raise WorkbenchError("too_large", "Judgment inputs exceed 32 MiB; use a scoped saved run.")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if not {"unique_id", "ds"} <= set(reader.fieldnames or []):
        raise WorkbenchError("invalid_judgment", f"{name} requires unique_id and ds.")
    rows = []
    for row in reader:
        if row["unique_id"] != series:
            continue
        if len(rows) >= 100000:
            raise WorkbenchError("too_large", "Too many model rows in this series; use a scoped run.")
        rows.append({**row, "ds": period(row["ds"])})
    return rows, {"name": resolved, "sha256": digest(raw)}


def _unique(rows, name):
    if len({row["ds"] for row in rows}) != len(rows):
        raise WorkbenchError("invalid_judgment", f"Duplicate series/period rows in {name}; no silent aggregation.")
    return {row["ds"]: row for row in rows}


def source(store, data):
    _, parent = store.run(data["runId"])
    parent_meta = json.loads(parent["metadata"])
    _, saved_raw = content(store, {"runId": data["runId"], "name": "judgment.json"})
    saved = json.loads(saved_raw) if saved_raw is not None else None
    base_id = saved["baseRunId"] if saved else data["runId"]
    _, base = store.run(base_id)
    manifest_raw = store.read_artifact(base_id, "manifest.json")
    manifest = json.loads(manifest_raw)
    selected, selected_hash = table(store, base_id, "forecast.csv", data["series"])
    historical, history_hash = table(store, base_id, "appendix/history.csv", data["series"])
    candidates, candidate_hash = table(store, base_id, "appendix/forecast_long.csv", data["series"], required=False)
    selected_map = _unique(selected, "saved forecast")
    _unique(historical, "history")
    if not selected or len(selected) > 500 or not historical:
        raise WorkbenchError("invalid_judgment", "Select a saved series with history and 1–500 forecast periods.")
    if max(row["ds"] for row in historical) >= min(selected_map):
        raise WorkbenchError("invalid_judgment", "Judgments apply to future periods only; history/forecast overlap is not allowed.")
    chosen = data.get("model", saved["settings"]["model"] if saved else "")
    names = sorted({row.get("model", "") for row in candidates} - {""})
    if chosen and chosen not in names:
        raise WorkbenchError("invalid_judgment", "That model has no saved future path in this series.")
    rows = [row for row in candidates if row.get("model") == chosen] if chosen else selected
    model_map = _unique(rows, chosen or "saved selection")
    if set(model_map) != set(selected_map):
        raise WorkbenchError("invalid_judgment", "This model does not cover every saved forecast period; no fill or model splicing is applied.")
    values = [{"ds": ds, "model_yhat": str(numeric(model_map[ds].get("yhat", ""))),
               "selected_yhat": str(numeric(selected_map[ds].get("yhat", "")))} for ds in sorted(selected_map)]
    model_options = []
    for name in ["", *names]:
        items = selected if not name else [row for row in candidates if row.get("model") == name]
        unique = {row["ds"]: row for row in items}
        complete = len(unique) == len(items) and set(unique) == set(selected_map)
        endpoint = unique.get(max(selected_map), {}).get("yhat")
        model_options.append({"id": name, "label": name or "Saved selected forecast", "complete": complete, "endpoint": endpoint})
    hashes = [selected_hash, history_hash, {"name": "manifest.json", "sha256": digest(manifest_raw)}]
    if candidate_hash:
        hashes.append(candidate_hash)
    fingerprint = digest(canonical_json({"baseRunId": base_id, "series": data["series"], "model": chosen, "hashes": hashes}).encode())
    with store.connect(readonly=True) as conn:
        versions = [dict(row) for row in conn.execute(
            """SELECT id, created_at createdAt, json_extract(metadata,'$.judgmentLabel') label,
                      json_extract(metadata,'$.parentRunId') parentRunId, json_extract(metadata,'$.judgmentModel') model
               FROM wb_runs WHERE json_extract(metadata,'$.judgmentBaseRunId')=?
               AND json_extract(metadata,'$.judgmentSeries')=? ORDER BY created_at DESC,id LIMIT 200""",
            (base_id, data["series"]),
        )]
    result = {"baseRunId": base_id, "baseName": base["name"], "series": data["series"], "model": chosen,
            "selectedModel": ", ".join(sorted({row.get("model", "") for row in selected} - {""})),
            "units": manifest["spec"].get("unit_label", ""), "rows": values,
            "history": [{"ds": row["ds"], "y": row.get("y", "")} for row in historical[-2000:]],
            "models": model_options, "savedSettings": saved["settings"] if saved else None,
            "versions": versions, "sourceHash": fingerprint, "hashes": hashes, "method": METHOD,
            "parentLabel": parent_meta.get("judgmentLabel", ""), "sourceSpec": manifest["spec"],
            "calendarLayer": None, "historyPeriods": sorted({row["ds"][:7] + "-01" for row in historical if row.get("y", "") != ""})}
    definition = data.get("calendar", saved["settings"].get("calendar") if saved else None)
    if definition is not None:
        result["rows"], result["calendarLayer"] = apply_layer(result, definition)
        if result["calendarLayer"]:
            result["sourceHash"] = digest(canonical_json({"sourceHash": fingerprint, "calendar": result["calendarLayer"]}).encode())
    return result


def calculate(rows, settings):
    error = next(Draft202012Validator(SETTINGS).iter_errors(settings), None)
    if error:
        raise WorkbenchError("invalid_judgment", error.message)
    periods = {row["ds"] for row in rows}
    start, end = settings["start"], settings["end"]
    if (start and start not in periods) or (end and end not in periods) or (start and end and start > end):
        raise WorkbenchError("invalid_judgment", "The bulk period range must match forecast periods and start before its end.")
    edits = {}
    for edit in settings["edits"]:
        if edit["ds"] not in periods or edit["ds"] in edits:
            raise WorkbenchError("invalid_judgment", "Row edits must have unique keys inside this forecast's future horizon.")
        edits[edit["ds"]] = edit
    bulk_percent, bulk_amount = numeric(settings["percent"]), numeric(settings["amount"])
    result = []
    with localcontext() as ctx:
        ctx.prec = 40
        cumulative = Decimal(0)
        for row in rows:
            ds, base = row["ds"], numeric(row["model_yhat"])
            edit = edits.get(ds, {})
            inside = (not start or ds >= start) and (not end or ds <= end)
            percent = numeric(edit.get("percent", ""), blank=True)
            amount = numeric(edit.get("amount", ""), blank=True)
            percent = percent if percent is not None else bulk_percent if inside else Decimal(0)
            amount = amount if amount is not None else bulk_amount if inside else Decimal(0)
            override = numeric(edit.get("override", ""), blank=True)
            adjustment_base = numeric(row.get("calendar_yhat", row["model_yhat"]))
            computed = adjustment_base * (1 + percent / 100) + amount
            adjusted = override if override is not None else computed
            numeric(str(adjusted))
            delta = adjusted - base
            cumulative += delta
            result.append({**row, **({"judgment_delta": str(adjusted - adjustment_base)} if "calendar_yhat" in row else {}),
                           "percent": str(percent), "amount": str(amount), "override": str(override) if override is not None else "",
                           "adjusted_yhat": str(adjusted), "delta": str(delta), "cumulative_delta": str(cumulative),
                           "delta_pct": str(delta / abs(base) * 100) if base else None, "note": edit.get("note", "")})
        totals = {key: str(sum((Decimal(row[key]) for row in result), Decimal(0))) for key in ("model_yhat", "selected_yhat", "adjusted_yhat", "delta")}
        total_base = Decimal(totals["model_yhat"])
        totals["delta_pct"] = str(Decimal(totals["delta"]) / abs(total_base) * 100) if total_base else None
        if rows and "calendar_yhat" in rows[0]:
            totals.update({key: str(sum((Decimal(row[key]) for row in result), Decimal(0)))
                           for key in ("calendar_yhat", "calendar_delta", "judgment_delta")})
    return result, totals


def preview(store, data):
    error = next(Draft202012Validator(SETTINGS).iter_errors(data["settings"]), None)
    if error:
        raise WorkbenchError("invalid_judgment", error.message)
    src = source(store, {**data, "model": data["settings"]["model"], "calendar": data["settings"].get("calendar")})
    if data.get("expectedSourceHash") and data["expectedSourceHash"] != src["sourceHash"]:
        raise WorkbenchError("stale_evidence", "The selected model/source changed. Reload it before saving.")
    rows, totals = calculate(src["rows"], data["settings"])
    settings = {**data["settings"]}
    if src["calendarLayer"]:
        settings["calendar"] = src["calendarLayer"]["settings"]
    return {**src, "rows": rows, "totals": totals, "settings": settings}


def import_rows(store, data):
    if len(data["text"].encode("utf-8")) > 128 * 1024:
        raise WorkbenchError("too_large", "Import is limited to 128 KiB; shorten notes or import fewer periods.")
    src = source(store, {**data, "calendar": data.get("calendar")})
    delimiter = "\t" if data.get("delimiter") == "tab" else ","
    reader = csv.DictReader(io.StringIO(data["text"].lstrip("\ufeff"), newline=""), delimiter=delimiter)
    fields = reader.fieldnames or []
    allowed = {"ds", "percent", "amount", "override", "note", "unique_id", "model", "base_run_id",
               "model_yhat", "selected_yhat", "adjusted_yhat", "delta", "cumulative_delta", "delta_pct",
               "calendar_yhat", "calendar_delta", "calendar_factor", "calendar_unscaled_yhat", "calendar_scaling_factor",
               "calendar_days", "weekdays", "weekends", "holiday_loss", "effective_days", "holiday_dates",
               "holiday_loss_source", "judgment_delta", "month_length_factor", "weekday_weekend_factor", "holiday_factor"}
    if set(fields) - allowed:
        raise WorkbenchError("invalid_judgment", f"Unknown import columns: {sorted(set(fields) - allowed)}. Use the exported worksheet headers.")
    if len(fields) != len(set(fields)) or not {"ds"} <= set(fields) or not {"percent", "amount", "override"}.intersection(fields):
        raise WorkbenchError("invalid_judgment", "Import needs unique headers: ds and at least one of percent, amount, override; note and unique_id are optional.")
    edits = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise WorkbenchError("invalid_judgment", "Every imported row must match the header column count.")
        if "unique_id" in row and row["unique_id"] != data["series"]:
            raise WorkbenchError("invalid_judgment", "Import includes another series; select one series per worksheet.")
        if "model" in row and row["model"] != src["model"]:
            raise WorkbenchError("invalid_judgment", "Import model differs from the selected baseline.")
        if "base_run_id" in row and row["base_run_id"] != src["baseRunId"]:
            raise WorkbenchError("invalid_judgment", "Import belongs to a different baseline version.")
        edits.append({"ds": period(row["ds"]), **{key: row.get(key, "") for key in ("percent", "amount", "override", "note")}})
        if len(edits) > 500:
            raise WorkbenchError("too_large", "Import is limited to 500 future periods.")
    if not edits:
        raise WorkbenchError("invalid_judgment", "The import contains no adjustment rows.")
    calculate(src["rows"], {**DEFAULTS, "model": src["model"], "edits": edits})
    return {"edits": edits, "rows": len(edits), "sha256": digest(data["text"].encode()), "sourceHash": src["sourceHash"]}


def csv_bytes(rows, *, escape=False):
    from nixtla_scaffold.workbench_explore import NUMBER as numeric_pattern
    def safe(value):
        text = "" if value is None else str(value)
        return "'" + text if escape and text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) and not numeric_pattern.fullmatch(text) else text
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    fields = list(rows[0])
    writer.writerow(fields)
    writer.writerows([safe(row[key]) for key in fields] for row in rows)
    return output.getvalue().encode("utf-8")


def export(store, data):
    result = preview(store, data)
    rows = [{"base_run_id": result["baseRunId"], "unique_id": data["series"], "model": result["model"], **row} for row in result["rows"]]
    identifier = str(uuid.uuid4())
    path, recipe_path = store.owned(f"exports/{identifier}.csv"), store.owned(f"exports/{identifier}.query.json")
    path.parent.mkdir(exist_ok=True)
    raw = csv_bytes(rows, escape=True)
    path.write_bytes(raw)
    recipe_path.write_text(canonical_json({"version": 1, "createdAt": timestamp(), "csvSha256": digest(raw),
        "baseRunId": result["baseRunId"], "parentRunId": data["runId"], "series": data["series"], "settings": result["settings"],
        "hashes": result["hashes"], "totals": result["totals"], "calendarLayer": result["calendarLayer"],
        "method": METHOD, "claim": "unvalidated_judgment_scenario"}), encoding="utf-8")
    return {"id": identifier, "path": str(path), "queryPath": str(recipe_path), "rows": len(rows)}


def save(store, data):
    result = preview(store, data)
    settings = result["settings"]
    if not settings["label"].strip() or not settings["reason"].strip():
        raise WorkbenchError("invalid_judgment", "Name the new version and record why you chose this model/adjustment.")
    recipe = {"version": 1, "baseRunId": result["baseRunId"], "parentRunId": data["runId"], "series": data["series"],
              "settings": settings, "hashes": result["hashes"], "sourceHash": result["sourceHash"], "method": METHOD,
              "totals": result["totals"], "calendarLayer": result["calendarLayer"], "claim": "unvalidated_judgment_scenario"}
    gate = {"status": "judgment_scenario", "planning_ready_claim_allowed": False,
            "reason": "Analyst judgment, not backtested or independently confirmed. Baseline accuracy and intervals do not transfer."}
    history, _ = table(store, result["baseRunId"], "appendix/history.csv", data["series"])
    future = [{"unique_id": data["series"], "ds": row["ds"], "model": "Judgment", "yhat": row["adjusted_yhat"]} for row in result["rows"]]
    # No fitted metrics, model policy receipts, or baseline intervals are copied.
    with tempfile.TemporaryDirectory(prefix="judgment-", dir=store.root) as temporary:
        root = Path(temporary)
        (root / "appendix").mkdir()
        (root / "audit").mkdir()
        (root / "manifest.json").write_text(canonical_json({"spec": result["sourceSpec"], "purpose": "judgment_scenario",
            "outputs": {"forecast": "forecast.csv", "history": "appendix/history.csv"}, "judgment": recipe}), encoding="utf-8")
        (root / "judgment.json").write_text(canonical_json(recipe), encoding="utf-8")
        for name in ("forecast.csv", "appendix/forecast_long.csv"):
            (root / name).write_bytes(csv_bytes(future))
        (root / "appendix/history.csv").write_bytes(csv_bytes([{"unique_id": data["series"], "ds": row["ds"], "y": row.get("y", "")} for row in history]))
        (root / "appendix/judgment_comparison.csv").write_bytes(csv_bytes([{"unique_id": data["series"], **row} for row in result["rows"]]))
        (root / "appendix/accuracy_gate.json").write_text(canonical_json(gate), encoding="utf-8")
        (root / "audit/backtest_predictions.csv").write_text("unique_id,ds,cutoff,y,Judgment\n", encoding="utf-8")
        (root / "audit/backtest_metrics.csv").write_text("unique_id,model,rmse,mae,bias\n", encoding="utf-8")
        (root / "audit/model_selection.csv").write_bytes(csv_bytes([{"unique_id": data["series"], "selected_model": "Judgment", "selection_reason": settings["reason"]}]))
        saved = store.import_run(root, name=result["baseName"], metadata={
            "purpose": "judgment_scenario", "adopted": False, "parentRunId": data["runId"],
            "judgmentBaseRunId": result["baseRunId"], "judgmentSeries": data["series"],
            "judgmentLabel": settings["label"], "judgmentModel": result["model"], "planningReady": False,
        })
    return {**saved, "label": settings["label"], "baseRunId": result["baseRunId"]}


def pipeline(store, data):
    _, raw = content(store, {"runId": data["runId"], "name": "judgment.json"})
    if raw is None:
        raise WorkbenchError("missing_evidence", "Save a named judgment version to inspect its finalization recipe.")
    recipe = json.loads(raw)
    if recipe["series"] != data["series"]:
        raise WorkbenchError("invalid_judgment", "Select the series saved in this judgment version.")
    rows, _ = table(store, data["runId"], "appendix/judgment_comparison.csv", data["series"])
    calculate(rows, recipe["settings"])
    if data.get("replay"):
        result = preview(store, {**data, "settings": recipe["settings"], "expectedSourceHash": recipe["sourceHash"]})
        _, saved = content(store, {"runId": data["runId"], "name": "appendix/judgment_comparison.csv"})
        replayed = csv_bytes([{"unique_id": data["series"], **row} for row in result["rows"]])
        if replayed != saved:
            raise WorkbenchError("stale_evidence", "Recipe replay differs from the saved comparison. No version was changed.")
    return {"recipe": recipe, "rows": rows, "verified": bool(data.get("replay"))}
