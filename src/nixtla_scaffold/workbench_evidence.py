"""Read saved analytical evidence without fitting, ranking or changing forecasts."""
from __future__ import annotations

import base64
import io
import json
import re

import pandas as pd

from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore, canonical_json, digest


def content(store: WorkbenchStore, data: dict) -> tuple[str, bytes | None]:
    name = data["name"]
    if "\\" in name or name.startswith("/") or ".." in name.split("/"):
        raise WorkbenchError("invalid_evidence", "Choose an original artifact from this saved version.")
    _, run = store.run(data["runId"])
    names = json.loads(run["hashes"])
    if name not in names:
        legacy = name.split("/")[-1]
        if name.split("/")[0] not in {"audit", "appendix"} or legacy not in names:
            return name, None
        name = legacy
    return name, store.read_artifact(data["runId"], name)


def evidence(store: WorkbenchStore, data: dict) -> dict:
    if not data["name"].endswith(".csv"):
        raise WorkbenchError("invalid_evidence", "Analytical tables must be saved CSV artifacts.")
    name, raw = content(store, data)
    result = {
        "name": name, "available": raw is not None, "rows": [], "columns": [],
        "total": 0, "offset": 0, "nextOffset": None, "models": [], "cutoffs": [],
        "seriesScope": "run", "bounds": {}, "sha256": digest(raw) if raw is not None else None,
        "message": "This artifact was not produced in this saved version." if raw is None else "",
    }
    if raw is None or not raw.strip():
        return result
    from nixtla_scaffold.workbench import records
    headers = pd.read_csv(io.BytesIO(raw), nrows=0).columns
    identifiers = {column: str for column in headers
                   if re.search(r"(^|_)(id|parent|child|version|model|name|sheet)$", column, re.I)}
    frame = pd.read_csv(io.BytesIO(raw), dtype=identifiers, keep_default_na=False,
                        na_values=[""], float_precision="round_trip")
    scope_column = data.get("seriesColumn") or next((key for key in ("unique_id", "parent_unique_id") if key in frame), "unique_id")
    if data.get("seriesColumn") and scope_column not in frame:
        raise WorkbenchError("invalid_scope", f"This artifact has no {scope_column} column.")
    if data.get("series") and scope_column in frame:
        frame = frame.loc[frame[scope_column].astype(str).eq(data["series"])].copy()
        result["seriesScope"] = "series"
    result["models"] = sorted(frame["model"].dropna().astype(str).unique().tolist()) if "model" in frame else []
    if data.get("models"):
        if "model" not in frame:
            raise WorkbenchError("invalid_scope", "Use column projection for a wide model table.")
        frame = frame.loc[frame["model"].astype(str).isin(data["models"])].copy()
    if "cutoff" in frame:
        cutoffs = sorted(frame["cutoff"].dropna().astype(str).unique().tolist())
        if len(cutoffs) > 2000:
            raise WorkbenchError("too_large", "More than 2,000 origins match. Narrow the series/model scope.")
        result["cutoffs"] = cutoffs
    if data.get("cutoff"):
        if "cutoff" not in frame:
            raise WorkbenchError("invalid_scope", "This artifact has no rolling-origin cutoff.")
        frame = frame.loc[pd.to_datetime(frame["cutoff"]).eq(pd.Timestamp(data["cutoff"]))].copy()
    if data.get("columns"):
        missing = set(data["columns"]) - set(frame)
        if missing:
            raise WorkbenchError("invalid_scope", f"Unknown evidence columns: {sorted(missing)}")
        frame = frame[data["columns"]]
    result["columns"] = list(frame)
    result["total"] = len(frame)
    for column in frame.select_dtypes(include="number"):
        values = frame[column].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if len(values):
            result["bounds"][column] = {"min": float(values.min()), "max": float(values.max())}
    offset, limit = data.get("offset", 0), data.get("limit", 500)
    if data.get("tail"):
        offset = max(0, len(frame) - offset - limit)
    result["offset"] = offset
    used = 0
    for row in records(frame.iloc[offset:offset + limit]):
        size = len(canonical_json(row).encode("utf-8"))
        if used + size > 1024 * 1024:
            if not result["rows"]:
                raise WorkbenchError("too_large", "One evidence row exceeds 1 MiB. Select fewer columns.")
            break
        result["rows"].append(row)
        used += size
    end = offset + len(result["rows"])
    result["nextOffset"] = end if end < len(frame) else None
    if not len(frame):
        result["message"] = "No saved evidence rows match this scope."
    return result


def image(store: WorkbenchStore, data: dict) -> dict:
    if not data["name"].lower().endswith((".png", ".jpg", ".jpeg")):
        raise WorkbenchError("invalid_evidence", "Only verified PNG and JPEG artifacts can be previewed.")
    name, raw = content(store, data)
    if raw is None:
        raise WorkbenchError("missing_evidence", "This image is not present in the saved version.")
    if len(raw) > 2 * 1024 * 1024:
        raise WorkbenchError("too_large", "Image preview exceeds 2 MiB. Open the original artifact locally.")
    if name.lower().endswith(".png") and raw.startswith(b"\x89PNG\r\n\x1a\n"):
        media = "image/png"
    elif name.lower().endswith((".jpg", ".jpeg")) and raw.startswith(b"\xff\xd8\xff"):
        media = "image/jpeg"
    else:
        raise WorkbenchError("invalid_evidence", "Only verified PNG and JPEG artifacts can be previewed.")
    return {"name": name, "sha256": digest(raw), "mediaType": media, "base64": base64.b64encode(raw).decode("ascii")}
