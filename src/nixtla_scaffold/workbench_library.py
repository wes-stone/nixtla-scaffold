"""Read-only explorer queries over the existing owned SQLite catalog."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd

from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest


def browse(store, data):
    section = data["section"]
    offset, limit = data.get("offset", 0), data.get("limit", 50)
    args = []
    if section == "forecasts":
        query = """
            SELECT 'family' kind, name id, name, name family,
                   COUNT(*) versionCount, MAX(created_at) createdAt
            FROM wb_runs GROUP BY name
        """
    elif section == "versions":
        if "family" not in data:
            raise WorkbenchError("scope_required", "Choose a forecast family.")
        query = """
            SELECT 'version' kind, id, id runId, name, name family, created_at createdAt,
                   json_extract(metadata,'$.adopted') adopted,
                   json_extract(metadata,'$.purpose') purpose,
                   json_extract(metadata,'$.judgmentLabel') versionLabel
            FROM wb_runs WHERE name=?
        """
        args = [data["family"]]
    elif section == "sources":
        query = """
            SELECT 'source' kind, id, id sourceId, name,
                   json_extract(metadata,'$.profile.rows') rows,
                   json_extract(metadata,'$.profile.series_count') seriesCount,
                   json_extract(metadata,'$.mapping.freq') frequency,
                   rowid sequence FROM wb_sources
        """
    elif section == "files":
        if "runId" not in data:
            raise WorkbenchError("scope_required", "Choose a saved forecast version.")
        store.run(data["runId"])
        query = """
            SELECT 'artifact' kind, r.id || ':' || f.key id, r.id runId,
                   f.key name, f.key artifactName, r.name family, f.value sha256
            FROM wb_runs r, json_each(r.hashes) f WHERE r.id=?
        """
        args = [data["runId"]]
    elif section == "notes":
        query = """
            SELECT 'note' kind, CAST(n.id AS TEXT) id, n.run_id runId,
                   substr(n.body,1,100) name, r.name family, n.created_at createdAt
            FROM wb_notes n JOIN wb_runs r ON r.id=n.run_id
        """
        if "family" in data:
            query += " WHERE r.name=?"
            args = [data["family"]]
    elif section == "jobs":
        query = """
            SELECT 'job' kind, id, kind name, status, created_at createdAt,
                   json_extract(result,'$.runId') runId,
                   json_extract(result,'$.sourceId') sourceId FROM wb_jobs
        """
    else:
        term = data.get("query", "").strip()
        if not term:
            return {"items": [], "total": 0, "offset": offset, "nextOffset": None}
        query = """
            SELECT * FROM (
                SELECT 'version' kind, id, name, name family, id runId,
                       NULL sourceId, NULL artifactName, created_at createdAt
                FROM wb_runs
                UNION ALL
                SELECT 'source', id, name, NULL, NULL, id, NULL, '' FROM wb_sources
                UNION ALL
                SELECT 'artifact', r.id || ':' || f.key, f.key, r.name, r.id,
                       NULL, f.key, r.created_at FROM wb_runs r, json_each(r.hashes) f
                UNION ALL
                SELECT 'note', CAST(n.id AS TEXT), substr(n.body,1,100), r.name, r.id,
                       NULL, NULL, n.created_at FROM wb_notes n JOIN wb_runs r ON r.id=n.run_id
                UNION ALL
                SELECT 'job', id, kind || ' · ' || status, '', json_extract(result,'$.runId'),
                       json_extract(result,'$.sourceId'), NULL, created_at FROM wb_jobs
            ) WHERE name LIKE ? ESCAPE '\\' OR family LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\'
        """
        literal = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        args = [literal] * 3
    order = "name,id" if section in {"forecasts", "files"} else "sequence DESC" if section == "sources" else "createdAt DESC,id"
    with store.connect(readonly=True) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM ({query})", args).fetchone()[0]
        items = [dict(row) for row in conn.execute(f"{query} ORDER BY {order} LIMIT ? OFFSET ?", [*args, limit, offset])]
    return {"items": items, "total": total, "offset": offset,
            "nextOffset": offset + len(items) if offset + len(items) < total else None}


def source_tuning_frame(frame, metadata):
    original_count = len(frame)
    ends = {str(row["unique_id"]): row["end"] for row in metadata["profile"].get("series", [])}
    if not ends:
        raise WorkbenchError("missing_profile", "This source has no saved tuning boundary; inspect it explicitly before previewing targets.")
    frame = frame[pd.to_datetime(frame["ds"]) <= pd.to_datetime(frame["unique_id"].map(ends))].copy()
    return frame, original_count - len(frame)


def source_view(store, data):
    path, metadata = store.source(data["sourceId"])
    frame = pd.read_csv(path, dtype={"unique_id": str}, keep_default_na=False, na_values=[""])
    frame, withheld = source_tuning_frame(frame, metadata)
    from nixtla_scaffold.workbench import records
    offset, limit = data.get("offset", 0), data.get("limit", 100)
    return {"sourceId": data["sourceId"], "name": metadata["originalName"], "mapping": metadata["mapping"],
            "horizon": metadata["confirmationHorizon"], "requiresRepair": metadata["requiresRepair"],
            "profile": {key: value for key, value in metadata["profile"].items() if key != "series"},
            "rows": records(frame.iloc[offset:offset + limit]), "columns": list(frame.columns),
            "total": len(frame), "withheldRows": withheld, "offset": offset,
            "nextOffset": offset + limit if offset + limit < len(frame) else None,
            "sha256": digest(path.read_bytes()), "readOnly": True}


def file_view(store, data):
    name = data["name"]
    content = store.read_artifact(data["runId"], name)
    root, _ = store.run(data["runId"])
    result = {"name": name, "path": str(root / name), "sha256": digest(content),
              "bytes": len(content), "readOnly": True, "rows": [], "columns": [], "total": 0,
              "nextOffset": None, "text": "", "truncated": False}
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        offset, limit = data.get("offset", 0), data.get("limit", 100)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        result["columns"] = reader.fieldnames or []
        size = 0
        for index, row in enumerate(reader):
            result["total"] = index + 1
            if index < offset or len(result["rows"]) >= limit:
                continue
            row_size = len(canonical_json(row).encode())
            if size + row_size > 64 * 1024:
                if not result["rows"]:
                    result["notice"] = "One row exceeds the 64 KiB preview limit. Use the verified local file for full values."
                break
            result["rows"].append(row)
            size += row_size
        # Count the rest even when the byte boundary ended this page early.
        result["total"] += sum(1 for _ in reader)
        end = offset + len(result["rows"])
        result["nextOffset"] = end if result["rows"] and end < result["total"] else None
        result["format"] = "table"
    elif suffix in {".json", ".jsonl", ".md", ".txt", ".yaml", ".py", ".ps1", ".cmd", ".html"}:
        result["format"] = "text"
        result["text"] = content[:64 * 1024].decode("utf-8-sig", errors="replace")
        result["truncated"] = len(content) > 64 * 1024
    else:
        result["format"] = "file"
        result["notice"] = "This format is not rendered inline. The verified local file is preserved at the path below."
    return result
