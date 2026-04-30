from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from nixtla_scaffold.data import canonicalize_forecast_frame, read_tabular_source


QUERY_EXTENSIONS = {
    "kusto": ".kql",
    "kql": ".kql",
    "dax": ".dax",
    "sql": ".sql",
}


def ingest_query_result(
    source: str | Path,
    output: str | Path,
    *,
    time_col: str = "ds",
    target_col: str = "y",
    id_col: str = "unique_id",
    id_value: str | None = None,
    source_kind: str = "mcp",
    query_file: str | Path | None = None,
    query_text: str | None = None,
    sheet: str | int | None = None,
) -> dict[str, Any]:
    """Turn an MCP/query export into the canonical unique_id/ds/y CSV contract."""

    raw = read_query_result_source(source, sheet=sheet)
    if id_col not in raw.columns and id_value:
        raw = raw.copy()
        raw["unique_id"] = id_value
        id_col = "unique_id"
    canonical = canonicalize_forecast_frame(raw, id_col=id_col, time_col=time_col, target_col=target_col)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_path, index=False)

    query_path = _write_query_artifact(output_path, source_kind, query_file=query_file, query_text=query_text)
    metadata = {
        "source_kind": source_kind,
        "input": str(source),
        "output": str(output_path),
        "query_file": str(query_path) if query_path else None,
        "columns": {
            "id_col": id_col,
            "time_col": time_col,
            "target_col": target_col,
            "id_value": id_value,
        },
        "rows": int(len(canonical)),
        "series_count": int(canonical["unique_id"].nunique()),
        "start": canonical["ds"].min().date().isoformat(),
        "end": canonical["ds"].max().date().isoformat(),
    }
    metadata_path = output_path.with_suffix(".source.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    metadata["metadata_file"] = str(metadata_path)
    return metadata


def read_query_result_source(source: str | Path, *, sheet: str | int | None = None) -> pd.DataFrame:
    path = Path(source)
    if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        return _read_json_result(path)
    return read_tabular_source(path, sheet=sheet)


def _read_json_result(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"query result JSON is empty: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.DataFrame.from_records(json.loads(line) for line in text.splitlines() if line.strip())

    payload = json.loads(text)
    if isinstance(payload, list):
        return pd.DataFrame.from_records(payload)
    if not isinstance(payload, dict):
        raise ValueError("query result JSON must be a list of records or an object with data/rows")

    data = payload.get("data", payload)
    if isinstance(data, list):
        return pd.DataFrame.from_records(data)
    if isinstance(data, dict):
        if "rows" in data and isinstance(data["rows"], list):
            return pd.DataFrame.from_records(data["rows"])
        if all(isinstance(value, list) for value in data.values()):
            return pd.DataFrame(data)
    if "rows" in payload and isinstance(payload["rows"], list):
        return pd.DataFrame.from_records(payload["rows"])
    raise ValueError("could not parse query result JSON; expected records, columnar data, or rows")


def _write_query_artifact(
    output_path: Path,
    source_kind: str,
    *,
    query_file: str | Path | None,
    query_text: str | None,
) -> Path | None:
    if query_file is None and query_text is None:
        return None
    extension = QUERY_EXTENSIONS.get(source_kind.lower(), ".query")
    query_path = output_path.with_suffix(extension)
    if query_file is not None:
        shutil.copyfile(Path(query_file), query_path)
    else:
        query_path.write_text(str(query_text).rstrip() + "\n", encoding="utf-8")
    return query_path
