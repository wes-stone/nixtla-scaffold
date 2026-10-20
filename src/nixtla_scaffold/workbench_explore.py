"""Reproducible, local-only table exploration over verified snapshots."""
from __future__ import annotations

import base64
import csv
from decimal import Decimal, localcontext
import io
import json
import re
import uuid

from jsonschema import Draft202012Validator
import pandas as pd

from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest, timestamp

DATASET_SCHEMA = {
    "oneOf": [
        {"type": "object", "additionalProperties": False, "required": ["kind", "runId", "name"],
         "properties": {"kind": {"const": "artifact"}, "runId": {"type": "string", "minLength": 1},
                        "name": {"type": "string", "minLength": 1}}},
        {"type": "object", "additionalProperties": False, "required": ["kind", "sourceId"],
         "properties": {"kind": {"const": "source"}, "sourceId": {"type": "string", "minLength": 1}}},
    ],
}
FIELDS = {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True}
QUERY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["version", "dataset", "mode", "select", "rows", "column", "value", "aggregate", "filters", "sort"],
    "properties": {
        "version": {"const": 1}, "dataset": DATASET_SCHEMA, "mode": {"enum": ["rows", "pivot"]},
        "select": {**FIELDS, "maxItems": 200}, "rows": {**FIELDS, "maxItems": 3},
        "column": {"type": "string"}, "value": {"type": "string"},
        "aggregate": {"enum": ["sum", "avg", "min", "max", "count", "count_distinct", "count_rows"]},
        "filters": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False, "required": ["column", "operator", "value"],
            "properties": {"column": {"type": "string"}, "operator": {"enum": [
                "eq", "neq", "contains", "gte", "lte", "is_null", "not_null"]},
                "value": {"type": "string", "maxLength": 1000}},
        }},
        "sort": {"type": "object", "additionalProperties": False, "required": ["column", "direction"],
                 "properties": {"column": {"type": "string"}, "direction": {"enum": ["asc", "desc"]}}},
    },
}
NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")
IDENTIFIER = re.compile(r"(^|_)(id|model|name|code|sku|zip|parent|child|version|sheet)$", re.I)
QUERY_PREFIX = "explore-query:"
MAX_ROWS = 250_000
MAX_CELLS = 2_000_000
MAX_PIVOT_CELLS = 100_000


def _validate(value, schema):
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error:
        raise WorkbenchError("invalid_query", f"Invalid exploration query: {error.message}")


def query(value):
    _validate(value, QUERY_SCHEMA)
    if len(canonical_json(value).encode()) > 32 * 1024:
        raise WorkbenchError("too_large", "Query exceeds 32 KiB. Reduce the selected fields or filters.")
    return value


def catalog(store, data):
    tables = []
    source_id = data.get("sourceId")
    if data.get("runId"):
        _, run = store.run(data["runId"])
        names = sorted(name for name in json.loads(run["hashes"]) if name.lower().endswith(".csv"))
        preferred = ["appendix/history.csv", "forecast.csv", "appendix/forecast_long.csv", "appendix/backtest_long.csv"]
        names.sort(key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name))
        tables = [{"label": name, "dataset": {"kind": "artifact", "runId": run["id"], "name": name}} for name in names]
        source_id = source_id or json.loads(run["metadata"]).get("sourceId")
    if source_id:
        _, metadata = store.source(source_id)
        tables.append({"label": f"Source snapshot · {metadata.get('originalName', source_id[:8])}",
                       "dataset": {"kind": "source", "sourceId": source_id}})
    return {"tables": tables}


def _read(store, dataset):
    _validate(dataset, DATASET_SCHEMA)
    metadata = None
    if dataset["kind"] == "artifact":
        if not dataset["name"].lower().endswith(".csv"):
            raise WorkbenchError("invalid_query", "Explore a registered CSV table, not an executable or arbitrary file.")
        raw = store.read_artifact(dataset["runId"], dataset["name"])
        label = dataset["name"]
    else:
        path, metadata = store.source(dataset["sourceId"])
        raw, label = path.read_bytes(), metadata.get("originalName", path.name)
    if len(raw) > 32 * 1024 * 1024:
        raise WorkbenchError("too_large", "This table exceeds the 32 MiB exploration limit. Use a smaller saved table.")
    csv.field_size_limit(32 * 1024 * 1024)
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = next(reader, [])
    if not headers or any(not col for col in headers) or len(headers) != len(set(headers)):
        raise WorkbenchError("invalid_table", "The saved table needs unique, nonempty column names.")
    if len(headers) > 200:
        raise WorkbenchError("too_large", "Explore tables with at most 200 columns; use a narrower saved table.")
    rows = []
    for values in reader:
        if not values:
            continue
        if len(values) != len(headers):
            raise WorkbenchError("invalid_table", "A saved CSV row has a different width from its header.")
        rows.append([None if value == "" else value for value in values])
        if len(rows) > MAX_ROWS or len(rows) * len(headers) > MAX_CELLS:
            raise WorkbenchError("too_large", "This table exceeds 250,000 rows or 2 million cells. No sample was substituted.")
    frame = pd.DataFrame(rows, columns=headers, dtype=object)
    withheld = 0
    if metadata is not None:
        from nixtla_scaffold.workbench_library import source_tuning_frame
        frame, withheld = source_tuning_frame(frame, metadata)
    types = {}
    for col in headers:
        values = frame[col].dropna()
        numeric = bool(len(values)) and not IDENTIFIER.search(col) and all(
            NUMBER.fullmatch(value) and not re.match(r"[+-]?0\d", value) and
            len(value) <= 128 and Decimal(value).is_finite() and abs(Decimal(value).adjusted()) <= 308
            for value in values)
        types[col] = "number" if numeric else "text"
    return frame, {"dataset": dataset, "label": label, "sha256": digest(raw), "rows": len(frame),
                   "withheldRows": withheld, "columns": [{"name": col, "type": types[col],
                    "nulls": int(frame[col].isna().sum())} for col in headers]}, types


def schema(store, data):
    return _read(store, data["dataset"])[1]


def _filter(frame, filters, types):
    for item in filters:
        col, op, value = item["column"], item["operator"], item["value"]
        if col not in types:
            raise WorkbenchError("invalid_query", f"Filter field is absent: {col}")
        source = frame[col]
        if op == "is_null":
            keep = source.isna()
        elif op == "not_null":
            keep = source.notna()
        elif op == "contains":
            keep = source.str.contains(value, regex=False, case=False, na=False)
        else:
            if types[col] == "number":
                if not NUMBER.fullmatch(value) or len(value) > 128 or abs(Decimal(value).adjusted()) > 308:
                    raise WorkbenchError("invalid_query", f"Enter a finite numeric filter for {col}.")
                source = source.map(lambda cell: Decimal(cell) if cell is not None else None)
                value = Decimal(value)
            compare = {"eq": lambda cell: cell == value, "neq": lambda cell: cell != value,
                       "gte": lambda cell: cell >= value, "lte": lambda cell: cell <= value}[op]
            keep = source.map(lambda cell: cell is not None and compare(cell))
        frame = frame.loc[keep].copy()
    return frame


def _aggregate(values, aggregation, numeric=False):
    present = [value for value in values if value is not None]
    if aggregation == "count_rows":
        return str(len(values))
    if aggregation == "count":
        return str(len(present))
    if aggregation == "count_distinct":
        return str(len({Decimal(value) if numeric else value for value in present}))
    if not present:
        return None
    numbers = [Decimal(value) for value in present]
    with localcontext() as context:
        context.prec = 768
        total = sum(numbers)
        if aggregation == "avg":
            context.prec = 34
        result = {"sum": lambda: total, "avg": lambda: total / len(numbers),
                  "min": lambda: min(numbers), "max": lambda: max(numbers)}[aggregation]()
    return str(result)


def _evaluate(store, definition):
    query(definition)
    frame, source, types = _read(store, definition["dataset"])
    frame = _filter(frame, definition["filters"], types)
    matched = len(frame)
    labels = {}
    if definition["mode"] == "rows":
        columns = definition["select"] or list(frame.columns)
        if not set(columns) <= set(types):
            raise WorkbenchError("invalid_query", "A selected field no longer exists in this snapshot.")
        frame = frame[columns].copy()
        output_types = {col: types[col] for col in columns}
    else:
        dimensions = definition["rows"]
        column, value, aggregation = definition["column"], definition["value"], definition["aggregate"]
        fields = [*dimensions, *([column] if column else []), *([value] if aggregation != "count_rows" else [])]
        if not set(fields) <= set(types) or (column and column in dimensions):
            raise WorkbenchError("invalid_query", "Choose existing fields and keep the pivot column separate from row fields.")
        if aggregation in {"sum", "avg", "min", "max"} and types.get(value) != "number":
            raise WorkbenchError("invalid_query", "This aggregation requires a numeric value field. Use count for text.")
        groups = {}
        column_values = set()
        def group_value(record, name):
            value = record[name]
            return Decimal(value) if value is not None and types[name] == "number" else value

        for record in frame.to_dict(orient="records"):
            row_key = tuple(group_value(record, name) for name in dimensions)
            pivot_key = group_value(record, column) if column else None
            column_values.add(pivot_key)
            groups.setdefault((row_key, pivot_key), []).append(record.get(value) if aggregation != "count_rows" else "1")
        row_keys = sorted({key[0] for key in groups}, key=lambda key: tuple((cell is not None, cell if cell is not None else "") for cell in key))
        if not dimensions and not row_keys and not column:
            row_keys = [()]
        pivots = sorted(column_values, key=lambda cell: (cell is not None, cell if cell is not None else "")) if column else [None]
        if len(pivots) > 60 or len(row_keys) * max(1, len(pivots)) > MAX_PIVOT_CELLS:
            raise WorkbenchError("too_large", "Pivot exceeds 60 columns or 100,000 cells. Filter categories or add fewer row groups.")
        used = set(dimensions)
        pivot_names = []
        for index, item in enumerate(pivots):
            if column:
                with localcontext() as context:
                    context.prec = 768
                    identity = ["number", str(item.normalize())] if isinstance(item, Decimal) else ["text", item]
                name = "__pivot_" + digest(canonical_json([column, identity]).encode())
            else:
                name = f"__value_{index}"
            while name in used:
                name += "_"
            used.add(name)
            pivot_names.append(name)
            category = str(item) if isinstance(item, Decimal) else json.dumps(item, ensure_ascii=False)
            labels[name] = f"{column} = {category}" if column else f"{aggregation}({'*' if aggregation == 'count_rows' else value})"
        records = []
        for row_key in row_keys:
            record = dict(zip(dimensions, (str(cell) if isinstance(cell, Decimal) else cell for cell in row_key)))
            for pivot_key, name in zip(pivots, pivot_names):
                values = groups.get((row_key, pivot_key))
                record[name] = _aggregate(values, aggregation, types.get(value) == "number") if values is not None else (
                    _aggregate([], aggregation) if not dimensions and not column else None)
            records.append(record)
        columns = [*dimensions, *pivot_names]
        frame = pd.DataFrame(records, columns=columns, dtype=object)
        output_types = {**{col: types[col] for col in dimensions}, **{col: "number" for col in pivot_names}}
    sort = definition["sort"]
    if sort["column"]:
        col = sort["column"]
        if col not in frame:
            raise WorkbenchError("invalid_query", "The sort field is not in this result. Clear sorting or select a result field.")
        frame = frame.sort_values(col, ascending=sort["direction"] == "asc", na_position="last", kind="stable",
                                  key=lambda values: values.map(lambda cell: Decimal(cell) if cell is not None else None)
                                  if output_types[col] == "number" else values)
    return frame, source, matched, labels, output_types


def preview(store, data):
    frame, source, matched, labels, types = _evaluate(store, data["query"])
    offset, limit = data.get("offset", 0), data.get("limit", 100)
    rows, size = [], 0
    for row in frame.iloc[offset:offset + limit].to_dict(orient="records"):
        row = {key: None if pd.isna(value) else value for key, value in row.items()}
        encoded = len(canonical_json(row).encode())
        if size + encoded > 1024 * 1024:
            if not rows:
                raise WorkbenchError("too_large", "One result row exceeds 1 MiB. Select fewer fields.")
            break
        rows.append(row)
        size += encoded
    end = offset + len(rows)
    return {"rows": rows, "columns": list(frame.columns), "labels": labels, "types": types,
            "total": len(frame), "offset": offset, "nextOffset": end if end < len(frame) else None,
            "matchedRows": matched, "source": source, "query": data["query"],
            "queryHash": digest(canonical_json(data["query"]).encode())}


def saved_queries(store, data):
    with store.connect(readonly=True) as conn:
        rows = conn.execute("SELECT id,revision,body FROM wb_state WHERE id LIKE ? ORDER BY id", (QUERY_PREFIX + "%",))
        items = [{**json.loads(row["body"]), "id": row["id"][len(QUERY_PREFIX):], "revision": row["revision"]} for row in rows]
    return {"queries": sorted(items, key=lambda item: item["updatedAt"], reverse=True)}


def save_query(store, data):
    definition = query(data["query"])
    _, source, _, _, _ = _evaluate(store, definition)
    name = data["name"].strip()
    if not name:
        raise WorkbenchError("invalid_query", "Give this query a name before saving.")
    identifier = data.get("id") or str(uuid.uuid4())
    body = {"name": name, "query": definition, "source": {"label": source["label"], "sha256": source["sha256"]},
            "updatedAt": timestamp()}
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT revision FROM wb_state WHERE id=?", (QUERY_PREFIX + identifier,)).fetchone()
        revision = row["revision"] if row else 0
        if data.get("id") and not row:
            raise WorkbenchError("not_found", "Saved query no longer exists. Save a new copy.")
        if data.get("expectedRevision", 0) != revision:
            raise WorkbenchError("revision_conflict", "This query changed in another panel. Reopen it before updating, or save a copy.")
        count = conn.execute("SELECT count(*) FROM wb_state WHERE id LIKE ?", (QUERY_PREFIX + "%",)).fetchone()[0]
        if not row and count >= 200:
            raise WorkbenchError("too_large", "This workspace already has 200 saved exploration queries. Update an existing query.")
        conn.execute("INSERT INTO wb_state VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,body=excluded.body",
                     (QUERY_PREFIX + identifier, revision + 1, canonical_json(body)))
    return {**body, "id": identifier, "revision": revision + 1}


def export(store, data):
    frame, source, matched, labels, _ = _evaluate(store, data["query"])
    identifier = str(uuid.uuid4())
    path = store.owned(f"exports/{identifier}.csv")
    recipe_path = store.owned(f"exports/{identifier}.query.json")
    path.parent.mkdir(exist_ok=True)
    escaped = 0

    def safe(value):
        nonlocal escaped
        if value is None or pd.isna(value):
            return ""
        value = str(value)
        if value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) and not NUMBER.fullmatch(value):
            escaped += 1
            return "'" + value
        return value

    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([safe(labels.get(col, col)) for col in frame.columns])
        for row in frame.itertuples(index=False, name=None):
            writer.writerow([safe(value) for value in row])
    receipt = {"version": 1, "query": data["query"], "source": source,
               "rows": len(frame), "matchedRows": matched, "createdAt": timestamp(),
               "csvSha256": digest(path.read_bytes()), "formulaEscapedCells": escaped,
               "semantics": "All matching previewable rows; blanks remain null. Exact decimal sums; averages use 34 significant digits."}
    with recipe_path.open("x", encoding="utf-8") as output:
        output.write(canonical_json(receipt))
    return {"id": identifier, "path": str(path), "queryPath": str(recipe_path), "rows": len(frame), "matchedRows": matched,
            "sha256": receipt["csvSha256"], "formulaEscapedCells": escaped, "bytes": path.stat().st_size}


def download(store, data):
    recipe_path = store.owned(f"exports/{data['id']}.query.json")
    if not recipe_path.is_file():
        raise WorkbenchError("not_found", "This exploration export no longer exists. Export the query again.")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    path = store.owned(f"exports/{data['id']}.csv") if data["kind"] == "csv" else recipe_path
    if path.stat().st_size > 4 * 1024 * 1024:
        raise WorkbenchError("too_large", f"This export exceeds the 4 MiB browser-transfer limit. The complete file is at {path}")
    raw = path.read_bytes()
    if data["kind"] == "csv" and digest(raw) != recipe["csvSha256"]:
        raise WorkbenchError("integrity_error", "The exported CSV changed after creation. Export the query again.")
    return {"name": path.name, "base64": base64.b64encode(raw).decode("ascii"),
            "mediaType": "text/csv;charset=utf-8" if data["kind"] == "csv" else "application/json"}
