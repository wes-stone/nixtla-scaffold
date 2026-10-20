import base64
import csv
from decimal import Decimal
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from nixtla_scaffold.workbench import dispatch
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore


def dataset(tmp_path, content):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text('{"spec":{"freq":"MS","horizon":1}}', encoding="utf-8")
    (run / "forecast.csv").write_text("unique_id,ds,yhat\nfixture,2026-01-01,1\n", encoding="utf-8")
    (run / "table.csv").write_text(content, encoding="utf-8")
    rid = store.import_run(run, name="Exploration fixture")["id"]
    return {"kind": "artifact", "runId": rid, "name": "table.csv"}


def query(source, **kwargs):
    return {"version": 1, "dataset": source, "mode": "rows", "select": [], "rows": [], "column": "",
            "value": "", "aggregate": "sum", "filters": [], "sort": {"column": "", "direction": "asc"}, **kwargs}


def preview(tmp_path, definition, **kwargs):
    return dispatch(tmp_path, "explore_preview", {"query": definition, **kwargs})


def test_full_snapshot_pivot_and_export_are_not_preview_page(tmp_path):
    source = dataset(tmp_path, "unique_id,month,model,y\n" + "".join(
        f"001,2026-{i % 2 + 1:02d}-01,Model {i % 3},{i}.00001\n" for i in range(250)))
    definition = query(source, mode="pivot", rows=["month"], column="model", value="y")
    result = preview(tmp_path, definition, limit=1)
    assert result["total"] == 2 and result["matchedRows"] == 250 and result["nextOffset"] == 1
    column = next(key for key, label in result["labels"].items() if label == 'model = "Model 0"')
    assert result["rows"][0][column] == str(sum(Decimal(f"{i}.00001") for i in range(0, 250, 6)))
    receipt = dispatch(tmp_path, "explore_export", {"query": definition})
    exported = list(csv.DictReader(Path(receipt["path"]).open(encoding="utf-8")))
    assert len(exported) == 2 and receipt["matchedRows"] == 250
    recipe = json.loads(Path(receipt["queryPath"]).read_text(encoding="utf-8"))
    assert recipe["query"] == definition and recipe["source"]["sha256"] == result["source"]["sha256"]
    assert dispatch(tmp_path, "catalog", {})["jobs"] == []


@pytest.mark.parametrize(("aggregation", "expected"), [
    ("sum", "9007199254740993.123456789012345680"), ("avg", "4503599627370496.561728394506172840"),
    ("min", "0.000000000000000001"), ("max", "9007199254740993.123456789012345679"),
    ("count", "2"), ("count_distinct", "2"), ("count_rows", "3"),
])
def test_decimal_precision_and_null_aggregates(tmp_path, aggregation, expected):
    source = dataset(tmp_path, "model,y\nA,9007199254740993.123456789012345679\nA,0.000000000000000001\nA,\nB,\nC,0\n")
    result = preview(tmp_path, query(source, mode="pivot", rows=["model"], value="y", aggregate=aggregation))
    assert Decimal(result["rows"][0]["__value_0"]) == Decimal(expected)
    assert result["rows"][1]["__value_0"] == ("1" if aggregation == "count_rows" else "0" if aggregation.startswith("count") else None)
    assert result["rows"][2]["__value_0"] == ("1" if aggregation.startswith("count") else "0")


def test_identifiers_missing_groups_and_collision_safe_pivot_labels(tmp_path):
    source = dataset(tmp_path, "unique_id,__value_0,category,y\n001,k,,1\nNA,k,null,0\n1,k,(empty),\n")
    result = preview(tmp_path, query(source))
    assert [row["unique_id"] for row in result["rows"]] == ["001", "NA", "1"]
    assert result["types"]["unique_id"] == "text"
    result = preview(tmp_path, query(source, mode="pivot", rows=["__value_0"], column="category", value="y"))
    assert len(result["columns"]) == 4 and len(set(result["columns"])) == 4
    assert len(set(result["labels"].values())) == 3
    assert result["rows"][0]["__value_0"] == "k"
    assert None in result["rows"][0].values() and "0" in result["rows"][0].values()


def test_filters_sorts_empty_results_and_raw_precision(tmp_path):
    source = dataset(tmp_path, "unique_id,y,note\n001,10.000000000000001,=HYPERLINK()\nNA,2,NA\n1,,empty\n")
    definition = query(source, filters=[{"column": "y", "operator": "gte", "value": "2"}],
                       sort={"column": "y", "direction": "asc"}, select=["unique_id", "y"])
    result = preview(tmp_path, definition)
    assert [row["unique_id"] for row in result["rows"]] == ["NA", "001"]
    assert result["rows"][1]["y"] == "10.000000000000001"
    assert preview(tmp_path, query(source, filters=[{"column": "note", "operator": "contains", "value": "hyperlink"}]))["total"] == 1
    assert preview(tmp_path, query(source, filters=[{"column": "y", "operator": "is_null", "value": ""}]))["rows"][0]["unique_id"] == "1"
    filters = [{"column": "y", "operator": "gte", "value": "100"}]
    assert preview(tmp_path, query(source, filters=filters))["rows"] == []
    assert preview(tmp_path, query(source, mode="pivot", value="y", filters=filters))["rows"] == [{"__value_0": None}]
    assert preview(tmp_path, query(source, mode="pivot", aggregate="count_rows", filters=filters))["rows"] == [{"__value_0": "0"}]
    assert preview(tmp_path, query(source, mode="pivot", column="note", value="y", filters=filters))["total"] == 0


def test_numeric_groups_and_distinct_counts_share_numeric_filter_semantics(tmp_path):
    source = dataset(tmp_path, "bucket,y\n1,0\n1.0,0.0\n2,2\n2.0,2.00\n")
    result = preview(tmp_path, query(source, mode="pivot", rows=["bucket"], value="y", aggregate="count_distinct"))
    assert result["total"] == 2
    assert [row["__value_0"] for row in result["rows"]] == ["1", "1"]
    pivot = preview(tmp_path, query(source, mode="pivot", column="bucket", aggregate="count_rows"))
    assert len(pivot["columns"]) == 2
    assert list(pivot["rows"][0].values()) == ["2", "2"]


def test_pivot_sort_stays_bound_to_category_when_an_earlier_category_disappears(tmp_path):
    source = dataset(tmp_path, "group,model,y\nx,A,1\nx,B,10\nx,C,100\ny,A,1\ny,B,20\ny,C,50\n")
    definition = query(source, mode="pivot", rows=["group"], column="model", value="y")
    initial = preview(tmp_path, definition)
    b_column = next(key for key, label in initial["labels"].items() if label == 'model = "B"')
    definition["sort"] = {"column": b_column, "direction": "asc"}
    definition["filters"] = [{"column": "model", "operator": "neq", "value": "A"}]
    result = preview(tmp_path, definition)
    assert result["labels"][b_column] == 'model = "B"'
    assert [row["group"] for row in result["rows"]] == ["x", "y"]
    definition["filters"] = [{"column": "model", "operator": "neq", "value": "B"}]
    with pytest.raises(WorkbenchError, match="sort field"):
        preview(tmp_path, definition)


def test_byte_bounded_pages_keep_exact_continuations(tmp_path):
    source = dataset(tmp_path, "unique_id,detail\n" + "".join(f"{index},{'x' * 70000}\n" for index in range(30)))
    definition = query(source)
    first = preview(tmp_path, definition)
    second = preview(tmp_path, definition, offset=first["nextOffset"])
    third = preview(tmp_path, definition, offset=second["nextOffset"])
    assert [len(page["rows"]) for page in [first, second, third]] == [14, 14, 2]
    assert [first["offset"], second["offset"], third["offset"]] == [0, 14, 28]
    assert third["nextOffset"] is None


def test_exports_formula_escape_headers_and_values_and_download_integrity(tmp_path):
    source = dataset(tmp_path, "unique_id,y,=header\n001,-12.12345678901234567890,=SUM(A1)\nNA,0,@text\n")
    receipt = dispatch(tmp_path, "explore_export", {"query": query(source)})
    assert receipt["formulaEscapedCells"] == 3
    exported = Path(receipt["path"]).read_text(encoding="utf-8")
    assert "'=header" in exported and "'=SUM" in exported and "-12.12345678901234567890" in exported
    downloaded = dispatch(tmp_path, "explore_download", {"id": receipt["id"], "kind": "csv"})
    assert base64.b64decode(downloaded["base64"]) == Path(receipt["path"]).read_bytes()
    definition = dispatch(tmp_path, "explore_download", {"id": receipt["id"], "kind": "query"})
    assert json.loads(base64.b64decode(definition["base64"]))["rows"] == 2
    Path(receipt["path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="changed"):
        dispatch(tmp_path, "explore_download", {"id": receipt["id"], "kind": "csv"})
    with pytest.raises(WorkbenchError):
        dispatch(tmp_path, "explore_download", {"id": "../outside", "kind": "csv"})


def test_saved_queries_survive_existing_workspace_reopen_and_conflicts(tmp_path):
    source = dataset(tmp_path, "unique_id,y\n001,2\n")
    dispatch(tmp_path, "save_state", {"expectedRevision": 0, "state": {"documents": []}})
    before = dispatch(tmp_path, "catalog", {})
    saved = dispatch(tmp_path, "explore_save", {"name": "By month", "query": query(source)})
    assert saved["revision"] == 1
    assert dispatch(tmp_path, "explore_queries", {})["queries"] == [saved]
    assert WorkbenchStore(tmp_path).initialized
    assert dispatch(tmp_path, "catalog", {})["revision"] == before["revision"]
    changed = dispatch(tmp_path, "explore_save", {"id": saved["id"], "expectedRevision": 1, "name": "Renamed", "query": query(source)})
    assert changed["revision"] == 2
    with pytest.raises(WorkbenchError, match="another panel"):
        dispatch(tmp_path, "explore_save", {"id": saved["id"], "expectedRevision": 1, "name": "Stale", "query": query(source)})
    assert dispatch(tmp_path, "explore_queries", {})["queries"][0]["name"] == "Renamed"
    catalog = dispatch(tmp_path, "explore_catalog", {"runId": source["runId"]})
    assert any(table["dataset"] == source for table in catalog["tables"])


def test_protected_source_never_exposed_to_schema_filter_pivot_or_export(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    path = tmp_path / "source.csv"
    pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2020-01-01", periods=48, freq="MS"), "y": range(48)}).to_csv(path, index=False)
    sid = dispatch(tmp_path, "profile", {"path": str(path), "horizon": 3, "holdoutWindows": 2})["sourceId"]
    source = {"kind": "source", "sourceId": sid}
    schema = dispatch(tmp_path, "explore_schema", {"dataset": source})
    assert schema["rows"] == 42 and schema["withheldRows"] == 6
    assert Decimal(preview(tmp_path, query(source, mode="pivot", aggregate="max", value="y"))["rows"][0]["__value_0"]) == 41
    assert preview(tmp_path, query(source, filters=[{"column": "y", "operator": "gte", "value": "42"}]))["total"] == 0
    receipt = dispatch(tmp_path, "explore_export", {"query": query(source)})
    assert receipt["rows"] == 42
    assert len(list(csv.DictReader(io.StringIO(Path(receipt["path"]).read_text())))) == 42
    assert dispatch(tmp_path, "catalog", {})["jobs"] == []


def test_invalid_queries_high_cardinality_and_hash_tampering_fail_explicitly(tmp_path):
    source = dataset(tmp_path, "unique_id,model,y\n" + "".join(f"001,M{i},{i}\n" for i in range(61)))
    for changes in [
        {"rows": ["absent"], "mode": "pivot", "value": "y"},
        {"mode": "pivot", "value": "unique_id"},
        {"select": ["absent"]},
        {"filters": [{"column": "y", "operator": "gte", "value": "NaN"}]},
        {"filters": [{"column": "y", "operator": "eq", "value": "1e9999"}]},
        {"sort": {"column": "absent", "direction": "asc"}},
    ]:
        with pytest.raises(WorkbenchError):
            preview(tmp_path, query(source, **changes))
    with pytest.raises(WorkbenchError, match="60 columns"):
        preview(tmp_path, query(source, mode="pivot", column="model", value="y"))
    with pytest.raises(WorkbenchError):
        preview(tmp_path, query({**source, "name": "../outside.csv"}))
    root, _ = WorkbenchStore(tmp_path).run(source["runId"])
    (root / "table.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="hash changed"):
        preview(tmp_path, query(source))
