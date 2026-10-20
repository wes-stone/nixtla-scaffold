import json

import pandas as pd
import pytest

from nixtla_scaffold.workbench import dispatch
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore
from nixtla_scaffold.data import load_forecast_dataset


def test_library_search_reaches_old_versions_sources_and_files_without_opening_blobs(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    with store.connect() as conn:
        conn.executemany("INSERT INTO wb_runs VALUES(?,?,?,?,?,?)", [
            (str(i), f"Forecast {i:03}", f"2026-01-{i:04}", "not-opened",
             json.dumps({"audit/rare-evidence.csv" if i == 0 else "forecast.csv": "hash"}), "{}") for i in range(205)])
        conn.executemany("INSERT INTO wb_sources VALUES(?,?,?,?,?)", [
            (str(i), "Legacy_100%" if i == 0 else f"Source {i}", "not-opened", "hash", "{}") for i in range(205)])
    assert not any(row["id"] == "0" for row in dispatch(tmp_path, "catalog", {})["runs"])
    families = dispatch(tmp_path, "library", {"section": "forecasts", "limit": 50})
    assert len(families["items"]) == 50 and families["total"] == 205 and families["nextOffset"] == 50
    old = dispatch(tmp_path, "library", {"section": "versions", "family": "Forecast 000"})
    assert old["items"][0]["id"] == "0"
    found = dispatch(tmp_path, "library", {"section": "search", "query": "rare-evidence"})
    assert found["total"] == 1 and found["items"][0]["artifactName"] == "audit/rare-evidence.csv"
    for query in ["Legacy_100%", "100%", "_"]:
        sources = dispatch(tmp_path, "library", {"section": "search", "query": query})
        assert sources["total"] == 1 and sources["items"][0]["sourceId"] == "0"
    assert dispatch(tmp_path, "library", {"section": "search", "query": "' OR 1=1 --"})["total"] == 0
    assert dispatch(tmp_path, "library", {"section": "files", "runId": "0"})["items"][0]["sha256"] == "hash"
    with pytest.raises(WorkbenchError, match="Choose"):
        dispatch(tmp_path, "library", {"section": "versions"})


def test_source_preview_keeps_reserved_targets_private_and_changes_no_state(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    path = tmp_path / "source.csv"
    pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2020-01-01", periods=48, freq="MS"), "y": range(48)}).to_csv(path, index=False)
    sid = dispatch(tmp_path, "profile", {"path": str(path), "horizon": 3, "holdoutWindows": 2})["sourceId"]
    first = dispatch(tmp_path, "source_view", {"sourceId": sid, "limit": 10})
    assert first["total"] == 42 and first["withheldRows"] == 6 and first["nextOffset"] == 10
    assert first["rows"][0]["unique_id"] == "001"
    assert load_forecast_dataset(store.source(sid)[0])["unique_id"].unique().tolist() == ["001"]
    last = dispatch(tmp_path, "source_view", {"sourceId": sid, "offset": 40, "limit": 10})
    assert last["nextOffset"] is None and last["rows"][-1]["y"] == 41
    catalog = dispatch(tmp_path, "catalog", {})
    assert catalog["revision"] == 0 and not catalog["jobs"]
    # A read-only preview does not consume the protected research allocation.
    assert dispatch(tmp_path, "start_job", {"kind": "research", "sourceId": sid, "requestKey": "after-preview",
                                           "spec": {"horizon": 3, "context": {"promotion_policy": {"minimum_confirmation_cutoffs": 2}}}})["created"]

@pytest.mark.parametrize("column,mapping", [("UNIQUE_ID", {}), ("account", {"id": "account"})])
@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_profile_and_shared_loader_preserve_distinct_text_series_ids(tmp_path, column, mapping, extension):
    frame = pd.DataFrame([
        {column: series, "ds": date, "y": i}
        for series in ["001", "1", "NA"]
        for i, date in enumerate(pd.date_range("2020-01-01", periods=24, freq="MS"))
    ])
    path = tmp_path / f"source.{extension}"
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False)
    assert set(load_forecast_dataset(path, id_col=column)["unique_id"]) == {"001", "1", "NA"}
    dispatch(tmp_path, "initialize", {})
    result = dispatch(tmp_path, "profile", {"path": str(path), "mapping": mapping, "horizon": 3})
    preview = dispatch(tmp_path, "source_view", {"sourceId": result["sourceId"]})
    assert set(row["unique_id"] for row in preview["rows"]) == {"001", "1", "NA"}
    assert preview["withheldRows"] == 9


def test_file_previews_page_exact_text_and_verify_hashes(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    path = tmp_path / "run"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 1}}), encoding="utf-8")
    (path / "forecast.csv").write_text("unique_id,ds,yhat\n001,2025-01-01,12345678901234.56789\n002,2025-01-01,\n", encoding="utf-8")
    (path / "notes.md").write_text("x" * 70000, encoding="utf-8")
    (path / "run.py").write_text("raise RuntimeError('must never execute')", encoding="utf-8")
    rid = store.import_run(path, name="Examples")["id"]
    page = dispatch(tmp_path, "file_view", {"runId": rid, "name": "forecast.csv", "limit": 1})
    assert page["rows"][0]["yhat"] == "12345678901234.56789" and page["rows"][0]["unique_id"] == "001"
    assert page["total"] == 2 and page["nextOffset"] == 1
    assert dispatch(tmp_path, "file_view", {"runId": rid, "name": "forecast.csv", "offset": 1})["rows"][0]["yhat"] == ""
    assert dispatch(tmp_path, "file_view", {"runId": rid, "name": "notes.md"})["truncated"]
    assert "must never execute" in dispatch(tmp_path, "file_view", {"runId": rid, "name": "run.py"})["text"]
    with pytest.raises(WorkbenchError, match="not part"):
        dispatch(tmp_path, "file_view", {"runId": rid, "name": "../outside"})
    root, _ = store.run(rid)
    (root / "forecast.csv").write_text("altered", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="hash changed"):
        dispatch(tmp_path, "file_view", {"runId": rid, "name": "forecast.csv"})


def test_file_csv_byte_limit_keeps_continuation_exact(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    path = tmp_path / "run"
    path.mkdir()
    (path / "manifest.json").write_text('{"spec":{"horizon":1,"freq":"MS"}}', encoding="utf-8")
    (path / "forecast.csv").write_text("unique_id,ds,yhat,detail\n" + "\n".join(f"x,2025-01-01,{i}," + "x" * 40000 for i in range(3)), encoding="utf-8")
    rid = store.import_run(path)["id"]
    first = dispatch(tmp_path, "file_view", {"runId": rid, "name": "forecast.csv"})
    assert first["total"] == 3 and first["nextOffset"] == 1
    second = dispatch(tmp_path, "file_view", {"runId": rid, "name": "forecast.csv", "offset": 1})
    assert second["total"] == 3 and second["nextOffset"] == 2
    assert second["rows"][0]["yhat"] == "1"
