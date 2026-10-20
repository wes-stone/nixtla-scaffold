import json
import subprocess
import sys
import time

import pandas as pd
import pytest

from nixtla_scaffold.workbench import dispatch
from nixtla_scaffold.workbench_store import WorkbenchStore


@pytest.mark.parametrize("series_count", [1, 50, 1000])
def test_scoped_pages_and_recent_history_remain_bounded(tmp_path, series_count):
    dispatch(tmp_path, "initialize", {})
    source = tmp_path / "panel"
    source.mkdir()
    names = [f"S{i:04d}-" + "long-series-name-" * 8 for i in range(series_count)]
    periods = 600 if series_count == 1 else 60
    forecast = pd.DataFrame({"unique_id": names, "ds": "2025-01-01", "yhat": 10.})
    forecast.to_csv(source / "forecast.csv", index=False)
    history = pd.DataFrame([{"unique_id": name, "ds": str(date.date()), "y": None if i == 20 else float(i)}
                            for name in names for i, date in enumerate(pd.date_range("2023-01-01", periods=periods, freq="D"))])
    history.to_csv(source / "history.csv", index=False)
    (source / "manifest.json").write_text(json.dumps({"spec": {"freq": "D", "horizon": 1}}), encoding="utf-8")
    run = dispatch(tmp_path, "import_run", {"path": str(source)})["id"]
    start = time.perf_counter()
    page = dispatch(tmp_path, "view", {"runId": run, "kind": "history", "series": names[-1], "limit": 25})
    assert len(page["rows"]) == 25 and page["total"] == periods and page["nextOffset"] == 25
    assert page["rows"][20]["y"] is None
    tail = dispatch(tmp_path, "view", {"runId": run, "kind": "history", "series": names[-1], "limit": 25, "tail": True})
    assert tail["offset"] == periods - 25 and tail["nextOffset"] is None and tail["rows"][-1]["y"] == periods - 1
    assert len(tail["series"]) == series_count
    assert len(json.dumps(tail).encode()) < 256 * 1024
    assert time.perf_counter() - start < 15
    exported = dispatch(tmp_path, "export_values", {"runId": run, "kind": "history", "series": names[-1]})
    assert exported["rows"] == periods


def test_import_recovers_after_process_dies_between_ledger_and_catalog_commit(tmp_path):
    dispatch(tmp_path, "initialize", {})
    source = tmp_path / "input"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 1}}), encoding="utf-8")
    (source / "forecast.csv").write_text("unique_id,ds,yhat\nx,2025-01-01,1\n", encoding="utf-8")
    code = (
        "import os,sys\nimport nixtla_scaffold.workbench_store as s\n"
        "register=s.register_run\n"
        "def interrupted(*args,**kwargs):\n register(*args,**kwargs)\n os._exit(91)\n"
        "s.register_run=interrupted\ns.WorkbenchStore(sys.argv[1]).import_run(sys.argv[2])\n"
    )
    crashed = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(source)], timeout=45, capture_output=True)
    assert crashed.returncode == 91, crashed.stderr.decode()
    assert not dispatch(tmp_path, "catalog", {})["runs"]
    recovered = WorkbenchStore(tmp_path).import_run(source)
    assert WorkbenchStore(tmp_path).import_run(source)["reused"]
    assert dispatch(tmp_path, "catalog", {})["runs"][0]["id"] == recovered["id"]


def test_source_catalog_pages_without_loading_series_profiles(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    with store.connect() as conn:
        conn.executemany("INSERT INTO wb_sources VALUES(?,?,?,?,?)",
                         [(str(i), f"Source {i}", f"sources/{i}", str(i),
                           json.dumps({"profile": {"series_count": 1000, "series": [{"unique_id": "hidden"}]}}))
                          for i in range(205)])
    first = dispatch(tmp_path, "catalog", {})
    assert len(first["sources"]) == 200 and first["sources"][0]["id"] == "204"
    assert first["totalSources"] == 205 and first["nextSourceOffset"] == 200
    assert "series" not in first["sources"][0]["metadata"]["profile"]
    last = dispatch(tmp_path, "catalog", {"sourceOffset": first["nextSourceOffset"]})
    assert len(last["sources"]) == 5 and last["nextSourceOffset"] is None
    assert not {row["id"] for row in first["sources"]} & {row["id"] for row in last["sources"]}
