import json
import base64
from pathlib import Path

import pytest

from nixtla_scaffold.workbench import dispatch
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore


@pytest.fixture
def saved(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    run = tmp_path / "run"
    (run / "appendix").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 1}}))
    (run / "forecast.csv").write_text("unique_id,ds,yhat\n001,2026-01-01,12345678901234.56789\n")
    (run / "appendix" / "backtest_long.csv").write_text(
        "unique_id,ds,cutoff,model,yhat,y_actual\n"
        "001,2026-01-01,2025-12-01,A,1.23456789012345,2\n"
        "001,2026-02-01,2026-01-01,A,2.23456789012345,3\n"
        "001,2026-01-01,2025-12-01,B,,2\n"
        "NA,2026-01-01,2025-12-01,A,5,6\n")
    (run / "appendix" / "hierarchy_rollup.csv").write_text(
        "parent_unique_id,child_unique_id,ds,parent_yhat\n001,0002,2026-01-01,3.333333333\n")
    (run / "appendix" / "model_win_rates.csv").write_text("model,win_rate\nA,0.5\n")
    rid = store.import_run(run, name="Evidence")["id"]
    return tmp_path, store, rid


def test_scoped_typed_paging_and_facets_do_not_execute_or_change_state(saved):
    root, store, rid = saved
    before = dispatch(root, "catalog", {})
    request = {"runId": rid, "name": "appendix/backtest_long.csv", "series": "001", "models": ["A"], "limit": 1}
    first = dispatch(root, "evidence", request)
    assert first["total"] == 2 and first["nextOffset"] == 1
    assert first["models"] == ["A", "B"] and len(first["cutoffs"]) == 2
    assert first["rows"][0]["unique_id"] == "001"
    assert first["rows"][0]["yhat"] == 1.23456789012345
    assert first["bounds"]["yhat"]["max"] == 2.23456789012345
    last = dispatch(root, "evidence", {**request, "offset": 1})
    assert last["nextOffset"] is None and last["rows"][0]["ds"] == "2026-02-01"
    cutoff = dispatch(root, "evidence", {**request, "cutoff": "2026-01-01", "columns": ["ds", "yhat"]})
    assert cutoff["total"] == 1 and cutoff["columns"] == ["ds", "yhat"]
    missing = dispatch(root, "evidence", {**request, "models": ["B"]})
    assert missing["rows"][0]["yhat"] is None
    after = dispatch(root, "catalog", {})
    assert after["jobs"] == before["jobs"] and after["revision"] == before["revision"]


def test_hierarchy_global_and_absent_evidence_are_explicit(saved):
    root, _, rid = saved
    hierarchy = dispatch(root, "evidence", {"runId": rid, "name": "appendix/hierarchy_rollup.csv", "series": "001", "seriesColumn": "parent_unique_id"})
    assert hierarchy["rows"][0]["child_unique_id"] == "0002" and hierarchy["seriesScope"] == "series"
    global_rows = dispatch(root, "evidence", {"runId": rid, "name": "appendix/model_win_rates.csv", "series": "001"})
    assert global_rows["seriesScope"] == "run" and global_rows["total"] == 1
    missing = dispatch(root, "evidence", {"runId": rid, "name": "finn/forecast.csv"})
    assert not missing["available"] and missing["message"]
    with pytest.raises(WorkbenchError, match="no parent"):
        dispatch(root, "evidence", {"runId": rid, "name": "forecast.csv", "seriesColumn": "parent"})


def test_exports_preserve_original_precision_and_full_filtered_scope(saved):
    root, _, rid = saved
    exported = dispatch(root, "export_evidence", {"runId": rid, "name": "forecast.csv", "series": "001"})
    assert exported["rows"] == 1
    assert "12345678901234.56789" in Path(exported["path"]).read_text()
    exported = dispatch(root, "export_evidence", {"runId": rid, "name": "appendix/backtest_long.csv", "series": "001", "models": ["A"]})
    assert exported["rows"] == 2
    assert "2.23456789012345" in Path(exported["path"]).read_text()


def test_bounds_security_and_hashes(saved):
    root, store, rid = saved
    for name in ["../forecast.csv", "appendix\\forecast.csv", "/forecast.csv", "manifest.json"]:
        with pytest.raises(WorkbenchError):
            dispatch(root, "evidence", {"runId": rid, "name": name})
    with pytest.raises(WorkbenchError):
        dispatch(root, "evidence", {"runId": rid, "name": "forecast.csv", "limit": 2001})
    with pytest.raises(WorkbenchError, match="Unknown"):
        dispatch(root, "evidence", {"runId": rid, "name": "forecast.csv", "columns": ["invented"]})
    path, _ = store.run(rid)
    (path / "forecast.csv").write_text("tampered")
    with pytest.raises(WorkbenchError, match="hash changed"):
        dispatch(root, "evidence", {"runId": rid, "name": "forecast.csv"})


def test_byte_bounded_evidence_retains_continuation_and_exact_export(saved):
    root, store, _ = saved
    source = root / "run"
    (source / "appendix" / "large.csv").write_text("unique_id,note\n001," + "a" * 600000 + "\n001," + "b" * 600000 + "\n")
    rid = store.import_run(source)["id"]
    request = {"runId": rid, "name": "appendix/large.csv", "series": "001"}
    first = dispatch(root, "evidence", request)
    assert len(first["rows"]) == 1 and first["nextOffset"] == 1
    second = dispatch(root, "evidence", {**request, "offset": first["nextOffset"]})
    assert second["rows"][0]["note"] == "b" * 600000 and second["nextOffset"] is None
    exported = dispatch(root, "export_evidence", request)
    assert exported["rows"] == 2 and "b" * 600000 in Path(exported["path"]).read_text()


def test_image_previews_are_verified_passive_and_bounded(saved):
    root, store, _ = saved
    source = root / "run"
    image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZt8AAAAASUVORK5CYII=")
    (source / "evidence.png").write_bytes(image)
    (source / "large.png").write_bytes(image + b"x" * (2 * 1024 * 1024))
    (source / "active.svg").write_text("<svg/>")
    rid = store.import_run(source)["id"]
    result = dispatch(root, "evidence_image", {"runId": rid, "name": "evidence.png"})
    assert result["mediaType"] == "image/png" and base64.b64decode(result["base64"]) == image
    with pytest.raises(WorkbenchError, match="2 MiB"):
        dispatch(root, "evidence_image", {"runId": rid, "name": "large.png"})
    with pytest.raises(WorkbenchError, match="PNG and JPEG"):
        dispatch(root, "evidence_image", {"runId": rid, "name": "active.svg"})
