import base64
import csv
from decimal import Decimal
import io
import json
from pathlib import Path

import pytest

from nixtla_scaffold.workbench import dispatch
from nixtla_scaffold.workbench_judgment import DEFAULTS, calculate
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore


def fixture(tmp_path):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    root = tmp_path / "fixture"
    (root / "appendix").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"spec": {"freq": "MS", "horizon": 2, "unit_label": "minutes"}}))
    (root / "forecast.csv").write_text("unique_id,ds,yhat,model\n001,2026-10-01,100,Conservative\n001,2026-11-01,120,Conservative\n")
    (root / "appendix/history.csv").write_text("unique_id,ds,y\n001,2026-08-01,90\n001,2026-09-01,100\n")
    (root / "appendix/forecast_long.csv").write_text(
        "unique_id,ds,model,yhat\n001,2026-10-01,Conservative,100\n001,2026-11-01,Conservative,120\n"
        "001,2026-10-01,Growth,200\n001,2026-11-01,Growth,300\n001,2026-10-01,Partial,250\n")
    run_id = store.import_run(root, name="Synthetic minutes")["id"]
    return {"runId": run_id, "series": "001"}


def test_model_choice_and_haircut(tmp_path):
    request = fixture(tmp_path)
    data = dispatch(tmp_path, "judgment_source", {**request, "model": "Growth"})
    assert data["selectedModel"] == "Conservative"
    assert data["rows"][0]["model_yhat"] == "200"
    assert not next(row for row in data["models"] if row["id"] == "Partial")["complete"]
    with pytest.raises(WorkbenchError, match="cover every"):
        dispatch(tmp_path, "judgment_source", {**request, "model": "Partial"})
    rows, totals = calculate(data["rows"], {**DEFAULTS, "percent": "-10"})
    assert [Decimal(row["adjusted_yhat"]) for row in rows] == [180, 270]
    assert Decimal(totals["adjusted_yhat"]) == 450
    assert Decimal(totals["delta_pct"]) == -10
    assert Decimal(totals["selected_yhat"]) == 220


def test_order_row_replacement_range_and_zero_override():
    base = [{"ds": f"2026-{month:02d}-01", "model_yhat": "100", "selected_yhat": "80"} for month in (10, 11, 12)]
    rows, total = calculate(base, {**DEFAULTS, "percent": "-10", "amount": "-5", "end": "2026-11-01", "edits": [
        {"ds": "2026-10-01", "percent": "20", "amount": "", "override": "", "note": "New launch"},
        {"ds": "2026-11-01", "percent": "", "amount": "", "override": "0", "note": "Closure"},
    ]})
    assert [Decimal(row["adjusted_yhat"]) for row in rows] == [115, 0, 100]
    assert [Decimal(row["cumulative_delta"]) for row in rows] == [15, -85, -85]
    assert Decimal(total["delta"]) == -85
    assert total["delta_pct"].startswith("-28.333333")


def test_decimal_exactness_zero_denominator_and_no_floor():
    rows, totals = calculate([{"ds": "2026-10-01", "model_yhat": "0.1", "selected_yhat": "0"}],
                             {**DEFAULTS, "percent": "-10", "amount": "-0.1"})
    assert Decimal(rows[0]["adjusted_yhat"]) == Decimal("-0.01")
    _, totals = calculate([{"ds": "2026-10-01", "model_yhat": "0", "selected_yhat": "0"}], DEFAULTS)
    assert totals["delta_pct"] is None
    with pytest.raises(WorkbenchError):
        calculate([{"ds": "2026-10-01", "model_yhat": "1e100", "selected_yhat": "1e100"}], {**DEFAULTS, "percent": "1e100"})


@pytest.mark.parametrize("change", [{"percent": ""}, {"amount": "NaN"}, {"percent": "1e999"}, {"start": "2027-01-01"},
    {"edits": [{"ds": "2020-01-01", "percent": "", "amount": "", "override": "10", "note": ""}]}])
def test_invalid_adjustments_are_explicit(change):
    with pytest.raises(WorkbenchError):
        calculate([{"ds": "2026-10-01", "model_yhat": "100", "selected_yhat": "80"}], {**DEFAULTS, **change})


def test_save_branch_export_and_existing_surfaces(tmp_path):
    request = fixture(tmp_path)
    src = dispatch(tmp_path, "judgment_source", {**request, "model": "Growth"})
    settings = {**DEFAULTS, "model": "Growth", "percent": "-10", "label": "Leadership", "reason": "Capacity risk"}
    save_request = {**request, "settings": settings, "expectedSourceHash": src["sourceHash"]}
    store = WorkbenchStore(tmp_path)
    before = store.read_artifact(request["runId"], "forecast.csv")
    version = dispatch(tmp_path, "judgment_save", save_request)
    assert dispatch(tmp_path, "judgment_save", save_request)["id"] == version["id"]
    assert store.read_artifact(request["runId"], "forecast.csv") == before
    viewed = dispatch(tmp_path, "view", {"runId": version["id"], "series": "001"})
    assert viewed["gate"]["planning_ready_claim_allowed"] is False
    assert viewed["rows"][0]["yhat"] == 180
    assert not any(key.startswith("yhat_lo") for key in viewed["rows"][0])
    assert dispatch(tmp_path, "view", {"runId": version["id"], "kind": "backtest"})["total"] == 0
    assert dispatch(tmp_path, "compare_versions", {"runId": version["id"], "referenceRunId": request["runId"]})["rows"][0]["delta"] == 80
    assert dispatch(tmp_path, "explore_catalog", {"runId": version["id"]})
    branch_src = dispatch(tmp_path, "judgment_source", {"runId": version["id"], "series": "001"})
    assert branch_src["baseRunId"] == request["runId"]
    assert branch_src["savedSettings"] == settings
    branch = dispatch(tmp_path, "judgment_save", {**save_request, "runId": version["id"],
        "settings": {**settings, "percent": "-20", "label": "Downside"}})
    branch_view = dispatch(tmp_path, "view", {"runId": branch["id"]})
    assert branch_view["rows"][0]["yhat"] == 160  # Against the model, not another haircut on 180.
    recipe = json.loads(store.read_artifact(branch["id"], "judgment.json"))
    assert recipe["parentRunId"] == version["id"] and recipe["baseRunId"] == request["runId"]
    exported = dispatch(tmp_path, "judgment_export", save_request)
    downloaded = dispatch(tmp_path, "explore_download", {"id": exported["id"], "kind": "csv"})
    raw = base64.b64decode(downloaded["base64"]).decode()
    imported = dispatch(tmp_path, "judgment_import", {**request, "model": "Growth", "text": raw})
    rows, total = calculate(src["rows"], {**DEFAULTS, "edits": imported["edits"]})
    assert Decimal(total["adjusted_yhat"]) == 450
    assert len(list(csv.DictReader(io.StringIO(raw)))) == 2
    assert not dispatch(tmp_path, "catalog", {})["jobs"]
    with pytest.raises(WorkbenchError, match="Refresh the underlying"):
        from nixtla_scaffold.workbench_operations import prepare
        prepare(store, {"runId": version["id"], "kind": "refresh", "approved": True})
    (tmp_path / "portable").mkdir()
    portable = WorkbenchStore(tmp_path / "portable")
    portable.initialize()
    reimported = portable.import_run(store.run(version["id"])[0])
    imported_metadata = json.loads(portable.run(reimported["id"])[1]["metadata"])
    assert imported_metadata["purpose"] == "judgment_scenario"
    assert imported_metadata["judgmentLabel"] == "Leadership"
    with pytest.raises(WorkbenchError, match="Refresh the underlying"):
        prepare(portable, {"runId": reimported["id"], "kind": "refresh", "approved": True})


@pytest.mark.parametrize("text", [
    "ds,override\n2026-09-01,90\n",
    "ds,override\n2026-10-01,90\n2026-10-01,90\n",
    "ds,override\n2026-10-01,=1+1\n",
    "ds,override\n2026-10-01,\n2026-11-01,90,extra\n",
    "ds,override,unique_id\n2026-10-01,90,002\n",
    "ds,override,model\n2026-10-01,90,Wrong\n",
])
def test_upload_rejects_wrong_scope_duplicate_dates_and_formulas(tmp_path, text):
    request = fixture(tmp_path)
    with pytest.raises(WorkbenchError):
        dispatch(tmp_path, "judgment_import", {**request, "model": "", "text": text})


def test_stale_source_and_unnamed_save(tmp_path):
    request = fixture(tmp_path)
    with pytest.raises(WorkbenchError, match="source changed"):
        dispatch(tmp_path, "judgment_save", {**request, "settings": DEFAULTS, "expectedSourceHash": "stale"})
    source = dispatch(tmp_path, "judgment_source", request)
    with pytest.raises(WorkbenchError, match="Name the new version"):
        dispatch(tmp_path, "judgment_save", {**request, "settings": DEFAULTS, "expectedSourceHash": source["sourceHash"]})


def test_import_size_is_bounded_by_bytes_not_characters(tmp_path):
    request = fixture(tmp_path)
    with pytest.raises(WorkbenchError, match="128 KiB"):
        dispatch(tmp_path, "judgment_import", {**request, "model": "", "text": "\u00e9" * 70000})


def layer_settings(**changes):
    from nixtla_scaffold.workbench_calendar_layer import DEFAULTS as CALENDAR
    return {**CALENDAR, "enabled": True, **changes}


def test_calendar_chain_holds_total_then_applies_final_judgment_and_replays(tmp_path):
    from decimal import localcontext
    request = fixture(tmp_path)
    definition = layer_settings(weekendWeight=0.5, holidayLosses=[{"ds": "2026-11-01", "days": 4}])
    src = dispatch(tmp_path, "judgment_source", {**request, "model": "Growth", "calendar": definition})
    assert src["calendarLayer"]["referenceActual"] == "100"
    assert src["calendarLayer"]["referenceExposure"] == "26.0"
    assert [row["effective_days"] for row in src["rows"]] == ["26.5", "21.5"]
    with localcontext() as context:
        context.prec = 40
        assert sum(Decimal(row["calendar_yhat"]) for row in src["rows"]) == 500
        for row in src["rows"]:
            product = Decimal(row["month_length_factor"]) * Decimal(row["weekday_weekend_factor"]) * Decimal(row["holiday_factor"])
            assert abs(product - Decimal(row["calendar_factor"])) < Decimal("1e-35")
    settings = {**DEFAULTS, "model": "Growth", "calendar": definition, "percent": "-10", "label": "Calendar base", "reason": "Synthetic exposure then haircut"}
    save_request = {**request, "settings": settings, "expectedSourceHash": src["sourceHash"]}
    saved = dispatch(tmp_path, "judgment_save", save_request)
    pipeline = dispatch(tmp_path, "judgment_pipeline", {"runId": saved["id"], "series": "001", "replay": True})
    assert pipeline["verified"]
    assert Decimal(pipeline["recipe"]["totals"]["adjusted_yhat"]) == 450
    assert pipeline["recipe"]["settings"]["calendar"]["referencePeriod"] == "2026-09-01"
    branch_source = dispatch(tmp_path, "judgment_source", {"runId": saved["id"], "series": "001"})
    assert branch_source["rows"] == src["rows"]
    assert branch_source["sourceHash"] == src["sourceHash"]
    branch = dispatch(tmp_path, "judgment_save", {**save_request, "runId": saved["id"], "settings": {
        **pipeline["recipe"]["settings"], "percent": "-20", "label": "Calendar downside"}})
    second = dispatch(tmp_path, "judgment_pipeline", {"runId": branch["id"], "series": "001", "replay": True})
    assert Decimal(second["recipe"]["totals"]["adjusted_yhat"]) == 400
    assert second["recipe"]["parentRunId"] == saved["id"]
    assert dispatch(tmp_path, "view", {"runId": request["runId"]})["rows"][0]["yhat"] == 100
    assert not dispatch(tmp_path, "view", {"runId": saved["id"]})["gate"]["planning_ready_claim_allowed"]
    assert dispatch(tmp_path, "view", {"runId": saved["id"], "kind": "backtest"})["total"] == 0
    assert not dispatch(tmp_path, "catalog", {})["jobs"]
    with pytest.raises(WorkbenchError, match="series"):
        dispatch(tmp_path, "judgment_pipeline", {"runId": saved["id"], "series": "another"})


def test_calendar_total_unlocked_holiday_overlap_and_override(tmp_path):
    request = fixture(tmp_path)
    definition = layer_settings(holdTotal=False, weekendWeight=0.5, holidayWeight=0,
                               holidayCalendar="custom", customDates="2026-10-03\n2026-10-03")
    src = dispatch(tmp_path, "judgment_source", {**request, "calendar": definition})
    first = src["rows"][0]
    assert first["holiday_loss"] == "0.5"
    assert first["effective_days"] == "26.0"
    assert Decimal(first["calendar_yhat"]) == 100
    definition["holidayLosses"] = [{"ds": "2026-10-01", "days": 3}]
    result = dispatch(tmp_path, "judgment_source", {**request, "calendar": definition})
    assert result["rows"][0]["effective_days"] == "23.5"  # Replaces dated loss, not 23.
    assert result["rows"][0]["holiday_loss_source"] == "monthly override"
    definition["holidayLosses"][0]["days"] = 0
    assert dispatch(tmp_path, "judgment_source", {**request, "calendar": definition})["rows"][0]["effective_days"] == "26.5"


@pytest.mark.parametrize("patch", [
    {"holidayLosses": [{"ds": "2026-11-01", "days": 31}]},
    {"holidayLosses": [{"ds": "2020-01-01", "days": 1}]},
    {"holidayLosses": [{"ds": "2026-11-01", "days": 1}] * 2},
    {"holidayLosses": [{"ds": "2026-11-01", "days": float("nan")}]},
    {"weekendWeight": float("nan")}, {"referencePeriod": "2027-01-01"},
    {"holidayCalendar": "custom", "customDates": "2026-02-30"},
])
def test_calendar_invalid_assumptions_fail_explicitly(tmp_path, patch):
    request = fixture(tmp_path)
    with pytest.raises(WorkbenchError):
        dispatch(tmp_path, "judgment_source", {**request, "calendar": layer_settings(**patch)})


def test_calendar_layers_reject_nonmonthly_normalized_negative_or_duplicate_inputs(tmp_path):
    from nixtla_scaffold.workbench_calendar_layer import apply_layer
    src = dispatch(tmp_path, "judgment_source", fixture(tmp_path))
    for change in [
        {"sourceSpec": {"freq": "D"}},
        {"sourceSpec": {"freq": "MS", "transform": {"normalization_factor_col": "days"}}},
        {"rows": [{**row, "model_yhat": "-1"} for row in src["rows"]]},
        {"history": [*src["history"], {"ds": "2026-09-30", "y": "100"}]},
        {"rows": [{**src["rows"][0], "ds": "2026-10-14"}]},
    ]:
        with pytest.raises(WorkbenchError):
            apply_layer({**src, **change}, layer_settings())


def test_calendar_source_guard_and_export_import_roundtrip(tmp_path):
    request = fixture(tmp_path)
    definition = layer_settings(holdTotal=False, weekendWeight=0.25)
    src = dispatch(tmp_path, "judgment_source", {**request, "model": "Growth", "calendar": definition})
    settings = {**DEFAULTS, "model": "Growth", "calendar": definition, "percent": "-10", "amount": "-1",
                "label": "Synthetic", "reason": "Layered assumptions"}
    with pytest.raises(WorkbenchError, match="source changed"):
        dispatch(tmp_path, "judgment_save", {**request, "settings": {**settings, "calendar": {**definition, "weekendWeight": 0.5}},
                                           "expectedSourceHash": src["sourceHash"]})
    exported = dispatch(tmp_path, "judgment_export", {**request, "settings": settings, "expectedSourceHash": src["sourceHash"]})
    raw = Path(exported["path"]).read_text()
    imported = dispatch(tmp_path, "judgment_import", {**request, "model": "Growth", "calendar": definition, "text": raw})
    assert imported["sourceHash"] == src["sourceHash"]
    assert calculate(src["rows"], {**DEFAULTS, "edits": imported["edits"]})[1] == calculate(src["rows"], settings)[1]
    recipe = json.loads(Path(exported["queryPath"]).read_text())
    assert recipe["calendarLayer"]["settings"]["referencePeriod"] == "2026-09-01"
    assert "calendar_yhat" in raw and "judgment_delta" in raw
    disabled = dispatch(tmp_path, "judgment_source", {**request, "calendar": {**definition, "enabled": False}})
    original = dispatch(tmp_path, "judgment_source", request)
    assert disabled["rows"] == original["rows"] and disabled["sourceHash"] == original["sourceHash"]


def test_legacy_judgment_pipeline_replay(tmp_path):
    request = fixture(tmp_path)
    source = dispatch(tmp_path, "judgment_source", request)
    saved = dispatch(tmp_path, "judgment_save", {**request, "settings": {**DEFAULTS, "label": "Legacy", "reason": "No calendar"},
                                               "expectedSourceHash": source["sourceHash"]})
    pipeline = dispatch(tmp_path, "judgment_pipeline", {"runId": saved["id"], "series": "001", "replay": True})
    assert pipeline["verified"] and pipeline["recipe"]["calendarLayer"] is None


def test_calendar_saved_receipt_is_portable_but_replay_requires_the_original_baseline(tmp_path):
    request = fixture(tmp_path)
    settings = {**DEFAULTS, "calendar": layer_settings(), "label": "Portable calendar", "reason": "Synthetic"}
    src = dispatch(tmp_path, "judgment_source", {**request, "calendar": settings["calendar"]})
    saved = dispatch(tmp_path, "judgment_save", {**request, "settings": settings, "expectedSourceHash": src["sourceHash"]})
    original_store = WorkbenchStore(tmp_path)
    portable_root = tmp_path / "portable-layer"
    portable_root.mkdir()
    portable = WorkbenchStore(portable_root)
    portable.initialize()
    imported = portable.import_run(original_store.run(saved["id"])[0])
    pipeline = dispatch(portable_root, "judgment_pipeline", {"runId": imported["id"], "series": "001"})
    assert pipeline["recipe"]["settings"]["calendar"]["enabled"]
    assert len(pipeline["rows"]) == 2
    assert not pipeline["verified"]
    with pytest.raises(WorkbenchError):
        dispatch(portable_root, "judgment_pipeline", {"runId": imported["id"], "series": "001", "replay": True})


def test_calendar_leap_month_month_end_and_zero_totals():
    from nixtla_scaffold.workbench_calendar_layer import apply_layer
    source = {"sourceSpec": {"freq": "ME"}, "history": [{"ds": "2024-01-31", "y": "31"}],
              "rows": [{"ds": "2024-02-29", "model_yhat": "31", "selected_yhat": "31"}]}
    rows, receipt = apply_layer(source, layer_settings(holdTotal=False))
    assert rows[0]["calendar_days"] == 29
    assert Decimal(rows[0]["calendar_yhat"]) == 29
    assert receipt["referenceActual"] == "31"
    source["rows"][0]["model_yhat"] = "0"
    rows, receipt = apply_layer(source, layer_settings())
    assert Decimal(rows[0]["calendar_yhat"]) == 0
    assert Decimal(receipt["calendarTotal"]) == 0
