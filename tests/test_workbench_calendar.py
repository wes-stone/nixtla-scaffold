import base64
from datetime import date
from decimal import Decimal
import json
from pathlib import Path

import pytest

from nixtla_scaffold.workbench import dispatch
from nixtla_scaffold.workbench_calendar import DEFAULTS, holidays, month_exposure
from nixtla_scaffold.workbench_store import WorkbenchError, WorkbenchStore


def saved_run(tmp_path, *, freq="MS", transform=None, history=None, forecast=None):
    store = WorkbenchStore(tmp_path)
    store.initialize()
    root = tmp_path / "fixture"
    (root / "appendix").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"spec": {"freq": freq, "horizon": 2, "transform": transform or {}}}), encoding="utf-8")
    (root / "appendix" / "history.csv").write_text(history or "unique_id,ds,y\n001,2024-01-01,3100\n001,2024-02-01,2900\n", encoding="utf-8")
    (root / "forecast.csv").write_text(forecast or "unique_id,ds,yhat\n001,2024-03-01,4000\n001,2024-04-01,4200\n", encoding="utf-8")
    run = store.import_run(root)["id"]
    return {"runId": run, "series": "001", "settings": dict(DEFAULTS)}


def test_leap_year_weekday_weekend_counts_and_neutral_calendar(tmp_path):
    request = saved_run(tmp_path)
    initial = dispatch(tmp_path, "catalog", {})
    result = dispatch(tmp_path, "calendar_preview", request)
    feb = result["rows"][1]
    assert (feb["days"], feb["weekdays"], feb["weekends"]) == (29, 21, 8)
    assert feb["exposure"] == "29"
    assert result["referencePeriod"] == "2024-02-01"
    march, april = result["rows"][2:]
    assert Decimal(march["calendar_scenario"]) == 3100
    assert Decimal(april["calendar_scenario"]) == 3000
    assert march["saved_forecast"] == "4000"
    assert Decimal(march["normalized_actual"] or 0) == 0
    assert all(Decimal(row["normalized_actual"]) == 2900 for row in result["rows"][:2])
    assert not result["forecastChanged"]
    after = dispatch(tmp_path, "catalog", {})
    assert initial["revision"] == after["revision"] and not after["jobs"]


def test_normal_and_century_february_lengths():
    for year, expected in [(2023, 28), (2024, 29), (2000, 29), (2100, 28)]:
        row = month_exposure(f"{year}-02-01", DEFAULTS, {})
        assert row["days"] == expected
        assert row["weekdays"] + row["weekends"] == expected


def test_us_observed_new_year_boundary_and_juneteenth():
    definition = {**DEFAULTS, "holidayCalendar": "us_federal"}
    dates = holidays(date(2021, 12, 1), date(2022, 1, 31), definition)
    assert date(2021, 12, 31) in dates
    assert date(2022, 1, 1) not in dates
    assert date(2021, 12, 24) in dates and date(2021, 12, 25) not in dates
    june = holidays(date(2021, 6, 1), date(2021, 6, 30), definition)
    assert date(2021, 6, 18) in june


def test_holiday_override_is_exclusive_on_weekends_and_duplicates_deduplicated(tmp_path):
    request = saved_run(tmp_path)
    request["settings"].update(holidayCalendar="custom", customDates="2024-03-02\n2024-03-02\n2024-03-04",
                               weekendWeight=0.5, holidayWeight=0.25)
    result = dispatch(tmp_path, "calendar_preview", request)
    march = result["rows"][2]
    assert (march["holiday_weekdays"], march["holiday_weekends"], march["holidays"]) == (1, 1, 2)
    assert march["ordinary_weekdays"] == 20 and march["ordinary_weekends"] == 9
    assert Decimal(march["exposure"]) == Decimal("25")
    assert Decimal(march["day_type_exposure"]) == Decimal("26")
    assert len(result["holidayDates"]) == 2
    for row in result["rows"]:
        product = Decimal(row["month_length_factor"]) * Decimal(row["weekday_weekend_factor"]) * Decimal(row["holiday_factor"])
        assert abs(product - Decimal(row["calendar_factor"])) < Decimal("1e-25")


def test_reference_normalization_uses_reference_holiday_and_mix(tmp_path):
    request = saved_run(tmp_path)
    request["settings"].update(referencePeriod="2024-01-01", holidayCalendar="custom", customDates="2024-01-01",
                               weekendWeight=0, holidayWeight=0)
    result = dispatch(tmp_path, "calendar_preview", request)
    assert result["referenceExposure"] == "22"
    assert Decimal(result["rows"][0]["calendar_factor"]) == 1
    assert abs(Decimal(result["rows"][2]["calendar_scenario"]) - Decimal(3100) * 21 / 22) < Decimal("1e-23")


def test_exports_roundtrip_assumptions_and_do_not_modify_inputs(tmp_path):
    request = saved_run(tmp_path)
    store = WorkbenchStore(tmp_path)
    original = store.read_artifact(request["runId"], "forecast.csv")
    receipt = dispatch(tmp_path, "calendar_export", request)
    assert receipt["rows"] == 4
    exported = Path(receipt["path"]).read_text(encoding="utf-8")
    assert "calendar_scenario" in exported and "normalized_actual" in exported
    metadata = json.loads(Path(receipt["queryPath"]).read_text())
    assert metadata["recipe"]["settings"]["referencePeriod"] == "2024-02-01"
    assert metadata["claim"] == "unvalidated_calendar_scenario"
    assert len(metadata["hashes"]) == 2
    file = dispatch(tmp_path, "explore_download", {"id": receipt["id"], "kind": "csv"})
    assert base64.b64decode(file["base64"]) == Path(receipt["path"]).read_bytes()
    assert store.read_artifact(request["runId"], "forecast.csv") == original
    assert not dispatch(tmp_path, "catalog", {})["jobs"]


@pytest.mark.parametrize("changes", [
    {"weekendWeight": -0.1}, {"holidayWeight": 4}, {"weekendWeight": float("nan")},
    {"referencePeriod": "2024-05-01"}, {"holidayCalendar": "custom", "customDates": ""},
    {"holidayCalendar": "custom", "customDates": "2024-02-30"},
    {"holidayCalendar": "custom", "customDates": "2024-01-01,Christmas"},
])
def test_invalid_assumptions_fail_explicitly(tmp_path, changes):
    request = saved_run(tmp_path)
    request["settings"].update(changes)
    with pytest.raises(WorkbenchError):
        dispatch(tmp_path, "calendar_preview", request)


@pytest.mark.parametrize("kwargs", [
    {"freq": "D"},
    {"transform": {"normalization_factor_col": "exposure"}},
    {"history": "unique_id,ds,y\n001,2024-01-01,31\n001,2024-01-31,31\n"},
    {"history": "unique_id,ds,y\n001,2024-01-15,31\n"},
    {"history": "unique_id,ds,y\n001,2024-01-01,-31\n"},
])
def test_unsupported_or_ambiguous_series_are_not_silently_adjusted(tmp_path, kwargs):
    request = saved_run(tmp_path, **kwargs)
    with pytest.raises(WorkbenchError):
        dispatch(tmp_path, "calendar_preview", request)


def test_null_actuals_month_end_ids_and_zero_reference_remain_distinct(tmp_path):
    request = saved_run(tmp_path, freq="ME", history="unique_id,ds,y\n001,2024-01-31,\n001,2024-02-29,0\n")
    result = dispatch(tmp_path, "calendar_preview", request)
    assert result["rows"][0]["observed"] is None and result["rows"][0]["normalized_actual"] is None
    assert Decimal(result["rows"][2]["calendar_scenario"]) == 0
    assert result["rows"][0]["unique_id"] == "001"

def test_latest_nonmissing_reference_and_negative_forecast_comparator(tmp_path):
    request = saved_run(tmp_path, history="unique_id,ds,y\n001,2024-01-01,3100\n001,2024-02-01,\n",
                        forecast="unique_id,ds,yhat\n001,2024-03-01,-50\n001,2024-04-01,\n")
    result = dispatch(tmp_path, "calendar_preview", request)
    assert result["referencePeriod"] == "2024-01-01"
    assert result["rows"][1]["normalized_actual"] is None
    assert result["rows"][2]["saved_forecast"] == "-50"
    assert Decimal(result["rows"][2]["scenario_minus_saved"]) == 3150
    assert result["rows"][3]["saved_forecast"] is None
    assert result["rows"][3]["scenario_minus_saved"] is None


def test_zero_exposure_and_tampering_fail(tmp_path):
    request = saved_run(tmp_path)
    request["settings"].update(holidayCalendar="custom", customDates="\n".join(f"2024-02-{day:02d}" for day in range(1, 30)),
                               holidayWeight=0)
    with pytest.raises(WorkbenchError, match="zero exposure"):
        dispatch(tmp_path, "calendar_preview", request)
    root, _ = WorkbenchStore(tmp_path).run(request["runId"])
    (root / "forecast.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="hash changed"):
        dispatch(tmp_path, "calendar_preview", request)
