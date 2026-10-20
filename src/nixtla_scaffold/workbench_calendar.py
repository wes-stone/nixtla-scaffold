"""Auditable calendar exposure scenarios; never mutate or refit a saved forecast."""
from __future__ import annotations

from calendar import monthrange
import csv
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, localcontext
import io
import json
import re
import uuid

from jsonschema import Draft202012Validator
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from nixtla_scaffold.workbench_evidence import content
from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest, timestamp

SETTINGS = {
    "type": "object", "additionalProperties": False,
    "required": ["referencePeriod", "weekendWeight", "holidayWeight", "holidayCalendar", "customDates"],
    "properties": {
        "referencePeriod": {"type": "string", "pattern": r"^(\d{4}-\d{2}-01)?$"},
        "weekendWeight": {"type": "number", "minimum": 0, "maximum": 3},
        "holidayWeight": {"type": "number", "minimum": 0, "maximum": 3},
        "holidayCalendar": {"enum": ["none", "us_federal", "custom"]},
        "customDates": {"type": "string", "maxLength": 12000},
    },
}
DEFAULTS = {"referencePeriod": "", "weekendWeight": 1, "holidayWeight": 1, "holidayCalendar": "none", "customDates": ""}
METHOD = (
    "Weekday weight = 1. Holidays replace (do not multiply) the ordinary day weight. "
    "Exposure = ordinary weekdays + ordinary weekends * weekend weight + holidays * holiday weight. "
    "Factor = exposure / reference exposure. Normalized actual = actual / factor. "
    "Factor bridge: let D = calendar days, B = weekdays + weekends * weekend weight "
    "(before replacing holiday weights), and E = final weighted exposure. "
    "Month length = D / D_ref; day mix = (B / D) / (B_ref / D_ref); "
    "holiday effect = (E / B) / (E_ref / B_ref). Their product is E / E_ref. "
    "A holiday factor above 1 can reflect reduced exposure in the reference month, "
    "even when the comparison month has no holidays. "
    "Calendar-only scenario = reference actual * factor, at a constant underlying daily rate. "
    "No trend, learned seasonality, fitted holiday effects or prediction intervals. "
    "The saved forecast is only a comparator and is never multiplied by these factors."
)


def settings(value):
    error = next(Draft202012Validator(SETTINGS).iter_errors(value), None)
    if error:
        raise WorkbenchError("invalid_calendar", error.message)
    for key in ("weekendWeight", "holidayWeight"):
        if not Decimal(str(value[key])).is_finite():
            raise WorkbenchError("invalid_calendar", "Calendar weights must be finite.")
    return value


def holidays(start: date, end: date, definition: dict) -> dict[date, str]:
    if definition["holidayCalendar"] == "none":
        return {}
    if definition["holidayCalendar"] == "us_federal":
        values = USFederalHolidayCalendar().holidays(start=start, end=end, return_name=True)
        return {day.date(): str(name) for day, name in values.items()}
    values = {}
    for line in definition["customDates"].splitlines():
        if not line.strip():
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", line.strip()):
            raise WorkbenchError("invalid_calendar", "Custom holidays need one YYYY-MM-DD date per line, without labels.")
        try:
            day = date.fromisoformat(line.strip())
        except ValueError as error:
            raise WorkbenchError("invalid_calendar", f"Invalid holiday date: {line.strip()}") from error
        if not 1900 <= day.year <= 2100:
            raise WorkbenchError("invalid_calendar", "Holiday dates must be between 1900 and 2100.")
        values[day] = "Custom holiday"
    if not values:
        raise WorkbenchError("invalid_calendar", "Enter at least one holiday date or choose No holiday calendar.")
    return {day: name for day, name in values.items() if start <= day <= end}


def month_exposure(period: str, definition: dict, dates: dict[date, str]) -> dict:
    start = date.fromisoformat(period)
    days = [start + timedelta(days=index) for index in range(monthrange(start.year, start.month)[1])]
    weekdays = sum(day.weekday() < 5 for day in days)
    holiday_weekdays = sum(day.weekday() < 5 and day in dates for day in days)
    holiday_weekends = sum(day.weekday() >= 5 and day in dates for day in days)
    weekend_weight = Decimal(str(definition["weekendWeight"]))
    holiday_weight = Decimal(str(definition["holidayWeight"]))
    nonholiday_weekdays = weekdays - holiday_weekdays
    nonholiday_weekends = len(days) - weekdays - holiday_weekends
    exposure = Decimal(nonholiday_weekdays) + nonholiday_weekends * weekend_weight + (holiday_weekdays + holiday_weekends) * holiday_weight
    day_type_exposure = Decimal(weekdays) + (len(days) - weekdays) * weekend_weight
    return {
        "days": len(days), "weekdays": weekdays, "weekends": len(days) - weekdays,
        "holiday_weekdays": holiday_weekdays, "holiday_weekends": holiday_weekends,
        "ordinary_weekdays": nonholiday_weekdays, "ordinary_weekends": nonholiday_weekends,
        "holidays": holiday_weekdays + holiday_weekends,
        "holiday_dates": "; ".join(f"{day.isoformat()} ({dates[day]})" for day in days if day in dates),
        "exposure": exposure, "day_type_exposure": day_type_exposure,
    }


def _table(store, run_id, name, series):
    actual_name, raw = content(store, {"runId": run_id, "name": name})
    if raw is None:
        raise WorkbenchError("missing_evidence", f"This run has no {name}. Calendar review requires saved history and forecast tables.")
    headers = pd.read_csv(io.BytesIO(raw), nrows=0).columns
    if not {"unique_id", "ds"} <= set(headers):
        raise WorkbenchError("invalid_calendar", f"{name} needs unique_id and ds columns.")
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    frame = frame.loc[frame.unique_id.eq(series)].copy()
    if frame.empty:
        raise WorkbenchError("invalid_calendar", f"No {series} rows in {name}. Select a saved series.")
    if len(frame) > 2000:
        raise WorkbenchError("too_large", "Calendar review supports up to 2,000 rows per series, without sampling.")
    periods = []
    for text in frame.ds:
        try:
            day = date.fromisoformat(text[:10])
        except ValueError as error:
            raise WorkbenchError("invalid_calendar", f"Invalid monthly timestamp in {name}: {text}") from error
        if not 1900 <= day.year <= 2100 or day.day not in {1, monthrange(day.year, day.month)[1]}:
            raise WorkbenchError("invalid_calendar", "Calendar review needs complete calendar months dated at month start or month end (1900–2100).")
        periods.append(day.replace(day=1).isoformat())
    frame["period"] = periods
    if frame.period.duplicated().any():
        raise WorkbenchError("invalid_calendar", f"Duplicate series/month rows in {name}; values are not silently aggregated.")
    return frame.sort_values("period"), {"name": actual_name, "sha256": digest(raw)}


def _value(text, *, nonnegative=True):
    if text == "":
        return None
    try:
        result = Decimal(text)
    except InvalidOperation as error:
        raise WorkbenchError("invalid_calendar", "Monthly flow values must be numeric.") from error
    if not result.is_finite() or (nonnegative and result < 0) or abs(result.adjusted()) > 100:
        raise WorkbenchError("invalid_calendar", "Calendar-rate review requires finite nonnegative monthly flow totals.")
    return result


def preview(store, data):
    definition = settings(data["settings"])
    manifest = json.loads(store.read_artifact(data["runId"], "manifest.json"))
    spec = manifest.get("spec", {})
    if spec.get("freq") not in {"MS", "M", "ME"}:
        raise WorkbenchError("invalid_calendar", "This adjuster is for complete monthly flow totals, not daily, weekly, fiscal, stock or rate series.")
    if (spec.get("transform") or {}).get("normalization_factor_col"):
        raise WorkbenchError("invalid_calendar", "This run already uses normalization factors. Use its original unnormalized run to avoid mixing calendar bases.")
    history, history_hash = _table(store, data["runId"], "appendix/history.csv", data["series"])
    forecast, forecast_hash = _table(store, data["runId"], "forecast.csv", data["series"])
    if "y" not in history or "yhat" not in forecast:
        raise WorkbenchError("invalid_calendar", "Saved tables require history y and forecast yhat.")
    actuals = dict(zip(history.period, map(_value, history.y)))
    predictions = dict(zip(forecast.period, (_value(value, nonnegative=False) for value in forecast.yhat)))
    if min(predictions) <= max(actuals):
        raise WorkbenchError("invalid_calendar", "Forecast months must follow saved history; overlapping months cannot be treated as future.")
    reference = definition["referencePeriod"] or next((period for period in reversed(actuals) if actuals[period] is not None), "")
    if reference not in actuals or actuals[reference] is None:
        raise WorkbenchError("invalid_calendar", "Choose an observed reference month with a nonmissing total.")
    periods = sorted(set(actuals) | set(predictions))
    start, end_month = date.fromisoformat(periods[0]), date.fromisoformat(periods[-1])
    end = end_month.replace(day=monthrange(end_month.year, end_month.month)[1])
    holiday_dates = holidays(start, end, definition)
    calendar = {period: month_exposure(period, definition, holiday_dates) for period in periods}
    ref = calendar[reference]
    if any(row["exposure"] <= 0 for row in calendar.values()):
        raise WorkbenchError("invalid_calendar", "Weights yield zero exposure for a month. Use positive weights for at least one day present in each month.")
    rows = []
    with localcontext() as context:
        context.prec = 34
        for period, cal in calendar.items():
            length_factor = Decimal(cal["days"]) / ref["days"]
            mix_factor = (cal["day_type_exposure"] / cal["days"]) / (ref["day_type_exposure"] / ref["days"])
            holiday_factor = (cal["exposure"] / cal["day_type_exposure"]) / (ref["exposure"] / ref["day_type_exposure"])
            factor = cal["exposure"] / ref["exposure"]
            observed = actuals.get(period)
            scenario = actuals[reference] * cal["exposure"] / ref["exposure"] if period in predictions else None
            saved = predictions.get(period)
            rows.append({
                "unique_id": data["series"], "ds": period, "scope": "forecast" if period in predictions else "history", **cal,
                "month_length_factor": length_factor, "weekday_weekend_factor": mix_factor, "holiday_factor": holiday_factor,
                "calendar_factor": factor, "observed": observed, "normalized_actual": observed * ref["exposure"] / cal["exposure"] if observed is not None else None,
                "saved_forecast": saved, "calendar_scenario": scenario,
                "scenario_minus_saved": scenario - saved if scenario is not None and saved is not None else None,
            })
    rows = [{key: str(value) if isinstance(value, Decimal) else value for key, value in row.items()} for row in rows]
    recipe = {"runId": data["runId"], "series": data["series"], "settings": {**definition, "referencePeriod": reference}}
    return {"rows": rows, "recipe": recipe, "referencePeriod": reference, "referenceActual": str(actuals[reference]),
            "referenceExposure": str(ref["exposure"]), "historyPeriods": list(actuals), "method": METHOD,
            "hashes": [history_hash, forecast_hash], "forecastChanged": False, "claim": "unvalidated_calendar_scenario",
            "holidayDates": [{"ds": day.isoformat(), "name": name} for day, name in sorted(holiday_dates.items())],
            "warnings": [
                "Weights are assumptions, not effects estimated from monthly totals. Daily usage is needed to estimate weekday/weekend/holiday rates reliably.",
                "Calendar-only projection holds the reference month's underlying rate constant; it is not a replacement for the model's growth or seasonal forecast.",
                "Use only complete calendar-month flow totals. This is not appropriate for balances, ARR, averages or partial months.",
                *([f"US federal observed dates use pandas {pd.__version__}; not a global, regional or company-closure calendar. No additional actual-date holiday is added when observance shifts."] if definition["holidayCalendar"] == "us_federal" else []),
                *(["No holiday calendar selected: holiday impact is not modeled."] if definition["holidayCalendar"] == "none" else []),
            ]}


def export(store, data):
    result = preview(store, data)
    identifier = str(uuid.uuid4())
    path = store.owned(f"exports/{identifier}.csv")
    recipe_path = store.owned(f"exports/{identifier}.query.json")
    path.parent.mkdir(exist_ok=True)
    from nixtla_scaffold.workbench_explore import NUMBER
    escaped = 0

    def safe(value):
        nonlocal escaped
        if value is None:
            return ""
        text = str(value)
        if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) and not NUMBER.fullmatch(text):
            escaped += 1
            return "'" + text
        return text

    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        fields = list(result["rows"][0])
        writer.writerow(fields)
        writer.writerows([safe(row[key]) for key in fields] for row in result["rows"])
    receipt = {key: value for key, value in result.items() if key != "rows"}
    receipt.update({"version": 1, "createdAt": timestamp(), "csvSha256": digest(path.read_bytes()), "rows": len(result["rows"])})
    with recipe_path.open("x", encoding="utf-8") as output:
        output.write(canonical_json(receipt))
    return {"id": identifier, "path": str(path), "queryPath": str(recipe_path), "rows": len(result["rows"]),
            "sha256": receipt["csvSha256"], "formulaEscapedCells": escaped}
