"""Explicit activity-day overlays on saved model paths, never statistical refits."""
from calendar import monthrange
from datetime import date
from decimal import Decimal, localcontext

from jsonschema import Draft202012Validator

from nixtla_scaffold import workbench_calendar as calendar
from nixtla_scaffold.workbench_store import WorkbenchError

SETTINGS = {
    **calendar.SETTINGS,
    "required": [*calendar.SETTINGS["required"], "enabled", "holdTotal", "holidayLosses"],
    "properties": {
        **calendar.SETTINGS["properties"],
        "enabled": {"type": "boolean"},
        "holdTotal": {"type": "boolean"},
        "holidayLosses": {"type": "array", "maxItems": 501, "items": {
            "type": "object", "additionalProperties": False, "required": ["ds", "days"],
            "properties": {"ds": {"type": "string", "pattern": r"^\d{4}-\d{2}-01$"},
                           "days": {"type": "number", "minimum": 0, "maximum": 31}},
        }},
    },
}
DEFAULTS = {**calendar.DEFAULTS, "enabled": False, "holdTotal": True, "holidayLosses": []}
METHOD = (
    "Calendar layer follows the chosen model and precedes final judgment. "
    "B = weekdays + weekend weight * weekends. E = B - holiday loss. "
    "A monthly holiday-loss override replaces the dated-calendar effect, never adds to it; "
    "otherwise dated holidays replace the ordinary day weight, including weekend holidays once. "
    "Unscaled calendar value = chosen model * E / reference E. "
    "Hold total scales these values to the chosen model horizon sum, with the final rounding residual reconciled. "
    "This reshapes the existing model; it is not FINN's independent constant-growth activity-day forecast. "
    "ARIMA/MSTL may already contain calendar seasonality: an added overlay can double-count it. "
    "Weights and holiday losses are explicit judgment, not fitted coefficients. No extra January rebound."
)


def month(value):
    try:
        day = date.fromisoformat(value[:10])
    except (ValueError, TypeError) as error:
        raise WorkbenchError("invalid_calendar", "Calendar layer needs valid monthly dates.") from error
    if not 1900 <= day.year <= 2100 or day.day not in (1, monthrange(day.year, day.month)[1]):
        raise WorkbenchError("invalid_calendar", "Calendar layer needs complete calendar months dated at month start or end.")
    return day.replace(day=1).isoformat()


def apply_layer(source, definition):
    error = next(Draft202012Validator(SETTINGS).iter_errors(definition), None)
    if error:
        raise WorkbenchError("invalid_calendar", error.message)
    if not definition["enabled"]:
        return source["rows"], None
    spec = source["sourceSpec"]
    if spec.get("freq") not in {"MS", "M", "ME"}:
        raise WorkbenchError("invalid_calendar", "Calendar layers require complete monthly consumption totals, not stocks, rates or other frequencies.")
    if (spec.get("transform") or {}).get("normalization_factor_col"):
        raise WorkbenchError("invalid_calendar", "This model already uses normalization factors. Do not apply a second calendar layer.")
    calendar.settings({key: definition[key] for key in calendar.DEFAULTS})
    historical = [row for row in source["history"] if row.get("y", "") != ""]
    history = {month(row["ds"]): row for row in historical}
    if len(history) != len(historical):
        raise WorkbenchError("invalid_calendar", "Calendar layers cannot aggregate duplicate history months.")
    reference = definition["referencePeriod"] or max(history, default="")
    if reference not in history:
        raise WorkbenchError("invalid_calendar", "Choose an observed reference month from the original model history.")
    calendar._value(history[reference]["y"])
    forecast_months = [month(row["ds"]) for row in source["rows"]]
    if len(set(forecast_months)) != len(forecast_months):
        raise WorkbenchError("invalid_calendar", "Calendar layers cannot aggregate duplicate forecast months.")
    if min(forecast_months) <= max(history):
        raise WorkbenchError("invalid_calendar", "Forecast months must follow observed history months.")
    periods = sorted({reference, *forecast_months})
    losses = {}
    for item in definition["holidayLosses"]:
        if item["ds"] not in periods or item["ds"] in losses:
            raise WorkbenchError("invalid_calendar", "Holiday losses must be unique months in the reference or forecast horizon.")
        value = Decimal(str(item["days"]))
        if not value.is_finite():
            raise WorkbenchError("invalid_calendar", "Holiday losses must be finite.")
        losses[item["ds"]] = value
    first, last = date.fromisoformat(periods[0]), date.fromisoformat(periods[-1])
    dates = calendar.holidays(first, last.replace(day=monthrange(last.year, last.month)[1]), definition)
    with localcontext() as context:
        context.prec = 40
        counts = {}
        for period in periods:
            cal = calendar.month_exposure(period, definition, dates)
            loss = losses.get(period, cal["day_type_exposure"] - cal["exposure"])
            effective = cal["day_type_exposure"] - loss
            if effective <= 0:
                raise WorkbenchError("invalid_calendar", f"Holiday loss leaves no positive activity days in {period}; reduce it.")
            counts[period] = {**cal, "holiday_loss": loss, "effective_days": effective,
                              "loss_source": "monthly override" if period in losses else "dated calendar" if definition["holidayCalendar"] != "none" else "not modeled"}
        ref = counts[reference]
        values = [calendar._value(row["model_yhat"]) for row in source["rows"]]
        if any(value is None for value in values):
            raise WorkbenchError("invalid_calendar", "Every chosen model period needs a finite nonnegative consumption total.")
        original_total = sum(values, Decimal(0))
        unscaled = [value * counts[period]["effective_days"] / ref["effective_days"] for value, period in zip(values, forecast_months)]
        unscaled_total = sum(unscaled, Decimal(0))
        scale = original_total / unscaled_total if definition["holdTotal"] and unscaled_total else Decimal(1)
        adjusted = [value * scale for value in unscaled]
        residual = Decimal(0)
        if definition["holdTotal"]:
            residual = original_total - sum(adjusted, Decimal(0))
            adjusted[-1] = original_total - sum(adjusted[:-1], Decimal(0))
        rows = []
        for row, period, raw, final in zip(source["rows"], forecast_months, unscaled, adjusted):
            cal = counts[period]
            calendar._value(str(final))
            rows.append({**row, "calendar_yhat": str(final), "calendar_delta": str(final - Decimal(row["model_yhat"])),
                         "calendar_factor": str(cal["effective_days"] / ref["effective_days"]),
                         "calendar_unscaled_yhat": str(raw), "calendar_scaling_factor": str(scale),
                         "calendar_days": cal["days"], "weekdays": cal["weekdays"], "weekends": cal["weekends"],
                         "month_length_factor": str(Decimal(cal["days"]) / ref["days"]),
                         "weekday_weekend_factor": str((cal["day_type_exposure"] / cal["days"]) / (ref["day_type_exposure"] / ref["days"])),
                         "holiday_factor": str((cal["effective_days"] / cal["day_type_exposure"]) / (ref["effective_days"] / ref["day_type_exposure"])),
                         "holiday_loss": str(cal["holiday_loss"]), "effective_days": str(cal["effective_days"]),
                         "holiday_dates": cal["holiday_dates"], "holiday_loss_source": cal["loss_source"]})
        receipt = {
            "version": 1, "method": METHOD, "settings": {**definition, "referencePeriod": reference},
            "referencePeriod": reference, "referenceExposure": str(ref["effective_days"]),
            "referenceHolidayLoss": str(ref["holiday_loss"]), "referenceActual": history[reference]["y"],
            "referenceCounts": {key: str(value) if isinstance(value, Decimal) else value for key, value in ref.items()},
            "modelTotal": str(original_total), "unscaledTotal": str(unscaled_total),
            "calendarTotal": str(sum(adjusted, Decimal(0))), "scalingFactor": str(scale), "roundingResidual": str(residual),
            "holidayDates": [{"ds": day.isoformat(), "name": name} for day, name in sorted(dates.items())],
            "warnings": ["Model seasonality may already include these effects; this is an optional, unvalidated overlay.",
                         "Monthly totals do not establish measured weekend or holiday coefficients.",
                         "Use only complete calendar-month consumption flows; no stock/rate conversion is inferred.",
                         "US federal uses observed dates only, not a global or company closure calendar."],
        }
    return rows, receipt
