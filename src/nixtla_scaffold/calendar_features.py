"""Origin-scoped calendar features shared by detection and native MLForecast."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from nixtla_scaffold.schema import CalendarSpec


def holiday_dates(calendar: CalendarSpec, start, end, cutoff) -> pd.DatetimeIndex:
    if calendar.known_as_of is not None:
        known = pd.Timestamp(calendar.known_as_of)
        origin = pd.Timestamp(cutoff)
        if known.tzinfo is not None:
            known = known.tz_convert("UTC").tz_localize(None)
        if origin.tzinfo is not None:
            origin = origin.tz_convert("UTC").tz_localize(None)
        if known > origin:
            return pd.DatetimeIndex([])
    if calendar.holiday_calendar == "us_federal":
        return USFederalHolidayCalendar().holidays(start=start, end=end)
    dates = pd.DatetimeIndex(calendar.custom_dates)
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


@dataclass(frozen=True)
class HolidayIndicator:
    dates: tuple[str, ...]
    shift: int = 0

    @property
    def __name__(self):
        return {0: "holiday", -1: "before_holiday", 1: "after_holiday"}[self.shift]

    def __call__(self, index):
        days = pd.DatetimeIndex(self.dates) + pd.Timedelta(days=self.shift)
        return np.asarray(pd.DatetimeIndex(index).normalize().isin(days), dtype=np.int8)


def native_calendar_features(history, spec, freq):
    """Calendar values are known future inputs; coefficients are learned by each fit."""
    if not spec.calendar.enabled:
        return []
    offset = pd.tseries.frequencies.to_offset(freq)
    if not isinstance(offset, (pd.offsets.Day, pd.offsets.BusinessDay)) or offset.n != 1:
        raise ValueError("Enhanced calendar features currently require daily or business-daily observations")
    start, cutoff = pd.to_datetime(history.ds).min(), pd.to_datetime(history.ds).max()
    end = cutoff + (spec.horizon + 2) * offset
    dates = holiday_dates(spec.calendar, start - pd.Timedelta(days=1), end, cutoff)
    features = ["is_month_start", "is_month_end"]
    if spec.calendar.holiday_calendar != "none":
        frozen = tuple(day.date().isoformat() for day in dates)
        features.extend(HolidayIndicator(frozen, shift) for shift in (0, -1, 1))
    return features
