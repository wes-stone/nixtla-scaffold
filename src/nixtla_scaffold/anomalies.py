"""Robust, cutoff-local seasonal diagnostics and auditable modeling-only repairs."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import time

import numpy as np
import pandas as pd

from nixtla_scaffold.calendar_features import holiday_dates
from nixtla_scaffold.transformations import _apply_target_transform_values, _inverse_target_transform_values

POINT_COLUMNS = [
    "unique_id", "cutoff", "stage", "ds", "observed_y", "expected_y", "proposed_y", "modeling_y",
    "score", "relative_deviation", "flagged", "eligible", "applied", "reason", "adjustment_id",
    "known_as_of", "approval_status", "output_scale",
]


def review_training(training, spec, profile, *, stage, observed=None):
    """Inspect one prefix once, shared across models. Never touch scoring actuals."""
    started = time.perf_counter()
    out = training.copy()
    rows, summaries = [], []
    if spec.anomaly.mode == "off":
        return out, {"version": 1, "mode": "off", "series": [], "points": []}
    offset = pd.tseries.frequencies.to_offset(profile.freq)
    daily = isinstance(offset, (pd.offsets.Day, pd.offsets.BusinessDay)) and offset.n == 1
    season = 5 if isinstance(offset, pd.offsets.BusinessDay) else 7
    for uid, full in out.groupby("unique_id", sort=True):
        group = full.sort_values("ds").tail(spec.anomaly.lookback)
        cutoff = pd.Timestamp(group.ds.max())
        summary = {"unique_id": str(uid), "cutoff": cutoff.isoformat(), "stage": stage,
                   "mode": spec.anomaly.mode, "history_rows": len(full), "review_rows": len(group),
                   "flagged": 0, "eligible": 0, "adjusted": 0, "status": "completed"}
        summary["training_sha256"] = sha256(group[["ds", "y"]].to_csv(index=False, lineterminator="\n").encode()).hexdigest()
        calendar_dates = pd.DatetimeIndex(full.ds)
        month_years = pd.DataFrame({"month": calendar_dates.month, "year": calendar_dates.year})
        month_support = month_years.groupby("month").year.nunique().reindex(range(1, 13), fill_value=0)
        summary["minimum_month_repetitions"] = int(month_support.min())
        summary["annual_effect_scope"] = ("at_least_two_repetitions_per_month_not_significance"
                                          if month_support.min() >= 2 else "limited_annual_support")
        summaries.append(summary)
        if not daily or len(group) < 8 * season:
            summary["status"] = "unsupported_calendar" if not daily else "insufficient_history"
            continue
        from statsforecast.mstl import mstl
        y = _inverse_target_transform_values(group.y.astype(float), spec.transform.target).to_numpy()
        if not np.isfinite(y).all():
            raise ValueError(f"{uid}: anomaly review requires finite origin-repaired training observations")
        use_log = bool((y >= 0).all())
        signal = np.log1p(y) if use_log else y
        decomposition = mstl(signal[:-season], period=season, stl_kwargs={"robust": True})
        expected_signal = np.empty_like(signal)
        expected_signal[:-season] = signal[:-season] - decomposition["remainder"].to_numpy()
        # STL can absorb endpoint shocks. Check the protected tail against prior matching weekdays.
        for pos in range(len(signal) - season, len(signal)):
            previous = np.arange(max(pos - 8 * season, pos % season), pos, season)
            values = signal[previous]
            slope = float(np.median(np.diff(values)))
            expected_signal[pos] = np.median(values + slope * ((pos - previous) / season))
        residual = signal - expected_signal
        center = float(np.median(residual[:-season]))
        # A 1% level floor makes near-deterministic histories usable without enormous numerical scores.
        scale_floor = .01 if use_log else max(float(np.median(np.abs(y))) * .01, 1e-8)
        scale = max(1.4826 * float(np.median(np.abs(residual[:-season] - center))), scale_floor)
        score = (residual - center) / scale
        expected = np.expm1(expected_signal) if use_log else expected_signal
        if use_log:
            expected = np.maximum(expected, 0)
        if not np.isfinite(expected).all():
            raise ValueError(f"{uid}: nonfinite native seasonal decomposition")
        denominator = np.maximum(np.abs(expected), max(float(np.median(np.abs(y))) * .01, 1e-8))
        deviation = np.abs(y - expected) / denominator
        flags = (np.abs(score) >= spec.anomaly.threshold) & (deviation >= spec.anomaly.minimum_relative_deviation)
        dates = pd.DatetimeIndex(group.ds)
        holidays = holiday_dates(spec.calendar, dates[0] - pd.Timedelta(days=1),
                                 cutoff + pd.Timedelta(days=1), cutoff)
        protected = dates.normalize().isin(holidays.union(holidays - pd.Timedelta(days=1)).union(holidays + pd.Timedelta(days=1)))
        observed_keys = None if observed is None else set(pd.to_datetime(
            observed.loc[observed.unique_id.astype(str).eq(str(uid)) & observed.y.notna(), "ds"],
        ))
        reasons = {}
        positions = np.flatnonzero(flags)
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        episodes = {int(pos): len(block) for block in groups for pos in block}
        for pos in positions:
            same_phase = int(np.sum(flags & (dates.dayofweek == dates[pos].dayofweek)))
            month_phase = dates.is_month_end if dates[pos].is_month_end else dates.day == dates[pos].day
            repeated_month_phase = int(np.sum(flags & month_phase)) >= 3
            before = np.arange(pos - 3 * season, pos, season)
            after = np.arange(pos + season, pos + 4 * season, season)
            before, after = before[before >= 0], after[after < len(y)]
            context_complete = len(before) >= 2 and len(after) >= 2
            context_shift = False
            if context_complete:
                left, right = float(np.median(y[before])), float(np.median(y[after]))
                context_shift = abs(right - left) / max(abs(left), abs(right), denominator[pos] * .01) >= spec.anomaly.minimum_relative_deviation
            if observed_keys is not None and dates[pos] not in observed_keys:
                reason = "structural_repair_not_anomaly"
            elif protected[pos]:
                reason = "declared_holiday_or_adjacent_day"
            elif pos >= len(group) - season:
                reason = "unresolved_tail"
            elif episodes[int(pos)] >= 3:
                reason = "sustained_episode_or_level_shift"
            elif same_phase >= 3:
                reason = "recurrent_weekday_pattern"
            elif repeated_month_phase:
                reason = "recurrent_monthly_pattern"
            elif not context_complete:
                reason = "insufficient_two_sided_context"
            elif context_shift:
                reason = "calendar_phase_or_level_shift"
            else:
                reason = "isolated_transient_candidate"
            reasons[int(pos)] = reason
        eligible = [pos for pos, reason in reasons.items() if reason == "isolated_transient_candidate"]
        if len(eligible) / len(group) > spec.anomaly.maximum_adjusted_fraction:
            reasons.update({pos: "adjustment_fraction_guard" for pos in eligible})
            eligible = []
        for pos, (index, point) in enumerate(group.iterrows()):
            can_adjust = pos in eligible
            applied = spec.anomaly.mode == "adjust" and can_adjust
            modeling_y = float(expected[pos] if applied else y[pos])
            if applied:
                out.loc[index, "y"] = float(_apply_target_transform_values(
                    pd.Series([modeling_y]), spec.transform.target).iloc[0])
            key = f"{uid}|{cutoff.isoformat()}|{point.ds}|{summary['training_sha256']}|{spec.anomaly.to_dict()}|{spec.calendar.to_dict()}"
            rows.append({
                "unique_id": str(uid), "cutoff": cutoff.isoformat(), "stage": stage,
                "ds": pd.Timestamp(point.ds).isoformat(), "observed_y": float(y[pos]),
                "expected_y": float(expected[pos]), "proposed_y": float(expected[pos]) if can_adjust else float(y[pos]),
                "modeling_y": modeling_y, "score": float(score[pos]), "relative_deviation": float(deviation[pos]),
                "flagged": bool(flags[pos]), "eligible": can_adjust, "applied": applied,
                "reason": reasons.get(pos, "within_expected_variation"),
                "adjustment_id": sha256(key.encode()).hexdigest()[:24] if flags[pos] else "",
                "known_as_of": cutoff.isoformat(),
                "approval_status": "modeling_only_not_actual_revision" if applied else "diagnostic_only",
                "output_scale": "normalized_units" if spec.transform.normalization_factor_col else "raw_units",
            })
        summary.update(flagged=len(positions), eligible=len(eligible),
                       adjusted=len(eligible) if spec.anomaly.mode == "adjust" else 0,
                       detector_transform="log1p" if use_log else "identity", residual_scale=scale,
                       seasonal_period=season)
    return out, {
        "version": 1, "mode": spec.anomaly.mode, "policy": spec.anomaly.to_dict(),
        "calendar": spec.calendar.to_dict(), "series": summaries, "points": rows,
        "seconds": time.perf_counter() - started,
        "method": "statsforecast.mstl robust STL; prior-weekday tail check; locally scaled seasonal residuals",
        "scope": "Training-only hypotheses, not verified data errors or certified anomaly probabilities.",
        "calendar_scope": "Explicit rule set only; no inferred geography or historical legal-vintage certification.",
        "actuals_changed": False,
        "data_vintage": "input_snapshot_historical_measurement_vintage_not_certified",
    }


def anomaly_adjustment_spec(spec):
    return replace(spec, anomaly=replace(spec.anomaly, mode="adjust"))


def enhanced_calendar_spec(spec):
    return replace(spec, calendar=replace(spec.calendar, enabled=True),
                   feature_recipe=replace(spec.feature_recipe, date_features_as_dummies=True))


def evidence_frames(report):
    prefixes = report.get("prefixes", [])
    points = [row for prefix in prefixes for row in prefix.get("points", [])]
    return {
        "points": pd.DataFrame(points, columns=POINT_COLUMNS),
        "series": pd.DataFrame([row for prefix in prefixes for row in prefix.get("series", [])],
                              columns=["unique_id", "cutoff", "stage", "mode", "history_rows", "review_rows",
                                       "flagged", "eligible", "adjusted", "status", "detector_transform",
                                       "residual_scale", "seasonal_period", "minimum_month_repetitions", "annual_effect_scope",
                                       "training_sha256"]),
    }
