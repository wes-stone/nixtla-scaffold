"""Experimental, timestamp-owned calibration using the genuine array API."""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json

import numpy as np
import pandas as pd

from nixtla_scaffold.schema import IntervalCalibrationSpec


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_backend():
    try:
        installed = version("conformalforecast")
    except ImportError as exc:
        raise ImportError("Experimental intervals require the pinned conformal extra in the selected runtime.") from exc
    if installed != "0.1.0":
        raise RuntimeError(f"ConformalForecast {installed} is unverified; this adapter requires 0.1.0.")
    return installed


class ResidualCalibrator:
    """Persist raw disjoint windows, not pickles or private upstream attributes."""

    def __init__(self, config: IntervalCalibrationSpec, *, horizon: int, freq: str, path_id: str):
        if config.method == "native" or type(horizon) is not int or horizon < 1 or not path_id:
            raise ValueError("An experimental method, positive horizon and exact point-path identity are required")
        self.version = require_backend()
        self.config, self.horizon, self.freq, self.path_id = config, horizon, freq, path_id
        self.windows: list[dict] = []

    def observe(self, *, cutoff, dates, predictions, truths, available_at, observed_at):
        cutoff, available_at, observed_at = map(pd.Timestamp, (cutoff, available_at, observed_at))
        dates = pd.DatetimeIndex(dates)
        expected = pd.date_range(cutoff, periods=self.horizon + 1, freq=self.freq)[1:]
        predictions, truths = np.asarray(predictions, dtype=float), np.asarray(truths, dtype=float)
        if dates.hasnans or not dates.equals(expected):
            raise ValueError("Calibration requires one complete, ordered, origin-aligned horizon")
        if predictions.shape != (self.horizon,) or truths.shape != (self.horizon,):
            raise ValueError("Calibration predictions and truths must match the complete horizon")
        if not np.isfinite(predictions).all() or not np.isfinite(truths).all():
            raise ValueError("Calibration cannot use nonfinite predictions or observations")
        if any(pd.isna(t) for t in (cutoff, available_at, observed_at)):
            raise ValueError("Calibration timestamps cannot be missing")
        if available_at < dates[-1] or available_at > observed_at:
            raise ValueError("A full calibration window must have landed before it is observed")
        if self.windows:
            previous = self.windows[-1]
            if dates[0] <= pd.Timestamp(previous["dates"][-1]) or cutoff <= pd.Timestamp(previous["cutoff"]):
                raise ValueError("Duplicate, overlapping or out-of-order calibration windows are forbidden")
            if observed_at < pd.Timestamp(previous["observed_at"]):
                raise ValueError("Calibration updates must have monotone observation timestamps")
        self.windows.append({
            "cutoff": cutoff.isoformat(), "dates": [t.isoformat() for t in dates],
            "predictions": predictions.tolist(), "truths": truths.tolist(),
            "available_at": available_at.isoformat(), "observed_at": observed_at.isoformat(),
        })
        self.windows = self.windows[-self.config.max_windows:]

    def interval(self, predictions, *, cutoff, level):
        from conformalforecast.base import InsufficientCalibrationSamples
        from conformalforecast.methods import SplitConformal, NonexchangeableConformalPrediction

        points = np.asarray(predictions, dtype=float)
        cutoff = pd.Timestamp(cutoff)
        if points.shape != (self.horizon,) or not np.isfinite(points).all():
            raise ValueError("Intervals require finite points for the complete configured horizon")
        if type(level) is not int or not 0 < level < 100:
            raise ValueError("Interval level must be an integer between 0 and 100")
        if pd.isna(cutoff) or any(pd.Timestamp(w["observed_at"]) > cutoff for w in self.windows):
            raise ValueError("Calibration state contains observations not available at this origin")
        receipt = {
            "method": f"conformalforecast_{self.config.method}", "engine_version": self.version,
            "config": asdict(self.config), "experimental": True, "level": level,
            "levels": [], "status": "insufficient_calibration", "path_id": self.path_id,
            "state_sha256": _hash(self.to_dict()),
            "calibration_age_days": (cutoff - pd.Timestamp(self.windows[-1]["dates"][-1])).total_seconds() / 86400 if self.windows else None,
            "calibration": {
                "windows": len(self.windows), "finite": True, "score_scale": "reporting_scale",
                "observed_residual_shape": [len(self.windows), self.horizon],
                "periods": [{"cutoff": w["cutoff"], "start": w["dates"][0], "end": w["dates"][-1],
                             "available_at": w["available_at"]} for w in self.windows],
            },
        }
        if not self.windows:
            receipt["reason"] = "No complete disjoint window has landed at this origin."
            return None, receipt
        preds = np.array([w["predictions"] for w in self.windows])[None, :, :]
        truths = np.array([w["truths"] for w in self.windows])[None, :, :]
        receipt["calibration"]["score_sha256"] = sha256(np.abs(preds - truths).astype("<f8").tobytes()).hexdigest()
        method = (SplitConformal(alpha=(100 - level) / 100) if self.config.method == "split_conformal" else
                  NonexchangeableConformalPrediction(alpha=(100 - level) / 100, rho=self.config.rho))
        try:
            fitted = method.fit(preds, truths)
        except InsufficientCalibrationSamples as exc:
            receipt["reason"] = str(exc)
            return None, receipt
        bounds = np.asarray(fitted.interval(points[None, None, :]))[0, 0]
        if (bounds.shape != (self.horizon, 2) or not np.isfinite(bounds).all()
                or not np.isfinite(bounds[:, 1] - bounds[:, 0]).all() or (bounds[:, 0] > bounds[:, 1]).any()):
            raise ValueError("ConformalForecast returned invalid interval bounds; no native-band substitution")
        receipt.update(status="experimental", levels=[level])
        return bounds, receipt

    def to_dict(self):
        return {
            "schema_version": 1, "engine_version": self.version, "config": asdict(self.config),
            "horizon": self.horizon, "freq": self.freq, "path_id": self.path_id, "windows": deepcopy(self.windows),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unknown calibration state version")
        state = cls(IntervalCalibrationSpec(**payload["config"]), horizon=payload["horizon"],
                    freq=payload["freq"], path_id=payload["path_id"])
        if payload["engine_version"] != state.version or len(payload["windows"]) > state.config.max_windows:
            raise ValueError("Calibration runtime or retained-window bound changed")
        for window in payload["windows"]:
            state.observe(**window)
        return state


def calibrate_path(backtest, future, config, *, model, horizon, freq, origin, levels, path_id):
    """One fixed series/path. Overlapping origins are scored but never double-counted."""
    cv, fc = backtest.copy(), future.copy()
    for frame in (cv, fc):
        frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
        if frame["ds"].isna().any():
            raise ValueError("Forecast target dates cannot be missing")
    cv["cutoff"] = pd.to_datetime(cv["cutoff"], errors="raise")
    if cv[["cutoff", "ds"]].isna().any().any() or cv.duplicated(["cutoff", "ds"]).any() or not cv.ds.gt(cv.cutoff).all():
        raise ValueError("Invalid or duplicate origin/target support")
    if not cv.empty and pd.Timestamp(origin) < cv.cutoff.max():
        raise ValueError("Future origin cannot precede backtest origins")
    if fc.ds.duplicated().any() or not pd.DatetimeIndex(fc.ds).equals(pd.date_range(origin, periods=horizon + 1, freq=freq)[1:]):
        raise ValueError("Future path must match the origin and full configured horizon")
    if "available_at" in cv:
        cv["available_at"] = pd.to_datetime(cv["available_at"], errors="raise")
        if cv.available_at.isna().any() or cv.available_at.lt(cv.ds).any():
            raise ValueError("Actual availability must be complete and cannot precede its target date")
    else:
        cv["available_at"] = cv.ds
    state = ResidualCalibrator(config, horizon=horizon, freq=freq, path_id=path_id)
    windows, excluded = [], []
    last_target = None
    groups = list(cv.groupby("cutoff", sort=True))
    for cutoff, group in groups:
        group = group.sort_values("ds")
        expected = pd.date_range(cutoff, periods=horizon + 1, freq=freq)[1:]
        if not pd.DatetimeIndex(group.ds).equals(expected) or not np.isfinite(group[[model, "y"]]).all().all():
            excluded.append({"cutoff": cutoff.isoformat(), "reason": "incomplete_or_nonfinite_horizon"})
        elif last_target is not None and group.ds.iloc[0] <= last_target:
            excluded.append({"cutoff": cutoff.isoformat(), "reason": "overlapping_calibration_targets"})
        else:
            windows.append(group)
            last_target = group.ds.iloc[-1]
    cursor, receipts = 0, []
    for frame in (cv, fc):
        for level in levels:
            frame[f"{model}-lo-{level}"] = np.nan
            frame[f"{model}-hi-{level}"] = np.nan
    for cutoff, group, stage in [*( (cutoff, group.sort_values("ds"), "backtest") for cutoff, group in groups),
                                 (pd.Timestamp(origin), fc, "future")]:
        while cursor < len(windows) and windows[cursor].available_at.max() <= cutoff:
            window = windows[cursor]
            state.observe(cutoff=window.cutoff.iloc[0], dates=window.ds,
                          predictions=window[model], truths=window.y,
                          available_at=window.available_at.max(), observed_at=cutoff)
            cursor += 1
        expected = pd.date_range(cutoff, periods=horizon + 1, freq=freq)[1:]
        if not pd.DatetimeIndex(group.ds).equals(expected) or not np.isfinite(group[model]).all():
            receipts.append({"stage": stage, "cutoff": cutoff.isoformat(), "status": "incomplete_point_path"})
            continue
        target = cv if stage == "backtest" else fc
        for level in levels:
            bounds, receipt = state.interval(group[model], cutoff=cutoff, level=level)
            receipts.append({**receipt, "stage": stage, "cutoff": cutoff.isoformat()})
            if bounds is not None:
                target.loc[group.index, [f"{model}-lo-{level}", f"{model}-hi-{level}"]] = bounds
    return cv, fc, {
        "receipts": receipts, "state": state.to_dict(), "excluded_windows": excluded,
        "availability": "declared_available_at" if "available_at" in backtest else "target_timestamp_assumed_not_certified",
        "claim": "experimental_retrospective_not_confirmation",
        "scope": "Marginal horizon intervals, not simultaneous or aggregate-total intervals.",
    }


def comparison_scores(paths, *, model, levels):
    """Identical finite support for both challengers and any available native control."""
    from utilsforecast.losses import winkler_score

    rows = []
    joint = pd.Series(True, index=next(iter(paths.values())).index)
    active = {name: frame for name, frame in paths.items() if any(
        {f"{model}-lo-{level}", f"{model}-hi-{level}"}.issubset(frame)
        and np.isfinite(frame[[f"{model}-lo-{level}", f"{model}-hi-{level}"]]).all(axis=1).any()
        for level in levels)}
    for frame in active.values():
        for level in levels:
            columns = ["y", model, f"{model}-lo-{level}", f"{model}-hi-{level}"]
            joint &= (np.isfinite(frame[columns]).all(axis=1)
                      & frame[columns[-2]].le(frame[columns[-1]])) if set(columns).issubset(frame) else False
    for level in levels:
        lo, hi = f"{model}-lo-{level}", f"{model}-hi-{level}"
        available = {name: frame for name, frame in paths.items()
                     if {lo, hi}.issubset(frame) and np.isfinite(frame[[lo, hi]]).all(axis=1).any()}
        if not available:
            continue
        common = pd.Series(True, index=next(iter(paths.values())).index)
        for frame in available.values():
            common &= np.isfinite(frame[["y", model, lo, hi]]).all(axis=1) & frame[lo].le(frame[hi])
        for name, frame in available.items():
            data = frame.loc[common].copy()
            data["horizon_step"] = frame.groupby("cutoff", sort=False)["ds"].rank(method="dense").loc[common].astype(int)
            scopes = [("overall", "", data), *(("horizon", str(k), g) for k, g in data.groupby("horizon_step")),
                      *(("window", pd.Timestamp(k).isoformat(), g) for k, g in data.groupby("cutoff"))]
            for scope, key, group in scopes:
                if group.empty:
                    continue
                # The scope is already selected; cutoff grouping would discard or reweight its observations.
                score_input = group[["y", model, lo, hi]].assign(unique_id="scope")
                score = float(winkler_score(score_input, models=[model], level=level)[model].item())
                wis_rows = group.loc[joint.reindex(group.index)]
                wis = None
                if not wis_rows.empty:
                    total = .5 * (wis_rows.y - wis_rows[model]).abs()
                    for nominal in levels:
                        lower, upper = wis_rows[f"{model}-lo-{nominal}"], wis_rows[f"{model}-hi-{nominal}"]
                        alpha = (100 - nominal) / 100
                        interval_score = upper - lower + 2 / alpha * (
                            (lower - wis_rows.y).clip(lower=0) + (wis_rows.y - upper).clip(lower=0))
                        total += alpha / 2 * interval_score
                    wis = float((total / (len(levels) + .5)).mean())
                rows.append({
                    "method": name, "model": model, "level": level, "scope": scope, "scope_key": key,
                    "observations": len(group), "cutoffs": group.cutoff.nunique(),
                    "coverage": float(group.y.between(group[lo], group[hi]).mean()),
                    "mean_width": float((group[hi] - group[lo]).mean()), "mean_interval_score": score,
                    "weighted_interval_score": wis, "wis_observations": len(wis_rows),
                    "compared_methods": ";".join(available), "metric_version": version("utilsforecast"),
                    "scoring_version": 2, "aggregation": "observation_mean",
                    "status": "experimental_paired_retrospective",
                })
    return rows


def apply_experimental_intervals(result, history, spec, *, freq):
    from nixtla_scaffold.models import model_columns
    from nixtla_scaffold.model_families import model_family
    if spec.interval_calibration.method == "native":
        return result
    require_backend()
    future, backtest = result.forecast.copy(), result.backtest_predictions.copy()
    for frame in (future, backtest):
        frame.drop(columns=[c for c in frame if "-lo-" in c or "-hi-" in c], inplace=True)
    evidence = json.loads(json.dumps(result.model_policy_resolution.get("candidate_evidence", {})))
    comparisons, records, messages = [], [], []
    for uid, group in result.forecast.groupby("unique_id", sort=True):
        observed = history.loc[history.unique_id.eq(uid)]
        origin = pd.Timestamp(observed.ds.max())
        base = result.backtest_predictions.loc[result.backtest_predictions.unique_id.eq(uid)].copy()
        if "available_at" in observed and "available_at" not in base:
            base = base.merge(observed[["ds", "available_at"]], on="ds", validate="many_to_one")
            base.index = backtest.loc[backtest.unique_id.eq(uid)].index
        for model in model_columns(future):
            if model_family(model) == "ensemble" or model not in base:
                messages.append(f"{uid}/{model}: experimental intervals unavailable; an exact frozen ensemble replay or model backtest is required.")
                continue
            paths = {"native": base}
            for method in ("split_conformal", "nexcp"):
                config = replace(spec.interval_calibration, method=method)
                cv, fc, record = calibrate_path(base, group.sort_values("ds"), config, model=model,
                                               horizon=spec.horizon, freq=freq, origin=origin,
                                               levels=spec.levels, path_id=f"{uid}/{model}")
                paths[method] = cv
                records.append({"unique_id": str(uid), "model": model, "method": method, **record})
                if method != spec.interval_calibration.method:
                    continue
                for destination, source in ((future, fc), (backtest, cv)):
                    for level in spec.levels:
                        for side in ("lo", "hi"):
                            column = f"{model}-{side}-{level}"
                            if column not in destination:
                                destination[column] = np.nan
                            destination.loc[source.index, column] = source[column]
                for row in evidence.get("fits", []):
                    if row["unique_id"] == str(uid) and row["model"] == model:
                        matches = [r for r in record["receipts"] if r["stage"] == row.get("stage", "future")
                                   and pd.Timestamp(r["cutoff"]) == pd.Timestamp(row["cutoff"])]
                        supported = [r for r in matches if r.get("levels")]
                        row["intervals"] = {
                            "method": f"conformalforecast_{method}", "experimental": True,
                            "engine_version": "0.1.0", "levels": [r["level"] for r in supported],
                            "calibration": supported[0]["calibration"] if supported else None,
                            "level_receipts": matches,
                        }
            comparisons.extend({"unique_id": str(uid), **row} for row in comparison_scores(paths, model=model, levels=spec.levels))
    messages.append("ConformalForecast intervals are experimental retrospective evidence; points, selection and protected confirmation requirements are unchanged. Cold-start or unsupported paths remain point-only.")
    return replace(result, forecast=future, backtest_predictions=backtest, warnings=(*result.warnings, *messages),
                   model_policy_resolution={**result.model_policy_resolution, "candidate_evidence": evidence,
                                            "interval_calibration": {"records": records, "comparison": comparisons}})
