"""Executed interval provenance; nominal bands are not a calibration certificate."""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import math

import numpy as np
import pandas as pd


def minimum_rank_samples(level: int) -> int:
    return math.ceil(100 / (100 - level))


def _available_levels(frame, alias):
    levels = []
    for name in frame:
        prefix = f"{alias}-lo-"
        if not name.startswith(prefix):
            continue
        level = int(name.removeprefix(prefix))
        upper = f"{alias}-hi-{level}"
        if upper in frame and not frame.empty and np.isfinite(frame[[name, upper]]).all().all() and frame[name].le(frame[upper]).all():
            levels.append(level)
    return sorted(levels)


def _calibration(values, periods):
    array = np.asarray(values, dtype="<f8")
    return {
        "observed_residual_shape": list(array.shape),
        "score_sha256": sha256(array.tobytes()).hexdigest(),
        "finite": bool(np.isfinite(array).all()),
        "windows": len(periods), "periods": periods, "score_scale": "model_scale",
    }


def native_interval_evidence(fitted, forecast, history, alias):
    levels = _available_levels(forecast, alias)
    receipt = {"method": "point_forecast_only", "levels": levels, "engine_version": version("statsforecast"),
               "calibration": None}
    if not levels:
        return receipt
    if type(fitted).__name__ == "AutoTBATS":
        receipt["method"] = "statsforecast_native_tbats"
        return receipt
    interval_model = fitted.trend_forecaster if type(fitted).__name__ == "MSTL" else fitted
    scores = getattr(interval_model, "_cs", None)
    configuration = getattr(interval_model, "prediction_intervals", None)
    if configuration is not None and scores is not None:
        rows, h = scores.shape
        dates = pd.DatetimeIndex(pd.to_datetime(history.ds)).sort_values()
        periods = []
        for index in range(rows):
            end = len(dates) - rows * h + index * h
            periods.append({"cutoff": dates[end - 1].isoformat(),
                            "targets": [value.isoformat() for value in dates[end:end + h]]})
        receipt.update(method=f"statsforecast_{configuration.method}", calibration=_calibration(scores, periods))
    elif configuration is None:
        receipt["method"] = "statsforecast_native"
    else:
        receipt["method"] = "unverified_execution"
    return receipt


def ml_interval_evidence(engine, forecast, uid, alias):
    levels = _available_levels(forecast, alias)
    receipt = {"method": "point_forecast_only", "levels": levels, "engine_version": version("mlforecast"),
               "calibration": None}
    if not levels:
        return receipt
    configuration = getattr(engine, "prediction_intervals", None)
    scores = getattr(engine, "_cs_df", None)
    if configuration is None or not isinstance(scores, pd.DataFrame) or alias not in scores:
        receipt["method"] = "unverified_execution"
        return receipt
    selected = scores.loc[scores["unique_id"].astype(str).eq(str(uid))].sort_values(["cutoff", "ds"])
    periods = [{"cutoff": pd.Timestamp(cutoff).isoformat(),
                "targets": [pd.Timestamp(value).isoformat() for value in group.ds]}
               for cutoff, group in selected.groupby("cutoff", sort=True)]
    receipt.update(method=f"mlforecast_{configuration.method}",
                   calibration=_calibration(selected[alias].to_numpy(), periods),
                   calibration_refit="fixed_first_calibration_fit")
    return receipt


def blend_candidate_evidence(future, backtest, receipt, horizon):
    from nixtla_scaffold.candidates import CandidateRecipe
    recipe = CandidateRecipe("Blend", "workbench_blend", {"class": "frozen_blend", "parameters": receipt["recipe"]},
                             {"horizon": horizon, "recipe_fingerprint": receipt["recipe_fingerprint"]})
    fits = []
    for uid, group in backtest.groupby("unique_id", sort=True):
        calibration = None
        if "partition" in group:
            calibration_rows = group.loc[group.partition.eq("calibration")].sort_values(["cutoff", "ds"])
            if not calibration_rows.empty:
                periods = [{"cutoff": pd.Timestamp(cutoff).isoformat(), "targets": [pd.Timestamp(t).isoformat() for t in rows.ds]}
                           for cutoff, rows in calibration_rows.groupby("cutoff", sort=True)]
                calibration = _calibration(np.abs(calibration_rows.Blend - calibration_rows.y), periods)
                calibration["score_scale"] = "reporting_scale"
        for stage, frame in (("future", future.loc[future.unique_id.astype(str).eq(str(uid))]),
                             ("backtest", group.loc[group.partition.eq("confirmation")] if "partition" in group else group)):
            levels = _available_levels(frame, "Blend")
            intervals = {"method": "blend_residual_order_statistic" if levels and calibration else "point_forecast_only",
                         "levels": levels, "calibration": calibration, "method_version": 1,
                         "confirmation_windows": 1 if calibration else 0}
            experimental = receipt.get("interval_calibration", {})
            method = receipt.get("calibration", {}).get("method", "")
            if method.startswith("conformalforecast_"):
                records = [r for r in experimental.get("records", []) if r["unique_id"] == str(uid)
                           and method == f"conformalforecast_{r['method']}"]
                rows = [r for record in records for r in record["receipts"]
                        if r["stage"] == stage and (stage == "future"
                            or pd.Timestamp(r["cutoff"]) == pd.Timestamp(group.cutoff.max()))]
                intervals = {"method": method, "experimental": True, "levels": levels,
                             "engine_version": "0.1.0", "level_receipts": rows,
                             "calibration": rows[0]["calibration"] if rows else None,
                             "confirmation_windows": 1}
            fits.append({"unique_id": str(uid), "model": "Blend", "stage": stage,
                         "cutoff": rows[0]["cutoff"] if intervals.get("experimental") and rows else pd.Timestamp(group.cutoff.max()).isoformat(),
                         "recipe_id": recipe.fingerprint, "status": "succeeded",
                         "fitted_structure_status": "not_applicable_frozen_blend", "intervals": intervals})
    return {"recipes": {recipe.fingerprint: recipe.to_dict()}, "fits": fits}


def interval_provenance(run, uid, model, *, stage="future"):
    fits = run.model_policy_resolution.get("candidate_evidence", {}).get("fits", [])
    selected = [row for row in fits if row["unique_id"] == str(uid) and row["model"] == model
                and row.get("stage", "future") == stage]
    intervals = [row["intervals"] for row in selected if "intervals" in row]
    methods = sorted({row["method"] for row in intervals})
    return {
        "method": methods[0] if len(methods) == 1 else "mixed:" + ",".join(methods) if methods else "unverified_legacy",
        "receipts": intervals,
    }


def interval_support(run, uid, model, level, *, stage):
    provenance = interval_provenance(run, uid, model, stage=stage)
    receipts = provenance["receipts"]
    if not receipts:
        return False
    for receipt in receipts:
        if receipt.get("experimental"):
            return False
        if level not in receipt["levels"]:
            return False
        method = receipt["method"]
        if method in {"statsforecast_native", "statsforecast_native_tbats"}:
            continue
        if method not in {"statsforecast_conformal_distribution", "mlforecast_conformal_distribution", "blend_residual_order_statistic"}:
            return False
        calibration = receipt.get("calibration")
        if not calibration or not calibration["finite"] or calibration["windows"] < minimum_rank_samples(level):
            return False
        if receipt.get("calibration_feature_scope") == "precomputed_prefix_features":
            return False
    return True
