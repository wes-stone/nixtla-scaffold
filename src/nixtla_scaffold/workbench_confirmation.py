"""Frozen-policy confirmation with prior out-of-sample blend residual calibration."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from nixtla_scaffold.blend import LEARNED, BlendRecipe, _predict, _score, _weights
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.research import _build_chronological_split
from nixtla_scaffold.workbench_store import WorkbenchError, canonical_json, digest

CALIBRATION_WINDOWS = 20


def _split(frame, horizon):
    split = _build_chronological_split(frame, horizon=horizon, confirmation_cutoffs=CALIBRATION_WINDOWS + 1)
    if not split.all_series_eligible:
        raise WorkbenchError("insufficient_history", "; ".join(split.reasons))
    return split


def prepare(store, data):
    from nixtla_scaffold.workbench import _json_artifact, _safe_spec
    path, metadata = store.source(data.get("sourceId", ""))
    spec = _safe_spec(_json_artifact(store, data.get("runId", ""), "manifest.json", {})["spec"])
    if spec.events or spec.hierarchy or spec.hierarchy_reconciliation != "none" or spec.transform.normalization_factor_col:
        raise WorkbenchError("postprocessing_evidence_required", "Blend confirmation needs a compatible unnormalized, unreconciled baseline; complex deployed paths keep their native evaluation.")
    if metadata.get("reservedWindows", 1) < CALIBRATION_WINDOWS + 1 or metadata["confirmationHorizon"] != spec.horizon:
        raise WorkbenchError("protected_source_required", "Prepare a new source with 20 calibration windows and one confirmation window reserved before profiling.")
    if metadata["mapping"]["freq"] != spec.freq:
        raise WorkbenchError("calendar_changed", "Confirmation must use the frozen recipe's calendar.")
    with store.connect(readonly=True) as conn:
        blend = conn.execute("SELECT recipe,receipt FROM wb_blends WHERE id=? AND run_id=?", (data.get("blendId"), data["runId"])).fetchone()
    if not blend or digest(blend["receipt"].encode()) != data.get("expectedReceipt"):
        raise WorkbenchError("stale_evidence", "Review the exact frozen recipe before opening confirmation.")
    recipe = BlendRecipe.from_dict(json.loads(blend["recipe"]))
    if recipe.policy == "horizon_greedy":
        raise WorkbenchError("unqualified_policy", "Horizon grouping remains a benchmark prototype, not an admitted confirmation policy.")
    if recipe.policy in LEARNED and recipe.objective != spec.selection_policy.objective:
        raise WorkbenchError("objective_mismatch", "The learned recipe must preserve the declared objective.")
    reference = data.get("referenceModel", "Naive")
    if reference not in recipe.members:
        raise WorkbenchError("reference_required", "Choose a reference from the frozen recipe's component set.")
    assert_unexposed(store, path, spec.horizon, CALIBRATION_WINDOWS + 1)
    return {**data, "spec": spec.to_dict(), "recipe": recipe.to_dict(), "referenceModel": reference,
            "sourceFingerprint": digest(path.read_bytes()), "calibrationWindows": CALIBRATION_WINDOWS}


def assert_unexposed(store, path, horizon, windows):
    with store.connect(readonly=True) as conn:
        previous_sources = list(conn.execute("SELECT id,metadata FROM wb_sources"))
        jobs = list(conn.execute("SELECT input FROM wb_jobs"))
    keys = pd.read_csv(path, usecols=["unique_id", "ds"], dtype={"unique_id": str})
    keys["ds"] = pd.to_datetime(keys["ds"])
    protected = _build_chronological_split(keys.assign(y=0.), horizon=horizon, confirmation_cutoffs=windows).receipt
    protected = protected[protected["role"] == "confirmation"][["unique_id", "ds"]]
    first = protected.groupby("unique_id")["ds"].min()
    for previous in previous_sources:
        for row in json.loads(previous["metadata"])["profile"].get("series", []):
            uid = str(row["unique_id"])
            if uid in first and row.get("end") and pd.Timestamp(row["end"]) >= first[uid]:
                raise WorkbenchError("confirmation_consumed", "Overlapping series/periods have already reached target profiling in this workspace. A new filename does not create untouched evidence.")
    for job in jobs:
        old = json.loads(job["input"])
        if not old.get("sourceId"):
            continue
        old_path, _ = store.source(old["sourceId"])
        used = pd.read_csv(old_path, usecols=["unique_id", "ds"], dtype={"unique_id": str})
        used["ds"] = pd.to_datetime(used["ds"])
        if not used.merge(protected, on=["unique_id", "ds"]).empty:
            raise WorkbenchError("confirmation_consumed", "A prior job has already used overlapping protected observations. Confirmation is consumed even if that job was interrupted.")


def _predictions(run, recipe):
    from nixtla_scaffold.ensemble import observable_backtest_rows
    if recipe.policy == "horizon_greedy":
        raise ValueError("Horizon grouping is not qualified for protected deployment.")
    rows = []
    for uid, future in run.all_models.groupby("unique_id", sort=True):
        prior = run.backtest_predictions[run.backtest_predictions["unique_id"].astype(str) == str(uid)]
        frame = future.sort_values("ds").copy()
        origin = pd.to_datetime(frame.ds).min() - pd.Timedelta(nanoseconds=1)
        prior = observable_backtest_rows(prior, origin)
        history = run.observed_history if not run.observed_history.empty else run.history
        history = history.loc[pd.to_datetime(history.ds).le(origin)].copy()
        if "available_at" in history:
            release = pd.to_datetime(history.available_at, errors="coerce")
            history.loc[release.isna() | release.gt(origin), "y"] = np.nan
        if run.spec.selection_policy.clip_negative:
            frame[list(recipe.members)] = frame[list(recipe.members)].clip(lower=0)
            prior[list(recipe.members)] = prior[list(recipe.members)].clip(lower=0)
        weights = _weights(recipe, prior, history=history, horizon=run.spec.horizon,
                           clip_negative=run.spec.selection_policy.clip_negative)
        frame["Blend"] = _predict(frame, weights)
        frame["weights"] = canonical_json(weights)
        frame["horizon_step"] = np.arange(1, len(frame) + 1)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _radii(calibration, levels):
    rows = []
    for (uid, horizon), group in calibration.groupby(["unique_id", "horizon_step"], sort=True):
        errors = np.sort(np.abs(group["Blend"].to_numpy() - group["y"].to_numpy()))
        for level in levels:
            rank = math.ceil((len(errors) + 1) * level / 100)
            if not len(errors) or rank > len(errors):
                continue
            rows.append({"unique_id": uid, "horizon_step": horizon, "level": level,
                         "radius": float(errors[rank - 1]), "samples": len(errors)})
    return pd.DataFrame(rows)


def _apply_bounds(frame, radii, levels, *, forecast=False):
    result = frame.copy()
    point = "yhat" if forecast else "Blend"
    for level in levels:
        selected = radii[radii["level"] == level] if not radii.empty else pd.DataFrame()
        if selected.empty:
            continue
        radius = result[["unique_id", "horizon_step"]].merge(selected, on=["unique_id", "horizon_step"], how="left", validate="many_to_one")["radius"].to_numpy()
        lo, hi = (f"yhat_lo_{level}", f"yhat_hi_{level}") if forecast else (f"Blend-lo-{level}", f"Blend-hi-{level}")
        result[lo], result[hi] = result[point] - radius, result[point] + radius
    return result


def execute(store, data, out):
    from nixtla_scaffold.workbench import _safe_spec
    path, _ = store.source(data["sourceId"])
    if digest(path.read_bytes()) != data["sourceFingerprint"]:
        raise WorkbenchError("source_changed", "The protected snapshot changed.")
    frame = pd.read_csv(path, dtype={"unique_id": str})
    frame["ds"] = pd.to_datetime(frame["ds"])
    if "available_at" in frame:
        raise WorkbenchError("vintage_review_required", "This source declares publication vintages. Use the vintage-aware native research path; blend confirmation will not silently treat revised targets as contemporaneous.")
    if frame["y"].isna().any() or not np.isfinite(frame["y"]).all():
        raise WorkbenchError("missing_confirmation", "Protected calibration and confirmation require finite observed targets; no holdout imputation is permitted.")
    spec = _safe_spec(data["spec"])
    from nixtla_scaffold.schema import IntervalCalibrationSpec
    fit_spec = replace(spec, interval_calibration=IntervalCalibrationSpec())
    recipe = BlendRecipe.from_dict(data["recipe"])
    split = _split(frame, spec.horizon)
    training = split.tuning_data.copy()
    evaluated = []
    for index, (name, window) in enumerate(split.confirmation_windows):
        if store.job(data["jobId"])["status"] == "cancel_requested":
            raise WorkbenchError("cancelled", "Stopped before opening the next reserved window.")
        store.event(data["jobId"], "running", f"Frozen policy: evaluating {'confirmation' if index == CALIBRATION_WINDOWS else 'calibration'} window {index + 1}/{CALIBRATION_WINDOWS + 1}.")
        run = run_forecast(training, fit_spec)
        predicted = _predictions(run, recipe)
        predicted["ds"] = pd.to_datetime(predicted["ds"])
        paired = window[["unique_id", "ds", "y"]].merge(predicted, on=["unique_id", "ds"], how="outer", validate="one_to_one")
        required = ["y", "Blend", *recipe.members]
        if len(paired) != len(window) or not np.isfinite(paired[required]).all().all():
            raise WorkbenchError("coverage_gap", "Every protected series/period must have the same finite candidate and reference support.")
        origins = training.groupby("unique_id")["ds"].max()
        paired["cutoff"] = paired["unique_id"].map(origins)
        paired["partition"] = "confirmation" if index == CALIBRATION_WINDOWS else "calibration"
        evaluated.append(paired)
        training = pd.concat([training, window], ignore_index=True)
    calibration = pd.concat(evaluated[:-1], ignore_index=True)
    radii = _radii(calibration, spec.levels)
    confirmation = _apply_bounds(evaluated[-1], radii, spec.levels)
    store.event(data["jobId"], "running", "Confirmation consumed. Refitting the unchanged policy for future periods; not searching new recipes.")
    final_run = run_forecast(frame, fit_spec)
    future = _predictions(final_run, recipe).rename(columns={"Blend": "yhat"})
    future["model"] = "Blend"
    future = _apply_bounds(future, radii, spec.levels, forecast=True)
    interval_evaluation = {}
    if spec.interval_calibration.method != "native":
        confirmation, future, interval_evaluation = _experimental_blend_intervals(
            calibration, confirmation, future, spec, recipe)
        final_run = replace(final_run, spec=spec, model_policy_resolution={
            **final_run.model_policy_resolution, "interval_calibration": interval_evaluation})
    if spec.interval_calibration.aggregate_totals:
        from nixtla_scaffold.finance import enrich_frozen_run
        final_run = enrich_frozen_run(replace(final_run, spec=spec))
    comparison, coverage = [], []
    for uid, group in confirmation.groupby("unique_id"):
        candidate = _score(group, "Blend")
        reference = _score(group, data["referenceModel"])
        primary_metric, secondary_metric = "rmse", "mae"
        if spec.selection_policy.objective == "monthly_landing":
            from nixtla_scaffold.finance import finance_scorecard
            scores, _ = finance_scorecard(frame, group, ["Blend", data["referenceModel"]], horizon=spec.horizon,
                                          clip_negative=spec.selection_policy.clip_negative)
            scored = scores.replace({np.nan: None}).set_index("model")
            candidate.update(scored.loc["Blend"].to_dict())
            reference.update(scored.loc[data["referenceModel"]].to_dict())
            primary_metric, secondary_metric = "landing_mape_pct", "next28_mape_pct"
        primary = ((reference[primary_metric] - candidate[primary_metric]) / reference[primary_metric]
                   if reference[primary_metric] is not None and candidate[primary_metric] is not None
                   and reference[primary_metric] > 0 else None)
        secondary = ((candidate[secondary_metric] - reference[secondary_metric]) / reference[secondary_metric]
                     if reference[secondary_metric] is not None and candidate[secondary_metric] is not None
                     and reference[secondary_metric] > 0 else None)
        policy = spec.context.promotion_policy if spec.context else None
        comparison.append({"unique_id": uid, "candidate": candidate, "reference": reference,
                           "primary_metric": primary_metric, "secondary_metric": secondary_metric,
                           "required_confirmation_cutoffs": policy.minimum_confirmation_cutoffs if policy else 1,
                           "primary_improvement": primary, "secondary_regression": secondary,
                           "passes_numeric_policy": primary is not None and secondary is not None
                           and primary >= (policy.minimum_primary_metric_improvement if policy else 0.02)
                           and secondary <= (policy.maximum_secondary_metric_regression if policy else 0.01)
                           and (policy.minimum_confirmation_cutoffs if policy else 1) <= 1})
        for level in spec.levels:
            lo, hi = f"Blend-lo-{level}", f"Blend-hi-{level}"
            if lo in group:
                for horizon, rows in group.groupby("horizon_step"):
                    valid = rows[[lo, hi]].notna().all(axis=1)
                    coverage.append({"unique_id": uid, "horizon_step": int(horizon), "level": level,
                                     "observations": int(valid.sum()),
                                     "coverage": float(rows.loc[valid, "y"].between(rows.loc[valid, lo], rows.loc[valid, hi]).mean()) if valid.any() else None})
    if not np.isfinite(future["yhat"]).all():
        raise WorkbenchError("coverage_gap", "Final frozen-policy refit is missing component support.")
    base = store.import_run(final_run.to_directory(out / "refit"), name=data.get("name") or "Confirmed ensemble", metadata={
        "sourceId": data["sourceId"], "parentRunId": data["runId"], "purpose": "confirmation_refit",
    })
    backtest = pd.concat([calibration, confirmation], ignore_index=True)
    metrics = pd.DataFrame([{"unique_id": uid, **_score(group, model)}
                            for uid, group in backtest.groupby("unique_id") for model in ["Blend", *recipe.members]])
    finance_evidence = {}
    if spec.selection_policy.objective == "monthly_landing":
        scores, windows = finance_scorecard(frame, backtest, list(metrics.model.unique()),
                                            horizon=spec.horizon, clip_negative=spec.selection_policy.clip_negative)
        metrics = metrics.merge(scores, on=["unique_id", "model"], how="left", validate="one_to_one")
        finance_evidence = {"finance_scorecard": {
            "contract": spec.selection_policy.contract(), "cohort": "frozen_reserved_replay",
            "scores": scores.replace({np.nan: None}).to_dict("records"),
            "windows": windows.replace({np.nan: None}).to_dict("records")}}
    split.receipt.to_csv(out / "partitions.csv", index=False)
    receipt = {
        "schema_version": "nixtla.workbench.blend.v1", "recipe": recipe.to_dict(), "recipe_fingerprint": recipe.fingerprint,
        "paired_rows": len(backtest), "excluded_rows": 0, "scoring": "frozen_policy_reserved_confirmation",
        "confirmation_status": "independently_reserved_within_workspace", "confirmation": comparison,
        "calibration": {"method": "absolute_oof_residual_order_statistic", "windows": CALIBRATION_WINDOWS, "radii": json.loads(radii.to_json(orient="records"))},
        "confirmation_coverage": coverage, "interval_status": "prior_residual_calibration" if not radii.empty else "insufficient_calibration",
        "source_fingerprint": data["sourceFingerprint"], "partition_hash": digest((out / "partitions.csv").read_bytes()),
        "promotion_recommended": False, "advisory_only": True,
        **finance_evidence,
        "limitations": ["Confirmation is scoped to recorded workspace exposure, not a global data-access certificate.",
                        "Twenty calibration windows and one confirmation origin; horizon-specific coverage estimates remain small-sample.",
                        "Empirical intervals assume residual stability; no component quantiles were averaged.",
                        "Numeric policy checks are not independent human review, a planning lock or automatic promotion."],
    }
    if interval_evaluation:
        receipt["interval_calibration"] = interval_evaluation
        receipt["interval_status"] = "experimental"
        receipt["calibration"]["method"] = f"conformalforecast_{spec.interval_calibration.method}"
        receipt["limitations"].append("Experimental marginal bands are not a planning certificate or simultaneous/month-total intervals. Per-level receipts govern support.")
    blend_id = digest((base["id"] + recipe.fingerprint).encode())
    destination = store.owned(f"blends/{blend_id}")
    with store.snapshot_lock():
        destination.mkdir(parents=True, exist_ok=False)
        backtest.to_csv(destination / "backtest.csv", index=False)
        future[["unique_id", "ds", "yhat", "model", "weights", *[c for c in future if c.startswith("yhat_")]]].to_csv(destination / "forecast.csv", index=False)
        metrics.to_csv(destination / "metrics.csv", index=False)
        (destination / "partitions.csv").write_bytes((out / "partitions.csv").read_bytes())
        receipt["artifact_hashes"] = {name: digest((destination / name).read_bytes()) for name in ("backtest.csv", "forecast.csv", "metrics.csv", "partitions.csv")}
        (destination / "receipt.json").write_text(canonical_json(receipt), encoding="utf-8")
        with store.connect() as conn:
            conn.execute("INSERT INTO wb_blends VALUES(?,?,?,?,?)", (blend_id, base["id"], canonical_json(recipe.to_dict()),
                         str(destination.relative_to(store.root)), canonical_json(receipt)))
    return {"runId": base["id"], "blendId": blend_id, "receipt": receipt, "confirmationConsumed": True,
            "stopReason": "frozen_candidate_confirmed_no_further_search"}


def _experimental_blend_intervals(calibration, confirmation, future, spec, recipe):
    from nixtla_scaffold.conformal import calibrate_path, comparison_scores
    backtest = pd.concat([calibration, confirmation], ignore_index=True)
    result, projected = confirmation.copy(), future.copy()
    bounds = [c for c in projected if c.startswith(("yhat_lo_", "yhat_hi_"))]
    projected.drop(columns=bounds, inplace=True)
    result.drop(columns=[c for c in result if c.startswith(("Blend-lo-", "Blend-hi-"))], inplace=True)
    records, comparison = [], []
    for uid, targets in future.groupby("unique_id", sort=True):
        prior = backtest.loc[backtest.unique_id.eq(uid)].copy()
        points = targets.rename(columns={"yhat": "Blend"})
        origin = pd.Timestamp(targets.ds.min()) - pd.tseries.frequencies.to_offset(spec.freq)
        paths = {"native": prior}
        for method in ("split_conformal", "nexcp"):
            cv, fc, record = calibrate_path(
                prior, points, replace(spec.interval_calibration, method=method),
                model="Blend", horizon=spec.horizon, freq=spec.freq, origin=origin,
                levels=spec.levels, path_id=f"{uid}/frozen_blend/{recipe.fingerprint}")
            records.append({"unique_id": str(uid), "model": "Blend", "method": method, **record})
            paths[method] = cv
            if method != spec.interval_calibration.method:
                continue
            selected = cv.loc[cv.partition.eq("confirmation")].set_index(["unique_id", "cutoff", "ds"])
            for level in spec.levels:
                for side in ("lo", "hi"):
                    col, out = f"Blend-{side}-{level}", f"yhat_{side}_{level}"
                    if col not in result:
                        result[col] = np.nan
                    if out not in projected:
                        projected[out] = np.nan
                    mask = result.unique_id.eq(uid)
                    keys = pd.MultiIndex.from_frame(result.loc[mask, ["unique_id", "cutoff", "ds"]])
                    result.loc[mask, col] = selected.reindex(keys)[col].to_numpy()
                    projected.loc[fc.index, out] = fc[col].to_numpy()
        comparison.extend({"unique_id": str(uid), **row}
                          for row in comparison_scores(paths, model="Blend", levels=spec.levels))
    return result, projected, {"scope": "frozen_blend_only", "records": records, "comparison": comparison}
