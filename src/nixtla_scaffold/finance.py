"""Original-scale flow-total evaluation shared by native, custom and blend paths."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd


FINANCE_METRICS = (
    "landing_mape_pct", "landing_bias_pct", "landing_worst_pct", "next28_mape_pct",
    "daily_wmape_pct", "daily_rmse", "daily_mae", "daily_bias",
)
WINDOW_COLUMNS = (
    "unique_id", "model", "cutoff", "kind", "start", "end", "offset",
    "forecast_days", "period_days", "actual", "prediction", "error_pct", "status",
)


def _daily(frame, keys):
    out = frame.copy()
    out["unique_id"] = out["unique_id"].astype(str)
    for column in ("ds", "cutoff"):
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="raise")
            if out[column].dt.tz is not None:
                raise ValueError("Finance totals require timezone-naive daily dates")
            if out[column].isna().any() or not out[column].eq(out[column].dt.normalize()).all():
                raise ValueError("Finance totals require non-null midnight daily dates")
    if out.duplicated(keys).any():
        raise ValueError(f"Finance totals require unique {keys}")
    return out


def _finite_total(values):
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("Finance total overflows the supported numeric range") from exc
    if not math.isfinite(total):
        raise ValueError("Finance total is not finite")
    return total


def _window(history, predictions, cutoff, start, end, model, *, clip_negative):
    dates = pd.date_range(start, end, freq="D")
    known_dates = dates[dates <= cutoff]
    future_dates = dates[dates > cutoff]
    actual = pd.to_numeric(history.reindex(dates)["y"], errors="coerce").to_numpy(float)
    known = pd.to_numeric(history.reindex(known_dates)["y"], errors="coerce").to_numpy(float)
    values = (pd.to_numeric(predictions.reindex(future_dates)[model], errors="coerce").to_numpy(float)
              if model in predictions else np.full(len(future_dates), np.nan))
    status = "complete"
    if not np.isfinite(actual).all():
        status = "incomplete_actuals"
    if "available_at" in history and len(known_dates):
        released = pd.to_datetime(history.reindex(known_dates)["available_at"], errors="coerce")
        if released.isna().any() or released.gt(cutoff).any():
            status = "actuals_not_available_at_origin"
    if not np.isfinite(values).all() or not np.isfinite(known).all():
        status = "incomplete_predictions_or_origin_actuals"
    if clip_negative:
        values = np.maximum(values, 0)
    truth = _finite_total(actual) if np.isfinite(actual).all() else np.nan
    predicted = _finite_total([*known, *values]) if np.isfinite(known).all() and np.isfinite(values).all() else np.nan
    error = (predicted - truth) / abs(truth) * 100 if status == "complete" and truth != 0 else np.nan
    if status == "complete" and truth != 0 and not math.isfinite(error):
        raise ValueError("Finance percentage error overflows the supported numeric range")
    if status == "complete" and truth == 0:
        status = "zero_actual_total"
    return {
        "forecast_days": len(future_dates), "period_days": len(dates), "actual": truth,
        "prediction": predicted, "error_pct": error, "status": status,
    }


def total_windows(history, backtest, models, *, horizon, clip_negative=False, kinds=("month", "next28")):
    """Freeze windows by keys/calendar, never by candidate success or observed errors."""
    history = _daily(history, ["unique_id", "ds"])
    backtest = _daily(backtest, ["unique_id", "cutoff", "ds"])
    if (backtest.ds <= backtest.cutoff).any():
        raise ValueError("Finance backtest targets must follow their origin")
    checked = backtest.merge(history[["unique_id", "ds", "y"]], on=["unique_id", "ds"],
                             suffixes=("", "_source"), how="left", validate="many_to_one")
    if not (checked.y.eq(checked.y_source) | (checked.y.isna() & checked.y_source.isna())).all():
        raise ValueError("Finance backtest actuals differ from the supplied history")
    rows = []
    for uid, cohort in backtest.groupby("unique_id", sort=True):
        actuals = history.loc[history.unique_id.eq(uid)].set_index("ds").sort_index()
        if actuals.empty:
            raise ValueError(f"{uid}: finance backtest has no source history")
        for cutoff, group in cohort.groupby("cutoff", sort=True):
            predictions = group.set_index("ds").sort_index()
            planned_end = cutoff + pd.Timedelta(days=horizon)
            windows = []
            for kind in kinds:
                if kind == "next28":
                    windows.append((kind, 0, cutoff + pd.Timedelta(days=1), cutoff + pd.Timedelta(days=28)))
                elif kind in {"month", "quarter"}:
                    period = cutoff.to_period("M" if kind == "month" else "Q-DEC")
                    for offset in (0, 1):
                        target = period + offset
                        windows.append((kind, offset, target.start_time, target.end_time.normalize()))
                else:
                    raise ValueError(f"Unknown aggregate period: {kind}")
            for kind, offset, start, end in windows:
                if end > actuals.index.max() or end > planned_end or start < actuals.index.min() or end <= cutoff:
                    continue
                for model in models:
                    rows.append({
                        "unique_id": uid, "model": model, "cutoff": cutoff, "kind": kind,
                        "start": start, "end": end, "offset": offset,
                        **_window(actuals, predictions, cutoff, start, end, model, clip_negative=clip_negative),
                    })
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def finance_scorecard(history, backtest, models, *, horizon, clip_negative=False):
    windows = total_windows(history, backtest, models, horizon=horizon, clip_negative=clip_negative)
    rows = []
    for uid, cohort in backtest.groupby("unique_id", sort=True):
        for model in models:
            selected = windows.loc[windows.unique_id.eq(str(uid)) & windows.model.eq(model)]
            landing = selected.loc[selected.kind.eq("month")]
            next28 = selected.loc[selected.kind.eq("next28")]
            row = {"unique_id": str(uid), "model": model,
                   "landing_windows": len(landing), "next28_windows": len(next28)}
            for group, prefix in ((landing, "landing"), (next28, "next28")):
                valid = not group.empty and group.status.eq("complete").all()
                errors = group.error_pct if valid else pd.Series([np.nan])
                row[f"{prefix}_mape_pct"] = float(errors.abs().mean())
                if prefix == "landing":
                    row["landing_bias_pct"] = float(errors.mean())
                    row["landing_worst_pct"] = float(errors.abs().max())
            row.update({name: np.nan for name in FINANCE_METRICS if name.startswith("daily_")})
            if model in cohort and np.isfinite(cohort[[model, "y"]].to_numpy(float)).all():
                points = cohort[["ds", "y", model]].copy()
                if points.groupby("ds").y.nunique().gt(1).any():
                    raise ValueError("Finance scoring requires identical actuals across origins")
                if clip_negative:
                    points[model] = points[model].clip(lower=0)
                averaged = points.groupby("ds", sort=True)[["y", model]].mean()
                errors = averaged[model] - averaged.y
                denominator = averaged.y.abs().sum()
                row.update(
                    daily_wmape_pct=float(100 * errors.abs().sum() / denominator) if denominator else np.nan,
                    daily_rmse=float(np.sqrt(np.mean(errors**2))), daily_mae=float(errors.abs().mean()),
                    daily_bias=float(errors.mean()),
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=["unique_id", "model", *FINANCE_METRICS, "landing_windows", "next28_windows"]), windows


def finance_cutoff_metrics(scored, history, *, horizon, clip_negative=False):
    frame = scored[["unique_id", "ds", "cutoff", "y_actual", "yhat"]].rename(columns={"y_actual": "y", "yhat": "Candidate"})
    frame = _daily(frame, ["unique_id", "cutoff", "ds"])
    windows = total_windows(history, frame, ["Candidate"], horizon=horizon, clip_negative=clip_negative)
    rows = []
    for (uid, cutoff), group in frame.groupby(["unique_id", "cutoff"], sort=True):
        subset = windows.loc[windows.unique_id.eq(str(uid)) & windows.cutoff.eq(cutoff)]
        record = {"unique_id": str(uid), "cutoff": cutoff}
        for kind, column in (("month", "landing_mape_pct"), ("next28", "next28_mape_pct")):
            selected = subset.loc[subset.kind.eq(kind)]
            record[column] = (float(selected.error_pct.abs().mean())
                              if not selected.empty and selected.status.eq("complete").all() else np.nan)
        rows.append(record)
    return pd.DataFrame(rows, columns=["unique_id", "cutoff", "landing_mape_pct", "next28_mape_pct"])


def apply_finance_objective(result, history, spec, *, freq):
    if spec.selection_policy.objective != "monthly_landing":
        return result
    if freq != "D":
        raise ValueError("Monthly landing objective requires daily flow observations (freq='D')")
    from nixtla_scaffold.models import model_columns
    models = [name for name in model_columns(result.backtest_predictions) if name not in {"cutoff", "y"}]
    scores, windows = finance_scorecard(
        history, result.backtest_predictions, models, horizon=spec.horizon,
        clip_negative=spec.selection_policy.clip_negative,
    )
    metrics = result.backtest_metrics.drop(columns=[*FINANCE_METRICS, "landing_windows", "next28_windows"], errors="ignore")
    metrics = metrics.merge(scores, on=["unique_id", "model"], how="left", validate="one_to_one")
    return replace(result, backtest_metrics=metrics,
                   model_policy_resolution={**result.model_policy_resolution, "finance_scorecard": {
                       "contract": spec.selection_policy.contract(),
                       "scores": scores.replace({np.nan: None}).to_dict("records"),
                       "windows": windows.replace({np.nan: None}).to_dict("records"),
                       "actual_vintage": "input snapshot; historical vintages not certified unless supplied",
                   }})


def clip_model_result(result):
    frames = []
    for original in (result.forecast, result.backtest_predictions):
        frame = original.copy()
        columns = frame.columns.difference(["unique_id", "ds", "cutoff", "y", "available_at"])
        frame[columns] = frame[columns].clip(lower=0)
        frames.append(frame)
    return replace(result, forecast=frames[0], backtest_predictions=frames[1])


def enrich_frozen_run(run):
    """Recompute portable evidence for an adopted exact replay, never its points."""
    if run.spec.selection_policy.objective != "monthly_landing" and not run.spec.interval_calibration.aggregate_totals:
        return run
    from nixtla_scaffold.models import ModelResult, model_columns
    history = run.observed_history if not run.observed_history.empty else run.history
    result = ModelResult(forecast=run.all_models, backtest_predictions=run.backtest_predictions,
                         backtest_metrics=run.backtest_metrics, engine=run.engine,
                         model_weights=run.model_weights,
                         model_policy_resolution=run.model_policy_resolution)
    result = apply_finance_objective(result, history, run.spec, freq=run.profile.freq)
    if run.spec.interval_calibration.aggregate_totals:
        result.model_policy_resolution["aggregate_intervals"] = aggregate_intervals(
            history, run.backtest_predictions, run.all_models,
            [name for name in model_columns(run.all_models) if name != "WeightedEnsemble"],
            horizon=run.spec.horizon, levels=run.spec.levels, clip_negative=run.spec.selection_policy.clip_negative,
        )
    selected = run.model_selection
    if run.spec.selection_policy.objective == "monthly_landing":
        scores = result.backtest_metrics[["unique_id", "model", *FINANCE_METRICS, "landing_windows", "next28_windows"]]
        selected = selected.drop(columns=[*FINANCE_METRICS, "landing_windows", "next28_windows"], errors="ignore").merge(
            scores.rename(columns={"model": "selected_model"}), on=["unique_id", "selected_model"],
            how="left", validate="one_to_one",
        )
    return replace(run, model_selection=selected, backtest_metrics=result.backtest_metrics,
                   model_policy_resolution=result.model_policy_resolution)


def aggregate_intervals(history, backtest, future, models, *, horizon, levels, clip_negative=False):
    """Split-conformal absolute total errors, not sums of marginal endpoints.

    Pools match remaining days, period length and current/next period offset.
    Only disjoint, fully landed forecast segments contribute. Exchangeability
    remains an assumption, not a certificate of time-series coverage.
    """
    history = _daily(history, ["unique_id", "ds"])
    future = _daily(future, ["unique_id", "ds"])
    windows = total_windows(history, backtest, models, horizon=horizon, clip_negative=clip_negative,
                            kinds=("month", "quarter"))
    rows = []
    for uid, group in future.groupby("unique_id", sort=True):
        actuals = history.loc[history.unique_id.eq(uid)].set_index("ds").sort_index()
        cutoff = actuals.index.max()
        predictions = group.set_index("ds").sort_index()
        for kind, freq in (("month", "M"), ("quarter", "Q-DEC")):
            for offset in (0, 1):
                period = cutoff.to_period(freq) + offset
                start, end = period.start_time, period.end_time.normalize()
                if end <= cutoff or start < actuals.index.min():
                    continue
                expected = pd.date_range(max(start, cutoff + pd.Timedelta(days=1)), end)
                known = pd.to_numeric(actuals.reindex(pd.date_range(start, min(end, cutoff))).y, errors="coerce")
                for model in models:
                    values = (pd.to_numeric(predictions.reindex(expected)[model], errors="coerce")
                              if model in predictions else pd.Series(np.nan, index=expected))
                    if clip_negative:
                        values = values.clip(lower=0)
                    finite = np.isfinite(values).all() and np.isfinite(known).all()
                    if "available_at" in actuals and len(known):
                        released = pd.to_datetime(actuals.reindex(known.index)["available_at"], errors="coerce")
                        finite = finite and released.notna().all() and released.le(cutoff).all()
                    point = _finite_total([*known, *values]) if finite else None
                    pool = windows.loc[
                        windows.unique_id.eq(uid) & windows.model.eq(model) & windows.kind.eq(kind)
                        & windows.offset.eq(offset) & windows.forecast_days.eq(len(expected))
                        & windows.period_days.eq((end - start).days + 1)
                        & windows.status.isin(["complete", "zero_actual_total"]) & windows.end.le(cutoff)
                    ].sort_values("cutoff")
                    errors, support, previous_end = [], [], None
                    for record in pool.to_dict("records"):
                        first = max(record["start"], record["cutoff"] + pd.Timedelta(days=1))
                        if previous_end is not None and first <= previous_end:
                            continue
                        if "available_at" in actuals:
                            released = pd.to_datetime(actuals.reindex(pd.date_range(record["start"], record["end"]))["available_at"], errors="coerce")
                            if released.isna().any() or released.gt(cutoff).any():
                                continue
                        errors.append(abs(_finite_total([record["prediction"], -record["actual"]])))
                        support.append({"cutoff": record["cutoff"].isoformat(), "start": first.isoformat(),
                                        "end": record["end"].isoformat()})
                        previous_end = record["end"]
                    for level in levels:
                        if type(level) is not int or not 0 < level < 100:
                            raise ValueError("Aggregate interval levels must be integers in (0, 100)")
                        rank = math.ceil((len(errors) + 1) * level / 100)
                        enough = len(errors) >= 20 and rank <= len(errors)
                        radius = float(np.partition(errors, rank - 1)[rank - 1]) if finite and enough else None
                        rows.append({
                            "unique_id": uid, "model": model, "kind": kind, "offset": offset,
                            "start": start.isoformat(), "end": end.isoformat(), "cutoff": cutoff.isoformat(),
                            "level": level, "point_total": point,
                            "lower": _finite_total([point, -radius]) if radius is not None else None,
                            "upper": _finite_total([point, radius]) if radius is not None else None,
                            "calibration_windows": len(errors), "order_statistic": rank,
                            "status": "experimental" if radius is not None else
                                      "incomplete_future_or_origin_actuals" if not finite else "insufficient_calibration",
                            "method": "split_conformal_absolute_total_error",
                            "assumptions": "matched duration and offset; exchangeable aggregate errors; no simultaneous or planning guarantee",
                            "support": support,
                        })
    return rows
