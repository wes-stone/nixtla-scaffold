"""Native in-sample diagnostics; never a cleaning or forecast-calibration rule."""
from __future__ import annotations

from inspect import signature

import numpy as np
import pandas as pd


LEVEL = 99
LOOKBACK = 182
VALUE_COLUMNS = ["training_y", "fitted_y", "lower_99", "upper_99"]


def fitted_diagnostics(model, history: pd.DataFrame) -> dict:
    report = {
        "method": "predict_in_sample", "level": LEVEL, "training_rows": len(history),
        "review_rows": min(LOOKBACK, len(history)), "finite_rows": 0, "flagged": 0,
        "status": "unavailable", "reason": "", "scale": "model_input_units", "points": [],
    }
    predict = getattr(model, "predict_in_sample", None)
    if not callable(predict):
        return {**report, "reason": "Native estimator does not expose in-sample predictions."}
    try:
        signature(predict).bind(level=[LEVEL])
    except TypeError:
        return {**report, "reason": "Native estimator's in-sample API does not accept interval levels."}
    try:
        native = predict(level=[LEVEL])
    except NotImplementedError:
        return {**report, "reason": "Native estimator does not implement in-sample predictions."}
    keys = ("fitted", f"fitted-lo-{LEVEL}", f"fitted-hi-{LEVEL}")
    if not all(key in native for key in keys):
        return {**report, "reason": "Native estimator did not return fitted values and both interval bounds."}
    arrays = [np.asarray(native[key], dtype=float).reshape(-1) for key in keys]
    if any(len(values) != len(history) for values in arrays):
        raise ValueError("Native fitted diagnostic length does not match its training series")
    ordered = history.sort_values("ds")
    frame = pd.DataFrame({
        "ds": pd.to_datetime(ordered.ds).map(pd.Timestamp.isoformat).to_numpy(),
        "training_y": ordered.y.to_numpy(dtype=float),
        **dict(zip(VALUE_COLUMNS[1:], arrays)),
    }).tail(LOOKBACK)
    finite = np.isfinite(frame[VALUE_COLUMNS]).all(axis=1)
    if (frame.loc[finite, "lower_99"] > frame.loc[finite, "upper_99"]).any():
        raise ValueError("Native fitted diagnostic returned reversed interval bounds")
    frame["flagged"] = finite & (
        frame.training_y.lt(frame.lower_99) | frame.training_y.gt(frame.upper_99)
    )
    frame["valid"] = finite
    report.update(status="available" if finite.any() else "unavailable",
                  finite_rows=int(finite.sum()), flagged=int(frame.flagged.sum()),
                  reason="" if finite.any() else "No finite native fitted intervals in the review window.",
                  points=frame.astype(object).where(pd.notna(frame) & ~frame.isin([np.inf, -np.inf]), None).to_dict("records"))
    return report


def fitted_frames(run) -> dict[str, pd.DataFrame]:
    from nixtla_scaffold.transformations import _inverse_target_transform_values

    summaries, points = [], []
    for fit in run.model_policy_resolution.get("candidate_evidence", {}).get("fits", []):
        native = fit.get("native_fitted")
        if native is None:
            continue
        context = {key: fit.get(key, "future" if key == "stage" else None)
                   for key in ("unique_id", "model", "cutoff", "stage")}
        summaries.append({**context, **{key: value for key, value in native.items() if key != "points"}})
        if context["stage"] == "future":
            points.extend({**context, **point} for point in native["points"])
    frame = pd.DataFrame(points, columns=["unique_id", "model", "cutoff", "stage", "ds",
                                         *VALUE_COLUMNS, "flagged", "valid"])
    for name in VALUE_COLUMNS:
        frame[name] = _inverse_target_transform_values(pd.to_numeric(frame[name]), run.spec.transform.target)
    frame["output_scale"] = "normalized_units" if run.spec.transform.normalization_factor_col else "raw_units"
    return {
        "points": frame,
        "summary": pd.DataFrame(summaries, columns=[
            "unique_id", "model", "cutoff", "stage", "method", "level", "training_rows",
            "review_rows", "finite_rows", "flagged", "status", "reason", "scale",
        ]),
    }
