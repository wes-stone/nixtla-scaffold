"""Bounded deterministic engineering fixture; never reads business inputs."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from nixtla_scaffold import run_forecast
from nixtla_scaffold.schema import ForecastSpec, EvaluationPolicy, SelectionPolicy, IntervalCalibrationSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    x = np.arange(468)
    data = pd.DataFrame({"unique_id": "engineering_daily_flow", "ds": pd.date_range("2025-07-01", periods=len(x)),
                         "y": 100 + x * .2 + 12 * np.sin(2 * np.pi * x / 7) + np.cos(x)})
    spec = ForecastSpec(horizon=81, freq="D", season_length=7, levels=(80, 95),
                        model_policy="all", model_allowlist=("Naive", "SeasonalNaive", "AutoETS", "Ridge"),
                        weighted_ensemble=True, evaluation=EvaluationPolicy(max_windows=13, step_size=7),
                        selection_policy=SelectionPolicy(objective="monthly_landing", clip_negative=True),
                        interval_calibration=IntervalCalibrationSpec(aggregate_totals=True))
    started = time.perf_counter()
    run = run_forecast(data, spec)
    fitted = time.perf_counter() - started
    assert len(run.forecast) == 81 and np.isfinite(run.forecast.yhat).all()
    assert run.backtest_predictions.cutoff.nunique() == 13
    before = hashlib.sha256(run.forecast.to_csv(index=False).encode()).hexdigest()
    control = run_forecast(data, replace(spec, interval_calibration=IntervalCalibrationSpec()))
    pd.testing.assert_frame_equal(run.forecast, control.forecast)
    run.to_directory(out / "run")
    data.to_csv(out / "input.csv", index=False)
    receipt = {
        "purpose": "deterministic engineering acceptance, not a business forecast or accuracy claim",
        "single_run_fit_score_seconds": fitted, "total_with_point_invariance_control_and_export_seconds": time.perf_counter() - started,
        "selected_model": run.model_selection.selected_model.item(), "origins": 13, "horizon": 81,
        "future_sha256": before, "aggregate_interval_point_invariance": True,
        "scorecard": run.model_policy_resolution["finance_scorecard"]["scores"],
        "runtime": run.model_policy_resolution.get("resolved_candidate_identity", {}),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps({"receipt": str(out / "receipt.json"), "single_run_fit_score_seconds": fitted}))


if __name__ == "__main__":
    main()
