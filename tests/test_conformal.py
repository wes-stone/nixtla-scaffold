from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.conformal import ResidualCalibrator, calibrate_path, comparison_scores
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec, IntervalCalibrationSpec, forecast_spec_from_dict


@pytest.fixture
def optional_backend():
    pytest.importorskip("conformalforecast")


def add_window(state, index, *, prediction=100., error=1.):
    cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(days=index * state.horizon)
    dates = pd.date_range(cutoff, periods=state.horizon + 1, freq="D")[1:]
    state.observe(cutoff=cutoff, dates=dates, predictions=np.full(state.horizon, prediction),
                  truths=np.full(state.horizon, prediction + error),
                  available_at=dates[-1], observed_at=dates[-1])


def test_interval_config_roundtrip():
    spec = ForecastSpec(interval_calibration=IntervalCalibrationSpec(method="nexcp"))
    assert forecast_spec_from_dict(spec.to_dict()) == spec
    assert forecast_spec_from_dict({}).interval_calibration.method == "native"
    for kwargs in ({"method": "magic"}, {"rho": float("nan")}, {"rho": True}, {"max_windows": 0}):
        with pytest.raises(ValueError):
            IntervalCalibrationSpec(**kwargs)


@pytest.mark.parametrize("method", ["split_conformal", "nexcp"])
def test_native_backend_parity_and_restore_continue(optional_backend, method):
    from conformalforecast.methods import SplitConformal, NonexchangeableConformalPrediction
    state = ResidualCalibrator(IntervalCalibrationSpec(method=method), horizon=2, freq="D", path_id="fixed/A")
    for i in range(25):
        add_window(state, i, error=1 + (i % 9))
    payload = json.loads(json.dumps(state.to_dict()))
    restored = ResidualCalibrator.from_dict(payload)
    payload["windows"][0]["truths"][0] = 999
    for i in range(25, 29):
        add_window(state, i, error=2 + i % 7)
        add_window(restored, i, error=2 + i % 7)
    points = np.array([90., 130.])
    cutoff = pd.Timestamp("2020-03-01")
    bounds, receipt = state.interval(points, cutoff=cutoff, level=95)
    replay, _ = restored.interval(points, cutoff=cutoff, level=95)
    np.testing.assert_array_equal(bounds, replay)
    preds = np.array([w["predictions"] for w in state.windows])[None]
    truths = np.array([w["truths"] for w in state.windows])[None]
    model = SplitConformal(alpha=.05) if method == "split_conformal" else NonexchangeableConformalPrediction(alpha=.05, rho=.99)
    expected = model.fit(preds, truths).interval(points[None, None])[0, 0]
    np.testing.assert_array_equal(bounds, expected)
    assert receipt["experimental"] and receipt["calibration"]["windows"] == 29
    assert len(receipt["calibration"]["score_sha256"]) == 64


def test_cold_start_and_invalid_update_order(optional_backend):
    state = ResidualCalibrator(IntervalCalibrationSpec(method="split_conformal"), horizon=2, freq="D", path_id="A")
    assert state.interval([1, 2], cutoff="2020-01-01", level=95)[0] is None
    add_window(state, 0)
    with pytest.raises(ValueError, match="overlapping|out-of-order"):
        add_window(state, 0)
    with pytest.raises(ValueError, match="not available"):
        state.interval([1, 2], cutoff="2020-01-01", level=80)
    with pytest.raises(ValueError, match="landed"):
        state.observe(cutoff="2020-01-03", dates=pd.date_range("2020-01-04", periods=2),
                      predictions=[1, 2], truths=[2, 3], available_at="2020-01-05", observed_at="2020-01-04")
    assert len(state.windows) == 1


def paths():
    rows = []
    for i in range(100):
        cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        for h in range(1, 3):
            rows.append({"unique_id": "A", "cutoff": cutoff, "ds": cutoff + pd.Timedelta(days=h),
                         "y": 100 + ((i + h) % 9), "Fixed": 100.,
                         "Fixed-lo-80": 90., "Fixed-hi-80": 110.,
                         "Fixed-lo-95": 85., "Fixed-hi-95": 115.})
    cv = pd.DataFrame(rows)
    origin = cv.ds.max()
    future = pd.DataFrame({"unique_id": "A", "ds": pd.date_range(origin, periods=3)[1:], "Fixed": [100., 100.]})
    return cv, future, origin


def test_later_targets_and_delayed_releases_cannot_leak(optional_backend):
    cv, future, origin = paths()
    args = dict(config=IntervalCalibrationSpec(method="nexcp"), model="Fixed", horizon=2, freq="D",
                origin=origin, levels=(80, 95), path_id="frozen-recipe-hash")
    before, fc, evidence = calibrate_path(cv, future, **args)
    altered = cv.copy()
    boundary = pd.Timestamp("2020-03-01")
    altered.loc[altered.ds > boundary, "y"] += 10000
    after, _, _ = calibrate_path(altered, future, **args)
    columns = ["Fixed-lo-80", "Fixed-hi-80", "Fixed-lo-95", "Fixed-hi-95"]
    pd.testing.assert_frame_equal(before.loc[before.cutoff <= boundary, columns], after.loc[after.cutoff <= boundary, columns])
    pd.testing.assert_series_equal(fc.Fixed, future.Fixed)
    assert evidence["excluded_windows"] and len(evidence["state"]["windows"]) == 50
    all_targets = [t for w in evidence["state"]["windows"] for t in w["dates"]]
    assert len(all_targets) == len(set(all_targets))
    delayed = cv.assign(available_at=cv.ds + pd.Timedelta(days=500))
    delay_cv, delay_fc, receipt = calibrate_path(delayed, future, **args)
    assert delay_cv[columns].isna().all().all() and delay_fc[columns].isna().all().all()
    assert receipt["state"]["windows"] == []


def test_matched_interval_metrics_and_wis(optional_backend):
    cv, future, origin = paths()
    candidates = {"native": cv}
    for method in ("split_conformal", "nexcp"):
        candidates[method], _, _ = calibrate_path(cv, future, IntervalCalibrationSpec(method=method),
            model="Fixed", horizon=2, freq="D", origin=origin, levels=(80, 95), path_id="Fixed")
    scores = pd.DataFrame(comparison_scores(candidates, model="Fixed", levels=(80, 95)))
    overall = scores.loc[scores.scope.eq("overall")]
    assert overall.groupby("level").observations.nunique().eq(1).all()
    assert overall.coverage.between(0, 1).all()
    assert overall.mean_interval_score.ge(0).all()
    assert overall.weighted_interval_score.gt(0).all()
    assert set(scores.scope) == {"overall", "horizon", "window"}


def test_interval_score_includes_later_cutoff_errors(optional_backend):
    frame = pd.DataFrame({
        "cutoff": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "ds": pd.to_datetime(["2026-01-02", "2026-01-03"]),
        "y": [0., 100.], "Fixed": [0., 0.],
        "Fixed-lo-80": [-1., -1.], "Fixed-hi-80": [1., 1.],
    })
    scores = comparison_scores({"native": frame}, model="Fixed", levels=(80,))
    for row in scores:
        expected = (2. if row["scope_key"] == "2026-01-01T00:00:00" else 992.) if row["scope"] == "window" else 497.
        assert row["mean_interval_score"] == pytest.approx(expected)


@pytest.mark.parametrize("excluded", [None, "missing", "inverted"])
def test_interval_scores_weight_observations_on_exact_support(optional_backend, excluded):
    native = pd.DataFrame({
        "cutoff": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-04"] * 2),
        "ds": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]),
        "y": [0., 2., -2., 100., -5.], "Fixed": 0.,
        "Fixed-lo-80": -1., "Fixed-hi-80": 1.,
        "Fixed-lo-95": -3., "Fixed-hi-95": 3.,
    })
    wide = native.copy()
    wide[["Fixed-lo-80", "Fixed-hi-80", "Fixed-lo-95", "Fixed-hi-95"]] = [-2., 2., -4., 4.]
    if excluded:
        wide.loc[1, ["Fixed-hi-80", "Fixed-hi-95"]] = np.nan if excluded == "missing" else -10.
    paths = {"native": native, "wide": wide}
    before = {name: frame.copy(deep=True) for name, frame in paths.items()}
    scores = comparison_scores(paths, model="Fixed", levels=(80, 95))
    row_scores = {
        ("native", 80): [2., 12., 12., 992., 42.],
        ("native", 95): [6., 6., 6., 3886., 86.],
        ("wide", 80): [4., 4., 4., 984., 34.],
        ("wide", 95): [8., 8., 8., 3848., 48.],
    }
    row_wis = {"native": [.14, .94, .94, 98.54, 3.54], "wide": [.24, .64, .64, 97.84, 2.84]}
    supported = {0, 2, 3, 4} if excluded else set(range(5))
    scopes = {
        ("overall", ""): set(range(5)),
        ("horizon", "1"): {0, 3}, ("horizon", "2"): {1, 4}, ("horizon", "3"): {2},
        ("window", "2026-01-01T00:00:00"): {0, 1, 2},
        ("window", "2026-01-04T00:00:00"): {3, 4},
    }
    assert len(scores) == 24
    for row in scores:
        indices = sorted(scopes[row["scope"], row["scope_key"]] & supported)
        expected = np.mean([row_scores[row["method"], row["level"]][i] for i in indices])
        assert row["mean_interval_score"] == pytest.approx(expected)
        assert row["weighted_interval_score"] == pytest.approx(np.mean([row_wis[row["method"]][i] for i in indices]))
        assert row["observations"] == row["wis_observations"] == len(indices)
        assert row["cutoffs"] == native.iloc[indices].cutoff.nunique()
        assert row["compared_methods"] == "native;wide"
        assert row["scoring_version"] == 2
        assert row["aggregation"] == "observation_mean"
    for name, frame in paths.items():
        pd.testing.assert_frame_equal(frame, before[name])


def test_forecast_integration_preserves_points_and_selection(optional_backend, tmp_path):
    from nixtla_scaffold.forecast import run_forecast
    from nixtla_scaffold.intervals import interval_support
    data = pd.concat([pd.DataFrame({"unique_id": uid, "ds": pd.date_range("2020-01-01", periods=70),
                                   "y": 100 + np.sin(np.arange(70)) * factor}) for uid, factor in (("A", 1), ("B", 2))])
    spec = ForecastSpec(model_policy="baseline", model_allowlist=("Naive",), horizon=1, freq="D",
                        weighted_ensemble=False, evaluation=EvaluationPolicy(max_windows=35, step_size=1))
    control = run_forecast(data, spec)
    variant = run_forecast(data, replace(spec, interval_calibration=IntervalCalibrationSpec(method="split_conformal")))
    pd.testing.assert_frame_equal(control.model_selection, variant.model_selection)
    pd.testing.assert_series_equal(control.forecast.yhat, variant.forecast.yhat)
    pd.testing.assert_series_equal(control.backtest_predictions.Naive, variant.backtest_predictions.Naive)
    assert np.isfinite(variant.all_models[["Naive-lo-95", "Naive-hi-95"]]).all().all()
    assert not interval_support(variant, "A", "Naive", 95, stage="future")
    variant.to_directory(tmp_path / "run")
    assert (tmp_path / "run" / "audit" / "interval_calibration.json").is_file()
    comparisons = pd.read_csv(tmp_path / "run" / "appendix" / "interval_comparison.csv")
    assert not comparisons.empty
    assert comparisons.scoring_version.eq(2).all()
    assert comparisons.aggregation.eq("observation_mean").all()
    from nixtla_scaffold.workbench import dispatch
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dispatch(workspace, "initialize", {})
    imported = dispatch(workspace, "import_run", {"path": str(tmp_path / "run")})
    for name in ("interval_comparison", "interval_support"):
        table = dispatch(workspace, "evidence", {"runId": imported["id"], "name": f"appendix/{name}.csv", "series": "A"})
        assert table["available"] and table["rows"] and table["seriesScope"] == "series"


def test_exact_frozen_blend_residuals_and_level_specific_support(optional_backend):
    from nixtla_scaffold.blend import BlendRecipe
    from nixtla_scaffold.workbench_confirmation import _experimental_blend_intervals, _radii, _apply_bounds
    records = []
    for i in range(21):
        cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 2)
        for h in (1, 2):
            records.append({"unique_id": "A", "cutoff": cutoff, "ds": cutoff + pd.Timedelta(days=h),
                            "horizon_step": h, "y": 100., "Blend": 100. + h + i / 10,
                            "partition": "calibration" if i < 20 else "confirmation"})
    backtest = pd.DataFrame(records)
    calibration, confirmation = backtest.iloc[:-2], backtest.iloc[-2:]
    radii = _radii(calibration, (80, 95))
    confirmation = _apply_bounds(confirmation, radii, (80, 95))
    origin = backtest.ds.max()
    future = pd.DataFrame({"unique_id": "A", "ds": pd.date_range(origin, periods=3)[1:],
                           "horizon_step": [1, 2], "yhat": [103., 104.], "model": "Blend"})
    spec = ForecastSpec(horizon=2, freq="D", interval_calibration=IntervalCalibrationSpec(method="nexcp"))
    scored, projected, audit = _experimental_blend_intervals(
        calibration, confirmation, future, spec, BlendRecipe(members=("Naive", "Drift")))
    pd.testing.assert_series_equal(scored.Blend, confirmation.Blend)
    pd.testing.assert_series_equal(projected.yhat, future.yhat)
    assert scored["Blend-lo-95"].isna().all()
    assert np.isfinite(scored["Blend-lo-80"]).all()
    assert np.isfinite(projected["yhat_lo_95"]).all()
    assert audit["scope"] == "frozen_blend_only"
    assert audit["comparison"]
    for record in audit["records"]:
        assert len(record["state"]["windows"]) == 21
        confirmed = [r for r in record["receipts"] if r["stage"] == "backtest" and r["cutoff"] == confirmation.cutoff.iloc[0].isoformat()]
        assert all(r["calibration"]["windows"] == 20 for r in confirmed)
