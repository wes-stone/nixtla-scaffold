from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.anomalies import anomaly_adjustment_spec, evidence_frames, review_training
from nixtla_scaffold.calendar_features import holiday_dates, native_calendar_features
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.schema import AnomalySpec, CalendarSpec, EvaluationPolicy, ForecastSpec, TransformSpec, forecast_spec_from_dict
from nixtla_scaffold.transformations import prepare_modeling_target


def sample(n=210):
    t = np.arange(n)
    return pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2024-01-01", periods=n),
                         "y": (100 + .1 * t) * np.where(t % 7 < 5, 1, .5) + np.sin(t) * .3})


def settings(**kwargs):
    return ForecastSpec(horizon=7, freq="D", model_policy="baseline", levels=(), verbose=False,
                        weighted_ensemble=False, evaluation=EvaluationPolicy(max_windows=3), **kwargs)


def review(data, spec=None):
    spec = spec or settings()
    transformed, _, _ = prepare_modeling_target(data, spec.transform)
    return review_training(transformed, spec, profile_dataset(transformed, spec), stage="future", observed=transformed)


@pytest.mark.parametrize("values", [
    {"mode": "magic"}, {"version": True}, {"lookback": 7}, {"lookback": True},
    {"threshold": float("nan")}, {"threshold": True}, {"maximum_adjusted_fraction": .5},
    {"minimum_relative_deviation": 0},
])
def test_invalid_anomaly_policy(values):
    with pytest.raises(ValueError):
        AnomalySpec(**values)


def test_typed_policy_roundtrip_and_old_saved_recipes():
    spec = settings(calendar=CalendarSpec(holiday_calendar="custom", custom_dates=("2024-07-04",),
                                         known_as_of="2023-01-01"))
    assert forecast_spec_from_dict(spec.to_dict()) == spec
    old = spec.to_dict()
    old.pop("anomaly")
    old.pop("calendar")
    assert forecast_spec_from_dict(old).anomaly.mode == "off"
    for key in ("calendar", "anomaly"):
        with pytest.raises(ValueError):
            forecast_spec_from_dict({**old, key: {"typo": True}})
        with pytest.raises(ValueError):
            forecast_spec_from_dict({**old, key: None})


def test_detection_does_not_change_data_and_adjustment_is_modeling_only():
    data = sample()
    data.loc[150, "y"] *= .1
    original = data.copy(deep=True)
    unchanged, report = review(data)
    pd.testing.assert_frame_equal(unchanged, original)
    point = next(row for row in report["points"] if row["ds"] == data.ds.iloc[150].isoformat())
    assert point["flagged"] and point["eligible"] and not point["applied"]
    assert report["series"][0]["eligible"] == 1
    adjusted, applied = review(data, anomaly_adjustment_spec(settings()))
    assert adjusted.y.iloc[150] > 100
    pd.testing.assert_frame_equal(data, original)
    assert applied["series"][0]["adjusted"] == 1
    assert json.dumps(applied, allow_nan=False)


def test_weekends_and_growth_are_not_outliers():
    _, report = review(sample())
    assert report["series"][0]["flagged"] == 0


@pytest.mark.parametrize("boundary", ["is_month_start", "is_month_end"])
def test_recurring_month_boundary_dips_are_not_cleaned(boundary):
    data = sample(350)
    data.loc[getattr(data.ds.dt, boundary), "y"] *= .2
    changed, report = review(data, anomaly_adjustment_spec(settings()))
    pd.testing.assert_frame_equal(changed, data)
    assert any(row["reason"] == "recurrent_monthly_pattern" for row in report["points"])


@pytest.mark.parametrize("positions, reason", [
    ([207], "unresolved_tail"), ([150, 151, 152, 153], "sustained_episode_or_level_shift"),
    ([120, 127, 134, 141], "recurrent_weekday_pattern"),
])
def test_ambiguous_events_are_protected(positions, reason):
    data = sample()
    data.loc[positions, "y"] *= .05
    adjusted, report = review(data, anomaly_adjustment_spec(settings()))
    assert any(row["reason"] == reason for row in report["points"])
    pd.testing.assert_frame_equal(data, adjusted)


def test_declared_holiday_and_availability():
    data = sample()
    date = data.ds.iloc[150].date().isoformat()
    data.loc[150, "y"] *= .1
    calendar = CalendarSpec(holiday_calendar="custom", custom_dates=(date,), known_as_of="2023-01-01")
    _, report = review(data, anomaly_adjustment_spec(settings(calendar=calendar)))
    assert report["series"][0]["adjusted"] == 0
    assert any(row["reason"] == "declared_holiday_or_adjacent_day" for row in report["points"])
    assert holiday_dates(replace(calendar, known_as_of="2025-01-01"), data.ds.min(), data.ds.max(), data.ds.max()).empty
    with pytest.raises(ValueError, match="known_as_of"):
        CalendarSpec(holiday_calendar="custom", custom_dates=(date,))


@pytest.mark.parametrize("target", ["none", "log", "log1p"])
def test_transform_and_normalization_scale_are_preserved(target):
    data = sample()
    data.loc[150, "y"] *= .1
    data["factor"] = 2.
    spec = anomaly_adjustment_spec(settings(transform=TransformSpec(target=target, normalization_factor_col="factor")))
    adjusted, report = review(data, spec)
    point = next(row for row in report["points"] if row["ds"] == data.ds.iloc[150].isoformat())
    assert point["observed_y"] == pytest.approx(data.y.iloc[150] / 2)
    assert point["output_scale"] == "normalized_units"
    expected, _, _ = prepare_modeling_target(data.assign(y=data.y.mask(data.index == 150, point["modeling_y"] * 2)), spec.transform)
    np.testing.assert_allclose(adjusted.y, expected.y)


def test_native_features_are_future_known_and_not_calendar_multipliers():
    spec = settings(calendar=CalendarSpec(enabled=True, holiday_calendar="us_federal"))
    features = native_calendar_features(sample(), spec, "D")
    holiday = next(feature for feature in features if getattr(feature, "__name__", "") == "holiday")
    from nixtla_scaffold.candidates import parameter_value
    assert parameter_value(holiday)["feature"] == "holiday"
    np.testing.assert_array_equal(holiday(pd.DatetimeIndex(["2024-07-04", "2024-07-05"])), [1, 0])
    with pytest.raises(ValueError, match="daily"):
        native_calendar_features(sample(), spec, "MS")


def test_origin_local_adjustments_never_change_actuals_or_future_prefixes(tmp_path):
    data = sample()
    data.loc[150, "y"] *= .1
    spec = anomaly_adjustment_spec(settings())
    run = run_forecast(data, spec)
    pd.testing.assert_series_equal(run.history.y.reset_index(drop=True), data.y, check_names=False)
    actuals = run.backtest_predictions.merge(data, on=["unique_id", "ds"], suffixes=("", "_source"))
    np.testing.assert_array_equal(actuals.y, actuals.y_source)
    report = run.model_policy_resolution["anomalies"]
    assert len(report["prefixes"]) == 4
    for prefix in report["prefixes"]:
        assert all(pd.Timestamp(row["ds"]) <= pd.Timestamp(row["cutoff"]) for row in prefix["points"])
    directory = run.to_directory(tmp_path / "run")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert (directory / manifest["outputs"]["anomaly_points"]).is_file()
    assert not evidence_frames(report)["points"].empty


def test_future_values_cannot_change_a_completed_origin():
    data = sample()
    cutoff = data.ds.iloc[180]
    prefix = data[data.ds <= cutoff]
    _, before = review(prefix)
    data.loc[data.ds > cutoff, "y"] *= 1000
    _, after = review(data[data.ds <= cutoff])
    assert before["points"] == after["points"]


def test_legacy_adjustment_fails_instead_of_leaking_full_frame_cleaning():
    with pytest.raises(ValueError, match="rolling-origin"):
        run_forecast(sample(), replace(settings(anomaly=AnomalySpec(mode="adjust")),
                                       evaluation=EvaluationPolicy(strategy="none")))


def test_native_ml_holiday_features_execute_and_keep_finite_paths():
    spec = replace(settings(calendar=CalendarSpec(enabled=True, holiday_calendar="us_federal")),
                   model_policy="mlforecast", model_allowlist=("Ridge",),
                   evaluation=EvaluationPolicy(max_windows=2))
    run = run_forecast(sample(), spec)
    assert np.isfinite(run.all_models["Ridge"]).all()
    recipes = run.model_policy_resolution["candidate_evidence"]["recipes"].values()
    features = [name for recipe in recipes for name in recipe.get("execution", {}).get("actual_feature_names", [])]
    assert any("holiday" in name for name in features)
    assert len(run.backtest_predictions) == 14


def test_automatic_diagnostic_hypotheses_replace_not_expand_improvement_slots():
    from nixtla_scaffold.analysis import _diagnostic_hypotheses
    data = sample()
    data.loc[150, "y"] *= .1
    run = run_forecast(data, settings())
    choices = _diagnostic_hypotheses(run, [{"policy": "equal"}, {"policy": "top_k"}])
    assert len(choices) == 2 and choices[0]["variant"] == "anomaly_adjustment"
    assert choices[1]["policy"] == "equal"
    assert choices[0]["spec"]["anomaly"]["mode"] == "adjust"


def test_automatic_treatment_is_persisted_unadopted_with_matched_actuals(tmp_path):
    from nixtla_scaffold.analysis import analyze
    from nixtla_scaffold.workbench_store import WorkbenchStore
    data = sample()
    data.loc[150, "y"] *= .1
    job = analyze(data, settings(), workspace=tmp_path, max_wall_clock_seconds=120)
    result = job["result"]
    assert job["status"] == "completed" and len(result["improvements"]) == 2
    trial = result["improvements"][0]
    receipt = trial["receipt"]
    assert receipt["variant"] == "anomaly_adjustment"
    assert receipt["exact_match"] and receipt["actual_mismatch_rows"] == 0
    assert not receipt["promotion_recommended"] and not receipt["adopted"]
    store = WorkbenchStore(tmp_path)
    assert store.read_artifact(trial["runId"], "analysis_pair.json")


def test_cli_new_flags_and_saved_override_helpers():
    from argparse import Namespace
    from nixtla_scaffold.cli import _anomaly_from_args, _calendar_from_args
    assert _anomaly_from_args(Namespace(clean_outliers=True), AnomalySpec(mode="off")).mode == "adjust"
    assert _anomaly_from_args(Namespace(anomaly_mode="off"), AnomalySpec()).mode == "off"
    assert _calendar_from_args(Namespace(enhanced_calendar=True), CalendarSpec()).enabled
    assert settings(anomaly=AnomalySpec(mode="adjust")).interval_calibration.method == "split_conformal"


def test_sustained_level_and_weekend_regimes_are_not_cleaned_away():
    data = sample(280)
    data.loc[data.index >= 210, "y"] *= 1.8
    changed, report = review(data, anomaly_adjustment_spec(settings()))
    pd.testing.assert_frame_equal(data, changed)
    data = sample(280)
    data.loc[(data.index >= 210) & (data.ds.dt.dayofweek >= 5), "y"] *= 2
    changed, report = review(data, anomaly_adjustment_spec(settings()))
    pd.testing.assert_frame_equal(data, changed)


def test_imputed_point_is_not_reclassified_as_a_measurement_error():
    data = sample()
    data.loc[150, "y"] *= .05
    observed = data.copy()
    observed.loc[150, "y"] = np.nan
    spec = anomaly_adjustment_spec(settings())
    changed, report = review_training(data, spec, profile_dataset(data, spec), stage="future", observed=observed)
    pd.testing.assert_frame_equal(changed, data)
    assert any(row["reason"] == "structural_repair_not_anomaly" for row in report["points"])


def test_adjusted_intervals_use_original_outer_targets_and_require_prior_support():
    data = sample(350)
    data.loc[[160, 230], "y"] *= .1
    spec = replace(settings(anomaly=AnomalySpec(mode="adjust")), levels=(80, 95),
                   evaluation=EvaluationPolicy(max_windows=22, step_size=7))
    run = run_forecast(data, spec)
    cv = run.backtest_predictions
    intervals = [column for column in cv if "-lo-" in column or "-hi-" in column]
    assert intervals
    assert cv.loc[cv.cutoff == cv.cutoff.min(), intervals].isna().all().all()
    assert np.isfinite(run.all_models[[name for name in intervals if name in run.all_models]]).all().all()
    source = data.set_index("ds").y
    np.testing.assert_array_equal(cv.y, cv.ds.map(source))
