from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from nixtla_scaffold import DriverSnapshotGrant, DriverSourceContract, KnownFutureRegressor, run_forecast
from nixtla_scaffold.drivers import DriverFeatureBundle, prepare_mlforecast_regressor_features
from nixtla_scaffold.ml_features import native_transforms
from nixtla_scaffold.models import _mlforecast_fit_predict_resilient, _mlforecast_cross_validation_resilient
from nixtla_scaffold.schema import FeatureRecipeSpec, ForecastSpec, EvaluationPolicy, forecast_spec_from_dict
from test_runtime_contracts import _cli_spec


def test_advanced_recipe_roundtrip_and_entrypoints(monkeypatch):
    from nixtla_scaffold.pipelines import _forecast_spec_from_config
    from nixtla_scaffold.workbench import _safe_spec
    recipe = FeatureRecipeSpec(mlforecast_strategy="direct", horizon_features={"1": ["event"]},
                               lookup_lag_columns=("event",), pooled_rolling_window=3,
                               pooled_partition_columns=("event",), pooled_series_ids=("A", "B"))
    spec = ForecastSpec(feature_recipe=recipe)
    assert forecast_spec_from_dict(spec.to_dict()).feature_recipe == recipe
    for resolve in (_safe_spec, _forecast_spec_from_config):
        assert resolve({"feature_recipe": recipe.to_dict()}).feature_recipe == recipe
    for command in ("forecast", "compare-models", "experiment", "optimize", "ingest"):
        result = _cli_spec(monkeypatch, command, "--mlforecast-strategy", "direct",
                           "--horizon-features", '{"1":["event"]}', "--lookup-lag-column", "event",
                           "--pooled-rolling-window", "3", "--pooled-partition-column", "event",
                           "--pooled-series-id", "A", "--pooled-series-id", "B",
                           "--interval-backend", "nexcp")
        assert result.feature_recipe == recipe
        assert result.interval_calibration.method == "nexcp"


@pytest.mark.parametrize("kwargs", [
    {"mlforecast_strategy": "automatic"},
    {"horizon_features": {"1": ["event"]}},
    {"mlforecast_strategy": "direct", "horizon_features": {"0": ["event"]}},
    {"mlforecast_strategy": "direct", "horizon_features": {"1": "event"}},
    {"mlforecast_strategy": "direct", "horizon_features": {"1": ["event", "event"]}},
    {"lookup_lag_columns": ["event"]},
    {"pooled_rolling_window": 3},
    {"pooled_partition_columns": ["event"]},
])
def test_invalid_advanced_recipes_are_not_silently_changed(kwargs):
    with pytest.raises(ValueError):
        FeatureRecipeSpec(**kwargs)


def fixture(tmp_path, *, recipe=None, ids=("A",)):
    dates = pd.date_range("2024-01-01", periods=100)
    snapshot = pd.concat([pd.DataFrame({
        "unique_id": uid, "ds": dates, "value": dates.dayofweek.astype(float),
        "available_at": dates[0], "period_end": dates, "vintage": "original", "units": "calendar-code",
    }) for uid in ids], ignore_index=True)
    path = tmp_path / "calendar.csv"
    snapshot.to_csv(path, index=False)
    contract = DriverSourceContract("calendar", "D", "calendar-code", "stock", "fixture-v1",
                                    "Previously observed outcomes in the same known calendar bucket")
    driver = KnownFutureRegressor("event", availability="calendar", mode="model_candidate", source=contract)
    grant = DriverSnapshotGrant("calendar", str(path), sha256(path.read_bytes()).hexdigest(), ids)
    history = pd.concat([pd.DataFrame({
        "unique_id": uid, "ds": dates[:94],
        "y": 100 + np.arange(94) * .1 + np.sin(np.arange(94)) + index * 50,
    }) for index, uid in enumerate(ids)], ignore_index=True)
    spec = ForecastSpec(horizon=3, freq="D", season_length=7, levels=(),
                        model_policy="mlforecast", model_allowlist=("Ridge",), weighted_ensemble=False,
                        regressors=(driver,), driver_snapshots=(grant,), train_known_future_regressors=True,
                        feature_recipe=recipe or FeatureRecipeSpec(), unit_label="USD",
                        evaluation=EvaluationPolicy(max_windows=3, step_size=3))
    return history, snapshot, spec


@pytest.mark.parametrize("validation", [False, True])
def test_direct_strategy_survives_native_and_point_only_fallback(tmp_path, monkeypatch, validation):
    import mlforecast
    real = mlforecast.MLForecast
    seen = []
    class FailIntervals(real):
        def fit(self, *args, **kwargs):
            seen.append(kwargs.get("max_horizon"))
            if kwargs.get("prediction_intervals") is not None:
                raise ValueError("deliberate interval failure")
            return super().fit(*args, **kwargs)
        def cross_validation(self, *args, **kwargs):
            seen.append(kwargs.get("max_horizon"))
            if kwargs.get("prediction_intervals") is not None:
                raise ValueError("deliberate interval failure")
            return super().cross_validation(*args, **kwargs)
    monkeypatch.setattr(mlforecast, "MLForecast", FailIntervals)
    history, _, _ = fixture(tmp_path)
    args = dict(history=history, freq="D", lags=[1, 7], date_features=["dayofweek"], horizon=3,
                levels=(80,), prediction_intervals_factory=object, lag_transforms=None,
                exogenous_features=DriverFeatureBundle(),
                feature_recipe=FeatureRecipeSpec(mlforecast_strategy="direct"))
    if validation:
        output, warnings, _ = _mlforecast_cross_validation_resilient([("Ridge", Ridge)],
                                                                    n_windows=2, step_size=3, **args)
    else:
        output, _, warnings = _mlforecast_fit_predict_resilient([("Ridge", Ridge)], **args)
        recipes = output.attrs["candidate_evidence"]["recipes"].values()
        assert all(r["estimator"]["class"] == "mlforecast.direct" for r in recipes)
    assert np.isfinite(output.Ridge).all()
    assert seen and all(h == 3 for h in seen)
    assert any("point" in warning for warning in warnings)


@pytest.mark.parametrize("kind", ["horizon", "lookup", "pooled_partition"])
def test_real_advanced_features_are_origin_local(tmp_path, kind):
    recipe = {
        "horizon": FeatureRecipeSpec(mlforecast_strategy="direct", horizon_features={1: ["event"], 3: ["event"]}),
        "lookup": FeatureRecipeSpec(mlforecast_strategy="direct", lookup_lag_columns=("event",)),
        "pooled_partition": FeatureRecipeSpec(pooled_rolling_window=14, pooled_partition_columns=("event",),
                                             pooled_series_ids=("A", "B")),
    }[kind]
    data, _, spec = fixture(tmp_path, recipe=recipe, ids=("A", "B"))
    run = run_forecast(data, spec)
    assert "Ridge" in run.all_models, run.warnings
    assert np.isfinite(run.all_models.Ridge).all()
    first = run.backtest_predictions.cutoff.min()
    original = run.backtest_predictions.loc[run.backtest_predictions.cutoff.eq(first), "Ridge"]
    changed = data.copy()
    changed.loc[changed.ds.gt(first), "y"] += 1e6
    replay = run_forecast(changed, spec)
    np.testing.assert_allclose(original, replay.backtest_predictions.loc[
        replay.backtest_predictions.cutoff.eq(first), "Ridge"])
    fits = run.model_policy_resolution["candidate_evidence"]
    descriptions = str(fits["recipes"])
    assert "native_feature_recipe" in descriptions
    assert "actual_feature_names" in descriptions
    if kind == "lookup":
        assert "lookup_lag" in descriptions.lower()
    if kind == "horizon":
        explanations = run.model_explainability
        assert set(explanations.horizon_step) == {1, 2, 3}
        assert set(explanations.loc[explanations.feature.eq("event"), "horizon_step"]) == {1, 3}
    if kind == "pooled_partition":
        assert "rolling_mean" in descriptions.lower()


def test_unseen_bucket_late_release_and_invalid_panel_are_rejected(tmp_path):
    recipe = FeatureRecipeSpec(mlforecast_strategy="direct", lookup_lag_columns=("event",),
                               pooled_rolling_window=2, pooled_series_ids=("A", "B"))
    history, _, spec = fixture(tmp_path, recipe=recipe, ids=("A", "B"))
    bundle = prepare_mlforecast_regressor_features(history, spec, freq="D", horizon=3,
        season_length=7, min_training_rows=14, forecast_origin=history.ds.max())
    assert native_transforms(history, spec, bundle, None)
    for invalid, match in (
        (replace(bundle, future=bundle.future.assign(event=99)), "no observed"),
        (replace(bundle, future=bundle.future.assign(event=np.nan)), "missing or nonfinite"),
        (replace(bundle, feature_columns=()), "as-of snapshot"),
    ):
        with pytest.raises(ValueError, match=match):
            native_transforms(history, spec, invalid, None)
    with pytest.raises(ValueError, match="cohort"):
        native_transforms(history.loc[history.unique_id.eq("A")], spec, bundle, None)
    with pytest.raises(ValueError, match="aligned"):
        native_transforms(history.iloc[1:], spec, bundle, None)
    with pytest.raises(ValueError, match="unit"):
        native_transforms(history, replace(spec, unit_label=None), bundle, None)
    with pytest.raises(ValueError, match="legacy"):
        native_transforms(history, replace(spec, evaluation=EvaluationPolicy(strategy="legacy")), bundle, None)


def test_late_driver_revision_cannot_change_earlier_horizon_fit(tmp_path):
    recipe = FeatureRecipeSpec(mlforecast_strategy="direct", horizon_features={1: ["event"], 2: ["event"]})
    history, snapshot, spec = fixture(tmp_path, recipe=recipe)
    control = run_forecast(history, spec)
    first = control.backtest_predictions.cutoff.min()
    revision = snapshot.copy().assign(value=snapshot.value + 1000, vintage="revision",
                                      available_at=first + pd.Timedelta(days=1))
    path = tmp_path / "calendar.csv"
    pd.concat([snapshot, revision], ignore_index=True).to_csv(path, index=False)
    revised = replace(spec, driver_snapshots=(replace(spec.driver_snapshots[0],
                                                     sha256=sha256(path.read_bytes()).hexdigest()),))
    output = run_forecast(history, revised)
    np.testing.assert_allclose(
        control.backtest_predictions.loc[control.backtest_predictions.cutoff.eq(first), "Ridge"],
        output.backtest_predictions.loc[output.backtest_predictions.cutoff.eq(first), "Ridge"])
