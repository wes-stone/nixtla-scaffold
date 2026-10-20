from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from nixtla_scaffold.drivers import DriverFeatureBundle
from nixtla_scaffold.models import _mlforecast_cross_validation_resilient, _mlforecast_fit_predict_resilient
from nixtla_scaffold.schema import FeatureRecipeSpec, ForecastSpec, forecast_spec_from_dict
from nixtla_scaffold.pipelines import _forecast_spec_from_config
from nixtla_scaffold.workbench import _safe_spec
from test_runtime_contracts import _cli_spec


def test_calendar_option_roundtrips_across_entrypoints(monkeypatch):
    original = ForecastSpec()
    assert original.feature_recipe.date_features_as_dummies is False
    configured = replace(original, feature_recipe=FeatureRecipeSpec(date_features_as_dummies=True))
    assert forecast_spec_from_dict(configured.to_dict()) == configured
    assert _safe_spec({"feature_recipe": {"date_features_as_dummies": True}}).feature_recipe == configured.feature_recipe
    assert _forecast_spec_from_config({"feature_recipe": {"date_features_as_dummies": True}}).feature_recipe == configured.feature_recipe
    for command in ("forecast", "compare-models", "experiment", "optimize", "ingest"):
        assert _cli_spec(monkeypatch, command, "--date-features-as-dummies").feature_recipe.date_features_as_dummies
    with pytest.raises(ValueError, match="boolean"):
        FeatureRecipeSpec(date_features_as_dummies="false")


@pytest.fixture
def history():
    dates = pd.date_range("2024-12-01", periods=90, freq="D")
    return pd.DataFrame({"unique_id": "calendar", "ds": dates, "y": 100 + np.arange(90) * .1 + 20 * (dates.dayofweek == 1)})


def test_real_calendar_encoding_and_future_perturbation(history):
    args = dict(freq="D", lags=[1, 7], date_features=["month", "dayofweek"], horizon=7,
                levels=(), prediction_intervals_factory=None, lag_transforms=None,
                exogenous_features=DriverFeatureBundle(), date_features_as_dummies=True)
    forecast, features, _ = _mlforecast_fit_predict_resilient([("Ridge", Ridge)], history=history, **args)
    assert len(forecast) == 7 and np.isfinite(forecast["Ridge"]).all()
    names = set(features["feature"])
    assert {"month_1", "dayofweek_0", "dayofweek_6"}.issubset(names)
    assert not {"month", "dayofweek"}.intersection(names)
    assert forecast.attrs["candidate_evidence"]
    cv, _, _ = _mlforecast_cross_validation_resilient(
        [("Ridge", Ridge)], history=history, n_windows=3, step_size=7, **args)
    changed = history.copy()
    first_cutoff = cv.cutoff.min()
    changed.loc[changed.ds > first_cutoff, "y"] += 100000
    replay, _, _ = _mlforecast_cross_validation_resilient(
        [("Ridge", Ridge)], history=changed, n_windows=3, step_size=7, **args)
    np.testing.assert_allclose(
        cv.loc[cv.cutoff == first_cutoff, "Ridge"], replay.loc[replay.cutoff == first_cutoff, "Ridge"])
    assert len(cv) == 21 and cv.cutoff.nunique() == 3


@pytest.mark.parametrize("validation", [False, True])
def test_calendar_encoding_survives_point_only_fallback(monkeypatch, history, validation):
    import mlforecast
    real = mlforecast.MLForecast
    constructed = []

    class FailIntervals(real):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs.get("date_features_as_dummies"))
            super().__init__(*args, **kwargs)

        def fit(self, *args, **kwargs):
            if kwargs.get("prediction_intervals") is not None:
                raise ValueError("deliberate interval failure")
            return super().fit(*args, **kwargs)

        def cross_validation(self, *args, **kwargs):
            if kwargs.get("prediction_intervals") is not None:
                raise ValueError("deliberate interval failure")
            return super().cross_validation(*args, **kwargs)

    monkeypatch.setattr(mlforecast, "MLForecast", FailIntervals)
    args = dict(history=history, freq="D", lags=[1, 7], date_features=["month"], horizon=7,
                levels=(80,), prediction_intervals_factory=object, lag_transforms=None,
                exogenous_features=DriverFeatureBundle(), date_features_as_dummies=True)
    if validation:
        result, warnings, _ = _mlforecast_cross_validation_resilient(
            [("Ridge", Ridge)], n_windows=2, step_size=7, **args)
    else:
        result, _, warnings = _mlforecast_fit_predict_resilient([("Ridge", Ridge)], **args)
    assert np.isfinite(result["Ridge"]).all()
    assert len(constructed) == 3 and all(constructed)
    assert any("point" in warning for warning in warnings)
