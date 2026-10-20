"""Frozen adapter qualification; deliberately independent of new search policies."""
from importlib import metadata

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.models import forecast_with_mlforecast, forecast_with_statsforecast
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.schema import ForecastSpec


STACK = {"statsforecast": "2.1.1", "mlforecast": "1.1.0",
         "scikit-learn": "1.7.2", "lightgbm": "4.7.0"}


def _history(n=36):
    t = np.arange(n, dtype=float)
    y = 100 + 0.7 * t + 4 * np.sin(t * np.pi / 2) + np.random.default_rng(184).normal(0, 0.3, n)
    return pd.DataFrame({"unique_id": ["A"] * n,
                         "ds": pd.date_range("2020-01-01", periods=n, freq="MS"), "y": y})


def _assert_predictions_equal(actual, expected, keys):
    columns = list(expected.columns)
    left = actual[columns].sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-7, atol=1e-8)
    points = [col for col in columns if col not in {*keys, "y"}]
    assert np.isfinite(left[points].to_numpy(dtype=float)).all()


def test_required_accuracy_stack_is_installed_without_dependency_skips():
    for name, version in STACK.items():
        assert metadata.version(name) == version
    import lightgbm
    import mlforecast
    import sklearn
    assert lightgbm.LGBMRegressor and mlforecast.MLForecast and sklearn.__version__


def test_source_identity_tracks_uncommitted_code_and_packaged_lock(tmp_path):
    from nixtla_scaffold.schema import _runtime_source_identity
    module = tmp_path / "engine.py"
    module.write_text("value = 1\n", encoding="utf-8")
    lock = tmp_path / "runtime.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    before = _runtime_source_identity(tmp_path)
    module.write_text("value = 2\n", encoding="utf-8")
    after = _runtime_source_identity(tmp_path)
    assert before["sha256"] != after["sha256"]
    assert before["lockfile_sha256"] == after["lockfile_sha256"]
    lock.write_text("version = 2\n", encoding="utf-8")
    assert _runtime_source_identity(tmp_path)["lockfile_sha256"] != before["lockfile_sha256"]


@pytest.mark.parametrize("alias", ["Naive", "AutoETS", "AutoARIMA", "AutoCES", "AutoTBATS", "MSTL"])
def test_statsforecast_adapter_matches_direct_native_points_bounds_and_origins(alias, record_property):
    from statsforecast import StatsForecast, models as native
    from statsforecast.utils import ConformalIntervals

    history = _history()
    spec = ForecastSpec(horizon=3, freq="MS", season_length=4, model_allowlist=(alias,),
                        weighted_ensemble=False, verbose=False, strict_cv_horizon=True,
                        run_mode="legacy", policy_version=0, paired_validation=False)
    result = forecast_with_statsforecast(history, profile_dataset(history, spec), spec)
    args = {} if alias == "Naive" else {"season_length": 4}
    args["alias"] = alias
    direct = StatsForecast(models=[getattr(native, alias)(**args)], freq="MS", n_jobs=1)
    intervals = {"level": [80, 95], "prediction_intervals": ConformalIntervals(h=3, n_windows=3)}
    future = direct.forecast(df=history, h=3, **intervals)
    cv = direct.cross_validation(df=history, h=3, n_windows=3, step_size=3, refit=True, **intervals)
    _assert_predictions_equal(result.forecast, future, ["unique_id", "ds"])
    _assert_predictions_equal(result.backtest_predictions, cv, ["unique_id", "cutoff", "ds"])
    assert result.backtest_predictions["cutoff"].nunique() == 3
    assert not any("failed" in warning or "point forecast only" in warning for warning in result.warnings)
    for level in (80, 95):
        assert f"{alias}-lo-{level}" in future and f"{alias}-hi-{level}" in future
    record_property("statsforecast_version", metadata.version("statsforecast"))
    record_property("constructor", repr(args))
    record_property("forecast_points", future[alias].tolist())
    errors = cv["y"].to_numpy() - cv[alias].to_numpy()
    record_property("retrospective_rmse", float(np.sqrt(np.mean(errors ** 2))))


@pytest.mark.parametrize("alias", ["Ridge", "LightGBM"])
@pytest.mark.parametrize("with_intervals", [False, True])
def test_ml_adapter_matches_independent_native_configuration(alias, with_intervals):
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast
    from mlforecast.utils import PredictionIntervals
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    history = _history(96)
    levels = (80, 95) if with_intervals else ()
    spec = ForecastSpec(horizon=3, freq="MS", season_length=4, model_allowlist=(alias,), levels=levels,
                        weighted_ensemble=False, verbose=False, strict_cv_horizon=True,
                        run_mode="legacy", policy_version=0, paired_validation=False)
    model = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if alias == "Ridge"
             else LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                n_jobs=1, verbose=-1, random_state=42))
    direct = MLForecast(models={alias: model}, freq="MS", lags=list(range(1, 14)), date_features=["month"])
    fit_kwargs = {"prediction_intervals": PredictionIntervals(n_windows=3, h=3)} if with_intervals else {}
    predict_kwargs = {"level": list(levels)} if with_intervals else {}
    direct.fit(history, **fit_kwargs)
    future = direct.predict(h=3, **predict_kwargs)
    cv = direct.cross_validation(history, h=3, n_windows=5, step_size=3, refit=True,
                                 **fit_kwargs, **predict_kwargs)
    result = forecast_with_mlforecast(history, profile_dataset(history, spec), spec)
    _assert_predictions_equal(result.forecast, future, ["unique_id", "ds"])
    _assert_predictions_equal(result.backtest_predictions, cv, ["unique_id", "cutoff", "ds"])
    assert not any("failed" in warning for warning in result.warnings)
