from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.native_fitted import fitted_diagnostics, fitted_frames
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec, TransformSpec


def history(n=210):
    t = np.arange(n)
    return pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2024-01-01", periods=n),
                         "y": 100 + .2 * t + 15 * np.sin(2 * np.pi * t / 7)})


def test_native_fitted_flags_match_documented_statsforecast_path():
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL

    data = history()
    data.loc[150, "y"] += 100
    direct = StatsForecast(models=[MSTL(season_length=7)], freq="D")
    future = direct.forecast(df=data, h=7, fitted=True, level=[99])
    expected = direct.forecast_fitted_values().tail(182)
    fitted = MSTL(season_length=7).fit(data.y.to_numpy())
    before = fitted.predict(7)["mean"].copy()
    report = fitted_diagnostics(fitted, data)
    actual = pd.DataFrame(report["points"])
    np.testing.assert_allclose(actual.fitted_y, expected.MSTL)
    np.testing.assert_allclose(actual.lower_99, expected["MSTL-lo-99"])
    np.testing.assert_allclose(actual.upper_99, expected["MSTL-hi-99"])
    np.testing.assert_array_equal(actual.flagged, ~expected.y.between(expected["MSTL-lo-99"], expected["MSTL-hi-99"]))
    np.testing.assert_array_equal(before, fitted.predict(7)["mean"])
    np.testing.assert_allclose(before, future.MSTL)
    assert report["method"] == "predict_in_sample" and report["review_rows"] == 182
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("target", ["none", "log1p"])
def test_saved_native_evidence_is_bounded_origin_local_and_in_reporting_units(tmp_path, target):
    data = history(112)
    settings = ForecastSpec(
        horizon=7, freq="D", season_length=7, model_allowlist=("AutoETS",), levels=(),
        transform=TransformSpec(target=target), weighted_ensemble=False,
        evaluation=EvaluationPolicy(max_windows=2, step_size=7), verbose=False,
    )
    run = run_forecast(data, settings)
    frames = fitted_frames(run)
    np.testing.assert_allclose(frames["points"].training_y, data.y, rtol=1e-12)
    assert frames["points"].output_scale.eq("raw_units").all()
    assert frames["points"].stage.eq("future").all()
    assert frames["summary"].stage.eq("backtest").sum() == 2
    fits = run.model_policy_resolution["candidate_evidence"]["fits"]
    for fit in fits:
        if fit["stage"] == "backtest":
            assert fit["native_fitted"]["points"] == []
        assert fit["configuration_admission"]["cutoff"] == fit["cutoff"]
    assert not any("-lo-" in column for column in run.all_models)
    run.to_directory(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for kind in ("points", "summary"):
        path = tmp_path / manifest["outputs"][f"native_fitted_{kind}"]
        assert path.is_file()
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype={"unique_id": str}, keep_default_na=False),
                                      frames[kind].fillna(""), check_dtype=False, check_exact=False)


def test_native_unavailable_and_invalid_output_are_explicit():
    data = history(10)
    assert fitted_diagnostics(object(), data)["status"] == "unavailable"

    class Unsupported:
        def predict_in_sample(self, level):
            raise NotImplementedError

    assert "does not implement" in fitted_diagnostics(Unsupported(), data)["reason"]

    class Broken:
        def predict_in_sample(self, level):
            return {"fitted": [1.], "fitted-lo-99": [0.], "fitted-hi-99": [2.]}

    with pytest.raises(ValueError, match="length"):
        fitted_diagnostics(Broken(), data)


def test_native_initial_missing_values_remain_explicit_and_json_safe():
    from statsforecast.models import Naive

    data = history(10)
    report = fitted_diagnostics(Naive().fit(data.y.to_numpy()), data)
    assert report["finite_rows"] < report["review_rows"]
    assert report["points"][0]["valid"] is False
    assert report["points"][0]["flagged"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("name,status", [("SeasonalWindowAverage", "unavailable"),
                                        ("WindowAverage", "unavailable"), ("CrostonClassic", "available")])
def test_native_methods_with_differing_fitted_apis_remain_forecast_candidates(name, status):
    settings = ForecastSpec(horizon=7, freq="D", season_length=7, model_allowlist=(name,), levels=(),
                            weighted_ensemble=False, evaluation=EvaluationPolicy(max_windows=1), verbose=False)
    data = history(112)
    if name == "CrostonClassic":
        data.loc[data.index % 4 != 0, "y"] = 0.
    run = run_forecast(data, settings)
    assert run.forecast.model.eq(name).all()
    assert np.isfinite(run.forecast.yhat).all()
    summary = fitted_frames(run)["summary"]
    assert summary.status.eq(status).all()
    if status == "unavailable":
        assert summary.reason.str.len().gt(0).all()
    else:
        assert summary.level.eq(99).all()


def test_normalized_fitted_evidence_uses_training_copy_and_does_not_rewrite_actuals():
    data = history(112)
    data["units"] = 2.
    run = run_forecast(data, ForecastSpec(
        horizon=7, freq="D", season_length=7, model_allowlist=("AutoETS",), levels=(),
        weighted_ensemble=False, evaluation=EvaluationPolicy(strategy="none"),
        transform=TransformSpec(normalization_factor_col="units"), verbose=False,
    ))
    points = fitted_frames(run)["points"]
    assert points.output_scale.eq("normalized_units").all()
    np.testing.assert_allclose(points.training_y, data.y / 2)
    np.testing.assert_allclose(run.history.y, data.y / 2)
    np.testing.assert_allclose(data.y, history(112).y)
