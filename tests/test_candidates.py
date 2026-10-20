from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.candidates import CandidateRecipe, contextualize_evidence, merge_evidence, parameter_value
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.models import _statsforecast_model_factories, forecast_with_statsforecast
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec, TransformSpec


def history(n=40):
    t = np.arange(n)
    return pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2020-01-01", periods=n, freq="MS"),
                         "y": 100 + 2 * t + 3 * np.sin(t * np.pi / 2)})


def test_native_structure_records_actual_fitted_ets_not_inferred_alias():
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    data = history()
    spec = ForecastSpec(freq="MS", season_length=4, horizon=3, model_allowlist=("AutoETS",), levels=(),
                        evaluation=EvaluationPolicy(strategy="none"))
    result = forecast_with_statsforecast(data, profile_dataset(data, spec), spec)
    direct = StatsForecast(models=[AutoETS(season_length=4, alias="AutoETS")], freq="MS")
    direct.fit(data)
    evidence = result.model_policy_resolution["candidate_evidence"]
    row = evidence["fits"][0]
    state = direct.fitted_[0, 0].model_
    assert row["fitted_structure"]["method"] == state["method"]
    assert row["fitted_structure"]["components"] == state["components"]
    assert row["fitted_structure"]["aicc"] == pytest.approx(state["aicc"])
    recipe = evidence["recipes"][row["recipe_id"]]
    assert recipe["estimator"]["parameters"]["model"] == "ZZZ"
    assert recipe["estimator"]["parameters"]["season_length"] == 4
    json.dumps(evidence, allow_nan=False)


def test_recipe_context_survives_merges_and_changes_identity():
    base = CandidateRecipe("A", "baseline", {"algorithm": "A"}, {"horizon": 3})
    evidence = {"recipes": {base.fingerprint: base.to_dict()},
                "fits": [{"model": "A", "recipe_id": base.fingerprint}]}
    revised = contextualize_evidence(evidence, {"training_window": 24})
    merged = merge_evidence(evidence, revised)
    assert len(merged["recipes"]) == 2
    assert merged["fits"][1]["recipe_id"] != base.fingerprint
    assert base.to_dict()["execution"] == {"horizon": 3}
    assert contextualize_evidence(evidence, {"training_window": 24}) == revised


@pytest.mark.parametrize("models", [("AutoETS", "Ridge"), ("Naive", "RandomWalkWithDrift")])
def test_actual_origin_recipes_retain_transform_window_and_ml_parameters(models):
    spec = ForecastSpec(freq="MS", season_length=4, horizon=3, model_allowlist=models, levels=(),
                        transform=TransformSpec(target="log1p"),
                        evaluation=EvaluationPolicy(max_windows=2, training_window=28))
    run = run_forecast(history(), spec)
    evidence = run.model_policy_resolution["candidate_evidence"]
    assert {row["stage"] for row in evidence["fits"]} == {"future", "backtest"}
    assert all(row["status"] == "succeeded" for row in evidence["fits"])
    for row in evidence["fits"]:
        recipe = evidence["recipes"][row["recipe_id"]]
        assert recipe["execution"]["training_window"] == 28
        assert recipe["execution"]["target_transform"]["target"] == "log1p"
        assert recipe["execution"]["point_semantics"].startswith("inverse_of_transformed")
    if "Ridge" in models:
        ridge = next(row for row in evidence["fits"] if row["model"] == "Ridge")
        assert "coef_" in ridge["fitted_structure"]
        assert ridge["pooling"].startswith("one_global")
    json.dumps(evidence, allow_nan=False)


def test_new_ladder_considers_ces_tbats_and_damped_trend_without_changing_legacy():
    data = history()
    spec = ForecastSpec(freq="MS", season_length=4, horizon=3)
    profile = profile_dataset(data, spec)
    new = dict(_statsforecast_model_factories(data, profile, spec))
    old = dict(_statsforecast_model_factories(data, profile, replace(spec, run_mode="legacy", policy_version=0)))
    assert {"AutoCES", "AutoTBATS", "AutoETS_Damped", "AutoETS_Undamped"} <= new.keys()
    assert not {"AutoCES", "AutoTBATS", "AutoETS_Damped", "AutoETS_Undamped"} & old.keys()
    assert new["AutoETS_Damped"]().damped is True
    assert new["AutoETS_Undamped"]().damped is False


def test_nonfinite_and_unsupported_parameter_values_are_explicit():
    assert parameter_value(float("inf")) == {"nonfinite": "inf"}
    assert "unavailable" in parameter_value(object())
    assert parameter_value(np.int64(42)) == 42
    assert parameter_value(np.zeros(300))["length"] == 300


@pytest.mark.parametrize("alias", ["MSTL", "AutoARIMA_MSTLFeatures", "StatsSklearn_Ridge",
                                  "AutoETS_Damped", "AutoETS_Undamped"])
def test_special_and_explicit_trend_adapters_record_real_parameters(alias):
    spec = ForecastSpec(freq="MS", season_length=4, horizon=3, model_allowlist=(alias,), levels=(80, 95),
                        evaluation=EvaluationPolicy(max_windows=1))
    run = run_forecast(history(), spec)
    evidence = run.model_policy_resolution["candidate_evidence"]
    fits = [row for row in evidence["fits"] if row["model"] == alias]
    assert {row["stage"] for row in fits} == {"future", "backtest"}
    assert all(row["status"] == "succeeded" for row in fits)
    assert all("class" in evidence["recipes"][row["recipe_id"]]["estimator"] for row in fits)
    if alias in {"AutoETS_Damped", "AutoETS_Undamped"}:
        assert all(row["fitted_structure"]["method"].startswith("ETS(") for row in fits)
    json.dumps(evidence, allow_nan=False)


def test_small_series_automatic_ml_ladder_is_regularized_not_a_large_learner_grid():
    from nixtla_scaffold.models import forecast_with_mlforecast
    data = history(30)
    spec = ForecastSpec(freq="MS", season_length=4, horizon=3, levels=(),
                        evaluation=EvaluationPolicy(strategy="none"))
    result = forecast_with_mlforecast(data, profile_dataset(data, spec), spec)
    assert "Ridge" in result.forecast
    assert "RandomForest" not in result.forecast and "HistGradientBoosting" not in result.forecast
    assert any("effective lagged training rows" in message for message in result.warnings)
    omitted = [row for row in result.model_policy_resolution["candidate_evidence"]["fits"]
               if row["model"] == "RandomForest"]
    assert omitted and all(row["status"] == "ineligible" for row in omitted)
    assert all("effective lagged training rows" in row["diagnostics"][0] for row in omitted)
    assert all("class" in result.model_policy_resolution["candidate_evidence"]["recipes"][row["recipe_id"]]["estimator"]
               for row in omitted)


def test_saved_fit_table_retains_constructor_and_interval_receipts(tmp_path):
    from nixtla_scaffold.outputs import build_candidate_fits
    run = run_forecast(history(), ForecastSpec(
        freq="MS", season_length=4, horizon=3, model_allowlist=("AutoETS",), levels=(),
        evaluation=EvaluationPolicy(max_windows=1),
    ))
    table = build_candidate_fits(run)
    assert {"constructor", "execution", "fitted_method", "calibration_windows"} <= set(table)
    assert table.fitted_method.str.startswith("ETS(").all()
    assert table.interval_method.eq("point_forecast_only").all()
    run.to_directory(tmp_path / "saved")
    manifest = json.loads((tmp_path / "saved" / "manifest.json").read_text())
    for key in ("candidate_fits", "candidate_evidence", "ensemble_comparisons"):
        assert (tmp_path / "saved" / manifest["outputs"][key]).is_file()
