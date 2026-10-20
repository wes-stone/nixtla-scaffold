from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.configuration import configuration_diagnostics, configuration_frames, seasonal_evidence
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.models import _mfles_search_grid, _statsforecast_model_factories
from nixtla_scaffold.profile import profile_dataset, season_length_for_freq
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec, ModelConfiguration, forecast_spec_from_dict


def history(n=112, changed=False):
    t = np.arange(n)
    shape = np.array([1.3, 1.4, 1.4, 1.4, 1.3, .3, .3])[t % 7]
    if changed:
        shape[t >= n // 2] = np.array([1.2, 1.3, 1.3, 1.3, 1.2, .65, .65])[t[t >= n // 2] % 7]
    return pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2024-01-01", periods=n),
                         "y": (100 + .2 * t) * shape})


def spec(**kwargs):
    return ForecastSpec(horizon=7, freq="D", season_length=7, model_policy="statsforecast",
                        model_allowlist=("MFLES", "AutoMFLES"), levels=(), verbose=False,
                        weighted_ensemble=False, evaluation=EvaluationPolicy(strategy="none"), **kwargs)


@pytest.mark.parametrize("freq, expected", [
    ("D", 7), ("B", 5), ("W-MON", 52), ("MS", 12), ("ME", 12),
    ("BMS", 12), ("BME", 12), ("QS-JAN", 4), ("h", 24), ("2h", 12),
    ("15min", 96), ("2MS", 6), ("2D", 1), ("2W", 26), ("YS", 1),
])
def test_calendar_prior_respects_units_and_multipliers(freq, expected):
    assert season_length_for_freq(freq, 1000) == expected


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "7"])
def test_invalid_season_is_not_silently_coerced(value):
    with pytest.raises(ValueError, match="season_length"):
        forecast_spec_from_dict({"season_length": value})


@pytest.mark.parametrize("settings", [
    {"seasonal_periods": [7, 7]}, {"seasonal_periods": [7, 2.5]},
    {"seasonal_periods": [7, 12, 28, 365]}, {"mfles_fourier_order": True},
    {"mfles_seasonality": "guess"}, {"automfles_metric": "monthly_landing"},
    {"automfles_test_size": 0}, {"automfles_windows": True}, {"automfles_step_size": 1.5},
])
def test_configuration_contract_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        ModelConfiguration(**settings)


def test_configuration_roundtrip_and_legacy_defaults():
    configured = spec(model_configuration=ModelConfiguration(
        seasonal_periods=(7, 28), mfles_seasonality="additive", mfles_fourier_order=3,
    ))
    assert forecast_spec_from_dict(configured.to_dict()) == configured
    assert forecast_spec_from_dict({"horizon": 7}).model_configuration == ModelConfiguration()


def test_phase_drift_uses_cycle_normalization_not_growth_in_levels():
    stable = history(224)
    accelerated = stable.assign(y=stable.y * np.repeat(np.linspace(1, 100, 32), 7))
    a, b = seasonal_evidence(stable, 7), seasonal_evidence(accelerated, 7)
    assert not a["shape_changed"] and not b["shape_changed"]
    assert a["status"] == b["status"] == "supported_heuristic"
    assert a["shape_drift"] == pytest.approx(b["shape_drift"])
    assert seasonal_evidence(history(224, changed=True), 7)["shape_changed"]


def test_missing_and_nonpositive_cycles_are_not_repaired_for_diagnostics():
    frame = history(224)
    frame.loc[10, "y"] = np.nan
    assert seasonal_evidence(frame, 7)["status"] == "missing_observations"
    frame.loc[10, "y"] = -1
    assert seasonal_evidence(frame, 7)["status"] == "relative_shape_unavailable_nonpositive_cycles"
    rows = configuration_diagnostics(frame, spec(), freq="D", season=7)
    assert json.dumps(rows, allow_nan=False)


def test_detector_is_evidence_not_a_silent_setting_override():
    frame = history(224)
    settings = replace(spec(), season_length=14)
    row = configuration_diagnostics(frame, settings, freq="D", season=14)[0]
    assert row["primary_period"] == 14 and row["period_source"] == "explicit"
    assert row["detected_period"] == 7
    assert settings.season_length == 14


def test_fixed_treatment_preserves_native_search_grid():
    from statsforecast.mfles import default_configs
    for periods in ([7], [7, 28], []):
        original = default_configs(periods or None)
        treated = default_configs(periods or None, _mfles_search_grid(periods, {"multiplicative": False}))
        assert [{k: v for k, v in row.items() if k != "multiplicative"} for row in treated] == original


@pytest.mark.parametrize("configuration", [
    ModelConfiguration(), ModelConfiguration(mfles_seasonality="additive"),
    ModelConfiguration(mfles_fourier_order=3),
    ModelConfiguration(automfles_metric="mae"), ModelConfiguration(automfles_metric="mse"),
])
def test_actual_mfles_adapter_matches_native_and_discloses_fitted_state(configuration):
    frame, settings = history(), spec(model_configuration=configuration)
    run = run_forecast(frame, settings)
    factories = dict(_statsforecast_model_factories(frame, profile_dataset(frame, settings), settings))
    for alias in settings.model_allowlist:
        native = factories[alias]().fit(frame.y.to_numpy())
        np.testing.assert_allclose(run.all_models[alias], native.predict(7)["mean"], rtol=1e-12, atol=1e-10)
        row = next(row for row in run.model_policy_resolution["configuration"]["models"] if row["model"] == alias)
        assert row["fit_status"] == "succeeded"
        assert row["internal_transform"] != "not_reported"
        if configuration.mfles_seasonality == "additive":
            assert row["fitted_multiplicative"] is False
        if alias == "AutoMFLES":
            assert row["native_config_status"] == "reconstructed_from_native_grid_and_observed_scores"
            assert row["native_search_metric"] == configuration.automfles_metric
    frames = configuration_frames(run.model_policy_resolution["configuration"])
    assert set(frames) == {"series", "models", "seasonality", "phases"}
    assert json.dumps(run.model_policy_resolution["configuration"], allow_nan=False)


def test_explicit_multiseasonal_periods_are_supported_or_excluded_never_silently_dropped():
    frame = history(112)
    settings = spec(model_configuration=ModelConfiguration(seasonal_periods=(7, 28)))
    factories = dict(_statsforecast_model_factories(frame, profile_dataset(frame, settings), settings))
    for alias in ("MSTL", "MFLES", "AutoMFLES", "AutoTBATS"):
        assert factories[alias]().season_length == [7, 28]
    short = frame.iloc[:56]
    factories = dict(_statsforecast_model_factories(short, profile_dataset(short, settings), settings))
    assert not set(factories).intersection({"MSTL", "MFLES", "AutoMFLES", "AutoTBATS"})
    assert "AutoETS" in factories


def test_aliased_explicit_fourier_order_is_rejected():
    frame = history()
    settings = spec(model_configuration=ModelConfiguration(mfles_fourier_order=4))
    with pytest.raises(ValueError, match="aliased"):
        _statsforecast_model_factories(frame, profile_dataset(frame, settings), settings)


@pytest.mark.parametrize("run_mode", ["legacy", "accuracy-first"])
def test_automfles_never_calls_an_untuned_first_configuration_an_auto_search(run_mode):
    frame = history(14)
    settings = replace(spec(), horizon=31, run_mode=run_mode, policy_version=0 if run_mode == "legacy" else 1)
    factories = dict(_statsforecast_model_factories(frame, profile_dataset(frame, settings), settings))
    assert "MFLES" in factories
    assert "AutoMFLES" not in factories


def test_configuration_treatments_preserve_control_and_do_not_broaden_families():
    from nixtla_scaffold.experiments import _variant_catalog

    control = spec()
    for name in ("mfles_additive", "mfles_identified_fourier"):
        variant = _variant_catalog()[name]
        assert variant.applicable(None, control)[0]
        treatment = variant.build_spec(control)
        assert treatment.model_allowlist == control.model_allowlist
        assert treatment.evaluation == control.evaluation
        assert treatment.feature_recipe == control.feature_recipe
        assert treatment.model_configuration != control.model_configuration
        for disallowed in (
            replace(control, model_policy="mlforecast", model_allowlist=("Ridge",)),
            replace(control, model_policy="baseline", model_allowlist=()),
            replace(control, model_allowlist=("AutoETS",)),
        ):
            assert not variant.applicable(None, disallowed)[0]


def test_configuration_report_persists_with_native_export(tmp_path):
    run = run_forecast(history(), spec())
    run.to_directory(tmp_path)
    report = json.loads((tmp_path / "audit" / "configuration.json").read_text())
    assert report == run.model_policy_resolution["configuration"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["outputs"]["configuration"] == "audit/configuration.json"
    for kind in ("series", "seasonality", "phases", "models"):
        assert manifest["outputs"][f"configuration_{kind}"] == f"audit/configuration_{kind}.csv"
        frame = pd.read_csv(tmp_path / "audit" / f"configuration_{kind}.csv", dtype={"unique_id": str})
        assert not frame.empty
        assert frame.unique_id.eq("001").all()


def test_automfles_explicit_inner_validation_is_retained():
    frame = history()
    settings = spec(model_configuration=ModelConfiguration(
        automfles_test_size=14, automfles_windows=3, automfles_step_size=14, automfles_metric="mae",
    ))
    native = dict(_statsforecast_model_factories(frame, profile_dataset(frame, settings), settings))["AutoMFLES"]()
    assert (native.test_size, native.n_windows, native.step_size, native.metric) == (14, 3, 14, "mae")


def test_each_origin_retains_only_its_own_diagnostics():
    settings = replace(spec(), model_allowlist=("AutoETS",),
                       evaluation=EvaluationPolicy(max_windows=2, step_size=7))
    run = run_forecast(history(70), settings)
    fits = run.model_policy_resolution["candidate_evidence"]["fits"]
    for fit in fits:
        diagnostic = fit["training_configuration"]
        assert pd.Timestamp(diagnostic["as_of"]) == pd.Timestamp(fit["cutoff"])
        assert diagnostic["history_rows"] <= 70


def test_profile_diagnoses_the_declared_training_window_before_modeling():
    frame = history(224, changed=True)
    settings = replace(spec(), evaluation=EvaluationPolicy(training_window=56))
    profile = profile_dataset(frame, settings)
    assert profile.rows == 224
    assert profile.configuration_diagnostics[0]["history_rows"] == 56
    assert profile.configuration_diagnostics[0]["available_history_rows"] == 224


def test_future_targets_do_not_change_historical_configurations_or_predictions():
    settings = replace(spec(), model_allowlist=("AutoETS", "SeasonalNaive"),
                       evaluation=EvaluationPolicy(max_windows=3, step_size=7))
    frame = history(70)
    changed = frame.copy()
    changed.loc[changed.index[-7:], "y"] *= 10
    first, second = run_forecast(frame, settings), run_forecast(changed, settings)
    columns = ["unique_id", "cutoff", "ds", "AutoETS", "SeasonalNaive"]
    pd.testing.assert_frame_equal(first.backtest_predictions[columns], second.backtest_predictions[columns])
    assert first.model_policy_resolution["configuration"]["series"][0]["as_of"] == frame.ds.max().isoformat()


def test_short_and_flat_histories_do_not_claim_seasonality():
    for frame in (history(14), history(112).assign(y=100.)):
        result = configuration_diagnostics(frame, spec(), freq="D", season=7)[0]
        assert result["primary_evidence"] != "supported_heuristic"
        assert not result["shape_changed"]


def test_configuration_admission_selects_an_eligible_alternative_without_clipping():
    from nixtla_scaffold.configuration import screen_candidates
    from nixtla_scaffold.models import select_champions
    from nixtla_scaffold.schema import SelectionPolicy

    data = history(224, changed=True)
    paths = pd.DataFrame({"unique_id": "001", "ds": pd.date_range(data.ds.max() + pd.Timedelta(days=1), periods=7),
                          "Unstable": data.y.tail(7).to_numpy() * 2.5,
                          "Stable": data.y.tail(7).to_numpy()})
    metrics = pd.DataFrame({"unique_id": ["001", "001"], "model": ["Unstable", "Stable"],
                            "rmse": [1., 2.], "mae": [1., 2.], "abs_bias": [0., 0.],
                            "selection_eligible": [True, True]})
    original = paths.copy(deep=True)
    screened, scores, receipts = screen_candidates(data, paths, metrics, spec(), freq="D", season=7)
    selected = select_champions(screened, scores)
    assert selected.selected_model.tolist() == ["Stable"]
    assert not next(row for row in receipts if row["model"] == "Unstable")["eligible"]
    pd.testing.assert_frame_equal(paths, original)
    assert metrics.selection_eligible.all()
    assert scores.loc[scores.model.eq("Unstable"), "rmse"].iloc[0] == 1.
    with pytest.raises(ValueError, match="no finite configuration-admissible"):
        screen_candidates(data, paths.drop(columns="Stable"), metrics, spec(), freq="D", season=7)
    warned = replace(spec(), selection_policy=SelectionPolicy(configuration_guard="warn"))
    screened, scores, receipts = screen_candidates(data, paths, metrics, warned, freq="D", season=7)
    assert select_champions(screened, scores).selected_model.tolist() == ["Unstable"]
    assert all(row["eligible"] for row in receipts)
    assert any(row["review_required"] for row in receipts)


def test_new_configuration_guard_and_saved_spec_compatibility():
    from nixtla_scaffold.presets import resolve_forecast_spec
    from nixtla_scaffold.schema import SelectionPolicy

    assert ForecastSpec().selection_policy.configuration_guard == "exclude"
    assert resolve_forecast_spec({"selection_policy": {"objective": "monthly_landing"}}).selection_policy.configuration_guard == "exclude"
    old = spec().to_dict()
    old["selection_policy"].pop("configuration_guard")
    assert forecast_spec_from_dict(old).selection_policy.configuration_guard == "warn"
    assert forecast_spec_from_dict(spec().to_dict()) == spec()
    for value in (True, None, "clip", {}):
        with pytest.raises(ValueError, match="configuration_guard"):
            SelectionPolicy(configuration_guard=value)


@pytest.mark.parametrize("backtested_alternative", [False, True])
def test_run_enforces_admission_without_resurrecting_an_unbacktested_fallback(monkeypatch, backtested_alternative):
    import nixtla_scaffold.forecast as engine
    from nixtla_scaffold.models import ModelResult, _metrics_from_cv

    data = history(224, changed=True)
    future = pd.DataFrame({"unique_id": "001", "ds": pd.date_range(data.ds.max() + pd.Timedelta(days=1), periods=7),
                           "AutoETS": data.y.tail(7).to_numpy() * 2.5, "Naive": data.y.tail(7).to_numpy()})
    cv = data.tail(7).assign(cutoff=data.ds.iloc[-8], AutoETS=lambda frame: frame.y)
    if backtested_alternative:
        cv["Naive"] = cv.y + 1.
    result = ModelResult(future, _metrics_from_cv(cv), cv, "test", pd.DataFrame())
    monkeypatch.setattr(engine, "forecast_with_policy", lambda *_: result)
    settings = replace(spec(), require_backtest=True)
    if not backtested_alternative:
        with pytest.raises(ValueError, match="require_backtest=True"):
            engine.run_forecast(data, settings)
    else:
        run = engine.run_forecast(data, settings)
        assert run.model_selection.selected_model.tolist() == ["Naive"]
        np.testing.assert_array_equal(run.forecast.yhat, future.Naive)
        pd.testing.assert_frame_equal(run.all_models, future)
        assert run.backtest_metrics.set_index("model").loc["AutoETS", "rmse"] == 0.
        assert not run.backtest_metrics.set_index("model").loc["AutoETS", "selection_eligible"]


def test_cli_configuration_guard_preserves_refresh_policy_and_honors_override():
    import argparse
    from nixtla_scaffold.cli import _add_finn_inspired_args, _selection_from_args
    from nixtla_scaffold.schema import SelectionPolicy

    parser = argparse.ArgumentParser()
    _add_finn_inspired_args(parser)
    legacy = SelectionPolicy(configuration_guard="warn")
    assert _selection_from_args(parser.parse_args([]), legacy) == legacy
    assert _selection_from_args(parser.parse_args(["--configuration-guard", "exclude"]), legacy).configuration_guard == "exclude"
    assert _selection_from_args(parser.parse_args([]), SelectionPolicy()).configuration_guard == "exclude"
