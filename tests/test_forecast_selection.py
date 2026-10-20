import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import ForecastSpec, run_forecast
from nixtla_scaffold.models import ModelResult, _metrics_from_cv, select_champions
from nixtla_scaffold.model_families import canonical_model_name, model_family


def candidate_frame():
    return pd.DataFrame({
        "unique_id": ["A"], "ds": pd.to_datetime(["2026-01-01"]),
        "AutoETS": [200.0], "Holt": [220.0], "Naive": [100.0],
    })


def scores():
    return pd.DataFrame({
        "unique_id": ["A", "A", "A"], "model": ["AutoETS", "Holt", "Naive"],
        "rmse": [10.0, 10.0, 20.0], "mae": [8.0, 8.0, 15.0],
        "abs_bias": [0.1, 0.1, 0.2], "bias": [0.1, 0.1, 0.2],
        "cv_windows": [None, 3, 4], "observations": [6, 6, 6],
    })


@pytest.mark.parametrize("seed", range(5))
def test_exact_ties_are_explicit_and_invariant_to_metric_row_order(seed):
    chosen = select_champions(candidate_frame(), scores().sample(frac=1, random_state=seed)).iloc[0]
    assert chosen.selected_model == "AutoETS"
    assert "tied available metrics: AutoETS, Holt" in chosen.selection_reason
    assert "not evidence of superiority" in chosen.selection_reason
    assert pd.isna(chosen.cv_windows), "Winner must not borrow Holt's validation metadata."


def test_selection_keeps_nullable_metadata_on_the_actual_winning_row():
    metrics = scores()
    metrics.loc[0, "rmse"] = 5.0
    chosen = select_champions(candidate_frame(), metrics).iloc[0]
    assert chosen.rmse == 5.0 and chosen.selected_model == "AutoETS"
    assert pd.isna(chosen.cv_windows)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf, np.nan])
def test_nonfinite_scores_cannot_win(invalid):
    metrics = scores()
    metrics.loc[0, "rmse"] = invalid
    chosen = select_champions(candidate_frame(), metrics).iloc[0]
    assert chosen.selected_model == "Holt"
    assert chosen.cv_windows == 3


def test_incomplete_future_cannot_win_even_with_perfect_backtest():
    future = pd.concat([candidate_frame(), candidate_frame().assign(ds=pd.Timestamp("2026-02-01"))], ignore_index=True)
    future.loc[1, "AutoETS"] = np.nan
    metrics = scores()
    metrics.loc[0, "rmse"] = 0.0
    assert select_champions(future, metrics).iloc[0].selected_model == "Holt"


def test_naive_guard_does_not_borrow_validation_metadata_from_another_baseline():
    future = candidate_frame().assign(SeasonalNaive=95.0)
    metrics = scores()
    metrics.loc[0, "rmse"] = 19.5
    metrics.loc[1, "rmse"] = 25.0
    metrics.loc[2, "cv_windows"] = None
    metrics = pd.concat([metrics, metrics.tail(1).assign(model="SeasonalNaive", rmse=30.0, cv_windows=8)])
    chosen = select_champions(future, metrics).iloc[0]
    assert chosen.selected_model == "Naive"
    assert pd.isna(chosen.cv_windows)
    assert "naive guard" in chosen.selection_reason


@pytest.mark.parametrize("alias,canonical", [("tbats", "AutoTBATS"), ("ces", "AutoCES")])
@pytest.mark.parametrize("season", [1, 4])
def test_extended_classical_candidates_are_registered_and_run_natively(alias, canonical, season):
    import statsforecast.models as native

    if not hasattr(native, canonical):
        pytest.skip(f"{canonical} is unavailable in this StatsForecast version")
    assert canonical_model_name(alias) == canonical
    assert model_family(canonical) == "statsforecast"
    history = pd.DataFrame({"unique_id": ["A"] * 18, "ds": pd.date_range("2024-01-01", periods=18, freq="MS"),
                            "y": [100.0 + 2 * i + 5 * (i % season) for i in range(18)]})
    run = run_forecast(history, ForecastSpec(horizon=2, freq="MS", season_length=season, levels=(80, 95),
                                           model_policy="statsforecast", model_allowlist=(canonical,),
                                           weighted_ensemble=False, paired_validation=True, verbose=False))
    assert run.engine == "statsforecast"
    assert run.forecast.model.tolist() == [canonical, canonical]
    assert np.isfinite(run.forecast.yhat).all()
    assert set(run.backtest_metrics.model) == {canonical}
    assert (run.backtest_metrics.cv_windows > 0).all()
    for level in (80, 95):
        assert np.isfinite(run.forecast[[f"yhat_lo_{level}", f"yhat_hi_{level}"]]).all().all()
        assert (run.forecast[f"yhat_lo_{level}"] <= run.forecast.yhat).all()
        assert (run.forecast.yhat <= run.forecast[f"yhat_hi_{level}"]).all()


def test_extended_classical_candidates_preserve_legacy_standard_ladder():
    import statsforecast.models as native
    from nixtla_scaffold.models import _statsforecast_model_factories
    from nixtla_scaffold.profile import profile_dataset

    history = pd.DataFrame({"unique_id": ["A"] * 36, "ds": pd.date_range("2024-01-01", periods=36, freq="MS"), "y": range(36)})
    standard = ForecastSpec(horizon=6, freq="MS", model_policy="standard", run_mode="legacy", policy_version=0)
    profile = profile_dataset(history, standard)
    assert not {"AutoTBATS", "AutoCES"} & {name for name, _ in _statsforecast_model_factories(history, profile, standard)}
    extended = ForecastSpec(horizon=6, freq="MS", model_policy="all")
    available = {name for name in ("AutoTBATS", "AutoCES") if hasattr(native, name)}
    assert available <= {name for name, _ in _statsforecast_model_factories(history, profile, extended)}


def test_explicit_unavailable_extended_model_cannot_be_silently_skipped(monkeypatch):
    import statsforecast.models as native
    from nixtla_scaffold.models import _statsforecast_model_factories
    from nixtla_scaffold.profile import profile_dataset

    monkeypatch.delattr(native, "AutoTBATS", raising=False)
    history = pd.DataFrame({"unique_id": ["A"] * 18, "ds": pd.date_range("2024-01-01", periods=18, freq="MS"), "y": range(18)})
    spec = ForecastSpec(horizon=2, freq="MS", season_length=1, model_allowlist=("Naive", "AutoTBATS"))
    with pytest.raises(ImportError, match="AutoTBATS.*unavailable"):
        _statsforecast_model_factories(history, profile_dataset(history, spec), spec)


def test_implicit_unavailable_extended_model_is_disclosed(monkeypatch):
    import statsforecast.models as native
    import nixtla_scaffold.models as engine
    from nixtla_scaffold.profile import profile_dataset

    monkeypatch.delattr(native, "AutoTBATS", raising=False)
    monkeypatch.delattr(native, "AutoCES", raising=False)
    monkeypatch.setattr(engine, "_statsforecast_model_factories", lambda *_: [("Naive", lambda: native.Naive())])
    history = pd.DataFrame({"unique_id": ["A"] * 18, "ds": pd.date_range("2024-01-01", periods=18, freq="MS"), "y": range(18)})
    spec = ForecastSpec(horizon=2, freq="MS", season_length=1, model_policy="all", levels=(), weighted_ensemble=False)
    result = engine.forecast_with_statsforecast(history, profile_dataset(history, spec), spec)
    for name in ("AutoTBATS", "AutoCES"):
        assert any(f"{name} skipped: unavailable" in warning for warning in result.warnings)


@pytest.mark.parametrize("loser_error", [10.0, 1000000.0])
@pytest.mark.parametrize("scale", [1.0, 1000000000.0])
def test_delivered_point_and_intervals_equal_scored_model_not_pool_shrinkage(monkeypatch, loser_error, scale):
    import nixtla_scaffold.forecast as engine

    history = pd.DataFrame({"unique_id": ["A"] * 12, "ds": pd.date_range("2025-01-01", periods=12, freq="MS"),
                            "y": np.arange(100.0, 112.0) * scale})
    future = pd.DataFrame({
        "unique_id": ["A", "A"], "ds": pd.date_range("2026-01-01", periods=2, freq="MS"),
        "AutoETS": np.array([200.0, 210.0]) * scale,
        "AutoETS-lo-80": np.array([180.0, 190.0]) * scale,
        "AutoETS-hi-80": np.array([220.0, 230.0]) * scale,
        "Holt": np.array([250.0, 260.0]) * scale,
    })
    cv = pd.DataFrame({
        "unique_id": ["A"] * 3, "ds": pd.date_range("2025-10-01", periods=3, freq="MS"),
        "cutoff": [pd.Timestamp("2025-09-01")] * 3,
        "y": np.array([109.0, 110.0, 111.0]) * scale,
        "AutoETS": np.array([109.0, 110.0, 111.0]) * scale,
        "Holt": (np.array([109.0, 110.0, 111.0]) + loser_error) * scale,
    })
    result = ModelResult(future, _metrics_from_cv(cv), cv, "test", pd.DataFrame())
    monkeypatch.setattr(engine, "forecast_with_policy", lambda *_: result)
    run = run_forecast(history, ForecastSpec(horizon=2, freq="MS", levels=(80,), weighted_ensemble=False,
                                           run_mode="legacy", policy_version=0))
    assert run.forecast.model.tolist() == ["AutoETS", "AutoETS"]
    np.testing.assert_array_equal(run.forecast.yhat, future.AutoETS)
    np.testing.assert_array_equal(run.forecast.yhat_lo_80, future["AutoETS-lo-80"])
    np.testing.assert_array_equal(run.forecast.yhat_hi_80, future["AutoETS-hi-80"])
    assert not any("shrinkage" in warning for warning in run.warnings)


def test_raw_lane_scores_the_exact_canonical_actuals(monkeypatch):
    import nixtla_scaffold.forecast as engine

    history = pd.DataFrame({"unique_id": ["A"] * 12, "ds": pd.date_range("2025-01-01", periods=12, freq="MS"),
                            "y": np.arange(1_000_000_000, 1_000_000_012, dtype="float64")})
    cv = history.tail(3).assign(cutoff=pd.Timestamp("2025-09-01"), AutoETS=lambda frame: frame.y + 1)
    cv["y"] = cv.y.astype("float32")
    future = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-01-01")], "AutoETS": [1_000_000_013.0]})
    result = ModelResult(future, _metrics_from_cv(cv), cv, "test", pd.DataFrame())
    monkeypatch.setattr(engine, "forecast_with_policy", lambda *_: result)
    run = run_forecast(history, ForecastSpec(horizon=1, freq="MS", weighted_ensemble=False, paired_validation=True,
                                           run_mode="legacy", policy_version=0))
    np.testing.assert_array_equal(run.backtest_predictions.y, history.tail(3).y)
    assert run.backtest_metrics.iloc[0].rmse == 1.0
    assert run.backtest_metrics.iloc[0].mae == 1.0


@pytest.mark.parametrize("season", [1, 4])
def test_scaled_error_reference_never_uses_holdout_or_future_actuals(season):
    from nixtla_scaffold.models import _error_scale_map

    history = pd.DataFrame({"unique_id": ["A"] * 12, "ds": pd.date_range("2025-01-01", periods=12, freq="MS"), "y": np.arange(12.0)})
    cv = pd.DataFrame({"unique_id": ["A", "A"], "cutoff": pd.to_datetime(["2025-06-01", "2025-09-01"])})
    expected = {"A": {"mase_scale": float(season), "rmsse_scale": float(season)}}
    assert _error_scale_map(history, season, cv) == expected
    history.loc[history.ds > cv.cutoff.min(), "y"] = [100, 400, 20, 5000, 10, 60000]
    assert _error_scale_map(history, season, cv) == expected
    without_origins = _error_scale_map(history, season, pd.DataFrame())
    assert without_origins == {"A": {"mase_scale": None, "rmsse_scale": None}}


def test_native_metrics_and_export_share_training_only_scales():
    from nixtla_scaffold.models import _error_scale_map
    from nixtla_scaffold.outputs import build_backtest_long

    history = pd.DataFrame({"unique_id": ["A"] * 18, "ds": pd.date_range("2025-01-01", periods=18, freq="MS"),
                            "y": [100.0 + i for i in range(12)] + [400, 800, 1600, 3200, 6400, 12800]})
    run = run_forecast(history, ForecastSpec(horizon=2, freq="MS", season_length=1, model_policy="baseline",
                                           weighted_ensemble=False, levels=(), paired_validation=True))
    scales = _error_scale_map(run.history, 1, run.backtest_predictions)["A"]
    exported = build_backtest_long(run)
    assert set(exported.mase_scale) == {scales["mase_scale"]}
    assert set(exported.rmsse_scale) == {scales["rmsse_scale"]}
    np.testing.assert_allclose(run.backtest_metrics.mase, run.backtest_metrics.mae / scales["mase_scale"])
    np.testing.assert_allclose(run.backtest_metrics.rmsse, run.backtest_metrics.rmse / scales["rmsse_scale"])
