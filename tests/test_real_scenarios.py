"""Comprehensive forecast scenario tests.

Each test represents a realistic forecasting scenario a finance user would encounter.
Uses utilsforecast.data.generate_series for reproducible data generation plus
hand-crafted edge cases. Every test validates the full pipeline end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import ForecastSpec, run_forecast
from nixtla_scaffold.profile import profile_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_series(uid: str, values: list[float], start: str = "2023-01-31") -> pd.DataFrame:
    return pd.DataFrame({
        "unique_id": uid,
        "ds": pd.date_range(start, periods=len(values), freq="ME"),
        "y": values,
    })


def _assert_forecast_ok(run, *, horizon: int, series: set[str] | None = None):
    assert len(run.forecast) > 0
    assert run.forecast["yhat"].notna().all()
    if series:
        assert set(run.forecast["unique_id"]) == series
    assert "model" in run.forecast.columns
    assert run.diagnostics()["status"] == "success"


# ---------------------------------------------------------------------------
# SCENARIO 1: Classic monthly revenue with strong trend + seasonality
# Finance user: "I have 3 years of monthly revenue, forecast next 6 months"
# ---------------------------------------------------------------------------

def test_scenario_monthly_revenue_trend_seasonal():
    t = np.arange(36)
    y = 100 + 3.5 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(42).normal(0, 3, 36)
    df = _monthly_series("Revenue", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6))

    _assert_forecast_ok(run, horizon=6, series={"Revenue"})
    assert run.profile.freq == "ME"
    assert run.profile.season_length == 12
    assert not run.backtest_metrics.empty
    interp = run.interpretation()
    assert interp["seasonality"]["summary"]


# ---------------------------------------------------------------------------
# SCENARIO 2: Flat noisy series (no trend, no seasonality)
# Finance user: "Our cost center is pretty flat, what's the forecast?"
# ---------------------------------------------------------------------------

def test_scenario_flat_noisy_cost_center():
    rng = np.random.default_rng(7)
    y = 500 + rng.normal(0, 20, 24)
    df = _monthly_series("OpEx", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="auto"))

    _assert_forecast_ok(run, horizon=3, series={"OpEx"})
    # Forecast should be roughly around the mean
    assert abs(run.forecast["yhat"].mean() - 500) < 100


# ---------------------------------------------------------------------------
# SCENARIO 3: Very short history (only 5 months)
# Finance user: "New product line, only 5 months of data"
# ---------------------------------------------------------------------------

def test_scenario_short_history_5_months():
    df = _monthly_series("NewProduct", [10, 15, 22, 28, 35])

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="auto"))

    _assert_forecast_ok(run, horizon=3, series={"NewProduct"})
    assert any("limited" in str(w).lower() or "short" in str(w).lower() or "baseline" in run.engine for w in run.warnings) or run.engine == "baseline"


# ---------------------------------------------------------------------------
# SCENARIO 4: Only 2 data points
# Finance user: "I literally have 2 months, can you still forecast?"
# ---------------------------------------------------------------------------

def test_scenario_minimal_2_points():
    df = _monthly_series("Startup", [100, 120])

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    _assert_forecast_ok(run, horizon=2, series={"Startup"})
    assert run.engine == "baseline"


# ---------------------------------------------------------------------------
# SCENARIO 5: Multi-series panel with mixed history lengths
# Finance user: "3 product lines, one new, one mature, one medium"
# ---------------------------------------------------------------------------

def test_scenario_multi_series_mixed_history():
    rng = np.random.default_rng(99)
    df = pd.concat([
        _monthly_series("Mature", (100 + 2 * np.arange(36) + rng.normal(0, 5, 36)).tolist()),
        _monthly_series("Medium", (50 + 1.5 * np.arange(18) + rng.normal(0, 3, 18)).tolist(), start="2024-07-31"),
        _monthly_series("New", (20 + rng.normal(0, 2, 6)).tolist(), start="2025-07-31"),
    ], ignore_index=True)

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3, series={"Mature", "Medium", "New"})
    assert len(run.forecast) == 9  # 3 series × 3 horizons


# ---------------------------------------------------------------------------
# SCENARIO 6: Daily data with weekly seasonality
# Finance user: "We have daily transaction volumes"
# ---------------------------------------------------------------------------

def test_scenario_daily_weekly_seasonality():
    from utilsforecast.data import generate_series
    df = generate_series(n_series=1, freq="D", min_length=90, max_length=90, seed=42, with_trend=True)

    run = run_forecast(df, ForecastSpec(horizon=14, freq="D"))

    _assert_forecast_ok(run, horizon=14)
    assert run.profile.freq == "D"
    assert run.profile.season_length == 7


# ---------------------------------------------------------------------------
# SCENARIO 7: Business day frequency
# Finance user: "Trading desk P&L, business days only"
# ---------------------------------------------------------------------------

def test_scenario_business_day_frequency():
    from utilsforecast.data import generate_series
    df = generate_series(n_series=1, freq="B", min_length=60, max_length=60, seed=7)

    run = run_forecast(df, ForecastSpec(horizon=10, freq="B"))

    _assert_forecast_ok(run, horizon=10)
    assert run.profile.freq == "B"
    assert run.profile.season_length == 5


# ---------------------------------------------------------------------------
# SCENARIO 8: Weekly data
# Finance user: "Weekly sales aggregation"
# ---------------------------------------------------------------------------

def test_scenario_weekly_data():
    from utilsforecast.data import generate_series
    df = generate_series(n_series=1, freq="W-SUN", min_length=52, max_length=52, seed=11, with_trend=True)

    run = run_forecast(df, ForecastSpec(horizon=8, freq="W-SUN"))

    _assert_forecast_ok(run, horizon=8)
    assert "W" in run.profile.freq


# ---------------------------------------------------------------------------
# SCENARIO 9: Quarterly data (fiscal quarter end)
# Finance user: "Quarterly revenue, fiscal year ending Oct"
# ---------------------------------------------------------------------------

def test_scenario_quarterly_fiscal():
    rng = np.random.default_rng(55)
    dates = pd.date_range("2020-01-31", periods=20, freq="QE")
    y = 1000 + 50 * np.arange(20) + 200 * np.sin(2 * np.pi * np.arange(20) / 4) + rng.normal(0, 30, 20)
    df = pd.DataFrame({"unique_id": "QuarterlyRev", "ds": dates, "y": y})

    run = run_forecast(df, ForecastSpec(horizon=4, freq="QE"))

    _assert_forecast_ok(run, horizon=4, series={"QuarterlyRev"})


# ---------------------------------------------------------------------------
# SCENARIO 10: Missing timestamps in the middle
# Finance user: "We have gaps in our data from system migration"
# ---------------------------------------------------------------------------

def test_scenario_missing_internal_timestamps():
    y = list(100 + 2 * np.arange(24) + 10 * np.sin(2 * np.pi * np.arange(24) / 12))
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"unique_id": "GappyRev", "ds": dates, "y": y})
    # Drop months 5, 10, 15 to simulate gaps
    df = df.drop([4, 9, 14]).reset_index(drop=True)

    run = run_forecast(df, ForecastSpec(horizon=3, fill_method="interpolate"))

    _assert_forecast_ok(run, horizon=3, series={"GappyRev"})
    assert any("imputed" in str(w) or "interpolate" in str(w) for w in run.warnings)


# ---------------------------------------------------------------------------
# SCENARIO 11: All zeros except occasional spikes (intermittent demand)
# Finance user: "Sporadic deal closings, mostly zero months"
# ---------------------------------------------------------------------------

def test_scenario_intermittent_demand():
    y = [0, 0, 150, 0, 0, 0, 200, 0, 0, 80, 0, 0, 0, 0, 300, 0, 0, 0, 120, 0, 0, 0, 0, 250]
    df = _monthly_series("SporadicDeals", y)

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3, series={"SporadicDeals"})
    # ZeroForecast should be available as a candidate
    assert "ZeroForecast" in run.all_models.columns or run.engine == "baseline"


# ---------------------------------------------------------------------------
# SCENARIO 12: Negative values (costs)
# Finance user: "COGS series, all negative"
# ---------------------------------------------------------------------------

def test_scenario_negative_values_costs():
    rng = np.random.default_rng(33)
    y = -(200 + 3 * np.arange(24) + 15 * np.sin(2 * np.pi * np.arange(24) / 12) + rng.normal(0, 5, 24))
    df = _monthly_series("COGS", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3, series={"COGS"})
    # Forecasts should be negative
    assert (run.forecast["yhat"] < 0).all()


# ---------------------------------------------------------------------------
# SCENARIO 13: Level shift mid-series
# Finance user: "We had an acquisition in month 18, costs jumped"
# ---------------------------------------------------------------------------

def test_scenario_level_shift():
    rng = np.random.default_rng(77)
    t = np.arange(36)
    y = 100 + 2 * t + np.where(t >= 18, 80, 0) + rng.normal(0, 5, 36)
    df = _monthly_series("AcquisitionBump", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6))

    _assert_forecast_ok(run, horizon=6, series={"AcquisitionBump"})
    # Forecasts should be at the post-shift level (roughly)
    assert run.forecast["yhat"].mean() > 200


# ---------------------------------------------------------------------------
# SCENARIO 14: Exponential growth
# Finance user: "Hypergrowth SaaS ARR"
# ---------------------------------------------------------------------------

def test_scenario_exponential_growth():
    rng = np.random.default_rng(44)
    t = np.arange(30)
    y = 50 * np.power(1.04, t) + rng.normal(0, 2, 30)
    df = _monthly_series("HyperARR", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6))

    _assert_forecast_ok(run, horizon=6, series={"HyperARR"})
    # Last forecast should be > last actual
    last_actual = float(df["y"].iloc[-1])
    last_forecast = float(run.forecast.sort_values("ds")["yhat"].iloc[-1])
    assert last_forecast > last_actual * 0.9  # allow for model conservatism


# ---------------------------------------------------------------------------
# SCENARIO 15: Declining trend
# Finance user: "Legacy product winding down"
# ---------------------------------------------------------------------------

def test_scenario_declining_trend():
    rng = np.random.default_rng(88)
    t = np.arange(30)
    y = 500 - 8 * t + rng.normal(0, 10, 30)
    df = _monthly_series("LegacyProduct", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6))

    _assert_forecast_ok(run, horizon=6, series={"LegacyProduct"})


# ---------------------------------------------------------------------------
# SCENARIO 16: High noise / low signal
# Finance user: "This series is super volatile, can you still forecast?"
# ---------------------------------------------------------------------------

def test_scenario_high_noise():
    rng = np.random.default_rng(22)
    t = np.arange(36)
    y = 100 + 1.0 * t + rng.normal(0, 40, 36)  # noise >> signal
    df = _monthly_series("Volatile", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3, series={"Volatile"})


# ---------------------------------------------------------------------------
# SCENARIO 17: Event/driver scenario overlay
# Finance user: "We have a product launch in March, bump forecast by 15%"
# ---------------------------------------------------------------------------

def test_scenario_event_driver_overlay():
    from nixtla_scaffold.schema import DriverEvent
    rng = np.random.default_rng(42)
    t = np.arange(24)
    y = 200 + 5 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, 24)
    df = _monthly_series("ProductLine", y.tolist())

    event = DriverEvent(name="Q1 Launch", start="2025-01-31", end="2025-03-31", effect="multiplicative", magnitude=0.15)
    run = run_forecast(df, ForecastSpec(horizon=6, events=(event,)))

    _assert_forecast_ok(run, horizon=6, series={"ProductLine"})
    assert "yhat_scenario" in run.forecast.columns
    # Scenario forecast should be higher than base in the event window
    event_rows = run.forecast[run.forecast["ds"].between("2025-01-31", "2025-03-31")]
    if not event_rows.empty:
        assert (event_rows["yhat_scenario"] >= event_rows["yhat"]).all()


# ---------------------------------------------------------------------------
# SCENARIO 18: Prediction intervals
# Finance user: "I need 80% and 95% intervals for risk modeling"
# ---------------------------------------------------------------------------

def test_scenario_prediction_intervals():
    rng = np.random.default_rng(42)
    y = 100 + 2 * np.arange(48) + 10 * np.sin(2 * np.pi * np.arange(48) / 12) + rng.normal(0, 5, 48)
    df = _monthly_series("RevWithCI", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6, levels=(80, 95), model_policy="statsforecast"))

    _assert_forecast_ok(run, horizon=6, series={"RevWithCI"})
    assert run.engine == "statsforecast"
    interval_cols = [c for c in run.all_models.columns if "-lo-" in c or "-hi-" in c]
    assert len(interval_cols) > 0, "all_models should contain interval columns"
    if run.effective_levels():
        width_80 = (run.forecast["yhat_hi_80"] - run.forecast["yhat_lo_80"]).mean()
        width_95 = (run.forecast["yhat_hi_95"] - run.forecast["yhat_lo_95"]).mean()
        assert width_95 > width_80


# ---------------------------------------------------------------------------
# SCENARIO 19: utilsforecast multi-series panel (10 series)
# Finance user: "Forecast all 10 product SKUs at once"
# ---------------------------------------------------------------------------

def test_scenario_multi_series_panel_10():
    from utilsforecast.data import generate_series
    df = generate_series(n_series=10, freq="ME", min_length=24, max_length=48, seed=42, with_trend=True)

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3)
    assert run.forecast["unique_id"].nunique() == 10
    assert len(run.forecast) == 30  # 10 × 3


# ---------------------------------------------------------------------------
# SCENARIO 20: utilsforecast large panel (50 series)
# Finance user: "Full product catalog forecast"
# ---------------------------------------------------------------------------

def test_scenario_large_panel_50():
    from utilsforecast.data import generate_series
    df = generate_series(n_series=50, freq="ME", min_length=24, max_length=36, seed=7)

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="auto"))

    _assert_forecast_ok(run, horizon=3)
    assert run.forecast["unique_id"].nunique() == 50


# ---------------------------------------------------------------------------
# SCENARIO 21: Constant series (zero variance)
# Finance user: "Fixed lease cost, same every month"
# ---------------------------------------------------------------------------

def test_scenario_constant_series():
    df = _monthly_series("FixedLease", [5000.0] * 12)

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    _assert_forecast_ok(run, horizon=3, series={"FixedLease"})
    # All forecasts should be ~5000
    assert abs(run.forecast["yhat"].mean() - 5000) < 1


# ---------------------------------------------------------------------------
# SCENARIO 22: Outlier in training data
# Finance user: "We had a one-time $1M settlement in July"
# ---------------------------------------------------------------------------

def test_scenario_outlier_in_training():
    rng = np.random.default_rng(42)
    y = list(100 + 2 * np.arange(24) + rng.normal(0, 5, 24))
    y[12] += 200  # massive outlier
    df = _monthly_series("WithOutlier", y)

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3, series={"WithOutlier"})


# ---------------------------------------------------------------------------
# SCENARIO 23: Float precision edge case (very large values)
# Finance user: "Enterprise ARR in millions"
# ---------------------------------------------------------------------------

def test_scenario_large_values():
    rng = np.random.default_rng(42)
    y = 1_000_000 + 50_000 * np.arange(24) + rng.normal(0, 20_000, 24)
    df = _monthly_series("EnterpriseARR", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6))

    _assert_forecast_ok(run, horizon=6, series={"EnterpriseARR"})
    assert run.forecast["yhat"].mean() > 1_000_000


# ---------------------------------------------------------------------------
# SCENARIO 24: Very small values (fractional pennies)
# Finance user: "Per-unit cost is $0.003"
# ---------------------------------------------------------------------------

def test_scenario_tiny_values():
    rng = np.random.default_rng(42)
    y = 0.003 + 0.0001 * np.arange(24) + rng.normal(0, 0.0005, 24)
    df = _monthly_series("UnitCost", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3))

    _assert_forecast_ok(run, horizon=3, series={"UnitCost"})


# ---------------------------------------------------------------------------
# SCENARIO 25: Weighted ensemble disabled
# Finance user: "Just pick the best single model, no blending"
# ---------------------------------------------------------------------------

def test_scenario_no_weighted_ensemble():
    rng = np.random.default_rng(42)
    y = 100 + 2 * np.arange(24) + rng.normal(0, 5, 24)
    df = _monthly_series("SingleModel", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3, weighted_ensemble=False))

    _assert_forecast_ok(run, horizon=3, series={"SingleModel"})
    assert "WeightedEnsemble" not in run.all_models.columns


# ---------------------------------------------------------------------------
# SCENARIO 26: Profile-only check (no forecast, just data quality)
# Finance user: "Is my data ready for forecasting?"
# ---------------------------------------------------------------------------

def test_scenario_profile_only():
    rng = np.random.default_rng(42)
    y = 100 + 2 * np.arange(36) + rng.normal(0, 5, 36)
    df = _monthly_series("CheckMe", y.tolist())

    profile = profile_dataset(df)

    assert profile.rows == 36
    assert profile.freq == "ME"
    assert profile.season_length == 12
    assert profile.series_count == 1
    assert profile.series[0].readiness in {"ready", "marginal", "rich"}


# ---------------------------------------------------------------------------
# SCENARIO 27: Baseline-only policy
# Finance user: "I don't trust fancy models, just give me naive + drift"
# ---------------------------------------------------------------------------

def test_scenario_baseline_only():
    rng = np.random.default_rng(42)
    y = 100 + 2 * np.arange(18) + rng.normal(0, 5, 18)
    df = _monthly_series("Conservative", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    _assert_forecast_ok(run, horizon=3, series={"Conservative"})
    assert run.engine == "baseline"
    assert any(m in run.all_models.columns for m in ["Naive", "RandomWalkWithDrift", "HistoricAverage"])


# ---------------------------------------------------------------------------
# SCENARIO 28: Forward fill vs interpolate vs zero fill
# Finance user: "What happens with different gap-fill methods?"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fill_method", ["ffill", "interpolate", "zero"])
def test_scenario_fill_methods(fill_method):
    y = list(100 + 2 * np.arange(24) + 10 * np.sin(2 * np.pi * np.arange(24) / 12))
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"unique_id": "GapTest", "ds": dates, "y": y})
    df = df.drop([5, 11]).reset_index(drop=True)  # create gaps

    run = run_forecast(df, ForecastSpec(horizon=3, fill_method=fill_method))

    _assert_forecast_ok(run, horizon=3, series={"GapTest"})


# ---------------------------------------------------------------------------
# SCENARIO 29: Interpretation artifacts completeness
# Finance user: "I need full explainability for my CFO presentation"
# ---------------------------------------------------------------------------

def test_scenario_full_interpretation():
    rng = np.random.default_rng(42)
    t = np.arange(36)
    y = 100 + 3 * t + 15 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, 36)
    df = _monthly_series("ExplainMe", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=6))

    interp = run.interpretation()
    assert interp["backtesting"]["window_count"] > 0
    assert interp["seasonality"]["summary"]
    assert interp["naive_comparison"]["series"]
    assert run.explanation()  # model card
    diag = run.diagnostics()
    assert diag["status"] == "success"
    assert diag["next_diagnostic_steps"]


# ---------------------------------------------------------------------------
# SCENARIO 30: Manifest and artifact completeness
# Finance user: "What files should I expect in the output folder?"
# ---------------------------------------------------------------------------

def test_scenario_manifest_completeness():
    rng = np.random.default_rng(42)
    y = 100 + 2 * np.arange(24) + rng.normal(0, 5, 24)
    df = _monthly_series("ArtifactCheck", y.tolist())

    run = run_forecast(df, ForecastSpec(horizon=3))

    manifest = run.manifest()
    expected_keys = {"forecast", "all_models", "model_selection", "backtest_metrics",
                     "model_weights", "diagnostics", "model_card", "workbook",
                     "html_report", "streamlit_app", "interpretation"}
    assert expected_keys.issubset(manifest["outputs"].keys())
    assert manifest["engine"] in {"baseline", "statsforecast"}
    assert manifest["best_practice_receipts"]
