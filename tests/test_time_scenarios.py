from __future__ import annotations

import pandas as pd

from nixtla_scaffold import DriverEvent, ForecastSpec, add_fiscal_calendar, aggregate_hierarchy_frame, hierarchy_coherence, run_forecast
from nixtla_scaffold.cli import main
from nixtla_scaffold.data import canonicalize_forecast_frame
from nixtla_scaffold.profile import profile_dataset, repair_time_index
from nixtla_scaffold.transformations import apply_event_adjustments


def _frame(dates, values=None, unique_id: str = "A") -> pd.DataFrame:
    values = values if values is not None else list(range(1, len(dates) + 1))
    return canonicalize_forecast_frame(pd.DataFrame({"unique_id": [unique_id] * len(dates), "ds": dates, "y": values}))


def test_daily_frequency_inference_and_horizon_rolls_forward() -> None:
    df = _frame(pd.date_range("2026-01-01", periods=10, freq="D"))

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert run.profile.freq == "D"
    assert run.forecast["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-11", "2026-01-12", "2026-01-13"]
    assert run.profile.season_length == 7


def test_business_day_frequency_skips_weekends_in_future_dates() -> None:
    df = _frame(pd.bdate_range("2026-01-05", periods=10))

    run = run_forecast(df, ForecastSpec(horizon=4, model_policy="baseline"))

    assert run.profile.freq == "B"
    assert run.forecast["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22"]
    assert not run.forecast["ds"].dt.dayofweek.isin([5, 6]).any()


def test_weekly_sunday_anchor_keeps_sunday_future_dates() -> None:
    df = _frame(pd.date_range("2026-01-04", periods=10, freq="W-SUN"))

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert run.profile.freq == "W-SUN"
    assert run.forecast["ds"].dt.day_name().tolist() == ["Sunday", "Sunday", "Sunday"]


def test_weekly_saturday_anchor_keeps_saturday_future_dates() -> None:
    df = _frame(pd.date_range("2026-01-03", periods=10, freq="W-SAT"))

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert run.profile.freq == "W-SAT"
    assert run.forecast["ds"].dt.day_name().tolist() == ["Saturday", "Saturday", "Saturday"]


def test_daily_missing_timestamp_is_counted_and_ffill_repaired() -> None:
    df = _frame(["2026-01-01", "2026-01-02", "2026-01-04"], [10, 20, 40])
    profile = profile_dataset(df, ForecastSpec(horizon=2))

    repaired, warnings = repair_time_index(df, profile, ForecastSpec(horizon=2, fill_method="ffill"))

    assert profile.freq == "D"
    assert profile.missing_timestamps == 1
    assert repaired["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert repaired.loc[repaired["ds"] == pd.Timestamp("2026-01-03"), "y"].iloc[0] == 20
    assert any("imputed 1 values" in warning for warning in warnings)


def test_single_timestamp_with_explicit_daily_freq_forecasts_successfully() -> None:
    df = _frame(["2026-01-01"], [10])

    run = run_forecast(df, ForecastSpec(horizon=2, freq="D", model_policy="baseline"))

    assert run.forecast["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]


def test_two_irregular_timestamps_raise_without_explicit_freq() -> None:
    df = _frame(["2026-01-01", "2026-01-10"], [10, 20])

    try:
        profile_dataset(df, ForecastSpec(horizon=2))
    except ValueError as exc:
        assert "pass --freq explicitly" in str(exc)
    else:
        raise AssertionError("expected irregular two-point history to require explicit frequency")


def test_daily_horizon_includes_feb_29_on_leap_year() -> None:
    df = _frame(pd.date_range("2024-02-25", periods=4, freq="D"))

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert "2024-02-29" in run.forecast["ds"].dt.strftime("%Y-%m-%d").tolist()


def test_daily_horizon_crosses_year_boundary_cleanly() -> None:
    df = _frame(pd.date_range("2025-12-28", periods=4, freq="D"))

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert run.forecast["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-01", "2026-01-02"]


def test_month_end_frequency_missing_month_and_future_dates() -> None:
    dates = list(pd.date_range("2024-01-31", periods=18, freq="ME"))
    dates.pop(6)
    df = _frame(dates)

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert run.profile.freq == "ME"
    assert run.profile.missing_timestamps == 0
    assert run.forecast["ds"].dt.is_month_end.all()
    assert any("imputed 1 values" in warning for warning in run.warnings)


def test_month_start_frequency_and_future_dates() -> None:
    dates = list(pd.date_range("2024-01-01", periods=18, freq="MS"))
    dates.pop(5)
    df = _frame(dates)

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert run.profile.freq == "MS"
    assert (run.forecast["ds"].dt.day == 1).all()


def test_quarter_end_frequency_and_future_dates() -> None:
    df = _frame(pd.date_range("2024-03-31", periods=8, freq="QE"))

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert run.profile.freq == "QE"
    assert run.profile.season_length == 4
    assert run.forecast["ds"].dt.is_quarter_end.all()


def test_explicit_quarter_start_preserves_qs_alignment() -> None:
    df = _frame(pd.date_range("2024-01-01", periods=8, freq="QS"))

    run = run_forecast(df, ForecastSpec(horizon=2, freq="QS", model_policy="baseline"))

    assert run.profile.freq == "QS"
    assert run.profile.season_length == 4
    assert run.forecast["ds"].dt.is_quarter_start.all()


def test_inferred_quarter_start_preserves_calendar_alignment() -> None:
    df = _frame(pd.date_range("2024-01-01", periods=8, freq="QS"))

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert run.profile.freq == "QS"
    assert run.profile.season_length == 4
    assert run.forecast["ds"].dt.is_quarter_start.all()


def test_inferred_fiscal_quarter_start_preserves_anchor() -> None:
    df = _frame(pd.date_range("2024-02-01", periods=6, freq="QS-NOV"))

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert run.profile.freq == "QS-NOV"
    assert run.forecast["ds"].dt.month.tolist() == [8, 11]
    assert run.forecast["ds"].dt.day.tolist() == [1, 1]


def test_inferred_fiscal_quarter_end_preserves_anchor() -> None:
    df = _frame(pd.date_range("2024-01-31", periods=6, freq="QE-OCT"))

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert run.profile.freq == "QE-OCT"
    assert run.forecast["ds"].dt.month.tolist() == [7, 10]
    assert run.forecast["ds"].dt.is_month_end.all()


def test_two_point_quarter_start_can_be_inferred() -> None:
    df = _frame(["2024-01-01", "2024-04-01"], [10, 12])

    profile = profile_dataset(df, ForecastSpec(horizon=1))

    assert profile.freq == "QS"
    assert profile.season_length == 4


def test_gappy_quarter_start_repairs_missing_quarter() -> None:
    df = _frame(["2024-01-01", "2024-04-01", "2024-10-01"], [10, 12, 18])

    profile = profile_dataset(df, ForecastSpec(horizon=1))
    repaired, warnings = repair_time_index(df, profile, ForecastSpec(horizon=1, fill_method="interpolate"))

    assert profile.freq == "QS"
    assert repaired["ds"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
    ]
    assert warnings == ["A: imputed 1 values with fill_method=interpolate"]


def test_missing_fiscal_month_is_repaired_across_boundary() -> None:
    df = _frame(["2026-03-31", "2026-05-31"], [100, 120])
    profile = profile_dataset(df, ForecastSpec(horizon=1, freq="ME"))
    repaired, _ = repair_time_index(df, profile, ForecastSpec(horizon=1, freq="ME", fill_method="interpolate"))

    fiscal = add_fiscal_calendar(repaired, fiscal_year_start_month=4)

    assert fiscal["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-03-31", "2026-04-30", "2026-05-31"]
    assert fiscal["fiscal_month"].tolist() == [12, 1, 2]
    assert fiscal["fiscal_quarter"].tolist() == [4, 1, 1]


def test_23_month_history_is_usable_not_rich() -> None:
    df = _frame(pd.date_range("2024-01-31", periods=23, freq="ME"))

    profile = profile_dataset(df, ForecastSpec(horizon=3))

    assert profile.series[0].readiness == "usable"
    assert any("short for seasonal backtesting" in warning for warning in profile.series[0].warnings)


def test_24_month_history_is_rich_for_seasonal_backtesting() -> None:
    df = _frame(pd.date_range("2024-01-31", periods=24, freq="ME"))

    profile = profile_dataset(df, ForecastSpec(horizon=3))

    assert profile.series[0].readiness == "rich"
    assert not profile.series[0].warnings


def test_36_month_history_scores_seasonal_models() -> None:
    df = _frame(pd.date_range("2023-01-31", periods=36, freq="ME"), [100 + (idx % 12) * 2 + idx for idx in range(36)])

    run = run_forecast(df, ForecastSpec(horizon=6, model_policy="baseline"))

    assert run.profile.season_length == 12
    assert "SeasonalNaive" in set(run.backtest_metrics["model"])


def test_zero_heavy_monthly_series_falls_back_when_wape_is_undefined() -> None:
    df = _frame(pd.date_range("2025-01-31", periods=12, freq="ME"), [0] * 12)

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert len(run.forecast) == 3
    assert run.forecast["yhat"].notna().all()
    reason = run.model_selection["selection_reason"].iloc[0]
    assert "backtest not available" in reason or "RMSE" in reason


def test_two_point_series_has_no_backtest_metrics_but_still_forecasts() -> None:
    df = _frame(pd.date_range("2026-01-31", periods=2, freq="ME"), [10, 12])

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert len(run.forecast) == 2
    assert run.backtest_metrics.empty
    assert "backtest not available" in run.model_selection["selection_reason"].iloc[0]


def test_three_point_series_unblocks_backtest_metrics() -> None:
    df = _frame(pd.date_range("2026-01-31", periods=3, freq="ME"), [10, 12, 15])

    run = run_forecast(df, ForecastSpec(horizon=1, model_policy="baseline"))

    assert not run.backtest_metrics.empty
    assert {"wape", "mae", "bias"}.issubset(run.model_selection.columns)


def test_hierarchy_sparse_child_aligns_with_parent_calendar_after_repair() -> None:
    raw = pd.DataFrame(
        {
            "region": ["NA", "NA", "NA", "NA", "NA"],
            "product": ["A", "A", "A", "B", "B"],
            "month": ["2026-01-31", "2026-02-28", "2026-03-31", "2026-01-31", "2026-03-31"],
            "revenue": [100, 110, 120, 50, 70],
        }
    )
    hierarchy = aggregate_hierarchy_frame(raw, hierarchy_cols=("region", "product"), time_col="month", target_col="revenue")

    run = run_forecast(hierarchy, ForecastSpec(horizon=1, freq="ME", model_policy="baseline", fill_method="zero"))

    history_dates = run.history.groupby("unique_id")["ds"].apply(lambda values: tuple(values.dt.strftime("%Y-%m-%d"))).to_dict()
    assert len(set(history_dates.values())) == 1
    assert not hierarchy_coherence(run.forecast).empty


def test_event_window_is_inclusive_on_start_and_end_dates() -> None:
    forecast = pd.DataFrame(
        {"unique_id": ["A"] * 4, "ds": pd.date_range("2026-01-31", periods=4, freq="ME"), "yhat": [100.0] * 4}
    )
    event = DriverEvent(name="Promo", start="2026-02-28", end="2026-03-31", effect="additive", magnitude=5)

    out = apply_event_adjustments(forecast, [event])

    assert out["yhat_scenario"].tolist() == [100.0, 105.0, 105.0, 100.0]
    assert out["event_names"].tolist() == ["", "Promo", "Promo", ""]


def test_event_outside_horizon_is_noop_but_adds_scenario_columns() -> None:
    forecast = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-01-31")], "yhat": [100.0]})
    event = DriverEvent(name="Future", start="2026-12-31", magnitude=0.5)

    out = apply_event_adjustments(forecast, [event])

    assert out["yhat_scenario"].tolist() == [100.0]
    assert out["event_adjustment"].tolist() == [0.0]
    assert out["event_names"].tolist() == [""]


def test_affected_unique_ids_limit_adjustments_to_selected_series() -> None:
    forecast = pd.DataFrame(
        {"unique_id": ["A", "B"], "ds": [pd.Timestamp("2026-01-31")] * 2, "yhat": [100.0, 100.0]}
    )
    event = DriverEvent(name="Launch", start="2026-01-31", magnitude=0.1, affected_unique_ids=("A",))

    out = apply_event_adjustments(forecast, [event]).set_index("unique_id")

    assert out.loc["A", "yhat_scenario"] == 110.0
    assert out.loc["B", "yhat_scenario"] == 100.0


def test_overlapping_events_compound_in_declaration_order() -> None:
    forecast = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-01-31")], "yhat": [100.0]})
    events = [
        DriverEvent(name="Promo", start="2026-01-31", effect="additive", magnitude=5),
        DriverEvent(name="Launch", start="2026-01-31", effect="multiplicative", magnitude=0.1),
    ]

    out = apply_event_adjustments(forecast, events)

    assert out["yhat_scenario"].iloc[0] == 115.5
    assert out["event_adjustment"].iloc[0] == 15.5
    assert out["event_names"].iloc[0] == "Promo; Launch"


def test_confidence_weights_event_magnitude_before_application() -> None:
    forecast = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-01-31")], "yhat": [100.0]})
    event = DriverEvent(name="Low confidence", start="2026-01-31", magnitude=0.2, confidence=0.5)

    out = apply_event_adjustments(forecast, [event])

    assert out["yhat_scenario"].iloc[0] == 110.0


def test_event_scenario_intervals_move_with_scenario_not_baseline() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": [pd.Timestamp("2026-01-31")],
            "yhat": [100.0],
            "yhat_lo_80": [90.0],
            "yhat_hi_80": [110.0],
        }
    )
    event = DriverEvent(name="Launch", start="2026-01-31", effect="additive", magnitude=5)

    out = apply_event_adjustments(forecast, [event])

    assert out["yhat_lo_80"].iloc[0] == 90.0
    assert out["yhat_hi_80"].iloc[0] == 110.0
    assert out["yhat_scenario_lo_80"].iloc[0] == 95.0
    assert out["yhat_scenario_hi_80"].iloc[0] == 115.0


def test_multiplicative_event_scenario_intervals_scale_with_point_forecast() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": [pd.Timestamp("2026-01-31")],
            "yhat": [100.0],
            "yhat_lo_80": [90.0],
            "yhat_hi_80": [110.0],
        }
    )
    event = DriverEvent(name="Launch", start="2026-01-31", effect="multiplicative", magnitude=0.2)

    out = apply_event_adjustments(forecast, [event])

    assert out["yhat_scenario"].iloc[0] == 120.0
    assert out["yhat_scenario_lo_80"].iloc[0] == 108.0
    assert out["yhat_scenario_hi_80"].iloc[0] == 132.0
    assert out["yhat_scenario_lo_80"].iloc[0] <= out["yhat_scenario"].iloc[0] <= out["yhat_scenario_hi_80"].iloc[0]


def test_event_no_match_is_audited_in_attrs_and_run_warnings() -> None:
    forecast = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2026-01-31")], "yhat": [100.0]})
    event = DriverEvent(name="Typo", start="2026-01-31", magnitude=0.1, affected_unique_ids=("missing",))

    out = apply_event_adjustments(forecast, [event])

    assert "matched 0 forecast rows" in out.attrs["event_warnings"][0]

    df = _frame(pd.date_range("2025-01-31", periods=8, freq="ME"))
    run = run_forecast(df, ForecastSpec(horizon=1, model_policy="baseline", events=(event,)))
    assert any("matched 0 forecast rows" in warning for warning in run.warnings)


def test_driver_event_rejects_invalid_effect_and_confidence() -> None:
    for kwargs in [
        {"name": "Bad", "start": "2026-01-31", "effect": "bad"},
        {"name": "Bad", "start": "2026-01-31", "confidence": -0.1},
        {"name": "Bad", "start": "2026-01-31", "confidence": 1.1},
    ]:
        try:
            DriverEvent(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid DriverEvent to fail: {kwargs}")


def test_cli_event_json_rejects_malformed_or_incomplete_payloads(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.csv"
    _frame(pd.date_range("2026-01-31", periods=3, freq="ME")).to_csv(input_path, index=False)

    malformed = main(["forecast", "--input", str(input_path), "--event", "{bad-json"])
    missing = main(["forecast", "--input", str(input_path), "--event", '{"name":"No start"}'])

    captured = capsys.readouterr()
    assert malformed == 2
    assert missing == 2
    assert "invalid --event JSON" in captured.err
    assert "--event missing required" in captured.err
