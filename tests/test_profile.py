from __future__ import annotations

import pandas as pd

from nixtla_scaffold.data import canonicalize_forecast_frame
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.schema import ForecastSpec


def test_profile_infers_monthly_frequency_and_missing_dates() -> None:
    df = canonicalize_forecast_frame(
        pd.DataFrame(
            {
                "unique_id": ["A", "A", "A"],
                "ds": ["2025-01-31", "2025-03-31", "2025-04-30"],
                "y": [10, 12, 13],
            }
        )
    )

    profile = profile_dataset(df, ForecastSpec(horizon=2))

    assert profile.freq == "ME"
    assert profile.missing_timestamps == 1
    assert profile.series_count == 1
    assert profile.season_length == 12


def test_loader_creates_single_series_when_unique_id_missing() -> None:
    df = canonicalize_forecast_frame(
        pd.DataFrame({"ds": ["2025-01-31", "2025-02-28"], "y": [1, 2]})
    )

    assert df["unique_id"].tolist() == ["series_1", "series_1"]


def test_two_point_monthly_series_infers_month_end() -> None:
    df = canonicalize_forecast_frame(
        pd.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": ["2025-01-31", "2025-02-28"],
                "y": [10, 11],
            }
        )
    )

    profile = profile_dataset(df, ForecastSpec(horizon=2))

    assert profile.freq == "ME"


def test_loader_rejects_missing_unique_id_with_repeated_timestamps() -> None:
    data = pd.DataFrame(
        {
            "product": ["A", "B"],
            "ds": ["2025-01-31", "2025-01-31"],
            "y": [10, 20],
        }
    )

    try:
        canonicalize_forecast_frame(data)
    except ValueError as exc:
        assert "Pass --id-col" in str(exc)
    else:
        raise AssertionError("expected missing unique_id multi-series input to fail")


def test_loader_rejects_blank_unique_id_values() -> None:
    data = pd.DataFrame(
        {
            "unique_id": ["A", "", None],
            "ds": ["2025-01-31", "2025-02-28", "2025-03-31"],
            "y": [10, 11, 12],
        }
    )

    try:
        canonicalize_forecast_frame(data)
    except ValueError as exc:
        assert "missing unique_id" in str(exc)
    else:
        raise AssertionError("expected blank unique_id values to fail")


def test_missing_time_column_error_suggests_mapping() -> None:
    data = pd.DataFrame({"Product": ["A"], "Month": ["2025-01-31"], "Revenue": [10]})

    try:
        canonicalize_forecast_frame(data)
    except ValueError as exc:
        message = str(exc)
        assert "Detected columns" in message
        assert "--time-col Month" in message
    else:
        raise AssertionError("expected missing time column to fail with mapping hint")


def test_single_row_requires_explicit_frequency() -> None:
    df = canonicalize_forecast_frame(pd.DataFrame({"ds": ["2025-01-31"], "y": [10]}))

    try:
        profile_dataset(df, ForecastSpec(horizon=2))
    except ValueError as exc:
        assert "pass --freq explicitly" in str(exc)
    else:
        raise AssertionError("expected single-row frequency inference to fail")


def test_duplicate_keys_fail_before_modeling() -> None:
    from nixtla_scaffold import run_forecast

    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": ["2025-01-31", "2025-01-31"],
            "y": [10, 11],
        }
    )

    try:
        run_forecast(df, ForecastSpec(horizon=1, freq="ME"))
    except ValueError as exc:
        assert "duplicate unique_id/ds" in str(exc)
    else:
        raise AssertionError("expected duplicate keys to fail")


def test_drop_fill_regularizes_then_drops_structural_gaps() -> None:
    from nixtla_scaffold.profile import repair_time_index

    df = canonicalize_forecast_frame(
        pd.DataFrame(
            {
                "unique_id": ["A", "A", "A"],
                "ds": ["2025-01-31", "2025-03-31", "2025-04-30"],
                "y": [10, 12, 13],
            }
        )
    )
    spec = ForecastSpec(horizon=2, fill_method="drop")
    profile = profile_dataset(df, spec)
    repaired, warnings = repair_time_index(df, profile, spec)

    assert len(repaired) == 3
    assert any("dropped 1 structurally missing" in warning for warning in warnings)


def test_invalid_y_fails_fast() -> None:
    data = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-31"], "y": ["not-a-number"]})

    try:
        canonicalize_forecast_frame(data)
    except ValueError as exc:
        assert "non-numeric" in str(exc)
    else:
        raise AssertionError("expected invalid y to fail")


def test_all_null_y_fails_fast() -> None:
    data = pd.DataFrame({"unique_id": ["A", "A"], "ds": ["2025-01-31", "2025-02-28"], "y": [None, None]})

    try:
        canonicalize_forecast_frame(data)
    except ValueError as exc:
        assert "no usable numeric y" in str(exc)
    else:
        raise AssertionError("expected all-null y to fail")


def test_empty_input_fails_fast() -> None:
    data = pd.DataFrame({"unique_id": [], "ds": [], "y": []})

    try:
        canonicalize_forecast_frame(data)
    except ValueError as exc:
        assert "forecast input is empty" in str(exc)
    else:
        raise AssertionError("expected empty input to fail")


def test_monthly_season_length_stays_canonical_for_short_history() -> None:
    df = canonicalize_forecast_frame(
        pd.DataFrame(
            {
                "unique_id": ["A"] * 12,
                "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
                "y": range(12),
            }
        )
    )

    profile = profile_dataset(df, ForecastSpec(horizon=3))

    assert profile.season_length == 12

