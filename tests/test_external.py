from __future__ import annotations

import json

import pandas as pd
import pytest

from nixtla_scaffold.external import (
    EXTERNAL_FORECAST_SCHEMA_VERSION,
    build_external_forecast_metadata,
    canonicalize_external_forecasts,
    load_external_forecasts,
)


def test_canonicalize_external_long_forecast_preserves_metadata() -> None:
    frame = pd.DataFrame(
        {
            "Product": ["Revenue", "Revenue"],
            "Month": ["2026-01-31", "2026-02-28"],
            "Finance Forecast": ["$1,100", "1,250"],
            "finance_model": ["Finance regression", "Finance regression"],
            "owner": ["fpna", "fpna"],
            "model_version": ["v1", "v1"],
            "notes": ["board triangulation", "board triangulation"],
        }
    )

    out = canonicalize_external_forecasts(
        frame,
        id_col="Product",
        time_col="Month",
        value_col="Finance Forecast",
        model_col="finance_model",
        source_id="fpna_workbook",
    )

    assert out["unique_id"].tolist() == ["Revenue", "Revenue"]
    assert out["model"].tolist() == ["Finance regression", "Finance regression"]
    assert out["yhat"].tolist() == [1100.0, 1250.0]
    assert out["family"].tolist() == ["external", "external"]
    assert out["is_external_forecast"].tolist() == [True, True]
    assert out["external_forecast_validation_required"].tolist() == [True, True]
    assert out["comparison_evidence_status"].tolist() == ["future_only_unscored", "future_only_unscored"]
    assert out["is_backtested"].tolist() == [False, False]
    assert out["owner"].tolist() == ["fpna", "fpna"]
    assert out["model_version"].tolist() == ["v1", "v1"]

    metadata = build_external_forecast_metadata(out)
    assert metadata["schema_version"] == EXTERNAL_FORECAST_SCHEMA_VERSION
    assert metadata["rows"] == 2
    assert metadata["models"] == ["Finance regression"]
    assert metadata["evidence_status_distribution"] == {"future_only_unscored": 2}
    assert not metadata["is_backtested"]
    assert "imported yhat values" in metadata["is_backtested_rationale"]


def test_load_external_forecasts_infers_file_model_and_source(tmp_path) -> None:
    path = tmp_path / "finance_plan.csv"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "ds": ["2026-01-31"],
            "yhat": [100],
        }
    ).to_csv(path, index=False)

    out = load_external_forecasts(path)

    assert out.loc[0, "model"] == "finance_plan"
    assert out.loc[0, "source_id"] == "finance_plan"
    assert out.loc[0, "source_file"] == str(path)


def test_canonicalize_external_forecasts_preserves_inbound_source_id() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "ds": ["2026-01-31"],
            "yhat": [100],
            "model": ["Plan"],
            "source_id": ["fpna_workbook"],
        }
    )

    out = canonicalize_external_forecasts(frame)

    assert out.loc[0, "source_id"] == "fpna_workbook"


def test_canonicalize_external_forecasts_allows_same_model_scenario_from_different_sources() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "ds": ["2026-01-31", "2026-01-31"],
            "yhat": [100, 105],
            "model": ["Plan", "Plan"],
            "scenario_name": ["base", "base"],
            "source_id": ["fpna_workbook", "sales_plan"],
        }
    )

    out = canonicalize_external_forecasts(frame)

    assert out["source_id"].tolist() == ["fpna_workbook", "sales_plan"]
    assert out["horizon_step"].tolist() == [1, 1]
    metadata = build_external_forecast_metadata(out)
    assert metadata["sources"] == ["fpna_workbook", "sales_plan"]


def test_canonicalize_external_forecasts_rejects_blank_inbound_source_id() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "ds": ["2026-01-31"],
            "yhat": [100],
            "model": ["Plan"],
            "source_id": [" "],
        }
    )

    with pytest.raises(ValueError, match="blank source_id"):
        canonicalize_external_forecasts(frame)


def test_package_root_exports_external_helpers() -> None:
    import nixtla_scaffold as ns

    assert ns.EXTERNAL_FORECAST_SCHEMA_VERSION == EXTERNAL_FORECAST_SCHEMA_VERSION
    assert ns.canonicalize_external_forecasts is canonicalize_external_forecasts
    assert ns.load_external_forecasts is load_external_forecasts


def test_canonicalize_external_wide_forecast_melts_date_columns() -> None:
    frame = pd.DataFrame(
        {
            "Product": ["Revenue"],
            "scenario_name": ["base"],
            "2026-01-31": [110.0],
            pd.Timestamp("2026-02-28"): [120.0],
        }
    )

    out = canonicalize_external_forecasts(
        frame,
        format="wide",
        model_name="FY seasonality allocator",
        source_id="excel_model",
        id_col="Product",
    )

    assert out[["unique_id", "model", "scenario_name", "ds", "yhat"]].to_dict("records") == [
        {
            "unique_id": "Revenue",
            "model": "FY seasonality allocator",
            "scenario_name": "base",
            "ds": pd.Timestamp("2026-01-31"),
            "yhat": 110.0,
        },
        {
            "unique_id": "Revenue",
            "model": "FY seasonality allocator",
            "scenario_name": "base",
            "ds": pd.Timestamp("2026-02-28"),
            "yhat": 120.0,
        },
    ]
    assert pd.to_datetime(out["source_column"]).tolist() == [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-28")]
    assert out["horizon_step"].tolist() == [1, 2]


def test_canonicalize_external_wide_forecast_defaults_single_series_id() -> None:
    frame = pd.DataFrame({"model": ["Regression triangulation"], "2026-01-31": [42]})

    out = canonicalize_external_forecasts(frame, format="wide", source_id="analyst_export")

    assert out.loc[0, "unique_id"] == "series_1"
    assert out.loc[0, "model"] == "Regression triangulation"
    assert out.loc[0, "yhat"] == 42.0


def test_external_forecast_with_forecast_origin_is_unscored_snapshot_not_backtest() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "forecast_origin": ["2025-12-31", "2025-12-31"],
            "ds": ["2026-01-31", "2026-02-28"],
            "yhat": [100, 105],
        }
    )

    out = canonicalize_external_forecasts(frame, model_name="Historical finance model")

    assert "cutoff" in out.columns
    assert out["comparison_evidence_status"].tolist() == [
        "historical_cutoff_labeled_unscored",
        "historical_cutoff_labeled_unscored",
    ]
    assert out["is_backtested"].tolist() == [False, False]
    assert out["backtest_status"].tolist() == ["not_backtested", "not_backtested"]
    assert "has not been independently verified" in out.loc[0, "status_message"]


def test_external_forecast_rejects_cutoff_on_or_after_target_date() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "cutoff": ["2026-01-31", "2026-03-31"],
            "ds": ["2026-01-31", "2026-02-28"],
            "yhat": [100, 105],
            "model": ["Bad plan", "Bad plan"],
        }
    )

    with pytest.raises(ValueError, match="cutoff before ds"):
        canonicalize_external_forecasts(frame)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"unique_id": ["Revenue"], "ds": ["not-a-date"], "yhat": [1], "model": ["Plan"]}),
            "invalid ds dates",
        ),
        (
            pd.DataFrame({"unique_id": ["Revenue"], "ds": ["2026-01-31"], "yhat": ["bad"], "model": ["Plan"]}),
            "non-numeric yhat",
        ),
        (
            pd.DataFrame({"unique_id": ["Revenue"], "ds": ["2026-01-31"], "yhat": [1], "model": [" "]}),
            "blank model",
        ),
        (
            pd.DataFrame(
                {
                    "unique_id": ["Revenue", "Revenue"],
                    "ds": ["2026-01-31", "2026-01-31"],
                    "yhat": [1, 2],
                    "model": ["Plan", "Plan"],
                }
            ),
            "duplicate external forecast rows",
        ),
    ],
)
def test_canonicalize_external_forecasts_fails_closed(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        canonicalize_external_forecasts(frame)


def test_wide_external_forecast_rejects_duplicate_parsed_dates() -> None:
    frame = pd.DataFrame(
        [["Plan", 100, 101]],
        columns=["model", "2026-01-31", pd.Timestamp("2026-01-31")],
    )

    with pytest.raises(ValueError, match="duplicate parsed date columns"):
        canonicalize_external_forecasts(frame, format="wide")


def test_external_forecast_rejects_invalid_horizon_step() -> None:
    frame = pd.DataFrame(
        {"unique_id": ["Revenue"], "ds": ["2026-01-31"], "yhat": [100], "model": ["Plan"], "horizon_step": [0]}
    )

    with pytest.raises(ValueError, match="invalid horizon_step"):
        canonicalize_external_forecasts(frame)


def test_external_forecast_metadata_is_json_serializable() -> None:
    out = canonicalize_external_forecasts(
        pd.DataFrame({"unique_id": ["Revenue"], "ds": ["2026-01-31"], "yhat": [100], "model": ["Plan"]})
    )

    payload = build_external_forecast_metadata(out)

    assert json.loads(json.dumps(payload))["rows"] == 1
