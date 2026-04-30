from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_example(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quickstart_csv_example_runs(tmp_path) -> None:
    module = _load_example("examples/quickstart_csv/forecast_quick.py")

    output_dir = module.run_example(tmp_path / "quickstart")

    assert (output_dir / "forecast.csv").exists()
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["spec"]["model_policy"] == "baseline"


def test_serious_finance_example_runs_with_transform_and_event(tmp_path) -> None:
    module = _load_example("examples/serious_finance_forecast/forecast_finance.py")

    output_dir = module.run_example(tmp_path / "finance")

    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "audit" / "target_transform_audit.csv").exists()
    forecast = pd.read_csv(output_dir / "forecast.csv")
    assert "yhat_scenario" in forecast.columns


def test_hierarchy_reconciliation_example_runs(tmp_path) -> None:
    module = _load_example("examples/hierarchy_reconciliation/forecast_hierarchy.py")

    output_dir = module.run_example(tmp_path / "hierarchy")

    assert (output_dir / "hierarchy_reconciliation.csv").exists()
    post = pd.read_csv(output_dir / "audit" / "hierarchy_coherence_post.csv")
    assert pd.to_numeric(post["gap"], errors="coerce").abs().max() < 1e-8


def test_dataframe_template_runs(tmp_path) -> None:
    module = _load_example("examples/python_api_templates/dataframe_forecast.py")

    output_dir = module.run_example(tmp_path / "dataframe")

    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "streamlit_app.py").exists()


def test_custom_finance_model_example_runs(tmp_path) -> None:
    module = _load_example("examples/custom_finance_model/forecast_custom.py")

    output_dir = module.run_example(tmp_path / "custom_finance")

    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "custom_model_contracts.csv").exists()
    assert (output_dir / "audit" / "custom_model_invocations.csv").exists()
    forecast_long = pd.read_csv(output_dir / "forecast_long.csv")
    assert "Custom_MoM_Growth_FY_Seasonal" in set(forecast_long["model"])
    contracts = pd.read_csv(output_dir / "custom_model_contracts.csv")
    assert contracts["status"].iloc[0] == "succeeded"
    invocations = pd.read_csv(output_dir / "audit" / "custom_model_invocations.csv")
    assert set(invocations["status"]) == {"succeeded"}


def test_custom_finance_model_allocates_calendar_annual_target() -> None:
    module = _load_example("examples/custom_finance_model/finance_seasonality_model.py")
    history = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2024-01-31", periods=12, freq="ME"),
            "y": [100.0] * 12,
        }
    )
    future_grid = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
        }
    )

    output = module.build_finance_forecast(history, future_grid=future_grid, annual_target=1200.0)

    assert abs(float(output["yhat"].sum()) - 1200.0) < 1e-6
    assert set(output.columns) == {"unique_id", "ds", "yhat"}
    assert output["yhat"].notna().all()


def test_custom_finance_model_ignores_partial_year_for_seasonality() -> None:
    module = _load_example("examples/custom_finance_model/finance_seasonality_model.py")
    complete_year = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2024-01-31", periods=12, freq="ME"),
            "y": [100.0] * 12,
        }
    )
    partial_year = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 2,
            "ds": pd.to_datetime(["2025-01-31", "2025-02-28"]),
            "y": [10000.0, 10000.0],
        }
    )
    future_grid = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2026-01-31", periods=12, freq="ME"),
        }
    )

    output = module.build_finance_forecast(
        pd.concat([complete_year, partial_year], ignore_index=True),
        future_grid=future_grid,
        annual_target=1200.0,
    )

    january = output.loc[pd.to_datetime(output["ds"]).dt.month == 1, "yhat"].iloc[0]
    assert abs(float(january) - 100.0) < 1e-6
