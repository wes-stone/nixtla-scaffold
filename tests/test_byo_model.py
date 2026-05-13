from __future__ import annotations

import json

import pandas as pd
import pytest

from nixtla_scaffold.byo_model import (
    BYO_MODEL_SCHEMA_VERSION,
    ingest_byo_model_forecasts,
    load_byo_model_forecasts,
    write_byo_model_comparison,
    write_byo_model_scores,
)


GROUP_COLS = ("ProductGroup", "ProductLine", "Product")


def _write_byo_workbook(path, *, include_cutoff: bool = False) -> None:
    base = pd.DataFrame(
        {
            "ProductGroup": ["Cloud", "Cloud"],
            "ProductLine": ["Workspace", "Workspace"],
            "Product": ["Seats", "Usage"],
            "ds": ["2026-01-31", "2026-01-31"],
            "yhat": [110.0, 55.0],
            **({"cutoff": ["2025-12-31", "2025-12-31"]} if include_cutoff else {}),
        }
    )
    bull = base.copy()
    bull["yhat"] = [120.0, 60.0]
    with pd.ExcelWriter(path) as writer:
        base.to_excel(writer, sheet_name="Base", index=False)
        bull.to_excel(writer, sheet_name="Bull", index=False)


def test_byo_model_ingest_multisheet_generates_scenarios_and_rollups(tmp_path) -> None:
    workbook = tmp_path / "finance_model.xlsx"
    _write_byo_workbook(workbook)

    result = ingest_byo_model_forecasts(workbook, sheets=("Base", "Bull"), group_cols=GROUP_COLS, output_dir=tmp_path / "out")
    result.to_directory(tmp_path / "out")

    forecasts = result.forecasts
    assert result.manifest["schema_version"] == BYO_MODEL_SCHEMA_VERSION
    assert set(forecasts["scenario_name"]) == {"Base", "Bull"}
    assert set(forecasts["sheet"].astype(str)) == {"Base", "Bull"}
    assert "ProductGroup=Cloud|ProductLine=Workspace|Product=Seats" in set(forecasts["unique_id"])
    assert "ProductGroup=Cloud|ProductLine=Workspace" in set(forecasts["unique_id"])
    assert "ProductGroup=Cloud" in set(forecasts["unique_id"])
    assert "Total" in set(forecasts["unique_id"])
    assert set(forecasts["external_rollup_source"]) == {"leaf", "derived_sum"}

    out = tmp_path / "out"
    assert (out / "byo_model_forecasts.csv").exists()
    assert json.loads((out / "byo_model_manifest.json").read_text())["operation"] == "ingest"


def test_byo_model_grouped_import_rejects_workbook_subtotal_rows(tmp_path) -> None:
    workbook = tmp_path / "finance_model.xlsx"
    subtotal = pd.DataFrame(
        {
            "ProductGroup": ["Cloud", "Total"],
            "ProductLine": ["Workspace", "Subtotal"],
            "Product": ["Seats", "All"],
            "ds": ["2026-01-31", "2026-01-31"],
            "yhat": [110.0, 165.0],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        subtotal.to_excel(writer, sheet_name="Base", index=False)

    with pytest.raises(ValueError, match="leaf-level rows only"):
        load_byo_model_forecasts(workbook, sheets=("Base",), group_cols=GROUP_COLS)


def test_byo_model_compare_writes_grouped_summary(tmp_path) -> None:
    workbook = tmp_path / "finance_model.xlsx"
    _write_byo_workbook(workbook)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "unique_id": [
                "ProductGroup=Cloud|ProductLine=Workspace|Product=Seats",
                "ProductGroup=Cloud|ProductLine=Workspace|Product=Usage",
                "ProductGroup=Cloud",
                "Total",
            ],
            "ds": ["2026-01-31"] * 4,
            "yhat": [100.0, 50.0, 150.0, 150.0],
            "model": ["ScaffoldChampion"] * 4,
        }
    ).to_csv(run_dir / "forecast.csv", index=False)

    result = write_byo_model_comparison(
        run_dir,
        workbook,
        output_dir=run_dir / "byo_model",
        sheets=("Base",),
        group_cols=GROUP_COLS,
        main_model_preference="Base",
    )

    assert (run_dir / "byo_model" / "forecast_comparison.csv").exists()
    assert (run_dir / "byo_model" / "byo_model_comparison_summary.csv").exists()
    assert result.comparison.manifest["alignment"]["rows_aligned"] >= 3
    assert result.manifest["main_model_preference"] == "Base"
    assert result.manifest["main_model_preference_scope"] == "display_only"
    assert "hierarchy_level" in result.byo_summary.columns
    assert "ProductGroup" in result.byo_summary.columns


def test_byo_model_score_rolls_actuals_and_summarizes_versions(tmp_path) -> None:
    workbook = tmp_path / "finance_snapshots.xlsx"
    _write_byo_workbook(workbook, include_cutoff=True)
    actuals = tmp_path / "actuals.csv"
    pd.DataFrame(
        {
            "ProductGroup": ["Cloud", "Cloud"],
            "ProductLine": ["Workspace", "Workspace"],
            "Product": ["Seats", "Usage"],
            "ds": ["2026-01-31", "2026-01-31"],
            "y": [100.0, 50.0],
        }
    ).to_csv(actuals, index=False)

    result = write_byo_model_scores(workbook, actuals, tmp_path / "scores", sheets=("Base", "Bull"), group_cols=GROUP_COLS)

    assert (tmp_path / "scores" / "external_backtest_long.csv").exists()
    assert (tmp_path / "scores" / "byo_model_score_summary.csv").exists()
    assert set(result.byo_summary["scenario_name"]) == {"Base", "Bull"}
    assert "Total" in set(result.scores.backtest_long["unique_id"])
    assert result.scores.manifest["metric_status"] == "scored"
