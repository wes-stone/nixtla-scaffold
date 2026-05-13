from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from nixtla_scaffold import ForecastSpec, refresh_pipeline, run_forecast, run_pipeline
from nixtla_scaffold.cli import main
from nixtla_scaffold.pipelines import PIPELINE_MANIFEST_FILE, PIPELINE_SUMMARY_FILE


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _write_extract_script(path, *, values: list[int] | None = None) -> None:
    values_literal = values or [100, 104, 109, 115, 122, 130, 139, 149]
    _write(
        path,
        f"""
        import argparse
        from pathlib import Path

        import pandas as pd

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", required=True)
        parser.add_argument("--query-file", default="")
        args = parser.parse_args()

        if args.query_file:
            Path(args.query_file).read_text(encoding="utf-8")

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({{
            "Product": ["Contoso"] * {len(values_literal)},
            "FiscalMonth": pd.date_range("2025-01-31", periods={len(values_literal)}, freq="ME"),
            "Sales": {values_literal!r},
        }}).to_csv(output, index=False)
        """,
    )


def _write_transform_script(path) -> None:
    _write(
        path,
        """
        import argparse
        from pathlib import Path

        import pandas as pd

        parser = argparse.ArgumentParser()
        parser.add_argument("--sales", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()

        raw = pd.read_csv(args.sales)
        canonical = raw.rename(columns={"Product": "unique_id", "FiscalMonth": "ds", "Sales": "y"})
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        canonical[["unique_id", "ds", "y"]].to_csv(output, index=False)
        """,
    )


def _write_pipeline_config(path, *, transform: bool = True, forecast_output: str = "forecast") -> None:
    id_col, time_col, target_col = ("unique_id", "ds", "y") if transform else ("Product", "FiscalMonth", "Sales")
    lines = [
        "schema_version: nixtla_scaffold.pipeline.v1",
        "name: contoso_dax_test",
        "extracts:",
        "  - name: sales",
        r"    script: scripts\extract_sales.py",
        r"    query_file: queries\sales.dax",
        r"    output: raw\sales.csv",
        "    args:",
        "      - --output",
        '      - "{output:sales}"',
        "      - --query-file",
        '      - "{query_file}"',
    ]
    if transform:
        lines.extend(
            [
                "transform:",
                r"  script: scripts\build_forecast_input.py",
                r"  output: prepared\transformed.csv",
                "  args:",
                "    - --sales",
                '    - "{output:sales}"',
                "    - --output",
                '    - "{output:forecast_input}"',
            ]
        )
    lines.extend(
        [
            "canonical:",
            r"  output: prepared\forecast_input.csv",
            f"  id_col: {id_col}",
            f"  time_col: {time_col}",
            f"  target_col: {target_col}",
            "forecast:",
            f"  output: {forecast_output}",
            "  preset: quick",
            "  horizon: 2",
            "  freq: ME",
            "  model_policy: baseline",
            "  verbose: false",
        ]
    )
    _write(path, "\n".join(lines))


def test_pipeline_run_executes_extract_transform_forecast_and_attaches_manifest(tmp_path) -> None:
    config = tmp_path / "pipeline.yaml"
    _write_extract_script(tmp_path / "scripts" / "extract_sales.py")
    _write_transform_script(tmp_path / "scripts" / "build_forecast_input.py")
    _write(tmp_path / "queries" / "sales.dax", "EVALUATE ROW(\"Sales\", 1)")
    _write_pipeline_config(config)

    output_dir = tmp_path / "pipeline_run"
    manifest = run_pipeline(config, output_dir)

    assert manifest["status"] == "succeeded"
    assert (output_dir / PIPELINE_MANIFEST_FILE).exists()
    assert (output_dir / "prepared" / "forecast_input.csv").exists()
    assert (output_dir / "prepared" / "forecast_input.source.json").exists()
    assert (output_dir / "forecast" / "forecast.csv").exists()
    assert (output_dir / "forecast" / "appendix" / "source_pipeline_manifest.json").exists()
    assert (output_dir / PIPELINE_SUMMARY_FILE).exists()
    assert (output_dir / "forecast" / "appendix" / "source_pipeline_summary.md").exists()
    assert [step["role"] for step in manifest["steps"]] == ["extract", "transform"]
    assert manifest["steps"][0]["query_files"][0]["sha256"]

    summary = (output_dir / PIPELINE_SUMMARY_FILE).read_text(encoding="utf-8")
    assert "```mermaid" in summary
    assert "flowchart LR" in summary
    assert "extract: sales" in summary
    assert "transform: transform" in summary
    assert "canonical input" in summary
    assert "forecast run" in summary

    run_manifest = json.loads((output_dir / "forecast" / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["outputs"]["source_pipeline_manifest"] == "appendix/source_pipeline_manifest.json"
    assert run_manifest["outputs"]["source_pipeline_summary"] == "appendix/source_pipeline_summary.md"
    assert run_manifest["source_pipeline"]["name"] == "contoso_dax_test"
    assert run_manifest["source_pipeline"]["pipeline_summary"] == str(output_dir / PIPELINE_SUMMARY_FILE)


def test_checked_in_contoso_dax_pipeline_executes_scaffold_extracts(tmp_path) -> None:
    config = Path(__file__).resolve().parents[1] / "examples" / "contoso_dax_pipeline" / "pipeline.yaml"
    output_dir = tmp_path / "contoso_dax_example"

    manifest = run_pipeline(config, output_dir, forecast=False)

    assert manifest["status"] == "succeeded"
    assert [step["name"] for step in manifest["steps"]] == [
        "sales_actuals",
        "product_dimension",
        "transform",
    ]
    assert manifest["steps"][0]["query_files"][0]["path"].endswith("contoso_sales_fact.dax")
    assert manifest["steps"][1]["query_files"][0]["path"].endswith("contoso_product_dimension.dax")
    sales_output = manifest["steps"][0]["outputs"]["sales_actuals"]
    assert sales_output["date_column"] == "DateKey"
    assert sales_output["start"] >= "2025-01-01"
    assert sales_output["end"] <= "2025-12-31"

    sales = pd.read_csv(output_dir / "raw" / "contoso_sales.csv")
    products = pd.read_csv(output_dir / "raw" / "contoso_products.csv")
    canonical = pd.read_csv(output_dir / "prepared" / "forecast_input.csv")

    assert {"SalesKey", "ProductKey", "DateKey", "Quantity", "Amount"}.issubset(sales.columns)
    assert len(sales) == 100
    assert {"ProductKey", "ProductName", "Category", "Price"}.issubset(products.columns)
    assert len(products) == 5
    assert list(canonical.columns) == ["unique_id", "ds", "y"]
    assert canonical["unique_id"].nunique() == 2
    assert len(canonical) > 0

    summary = (output_dir / PIPELINE_SUMMARY_FILE).read_text(encoding="utf-8")
    assert "```mermaid" in summary
    assert "extract: sales_actuals" in summary
    assert "extract: product_dimension" in summary
    assert "Prepared input" in summary


def test_checked_in_contoso_kql_pipeline_executes_scaffold_extracts(tmp_path) -> None:
    config = Path(__file__).resolve().parents[1] / "examples" / "contoso_kql_pipeline" / "pipeline.yaml"
    output_dir = tmp_path / "contoso_kql_example"

    manifest = run_pipeline(config, output_dir, forecast=False)

    assert manifest["status"] == "succeeded"
    assert [step["name"] for step in manifest["steps"]] == [
        "revenue_actuals",
        "product_dimension",
        "transform",
    ]
    assert manifest["steps"][0]["query_files"][0]["path"].endswith("contoso_revenue_by_month.kql")
    assert manifest["steps"][1]["query_files"][0]["path"].endswith("contoso_product_dimension.kql")

    revenue = pd.read_csv(output_dir / "raw" / "contoso_revenue.csv")
    products = pd.read_csv(output_dir / "raw" / "contoso_products.csv")
    canonical = pd.read_csv(output_dir / "prepared" / "forecast_input.csv")

    assert {"Month", "ProductCategoryName", "Revenue", "TotalCost", "Transactions"}.issubset(revenue.columns)
    assert len(revenue) == 288
    assert {"ProductKey", "ProductName", "ProductCategoryName", "ProductSubcategoryName", "Manufacturer"}.issubset(
        products.columns
    )
    assert len(products) == 16
    assert list(canonical.columns) == ["unique_id", "ds", "y"]
    assert canonical["unique_id"].nunique() == 8
    assert len(canonical) == 288
    assert canonical["ds"].min() == "2007-01-31"
    assert canonical["ds"].max() == "2009-12-31"

    summary = (output_dir / PIPELINE_SUMMARY_FILE).read_text(encoding="utf-8")
    assert "```mermaid" in summary
    assert "extract: revenue_actuals" in summary
    assert "extract: product_dimension" in summary
    assert "Prepared input" in summary


def test_pipeline_cli_can_prepare_single_extract_without_forecast(tmp_path) -> None:
    config = tmp_path / "pipeline.yaml"
    _write_extract_script(tmp_path / "scripts" / "extract_sales.py")
    _write(tmp_path / "queries" / "sales.dax", "EVALUATE ROW(\"Sales\", 1)")
    _write_pipeline_config(config, transform=False)

    output_dir = tmp_path / "pipeline_prepare"
    exit_code = main(["pipeline", "run", "--config", str(config), "--output", str(output_dir), "--no-forecast"])

    assert exit_code == 0
    assert (output_dir / "prepared" / "forecast_input.csv").exists()
    assert (output_dir / PIPELINE_SUMMARY_FILE).exists()
    assert not (output_dir / "forecast").exists()
    manifest = json.loads((output_dir / PIPELINE_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["forecast"] == {}
    assert manifest["outputs"]["pipeline_summary"] == str(output_dir / PIPELINE_SUMMARY_FILE)
    assert "Prepared input" in (output_dir / PIPELINE_SUMMARY_FILE).read_text(encoding="utf-8")


def test_pipeline_failure_writes_failed_manifest(tmp_path) -> None:
    config = tmp_path / "pipeline.yaml"
    _write(
        tmp_path / "scripts" / "fail.py",
        """
        import sys

        print("extract failed")
        sys.exit(7)
        """,
    )
    _write(
        config,
        r"""
        schema_version: nixtla_scaffold.pipeline.v1
        name: failing_pipeline
        extracts:
          - name: bad_extract
            script: scripts\fail.py
            output: raw\missing.csv
        forecast:
          enabled: false
        """,
    )

    output_dir = tmp_path / "pipeline_failure"
    with pytest.raises(RuntimeError):
        run_pipeline(config, output_dir, forecast=False)

    manifest = json.loads((output_dir / PIPELINE_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["steps"][0]["exit_code"] == 7
    assert manifest["error"]["type"] == "RuntimeError"
    summary = (output_dir / PIPELINE_SUMMARY_FILE).read_text(encoding="utf-8")
    assert "```mermaid" in summary
    assert "failed: no canonical input" in summary
    assert "pipeline extract 'bad_extract' failed with exit code 7" in summary


def test_pipeline_refresh_reuses_previous_spec_and_writes_refresh_delta(tmp_path) -> None:
    previous_history = pd.DataFrame(
        {
            "unique_id": ["Contoso"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 104, 109, 115, 122, 130, 139, 149],
        }
    )
    previous_run = run_forecast(previous_history, ForecastSpec(horizon=2, freq="ME", model_policy="baseline", verbose=False)).to_directory(
        tmp_path / "previous_run"
    )

    config = tmp_path / "pipeline.yaml"
    _write_extract_script(tmp_path / "scripts" / "extract_sales.py", values=[101, 106, 112, 119, 127, 136, 146, 157, 169])
    _write_transform_script(tmp_path / "scripts" / "build_forecast_input.py")
    _write(tmp_path / "queries" / "sales.dax", "EVALUATE ROW(\"Sales\", 1)")
    _write_pipeline_config(config, forecast_output="refreshed_forecast")

    output_dir = tmp_path / "pipeline_refresh"
    manifest = refresh_pipeline(config, previous_run, output_dir)

    refreshed = output_dir / "refreshed_forecast"
    assert manifest["forecast"]["mode"] == "refresh"
    assert refreshed.exists()
    assert (refreshed / "appendix" / "refresh_delta.csv").exists()
    assert (refreshed / "refresh_manifest.json").exists()
    assert (refreshed / "appendix" / "source_pipeline_manifest.json").exists()
    assert (refreshed / "appendix" / "source_pipeline_summary.md").exists()
