from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold.cli import main
from nixtla_scaffold.connectors import ingest_query_result


def test_ingest_columnar_mcp_result_to_canonical_csv(tmp_path) -> None:
    result_path = tmp_path / "kusto_result.json"
    query_path = tmp_path / "premium.kql"
    output_path = tmp_path / "premium_input.csv"
    result_path.write_text(
        json.dumps(
            {
                "format": "columnar",
                "data": {
                    "day_dt": ["2025-06-30", "2025-07-31", "2025-08-31"],
                    "ARR_30day_avg": [100.0, 112.0, 125.0],
                },
            }
        ),
        encoding="utf-8",
    )
    query_path.write_text("database('copilot').copilot_revenue_daily\n", encoding="utf-8")

    metadata = ingest_query_result(
        result_path,
        output_path,
        source_kind="kusto",
        query_file=query_path,
        id_value="Premium Overage ARR",
        time_col="day_dt",
        target_col="ARR_30day_avg",
    )

    canonical = pd.read_csv(output_path)
    assert canonical.columns.tolist() == ["unique_id", "ds", "y"]
    assert canonical["unique_id"].unique().tolist() == ["Premium Overage ARR"]
    assert metadata["rows"] == 3
    assert output_path.with_suffix(".source.json").exists()
    assert output_path.with_suffix(".kql").read_text(encoding="utf-8").startswith("database")


def test_profile_accepts_frequency_hints_for_user_smoke_flow(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 2,
            "ds": ["2026-01-31", "2026-03-31"],
            "y": [100, 120],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(["profile", "--input", str(input_path), "--freq", "ME"])

    assert exit_code == 0
    assert '"freq": "ME"' in capsys.readouterr().out


def test_ingest_cli_can_forecast_immediately_and_write_reports(tmp_path) -> None:
    result_path = tmp_path / "dax_export.jsonl"
    output_path = tmp_path / "dax_input.csv"
    run_dir = tmp_path / "run"
    rows = [
        {"FiscalMonth": "2025-01-31", "ARR": 100},
        {"FiscalMonth": "2025-02-28", "ARR": 105},
        {"FiscalMonth": "2025-03-31", "ARR": 109},
        {"FiscalMonth": "2025-04-30", "ARR": 114},
        {"FiscalMonth": "2025-05-31", "ARR": 120},
    ]
    result_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(result_path),
            "--source",
            "dax",
            "--id-value",
            "ARR",
            "--time-col",
            "FiscalMonth",
            "--target-col",
            "ARR",
            "--output",
            str(output_path),
            "--forecast-output",
            str(run_dir),
            "--freq",
            "ME",
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert (run_dir / "forecast.csv").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "report_base64.txt").exists()
    assert (run_dir / "streamlit_app.py").exists()


def test_ingest_custom_flags_require_forecast_output(tmp_path, capsys) -> None:
    result_path = tmp_path / "dax_export.jsonl"
    output_path = tmp_path / "dax_input.csv"
    result_path.write_text(json.dumps({"FiscalMonth": "2025-01-31", "ARR": 100}) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(result_path),
            "--source",
            "dax",
            "--id-value",
            "ARR",
            "--time-col",
            "FiscalMonth",
            "--target-col",
            "ARR",
            "--output",
            str(output_path),
            "--custom-model",
            "demo:model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "custom model flags require --forecast-output" in captured.err
    assert not output_path.exists()


def test_ingest_forecast_failure_diagnostics_use_forecast_output_dir(tmp_path, capsys) -> None:
    result_path = tmp_path / "dax_export.jsonl"
    output_path = tmp_path / "dax_input.csv"
    run_dir = tmp_path / "forecast run"
    rows = [
        {"FiscalMonth": "2025-01-31", "ARR": 100},
        {"FiscalMonth": "2025-02-28", "ARR": 105},
    ]
    result_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(result_path),
            "--source",
            "dax",
            "--id-value",
            "ARR",
            "--time-col",
            "FiscalMonth",
            "--target-col",
            "ARR",
            "--output",
            str(output_path),
            "--forecast-output",
            str(run_dir),
            "--freq",
            "ME",
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
            "--require-backtest",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Failure diagnostics written" in captured.err
    assert output_path.exists()
    assert (run_dir / "failure_diagnostics.json").exists()
    assert (run_dir / "failure_diagnostics.md").exists()
    assert not (tmp_path / "failure_diagnostics.json").exists()
