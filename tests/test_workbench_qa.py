from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold.cli import main
from nixtla_scaffold.workbench_qa import GOLDEN_SCENARIOS, SCENARIO_ALIASES, run_workbench_qa


def test_workbench_qa_generates_golden_runs_and_summary(tmp_path) -> None:
    payload = run_workbench_qa(
        output_dir=tmp_path / "qa",
        scenarios=("monthly_basic", "hierarchy_reconciled", "normalized_target_forecast", "custom_model_challenger"),
        model_policy="baseline",
        app_test=False,
        app_test_timeout_seconds=17,
    )

    summary = payload["summary"]
    assert summary["count"] == 4
    assert summary["passed"] == 4
    assert summary["min_usability_score"] >= 90
    assert summary["app_test_timeout_seconds"] == 17
    assert summary["max_streamlit_app_bytes"] > 0
    assert summary["max_csv_artifact_rows"] > 0
    assert summary["performance_status_counts"]["not_measured"] == 4
    summary_frame = pd.read_csv(tmp_path / "qa" / "workbench_qa_summary.csv")
    assert {
        "scenario",
        "status",
        "usability_score",
        "streamlit_compile_status",
        "streamlit_compile_seconds",
        "app_test_timeout_seconds",
        "streamlit_app_bytes",
        "csv_artifact_rows",
        "performance_status",
    }.issubset(summary_frame.columns)
    assert set(summary_frame["app_test_timeout_seconds"]) == {17}
    assert set(summary_frame["scenario"]) == {"monthly_basic", "hierarchy_reconciled", "normalized_target_forecast", "custom_model_challenger"}
    perf_frame = pd.read_csv(tmp_path / "qa" / "workbench_perf_summary.csv")
    assert {"scenario", "performance_status", "streamlit_app_bytes", "csv_artifact_rows"}.issubset(perf_frame.columns)
    assert (tmp_path / "qa" / "workbench_perf_summary.json").exists()
    assert (tmp_path / "qa" / "hierarchy_reconciled" / "hierarchy_reconciliation.csv").exists()
    assert (tmp_path / "qa" / "normalized_target_forecast" / "audit" / "target_transform_audit.csv").exists()
    assert (tmp_path / "qa" / "custom_model_challenger" / "custom_model_contracts.csv").exists()
    assert (tmp_path / "qa" / "custom_model_challenger" / "audit" / "custom_model_invocations.csv").exists()


def test_workbench_qa_runs_streamlit_app_test_when_requested(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("streamlit")

    payload = run_workbench_qa(
        output_dir=tmp_path / "qa",
        scenarios=("limited_history_new_product",),
        model_policy="baseline",
        app_test=True,
    )

    result = payload["results"][0]
    assert result["status"] == "passed"
    assert result["streamlit_compile_status"] == "passed"
    assert result["app_test_status"] == "passed"
    assert result["performance_status"] == "passed"
    assert result["app_test_seconds"] > 0


def test_workbench_qa_cli_writes_summary(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "workbench-qa",
            "--scenarios",
            "short_history",
            "--model-policy",
            "baseline",
            "--no-app-test",
            "--app-test-timeout",
            "23",
            "--output",
            str(tmp_path / "qa"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["count"] == 1
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["app_test_timeout_seconds"] == 23
    assert payload["summary"]["scenarios"] == ["short_history"]
    assert (tmp_path / "qa" / "short_history" / "forecast.csv").exists()
    assert (tmp_path / "qa" / "workbench_qa_summary.json").exists()
    assert (tmp_path / "qa" / "workbench_perf_summary.json").exists()


def test_workbench_qa_defaults_use_clear_canonical_scenario_names() -> None:
    assert GOLDEN_SCENARIOS == (
        "monthly_basic",
        "limited_history_new_product",
        "hierarchy_reconciled",
        "normalized_target_forecast",
        "custom_model_challenger",
    )
    assert SCENARIO_ALIASES["short_history"] == "limited_history_new_product"
    assert SCENARIO_ALIASES["transform_normalized"] == "normalized_target_forecast"


def test_workbench_qa_rejects_invalid_app_test_timeout(tmp_path) -> None:
    pytest = __import__("pytest")

    with pytest.raises(ValueError, match="app_test_timeout_seconds must be >= 1"):
        run_workbench_qa(output_dir=tmp_path / "qa", scenarios=("limited_history_new_product",), app_test_timeout_seconds=0)
