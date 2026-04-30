from __future__ import annotations

import json

import yaml

from nixtla_scaffold.cli import main
from nixtla_scaffold.setup import SetupAnswers, create_forecast_setup, setup_questions


def test_setup_questions_cover_agent_intake_topics() -> None:
    questions = {item["id"]: item for item in setup_questions(SetupAnswers(data_source="kusto", mcp_regressor_search=True))}

    assert set(questions) == {
        "preset",
        "data_source",
        "series_count",
        "intervals",
        "model_families",
        "exploration_mode",
        "mcp_regressor_search",
        "outputs",
    }
    assert questions["preset"]["answer"] == "finance"
    assert "Where is the data coming from?" in questions["data_source"]["question"]
    assert "Prediction intervals" in questions["intervals"]["caveat"]
    assert "MLForecast" in questions["model_families"]["caveat"]
    assert "NeuralForecast is research-only" in questions["model_families"]["caveat"]
    assert questions["mcp_regressor_search"]["answer"] is True
    assert "leakage" in questions["mcp_regressor_search"]["caveat"]


def test_create_forecast_setup_writes_agent_workspace(tmp_path) -> None:
    artifact = create_forecast_setup(
        tmp_path / "premium_overage",
        SetupAnswers(
            name="premium_overage",
            data_source="kusto",
            series_count="single",
            target_name="ARR_30day_avg",
            time_col="day_dt",
            id_value="Premium Overage ARR",
            horizon=6,
            freq="ME",
            intervals="auto",
            model_families=("statsforecast", "mlforecast", "hierarchicalforecast"),
            exploration_mode=True,
            mcp_regressor_search=True,
            outputs=("all",),
        ),
    )

    assert (artifact.workspace / "queries").is_dir()
    assert (artifact.workspace / "data" / "raw").is_dir()
    assert (artifact.workspace / "data" / "canonical").is_dir()
    assert (artifact.workspace / "outputs").is_dir()
    assert artifact.files["config"].exists()
    assert artifact.files["questions"].exists()
    assert artifact.files["agent_brief"].exists()
    assert artifact.files["query_template"].suffix == ".kql"
    config = yaml.safe_load(artifact.files["config"].read_text(encoding="utf-8"))
    assert config["answers"]["data_source"] == "kusto"
    assert config["answers"]["model_families"] == ["statsforecast", "mlforecast", "hierarchicalforecast"]
    assert config["forecast_preset"] == "finance"
    assert config["forecast_spec"]["horizon"] == 6
    assert config["forecast_spec"]["freq"] == "ME"
    assert "preset_catalog" in config
    assert "MLForecast" in config["policy"]["model_family_caveat"]
    assert config["policy"]["mcp_regressor_search_allowed"] is True
    assert "nixtla-scaffold ingest" in artifact.next_commands[0]
    assert "--preset finance" in artifact.next_commands[0]
    assert "--forecast-output" in artifact.next_commands[0]
    brief = artifact.files["agent_brief"].read_text(encoding="utf-8")
    assert "Required exploration checks" in brief
    assert "Model families" in brief
    assert "MCP regressor search allowed" in brief


def test_setup_cli_writes_questions_and_next_commands(tmp_path, capsys) -> None:
    workspace = tmp_path / "setup"

    exit_code = main(
        [
            "setup",
            "--workspace",
            str(workspace),
            "--name",
            "dax_arr",
            "--data-source",
            "dax",
            "--preset",
            "strict",
            "--series-count",
            "many",
            "--target-name",
            "ARR",
            "--time-col",
            "FiscalMonth",
            "--id-col",
            "Product",
            "--horizon",
            "6",
            "--freq",
            "ME",
            "--intervals",
            "intervals",
            "--model-families",
            "statsforecast",
            "mlforecast",
            "hierarchicalforecast",
            "--mcp-regressor-search",
            "--outputs",
            "excel",
            "html",
            "diagnostics",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == str(workspace)
    assert (workspace / "forecast_setup.yaml").exists()
    assert (workspace / "questions.json").exists()
    assert (workspace / "agent_brief.md").exists()
    assert (workspace / "queries" / "source.dax").exists()
    assert "nixtla-scaffold ingest" in payload["next_commands"][0]
    config = yaml.safe_load((workspace / "forecast_setup.yaml").read_text(encoding="utf-8"))
    assert config["forecast_preset"] == "strict"
    assert config["forecast_spec"]["strict_cv_horizon"] is True
