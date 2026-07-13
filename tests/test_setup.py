from __future__ import annotations

import json

import yaml

from nixtla_scaffold.cli import main
from nixtla_scaffold.schema import ResearchBudget
from nixtla_scaffold.setup import SetupAnswers, create_forecast_setup, setup_questions


def test_setup_questions_cover_agent_intake_topics() -> None:
    questions = {item["id"]: item for item in setup_questions(SetupAnswers(data_source="kusto", mcp_regressor_search=True))}

    assert set(questions) == {
        "decision",
        "target_context",
        "preset",
        "data_source",
        "input_path",
        "series_count",
        "intervals",
        "model_families",
        "research_budget",
        "source_discovery",
        "exploration_mode",
        "mcp_regressor_search",
        "outputs",
    }
    assert questions["preset"]["answer"] == "standard"
    assert "Where is the data coming from?" in questions["data_source"]["question"]
    assert "Prediction intervals" in questions["intervals"]["caveat"]
    assert "MLForecast" in questions["model_families"]["caveat"]
    assert "NeuralForecast is research-only" in questions["model_families"]["caveat"]
    assert questions["mcp_regressor_search"]["answer"] is True
    assert "leakage" in questions["mcp_regressor_search"]["caveat"]
    assert questions["research_budget"]["answer"]["profile"] == "balanced"
    assert questions["source_discovery"]["answer"] is True


def test_setup_legacy_aliases_canonicalize_in_generated_config() -> None:
    answers = SetupAnswers(preset="finance", model_families=("auto",))

    assert answers.preset == "standard"
    assert answers.model_families == ("light",)
    assert next(item for item in setup_questions(answers) if item["id"] == "preset")["answer"] == "standard"


def test_create_forecast_setup_writes_agent_workspace(tmp_path) -> None:
    artifact = create_forecast_setup(
        tmp_path / "usage_overage",
        SetupAnswers(
            name="usage_overage",
            data_source="kusto",
            series_count="single",
            target_name="ARR_30day_avg",
            time_col="day_dt",
            id_value="Usage Overage ARR",
            horizon=6,
            freq="ME",
            intervals="auto",
            model_families=("statsforecast", "mlforecast", "hierarchicalforecast"),
            exploration_mode=True,
            source_discovery=True,
            mcp_regressor_search=True,
            research_budget=ResearchBudget(profile="deep"),
            decision="Set the hosted-compute plan",
            audience="Finance leadership",
            target_semantics="Monthly usage overage ARR",
            units="USD ARR",
            grain="monthly",
            refresh_cadence="monthly",
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
    assert artifact.files["context"].exists()
    assert artifact.files["query_template"].suffix == ".kql"
    config = yaml.safe_load(artifact.files["config"].read_text(encoding="utf-8"))
    assert config["answers"]["data_source"] == "kusto"
    assert config["answers"]["model_families"] == ["statsforecast", "mlforecast", "hierarchicalforecast"]
    assert config["forecast_preset"] == "standard"
    assert config["forecast_spec"]["horizon"] == 6
    assert config["forecast_spec"]["freq"] == "ME"
    assert config["forecast_spec"]["context"]["research_budget"]["profile"] == "deep"
    assert config["forecast_context"]["decision"] == "Set the hosted-compute plan"
    assert "preset_catalog" in config
    assert "MLForecast" in config["policy"]["model_family_caveat"]
    assert config["policy"]["mcp_regressor_search_allowed"] is True
    assert "nixtla-scaffold ingest" in artifact.next_commands[0]
    assert "--preset standard" in artifact.next_commands[0]
    assert "--forecast-output" in artifact.next_commands[0]
    assert "--context-file" in artifact.next_commands[0]
    context = json.loads(artifact.files["context"].read_text(encoding="utf-8"))
    assert context["sources"][0]["status"] == "planned"
    assert context["research_budget"]["max_source_queries"] == 30
    brief = artifact.files["agent_brief"].read_text(encoding="utf-8")
    assert "Required exploration checks" in brief
    assert "Model families" in brief
    assert "MCP regressor search allowed" in brief
    assert "bounded schema/count/sample/aggregate queries" in brief
    assert "generated run_streamlit.ps1 / streamlit_app.py first" in brief
    assert "Do not stitch per-series winners" in brief
    assert "Do not invoke Excel COM" in brief
    assert "installed skill and executable package disagree" in brief


def test_setup_uses_real_single_series_input_and_emits_signal_artifacts(tmp_path) -> None:
    input_path = tmp_path / "source data" / "air passengers.csv"
    input_path.parent.mkdir()
    input_path.write_text("month,passengers\n2025-01-31,100\n", encoding="utf-8")

    artifact = create_forecast_setup(
        tmp_path / "air workspace",
        SetupAnswers(
            name="air_passengers",
            data_source="csv",
            input_path=str(input_path),
            series_count="single",
            target_name="passengers",
            time_col="month",
            id_value="AirPassengers",
            horizon=12,
            freq="ME",
            preset="accuracy-first",
            research_budget=ResearchBudget(profile="balanced"),
        ),
    )

    command = artifact.next_commands[0]
    context = json.loads(artifact.files["context"].read_text(encoding="utf-8"))
    assert f'--input "{input_path}"' in command
    assert '--id-value "AirPassengers"' in command
    assert "--time-col month --target-col passengers" in command
    assert context["sources"][0]["status"] == "available"
    assert context["sources"][0]["provenance"] == str(input_path.resolve())
    assert artifact.files["signal_needs"].exists()
    assert artifact.files["signal_probe_ledger"].exists()
    assert artifact.files["signal_contracts"].exists()


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
            "--research-budget",
            "time-boxed",
            "--decision",
            "Set plan",
            "--audience",
            "CFO",
            "--target-semantics",
            "Monthly ARR",
            "--units",
            "USD",
            "--grain",
            "monthly by product",
            "--refresh-cadence",
            "monthly",
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
    assert (workspace / "forecast_context.json").exists()
    assert (workspace / "queries" / "source.dax").exists()
    assert "nixtla-scaffold ingest" in payload["next_commands"][0]
    config = yaml.safe_load((workspace / "forecast_setup.yaml").read_text(encoding="utf-8"))
    assert config["forecast_preset"] == "strict"
    assert config["forecast_spec"]["strict_cv_horizon"] is True
    assert config["forecast_context"]["research_budget"]["profile"] == "time-boxed"
    assert config["forecast_context"]["audience"] == "CFO"
