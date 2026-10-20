from __future__ import annotations

import json

import yaml
import pytest

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
    assert questions["preset"]["answer"] == "accuracy-first"
    assert "Where is the data coming from?" in questions["data_source"]["question"]
    assert "Prediction intervals" in questions["intervals"]["caveat"]
    assert "MLForecast" in questions["model_families"]["caveat"]
    assert "neuralforecast_research" not in questions["model_families"]["options"]
    assert questions["mcp_regressor_search"]["answer"] is True
    assert "leakage" in questions["mcp_regressor_search"]["caveat"]
    assert questions["research_budget"]["answer"]["profile"] == "balanced"
    assert questions["source_discovery"]["answer"] is True
    assert all(item["reference_only"] for item in questions.values())


def test_setup_rejects_retired_neural_family() -> None:
    with pytest.raises(ValueError, match="neuralforecast_research"):
        SetupAnswers(model_families=("neuralforecast_research",))


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
    assert config["forecast_preset"] == "accuracy-first"
    assert config["forecast_spec"]["horizon"] == 6
    assert config["forecast_spec"]["freq"] == "ME"
    assert config["forecast_spec"]["context"]["research_budget"]["profile"] == "deep"
    assert config["forecast_context"]["decision"] == "Set the hosted-compute plan"
    assert "preset_catalog" in config
    assert "MLForecast" in config["policy"]["model_family_caveat"]
    assert config["policy"]["mcp_regressor_search_allowed"] is True
    assert "nixtla-scaffold ingest" in artifact.next_commands[0]
    assert "--preset accuracy-first" in artifact.next_commands[0]
    assert "--forecast-output" in artifact.next_commands[0]
    assert "--context-file" in artifact.next_commands[0]
    context = json.loads(artifact.files["context"].read_text(encoding="utf-8"))
    assert context["sources"][0]["status"] == "planned"
    assert context["research_budget"]["max_source_queries"] == 30
    brief = artifact.files["agent_brief"].read_text(encoding="utf-8")
    assert "Required exploration checks" in brief
    assert "Model families" in brief
    assert "MCP regressor search allowed" in brief
    assert "explicit scope and approval" in brief
    assert "Forecast Workbench canvas" in brief
    assert "compatibility viewer" in brief
    assert "optional reference, not an interview" in brief
    assert "Do not stitch per-series winners" in brief
    assert "Do not invoke Excel COM" in brief
    assert "installed skill and executable package disagree" in brief
    assert "Which model families should be considered?" not in brief
    assert "never fabricated opt-outs" in brief
    assert "two iterations, one variant each, 15 minutes" in brief
    assert "before inspecting targets" in brief


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
            "--research-max-iterations",
            "2",
            "--research-max-variants-per-iteration",
            "1",
            "--research-max-wall-clock-minutes",
            "15",
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
    assert config["forecast_context"]["research_budget"]["max_iterations"] == 2
    assert config["forecast_context"]["research_budget"]["max_variants_per_iteration"] == 1
    assert config["forecast_context"]["research_budget"]["max_wall_clock_minutes"] == 15
    assert config["forecast_context"]["audience"] == "CFO"


def test_interactive_setup_profiles_canonical_source_without_questions(tmp_path, monkeypatch) -> None:
    from nixtla_scaffold.setup import interactive_answers

    source = tmp_path / "source.csv"
    source.write_text(
        "unique_id,ds,y\nrevenue,2025-01-01,100\nrevenue,2025-02-01,110\nrevenue,2025-03-01,120\n",
        encoding="utf-8",
    )
    def unexpected_prompt(*args):
        raise AssertionError("An unambiguous supplied source must not trigger an intake question.")
    monkeypatch.setattr("builtins.input", unexpected_prompt)
    answers = interactive_answers(SetupAnswers(input_path=str(source), horizon=3, units="USD"))
    assert answers.horizon == 3
    assert answers.freq == "MS"
    assert answers.units == "USD"
    assert answers.target_name == "y"
    assert answers.time_col == "ds"


@pytest.mark.parametrize(
    "header,rows,answer,expected_prompt",
    [
        ("Product,Month,Revenue", "A,2025-01-01,1\nA,2025-02-01,2\nA,2025-03-01,3", None, None),
        ("ds,Revenue,Costs", "2025-01-01,1,4\n2025-02-01,2,5\n2025-03-01,3,6", "Revenue", "numeric column"),
        ("ds,y", "2025-01-01,1\n2025-03-01,2\n2025-04-01,3", "MS", "frequency"),
    ],
)
def test_analyst_first_intake_question_budget(tmp_path, monkeypatch, header, rows, answer, expected_prompt):
    from nixtla_scaffold.setup import interactive_answers

    path = tmp_path / "selected.csv"
    path.write_text(header + "\n" + rows + "\n", encoding="utf-8")
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        assert answer is not None and len(prompts) == 1
        assert expected_prompt in prompt
        return answer

    monkeypatch.setattr("builtins.input", ask)
    result = interactive_answers(SetupAnswers(input_path=str(path), horizon=6))
    assert len(prompts) == (0 if answer is None else 1)
    assert result.freq == "MS" and result.horizon == 6
    assert result.units == result.audience == result.decision == ""


def test_optional_context_does_not_get_fabricated_to_complete_setup(tmp_path):
    artifact = create_forecast_setup(tmp_path / "setup", SetupAnswers())
    context = json.loads(artifact.files["context"].read_text(encoding="utf-8"))
    assert context["decision"] == context["audience"] == context["units"] == ""
    assert context["accuracy_policy"]["require_context_discovery"]
    assert context["accuracy_policy"]["allow_directional_baseline"]
    brief = artifact.files["agent_brief"].read_text(encoding="utf-8")
    assert "not a protected research sequence" in brief
    assert "do not weaken AccuracyPolicy" in brief
