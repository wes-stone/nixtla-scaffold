from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from nixtla_scaffold.presets import PRESET_NAMES, forecast_spec_preset, preset_catalog


DATA_SOURCES = ("csv", "excel", "kusto", "dax", "sql", "dataframe", "unknown")
SERIES_COUNTS = ("single", "few", "many", "hierarchy", "unknown")
INTERVAL_MODES = ("auto", "point", "intervals")
MODEL_FAMILIES = ("auto", "baseline", "statsforecast", "mlforecast", "hierarchicalforecast", "neuralforecast_research")


@dataclass(frozen=True)
class SetupAnswers:
    name: str = "forecast"
    data_source: str = "unknown"
    series_count: str = "unknown"
    target_name: str = "y"
    time_col: str = "ds"
    id_col: str = "unique_id"
    id_value: str | None = None
    preset: str = "finance"
    horizon: int = 12
    freq: str | None = None
    intervals: str = "auto"
    model_families: tuple[str, ...] = ("auto",)
    exploration_mode: bool = True
    mcp_regressor_search: bool = False
    outputs: tuple[str, ...] = ("all",)
    hierarchy_cols: tuple[str, ...] = ()
    query_file: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {DATA_SOURCES}")
        if self.series_count not in SERIES_COUNTS:
            raise ValueError(f"series_count must be one of {SERIES_COUNTS}")
        if self.intervals not in INTERVAL_MODES:
            raise ValueError(f"intervals must be one of {INTERVAL_MODES}")
        if self.preset not in PRESET_NAMES:
            raise ValueError(f"preset must be one of {PRESET_NAMES}")
        invalid_models = [model for model in self.model_families if model not in MODEL_FAMILIES]
        if invalid_models:
            raise ValueError(f"model_families values must be in {MODEL_FAMILIES}, got {invalid_models}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outputs"] = list(self.outputs)
        data["hierarchy_cols"] = list(self.hierarchy_cols)
        data["model_families"] = list(self.model_families)
        return data


@dataclass(frozen=True)
class SetupArtifact:
    workspace: Path
    files: dict[str, Path] = field(default_factory=dict)
    next_commands: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "files": {key: str(value) for key, value in self.files.items()},
            "next_commands": list(self.next_commands),
        }


def create_forecast_setup(workspace: str | Path, answers: SetupAnswers) -> SetupArtifact:
    root = Path(workspace)
    for folder in [
        root,
        root / "queries",
        root / "data" / "raw",
        root / "data" / "canonical",
        root / "outputs",
        root / "reports",
        root / "notes",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    questions = setup_questions(answers)
    next_commands = tuple(_next_commands(root, answers))
    config = _config_payload(root, answers, next_commands)
    brief = _agent_brief(root, answers, questions, next_commands)
    query_template = _query_template(answers)

    files = {
        "config": root / "forecast_setup.yaml",
        "questions": root / "questions.json",
        "agent_brief": root / "agent_brief.md",
    }
    files["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    files["questions"].write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")
    files["agent_brief"].write_text(brief, encoding="utf-8")
    if query_template:
        query_path = root / "queries" / f"source{_query_extension(answers.data_source)}"
        query_path.write_text(query_template, encoding="utf-8")
        files["query_template"] = query_path

    return SetupArtifact(workspace=root, files=files, next_commands=next_commands)


def setup_questions(answers: SetupAnswers | None = None) -> list[dict[str, Any]]:
    selected = answers or SetupAnswers()
    return [
        {
            "id": "preset",
            "question": "Which forecast preset should we start from?",
            "answer": selected.preset,
            "options": list(PRESET_NAMES),
            "why_it_matters": "Presets keep the first run simple while preserving strict and hierarchy modes for serious planning.",
        },
        {
            "id": "data_source",
            "question": "Where is the data coming from?",
            "answer": selected.data_source,
            "options": list(DATA_SOURCES),
            "why_it_matters": "Determines whether the next step is direct forecast, ingest from MCP/query export, or workbook column mapping.",
        },
        {
            "id": "series_count",
            "question": "How many different things are we forecasting?",
            "answer": selected.series_count,
            "options": list(SERIES_COUNTS),
            "why_it_matters": "Single series can use a simpler path; many series and hierarchy need stricter sufficiency, sparse-history, and coherence checks.",
        },
        {
            "id": "intervals",
            "question": "Do we want a point forecast, prediction intervals, or automatic interval gating?",
            "answer": selected.intervals,
            "options": list(INTERVAL_MODES),
            "caveat": "Prediction intervals are only shown when history and model support them; short histories should not display false precision.",
        },
        {
            "id": "model_families",
            "question": "Which model families should be considered?",
            "answer": list(selected.model_families),
            "options": list(MODEL_FAMILIES),
            "caveat": "Current production engine supports baseline and StatsForecast directly; MLForecast and HierarchicalForecast are setup/roadmap choices unless optional dependencies and future-driver/reconciliation contracts are ready. NeuralForecast is research-only.",
        },
        {
            "id": "exploration_mode",
            "question": "Should the agent run exploration mode before forecasting?",
            "answer": selected.exploration_mode,
            "why_it_matters": "Exploration profiles gaps, duplicates, zeros, anomalies, seasonality hints, hierarchy shape, and candidate grains before modeling.",
        },
        {
            "id": "mcp_regressor_search",
            "question": "Can the agent use MCPs to look for possible regressors/drivers?",
            "answer": selected.mcp_regressor_search,
            "caveat": "Candidate regressors must be known for future periods and checked for leakage before they become model features; otherwise use them as scenario notes.",
        },
        {
            "id": "outputs",
            "question": "Which outputs do we want?",
            "answer": list(selected.outputs),
            "options": ["all", "csv", "excel", "html", "base64_html", "streamlit", "diagnostics", "model_card"],
            "why_it_matters": "Stakeholder reporting often needs Excel/HTML, while agent handoff needs JSON diagnostics and base64 HTML.",
        },
    ]


def interactive_answers(defaults: SetupAnswers | None = None) -> SetupAnswers:
    current = defaults or SetupAnswers()
    current = replace(current, preset=_prompt_choice("Which forecast preset should we start from?", PRESET_NAMES, current.preset))
    current = replace(current, data_source=_prompt_choice("Where is the data coming from?", DATA_SOURCES, current.data_source))
    current = replace(current, series_count=_prompt_choice("How many different things are we forecasting?", SERIES_COUNTS, current.series_count))
    current = replace(current, target_name=_prompt_text("Target/metric column or name", current.target_name))
    current = replace(current, time_col=_prompt_text("Date/time column", current.time_col))
    current = replace(current, id_col=_prompt_text("Series ID column", current.id_col))
    current = replace(current, horizon=int(_prompt_text("Forecast horizon", str(current.horizon))))
    current = replace(current, freq=_prompt_text("Frequency hint (blank for auto)", current.freq or "") or None)
    current = replace(current, intervals=_prompt_choice("Point forecast or intervals?", INTERVAL_MODES, current.intervals))
    models = _prompt_text("Model families, comma-separated", ",".join(current.model_families))
    current = replace(current, model_families=tuple(item.strip() for item in models.split(",") if item.strip()))
    current = replace(current, exploration_mode=_prompt_bool("Run exploration mode first?", current.exploration_mode))
    current = replace(current, mcp_regressor_search=_prompt_bool("Allow MCP search for candidate regressors?", current.mcp_regressor_search))
    outputs = _prompt_text("Outputs, comma-separated", ",".join(current.outputs))
    return replace(current, outputs=tuple(item.strip() for item in outputs.split(",") if item.strip()))


def _config_payload(root: Path, answers: SetupAnswers, next_commands: tuple[str, ...]) -> dict[str, Any]:
    return {
        "setup_version": 1,
        "workspace": str(root),
        "answers": answers.to_dict(),
        "forecast_preset": answers.preset,
        "forecast_spec": forecast_spec_preset(
            answers.preset,
            horizon=answers.horizon,
            freq=answers.freq,
        ).to_dict(),
        "preset_catalog": preset_catalog(),
        "policy": {
            "timegpt": "excluded",
            "exploration_first": answers.exploration_mode,
            "mcp_regressor_search_allowed": answers.mcp_regressor_search,
            "model_family_caveat": "Baseline, StatsForecast, MLForecast, and hierarchy reconciliation are active when optional dependencies/data contracts support them. NeuralForecast is research-only.",
            "interval_caveat": "Prediction intervals are gated by available history and model support.",
            "regressor_caveat": "Only use regressors as model features when future values are known and leakage checks pass.",
        },
        "paths": {
            "queries": "queries",
            "raw_data": "data/raw",
            "canonical_data": "data/canonical",
            "outputs": "outputs",
            "reports": "reports",
            "notes": "notes",
        },
        "next_commands": list(next_commands),
    }


def _agent_brief(
    root: Path,
    answers: SetupAnswers,
    questions: list[dict[str, Any]],
    next_commands: tuple[str, ...],
) -> str:
    lines = [
        f"# Forecast setup brief: {answers.name}",
        "",
        "Use this file as the agent intake before running a forecast. Do not skip exploration when the grain, frequency, or metric definition is uncertain.",
        "",
        "## Intake answers",
        "",
    ]
    for item in questions:
        lines.append(f"- **{item['question']}** `{item['answer']}`")
        if "caveat" in item:
            lines.append(f"  Caveat: {item['caveat']}")
    lines.extend(
        [
            "",
            "## Agent operating mode",
            "",
            f"- Data source: `{answers.data_source}`",
            f"- Forecast preset: `{answers.preset}`",
            f"- Series count: `{answers.series_count}`",
            f"- Forecast horizon: `{answers.horizon}`",
            f"- Frequency hint: `{answers.freq or 'auto'}`",
            f"- Model families: `{', '.join(answers.model_families)}`",
            f"- Exploration mode: `{answers.exploration_mode}`",
            f"- MCP regressor search allowed: `{answers.mcp_regressor_search}`",
            f"- Requested outputs: `{', '.join(answers.outputs)}`",
            "",
            "## Required exploration checks",
            "",
            "1. Confirm the metric definition and whether it is raw, adjusted, billed, consumed, trial, internal, or external.",
            "2. Confirm the forecast grain and whether each row maps to one `unique_id`, `ds`, `y` point.",
            "3. Profile missing dates, duplicate keys, sparse histories, zeros, negatives, and outliers.",
            "4. Check whether hierarchy or grouped rollups are needed before stakeholder use.",
            "5. Choose model families deliberately: baselines for sanity, StatsForecast for classical production, MLForecast only when future regressors are known, HierarchicalForecast for reconciliation, NeuralForecast for research only.",
            "6. List possible drivers/events and separate known future assumptions from historical-only explanatory variables.",
            "7. Decide whether intervals are appropriate after seeing history length and backtest windows.",
            "",
            "## Next commands",
            "",
        ]
    )
    lines.extend(f"```powershell\n{command}\n```" for command in next_commands)
    if answers.notes:
        lines.extend(["", "## Notes", "", answers.notes])
    lines.append("")
    return "\n".join(lines)


def _next_commands(root: Path, answers: SetupAnswers) -> list[str]:
    run_dir = root / "outputs" / "forecast_run"
    canonical = root / "data" / "canonical" / "forecast_input.csv"
    raw = root / "data" / "raw" / _raw_placeholder(answers)
    freq_part = f" --freq {answers.freq}" if answers.freq else ""
    preset_part = f" --preset {answers.preset}"
    horizon_part = f" --horizon {answers.horizon}"
    model_policy_part = _model_policy_part(answers)
    if answers.data_source in {"kusto", "dax", "sql"}:
        source_kind = answers.data_source
        query_file = root / "queries" / f"source{_query_extension(answers.data_source)}"
        id_part = f' --id-value "{answers.id_value}"' if answers.id_value else f" --id-col {answers.id_col}"
        ingest = (
            f"nixtla-scaffold ingest --input {raw} --source {source_kind} --query-file {query_file}"
            f"{id_part} --time-col {answers.time_col} --target-col {answers.target_name}"
            f" --output {canonical} --forecast-output {run_dir}{preset_part}{freq_part}{horizon_part}{model_policy_part}"
        )
        return [ingest, f"nixtla-scaffold explain --run {run_dir}", f"nixtla-scaffold report --run {run_dir}"]

    profile = f"nixtla-scaffold profile --input {raw} --id-col {answers.id_col} --time-col {answers.time_col} --target-col {answers.target_name}{freq_part}"
    forecast = (
        f"nixtla-scaffold forecast --input {raw} --id-col {answers.id_col} --time-col {answers.time_col}"
        f" --target-col {answers.target_name}{preset_part}{freq_part}{horizon_part}{model_policy_part} --output {run_dir}"
    )
    return [profile, forecast, f"nixtla-scaffold explain --run {run_dir}", f"nixtla-scaffold report --run {run_dir}"]


def _raw_placeholder(answers: SetupAnswers) -> str:
    if answers.data_source == "excel":
        return "source_workbook.xlsx"
    if answers.data_source in {"kusto", "dax", "sql"}:
        return "query_result.json"
    return "source_data.csv"


def _query_template(answers: SetupAnswers) -> str:
    if answers.data_source == "kusto":
        return (
            "// Replace with a grain-safe KQL query that returns the configured time and target columns.\n"
            f"// Required result columns: {answers.time_col}, {answers.target_name}"
            + ("" if answers.id_value else f", {answers.id_col}")
            + "\n"
        )
    if answers.data_source == "dax":
        return (
            "-- Replace with a DAX query that returns one row per forecast period and series.\n"
            f"-- Required result columns: {answers.time_col}, {answers.target_name}"
            + ("" if answers.id_value else f", {answers.id_col}")
            + "\n"
        )
    if answers.data_source == "sql":
        return (
            "-- Replace with a SQL query that returns one row per forecast period and series.\n"
            f"-- Required result columns: {answers.time_col}, {answers.target_name}"
            + ("" if answers.id_value else f", {answers.id_col}")
            + "\n"
        )
    return ""


def _model_policy_part(answers: SetupAnswers) -> str:
    families = set(answers.model_families)
    if "baseline" in families and len(families) == 1:
        return " --model-policy baseline"
    if "statsforecast" in families and families.isdisjoint({"mlforecast", "hierarchicalforecast", "neuralforecast_research"}):
        return " --model-policy statsforecast"
    return ""


def _query_extension(data_source: str) -> str:
    return {"kusto": ".kql", "dax": ".dax", "sql": ".sql"}.get(data_source, ".query")


def _prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    value = _prompt_text(f"{prompt} ({'/'.join(choices)})", default)
    if value not in choices:
        raise ValueError(f"{prompt} must be one of {choices}")
    return value


def _prompt_bool(prompt: str, default: bool) -> bool:
    value = _prompt_text(f"{prompt} (y/n)", "y" if default else "n").lower()
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"{prompt} must be yes or no")


def _prompt_text(prompt: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default
