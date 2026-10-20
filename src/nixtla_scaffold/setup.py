from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from nixtla_scaffold.presets import PRESET_NAMES, canonical_preset_name, forecast_spec_preset, preset_catalog
from nixtla_scaffold.schema import ContextSource, ForecastContext, ResearchBudget
from nixtla_scaffold.signals import build_initial_signal_needs, write_signal_artifacts


DATA_SOURCES = ("csv", "excel", "kusto", "dax", "sql", "dataframe", "unknown")
SERIES_COUNTS = ("single", "few", "many", "hierarchy", "unknown")
INTERVAL_MODES = ("auto", "point", "intervals")
MODEL_FAMILIES = ("standard", "light", "auto", "baseline", "statsforecast", "mlforecast", "hierarchicalforecast")


@dataclass(frozen=True)
class SetupAnswers:
    name: str = "forecast"
    data_source: str = "unknown"
    input_path: str | None = None
    series_count: str = "unknown"
    target_name: str = "y"
    time_col: str = "ds"
    id_col: str = "unique_id"
    id_value: str | None = None
    preset: str = "accuracy-first"
    horizon: int = 6
    freq: str | None = None
    intervals: str = "auto"
    model_families: tuple[str, ...] = ("standard",)
    exploration_mode: bool = True
    source_discovery: bool = True
    mcp_regressor_search: bool = False
    research_budget: ResearchBudget = field(default_factory=ResearchBudget)
    decision: str = ""
    audience: str = ""
    target_semantics: str = ""
    units: str = ""
    grain: str = ""
    refresh_cadence: str = ""
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
        if self.input_path is not None and not str(self.input_path).strip():
            raise ValueError("input_path cannot be blank")
        object.__setattr__(self, "preset", canonical_preset_name(self.preset))
        canonical_families = tuple(dict.fromkeys("light" if model == "auto" else model for model in self.model_families))
        object.__setattr__(self, "model_families", canonical_families)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outputs"] = list(self.outputs)
        data["hierarchy_cols"] = list(self.hierarchy_cols)
        data["model_families"] = list(self.model_families)
        data["research_budget"] = self.research_budget.to_dict()
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
    context = _forecast_context(root, answers)
    config = _config_payload(root, answers, next_commands, context)
    brief = _agent_brief(root, answers, questions, next_commands)
    query_template = _query_template(answers)

    files = {
        "config": root / "forecast_setup.yaml",
        "questions": root / "questions.json",
        "agent_brief": root / "agent_brief.md",
        "context": root / "forecast_context.json",
    }
    files["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    files["questions"].write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")
    files["agent_brief"].write_text(brief, encoding="utf-8")
    files["context"].write_text(json.dumps(context.to_dict(), indent=2) + "\n", encoding="utf-8")
    files.update(write_signal_artifacts(context, root))
    if query_template:
        query_path = root / "queries" / f"source{_query_extension(answers.data_source)}"
        query_path.write_text(query_template, encoding="utf-8")
        files["query_template"] = query_path

    return SetupArtifact(workspace=root, files=files, next_commands=next_commands)


def setup_questions(answers: SetupAnswers | None = None) -> list[dict[str, Any]]:
    """Return optional context fields for saved setups, not a mandatory interview."""
    selected = answers or SetupAnswers()
    questions = [
        {
            "id": "decision",
            "question": "What decision will this forecast support, and who is the audience?",
            "answer": {"decision": selected.decision, "audience": selected.audience},
            "why_it_matters": "Accuracy claims should be calibrated to the decision, audience, and cost of error.",
        },
        {
            "id": "target_context",
            "question": "What exactly is the target, its units, and its time grain?",
            "answer": {
                "semantics": selected.target_semantics or selected.target_name,
                "units": selected.units,
                "grain": selected.grain or selected.freq or "unknown",
            },
            "why_it_matters": "Prevents plausible-looking forecasts of the wrong metric, unit, or grain.",
        },
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
            "id": "input_path",
            "question": "What existing file should the generated commands use?",
            "answer": selected.input_path or "",
            "why_it_matters": "Using the real path keeps the generated first command executable instead of pointing at a placeholder.",
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
                "caveat": "Standard is the default serious model policy: StatsForecast, feasible MLForecast, and optional smooth ADAM when installed. Light is the slim path without smooth by default.",
        },
        {
            "id": "research_budget",
            "question": "How much bounded research should the agent perform?",
            "answer": selected.research_budget.to_dict(),
            "options": ["time-boxed", "balanced", "deep", "custom"],
            "why_it_matters": "The agent must search beyond the first baseline without running an unbounded sweep.",
        },
        {
            "id": "source_discovery",
            "question": "Should the agent run bounded read-only discovery across relevant connected sources?",
            "answer": selected.source_discovery,
            "caveat": "An opt-out is allowed but is recorded in the context receipt.",
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
            "options": ["streamlit", "all", "csv", "excel", "html", "base64_html", "diagnostics", "model_card"],
            "why_it_matters": (
                "Forecast runs generate the standard Streamlit, CSV, Excel, HTML, and "
                "diagnostic artifacts automatically. This answer sets review priority; "
                "it does not authorize a replacement dashboard or Office automation."
            ),
        },
    ]
    return [{**question, "reference_only": True} for question in questions]


def interactive_answers(defaults: SetupAnswers | None = None) -> SetupAnswers:
    import pandas as pd
    from nixtla_scaffold.data import read_tabular_source

    current = defaults or SetupAnswers()
    if not current.input_path:
        path = _prompt_text("Selected input file", "")
        if not path:
            raise ValueError("Select a source file before interactive setup.")
        current = replace(current, input_path=path)
    raw = read_tabular_source(current.input_path)
    if raw.empty:
        raise ValueError("Selected source has no rows.")
    columns = list(raw.columns)
    dates = [col for col in columns if str(col).lower() in {"ds", "date", "month", "time", "period"}]
    time_col = current.time_col if current.time_col in columns else dates[0] if len(dates) == 1 else _prompt_text("Which column is the time axis?", "")
    ids = [col for col in columns if str(col).lower() in {"unique_id", "series", "series_id", "product"}]
    id_col = current.id_col if current.id_col in columns else ids[0] if len(ids) == 1 else current.id_col
    numeric = [col for col in columns if col not in {time_col, id_col} and pd.api.types.is_numeric_dtype(raw[col])]
    target = current.target_name if current.target_name in columns else numeric[0] if len(numeric) == 1 else _prompt_text("Which numeric column should be forecast?", "")
    if time_col not in columns or target not in columns:
        raise ValueError("The time and target columns must exist in the selected source.")
    calendar = pd.to_datetime(raw[time_col], errors="raise").drop_duplicates().sort_values()
    freq = current.freq or (pd.infer_freq(calendar) if len(calendar) >= 3 else None)
    if not freq:
        freq = _prompt_text("The observed calendar is irregular. Which forecast frequency should be used?", "")
    if not freq:
        raise ValueError("An explicit frequency is required for an irregular calendar.")
    print(f"Observed {len(raw)} rows; time={time_col}, target={target}, frequency={freq}. Horizon={current.horizon}.")
    return replace(current, time_col=time_col, target_name=target, id_col=id_col, freq=freq,
                   series_count="few" if id_col in columns and raw[id_col].nunique() > 1 else "single")


def _config_payload(
    root: Path,
    answers: SetupAnswers,
    next_commands: tuple[str, ...],
    context: ForecastContext,
) -> dict[str, Any]:
    return {
        "setup_version": 1,
        "workspace": str(root),
        "answers": answers.to_dict(),
        "forecast_preset": answers.preset,
        "forecast_spec": forecast_spec_preset(
            answers.preset,
            horizon=answers.horizon,
            freq=answers.freq,
            context=context,
        ).to_dict(),
        "forecast_context": context.to_dict(),
        "preset_catalog": preset_catalog(),
        "policy": {
            "timegpt": "excluded",
            "exploration_first": answers.exploration_mode,
            "source_discovery": answers.source_discovery,
            "mcp_regressor_search_allowed": answers.mcp_regressor_search,
            "research_budget": answers.research_budget.to_dict(),
            "model_family_caveat": "Baseline, StatsForecast, MLForecast, and hierarchy reconciliation are active when optional dependencies/data contracts support them.",
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
        "Inspect the selected data first. These saved context fields are optional reference, not an interview. Ask only for consequential ambiguity or authorization; otherwise proceed within the bounded local forecast request.",
        "",
        "## Supplied context and optional reference",
        "",
    ]
    for item in questions:
        lines.append(f"- **{item['id'].replace('_', ' ').capitalize()}**: `{item['answer']}`")
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
            f"- Decision / audience: `{answers.decision or 'not recorded'}` / `{answers.audience or 'not recorded'}`",
            f"- Target semantics / units / grain: `{answers.target_semantics or answers.target_name}` / `{answers.units or 'not recorded'}` / `{answers.grain or answers.freq or 'not recorded'}`",
            f"- Refresh cadence: `{answers.refresh_cadence or 'not recorded'}`",
            f"- Exploration mode: `{answers.exploration_mode}`",
            f"- Bounded connected-source discovery: `{answers.source_discovery}`",
            f"- MCP regressor search allowed: `{answers.mcp_regressor_search}`",
            f"- Research budget: `{answers.research_budget.profile}` — {answers.research_budget.to_dict()}",
            f"- Requested outputs: `{', '.join(answers.outputs)}`",
            "",
            "## Required exploration checks",
            "",
            "1. Establish metric meaning from supplied definitions and metadata. Preserve unknown units, audience and business meaning; do not invent answers or block a directional baseline on optional context.",
            "2. Derive grain, date coverage and series keys from the selected source. Ask one focused question only for an unresolved competing target, calendar or consequential aggregation/repair.",
            "3. Before target-dependent profiling, choose the evidence path: ordinary full-history forecasting is retrospective; confirmatory research must reserve its chronological split before inspecting targets. Never claim a previously viewed tail is untouched.",
            "4. Check whether hierarchy or grouped rollups are needed before stakeholder use.",
            "5. Choose model families deliberately: baselines for sanity, StatsForecast for classical production, MLForecast for feasible lag/calendar ML and audited future-regressor experiments, HierarchicalForecast for reconciliation.",
            "6. Investigate drivers/events from supplied evidence, not a feature interview. Unknown future events mean not modeled, not that no events exist.",
            "7. Decide whether intervals are appropriate after seeing history length and backtest windows.",
            "8. Obtain explicit scope and approval before accessing a new connected source. Use bounded queries only within that scope and record source dispositions in forecast_context.json.",
            "9. Never copy discovered drivers into model regressors. Admit them only after future-availability, latency, leakage, and rolling-origin evidence pass.",
            "10. Read appendix/accuracy_gate.json after the run. Directional-only output must not be described as planning-ready.",
            "11. Work signal_needs.json within approved scope. Record unavailable routes honestly, never fabricated opt-outs or satisfied needs; keep incomplete discovery explicit without blocking an otherwise valid directional baseline.",
            "12. Prefer the installed Forecast Workbench canvas for supported review and ensemble paths, but continue with the matching CLI when absent. Retain run_streamlit.ps1 / streamlit_app.py as the compatibility viewer.",
            "13. Do not stitch per-series winners from different experiment runs into a hybrid official forecast. One official forecast must come from one frozen spec and its within-run selection.",
            "14. forecast.xlsx and output/forecast_review.xlsx are automatic artifacts. Do not invoke Excel COM, LibreOffice, or custom workbook automation unless the user explicitly asks for a custom workbook.",
            "15. Stop if the installed skill and executable package disagree; use a matching worktree or Git ref instead of hand-building missing capabilities.",
            "16. Do not ask users to choose profiling, models, intervals, outputs or a routine research budget. Agent-created setups should explicitly use time-boxed bounds: two iterations, one variant each, 15 minutes. A larger saved budget is not new authorization to spend it.",
            "17. Unknown required context fields stay empty; do not weaken AccuracyPolicy or populate placeholders to get planning-ready status. Preserve independent analysis while resolving only consequential blockers.",
            "",
            "## Next commands",
            "",
            "These are ordinary full-history commands, not a protected research sequence. For confirmatory work, use optimize's chronological split before any full-target profile or ingest-with-forecast. Do not run these first and relabel the same tail as untouched.",
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
    raw = Path(answers.input_path) if answers.input_path else root / "data" / "raw" / _raw_placeholder(answers)
    freq_part = f" --freq {answers.freq}" if answers.freq else ""
    preset_part = f" --preset {answers.preset}"
    context_part = f" --context-file {_quote_path(root / 'forecast_context.json')}"
    horizon_part = f" --horizon {answers.horizon}"
    model_policy_part = _model_policy_part(answers)
    if answers.data_source in {"kusto", "dax", "sql"} or answers.series_count == "single":
        source_kind = answers.data_source
        query_file = root / "queries" / f"source{_query_extension(answers.data_source)}"
        injected_id = answers.id_value or (answers.name if answers.series_count == "single" else None)
        id_part = f' --id-value "{injected_id}"' if injected_id else f" --id-col {answers.id_col}"
        query_part = (
            f" --query-file {_quote_path(query_file)}"
            if answers.data_source in {"kusto", "dax", "sql"}
            else ""
        )
        ingest = (
            f"nixtla-scaffold ingest --input {_quote_path(raw)} --source {source_kind}{query_part}"
            f"{id_part} --time-col {answers.time_col} --target-col {answers.target_name}"
            f" --output {_quote_path(canonical)} --forecast-output {_quote_path(run_dir)}"
            f"{preset_part}{context_part}{freq_part}{horizon_part}{model_policy_part}"
        )
        return [
            ingest,
            f"nixtla-scaffold explain --run {_quote_path(run_dir)}",
            f"nixtla-scaffold report --run {_quote_path(run_dir)}",
        ]

    profile = (
        f"nixtla-scaffold profile --input {_quote_path(raw)} --id-col {answers.id_col}"
        f" --time-col {answers.time_col} --target-col {answers.target_name}{freq_part}"
    )
    forecast = (
        f"nixtla-scaffold forecast --input {_quote_path(raw)} --id-col {answers.id_col} --time-col {answers.time_col}"
        f" --target-col {answers.target_name}{preset_part}{context_part}{freq_part}{horizon_part}{model_policy_part}"
        f" --output {_quote_path(run_dir)}"
    )
    return [
        profile,
        forecast,
        f"nixtla-scaffold explain --run {_quote_path(run_dir)}",
        f"nixtla-scaffold report --run {_quote_path(run_dir)}",
    ]


def _forecast_context(root: Path, answers: SetupAnswers) -> ForecastContext:
    input_path = Path(answers.input_path) if answers.input_path else None
    source_status = (
        "available"
        if input_path is not None and input_path.exists()
        else ("planned" if answers.source_discovery else "unavailable")
    )
    source = ContextSource(
        source_id=f"target_{answers.data_source}",
        kind=answers.data_source,
        status=source_status,
        provenance=str(input_path.resolve()) if input_path is not None and input_path.exists() else "",
        query_ref=str(root / "queries" / f"source{_query_extension(answers.data_source)}")
        if answers.data_source in {"kusto", "dax", "sql"}
        else "",
        notes="Primary target source; additional source discovery stays within explicit authorization and is not a prerequisite for a directional local baseline.",
    )
    return ForecastContext(
        decision=answers.decision,
        audience=answers.audience,
        target_semantics=answers.target_semantics or answers.target_name,
        units=answers.units,
        grain=answers.grain or answers.freq or "",
        requested_horizon=answers.horizon,
        refresh_cadence=answers.refresh_cadence,
        hierarchy_required=answers.series_count == "hierarchy",
        source_discovery_enabled=answers.source_discovery,
        sources=(source,),
        signal_needs=build_initial_signal_needs(
            target_semantics=answers.target_semantics or answers.target_name,
            grain=answers.grain or answers.freq or "",
            source_discovery_enabled=answers.source_discovery,
        ),
        research_budget=answers.research_budget,
    )


def _quote_path(path: str | Path) -> str:
    return f'"{Path(path)}"'


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
    if "statsforecast" in families and families.isdisjoint({"mlforecast", "hierarchicalforecast"}):
        return " --model-policy statsforecast"
    if "standard" in families and len(families) == 1:
        return " --model-policy standard"
    if "light" in families and len(families) == 1:
        return " --model-policy light"
    if "mlforecast" in families and len(families) == 1:
        return " --model-policy mlforecast"
    if "auto" in families and len(families) == 1:
        return " --model-policy light"
    if "standard" in families:
        return ""
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
