"""Simple, explainable forecasting scaffolding on the Nixtla ecosystem."""

from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.best_practices import best_practice_receipts
from nixtla_scaffold.connectors import ingest_query_result
from nixtla_scaffold.comparisons import (
    FORECAST_COMPARISON_SCHEMA_VERSION,
    ForecastComparisonResult,
    compare_forecasts,
    write_forecast_comparison,
)
from nixtla_scaffold.custom_models import CUSTOM_MODEL_SCHEMA_VERSION, append_custom_model_result, forecast_with_custom_model
from nixtla_scaffold.diagnostics import build_run_diagnostics
from nixtla_scaffold.external import (
    EXTERNAL_FORECAST_SCHEMA_VERSION,
    build_external_forecast_metadata,
    canonicalize_external_forecasts,
    load_external_forecasts,
)
from nixtla_scaffold.external_scoring import (
    EXTERNAL_SCORE_SCHEMA_VERSION,
    ExternalForecastScoreResult,
    score_external_forecasts,
    write_external_forecast_scores,
)
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.headline import ExecutiveHeadline, build_executive_headline
from nixtla_scaffold.hierarchy import aggregate_hierarchy_frame, hierarchy_coherence, hierarchy_summary, reconcile_hierarchy_forecast
from nixtla_scaffold.knowledge import load_agent_skill
from nixtla_scaffold.ledger import (
    DEFAULT_LEDGER_PATH,
    LEDGER_SCHEMA_VERSION,
    LedgerResult,
    compare_versions,
    export_ledger,
    ingest_actuals,
    ingest_adjustments,
    init_ledger,
    lock_version,
    register_run,
)
from nixtla_scaffold.presets import PRESET_NAMES, forecast_spec_preset, preset_catalog
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.release_gates import OPTIONAL_EXTRAS, ReleaseGateResult, format_release_gate_console_summary, run_release_gates
from nixtla_scaffold.reports import build_html_report, build_streamlit_app, write_report_artifacts
from nixtla_scaffold.scenario_lab import run_scenario_lab
from nixtla_scaffold.schema import CustomModelSpec, DataProfile, DriverEvent, ForecastRun, ForecastSpec, KnownFutureRegressor, SeriesProfile, TransformSpec
from nixtla_scaffold.setup import SetupAnswers, create_forecast_setup, setup_questions
from nixtla_scaffold.transformations import add_fiscal_calendar, apply_event_adjustments, label_anomalies, normalize_by_factor
from nixtla_scaffold.workbench_qa import GOLDEN_SCENARIOS, SCENARIO_ALIASES, SCENARIO_DESCRIPTIONS, WORKBENCH_QA_SCENARIOS, run_workbench_qa

__all__ = [
    "DataProfile",
    "ForecastRun",
    "ForecastSpec",
    "GOLDEN_SCENARIOS",
    "SCENARIO_ALIASES",
    "SCENARIO_DESCRIPTIONS",
    "WORKBENCH_QA_SCENARIOS",
    "PRESET_NAMES",
    "OPTIONAL_EXTRAS",
    "ExecutiveHeadline",
    "EXTERNAL_FORECAST_SCHEMA_VERSION",
    "EXTERNAL_SCORE_SCHEMA_VERSION",
    "FORECAST_COMPARISON_SCHEMA_VERSION",
    "CUSTOM_MODEL_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "CustomModelSpec",
    "ExternalForecastScoreResult",
    "ForecastComparisonResult",
    "LedgerResult",
    "ReleaseGateResult",
    "DriverEvent",
    "KnownFutureRegressor",
    "SeriesProfile",
    "SetupAnswers",
    "TransformSpec",
    "add_fiscal_calendar",
    "aggregate_hierarchy_frame",
    "apply_event_adjustments",
    "best_practice_receipts",
    "build_run_diagnostics",
    "build_external_forecast_metadata",
    "build_html_report",
    "build_executive_headline",
    "build_streamlit_app",
    "append_custom_model_result",
    "compare_forecasts",
    "create_forecast_setup",
    "forecast_spec_preset",
    "forecast_with_custom_model",
    "format_release_gate_console_summary",
    "hierarchy_coherence",
    "hierarchy_summary",
    "ingest_query_result",
    "ingest_actuals",
    "ingest_adjustments",
    "init_ledger",
    "label_anomalies",
    "load_agent_skill",
    "load_forecast_dataset",
    "load_external_forecasts",
    "canonicalize_external_forecasts",
    "normalize_by_factor",
    "profile_dataset",
    "preset_catalog",
    "reconcile_hierarchy_forecast",
    "register_run",
    "run_forecast",
    "run_release_gates",
    "run_scenario_lab",
    "run_workbench_qa",
    "score_external_forecasts",
    "setup_questions",
    "write_external_forecast_scores",
    "write_report_artifacts",
    "write_forecast_comparison",
    "lock_version",
    "compare_versions",
    "export_ledger",
    "DEFAULT_LEDGER_PATH",
]

