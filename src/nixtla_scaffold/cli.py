from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from nixtla_scaffold.byo_model import write_byo_model_comparison, write_byo_model_ingest, write_byo_model_scores
from nixtla_scaffold.comparisons import write_forecast_comparison
from nixtla_scaffold.challengers import run_challengers
from nixtla_scaffold.connectors import ingest_query_result
from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.diagnostics import write_failure_diagnostics
from nixtla_scaffold.drivers import parse_driver_events, parse_known_future_regressors
from nixtla_scaffold.experiments import EXPERIMENT_VARIANTS, compare_models, run_experiment, run_optimizer
from nixtla_scaffold.external_scoring import write_external_forecast_scores
from nixtla_scaffold.finn_bridge import canonicalize_finn_forecasts, check_finn_environment, compare_finn_forecasts, run_finn_bridge, score_finn_forecasts
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.hierarchy import aggregate_hierarchy_frame, hierarchy_summary
from nixtla_scaffold.knowledge import check_agent_skill, format_knowledge, load_agent_skill, search_knowledge, sync_agent_skill
from nixtla_scaffold.ledger import (
    DEFAULT_LEDGER_PATH,
    compare_versions,
    export_ledger,
    ingest_actuals,
    ingest_adjustments,
    init_ledger,
    lock_version,
    register_run,
)
from nixtla_scaffold.ops import (
    build_doctor_payload,
    build_status_payload,
    run_operating_loop,
    write_doctor_outputs,
    write_drift_report,
    write_status_outputs,
)
from nixtla_scaffold.pipelines import refresh_pipeline, run_pipeline
from nixtla_scaffold.presets import PRESET_NAMES, forecast_spec_preset, preset_catalog
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.release_gates import OPTIONAL_EXTRAS, format_release_gate_console_summary, run_release_gates
from nixtla_scaffold.reports import write_report_artifacts_from_directory
from nixtla_scaffold.refresh import write_refresh_artifacts
from nixtla_scaffold.scenario_lab import run_scenario_lab
from nixtla_scaffold.schema import (
    ChallengerSpec,
    CleaningSpec,
    CustomModelSpec,
    EnsembleSpec,
    FeatureRecipeSpec,
    ForecastContext,
    ForecastSpec,
    ParallelSpec,
    ResearchBudget,
    TransformSpec,
    forecast_spec_from_dict,
    load_forecast_context,
)
from nixtla_scaffold.setup import (
    DATA_SOURCES,
    INTERVAL_MODES,
    MODEL_FAMILIES,
    SERIES_COUNTS,
    SetupAnswers,
    create_forecast_setup,
    interactive_answers,
)
from nixtla_scaffold.workbench_qa import GOLDEN_SCENARIOS, WORKBENCH_QA_SCENARIOS, run_workbench_qa

MODEL_POLICY_CHOICES = ["standard", "light", "auto", "baseline", "statsforecast", "mlforecast", "all"]
MODEL_POLICY_HELP = (
    "Model family policy: standard runs StatsForecast, attempts feasible MLForecast, and attempts optional smooth ADAM; "
    "light runs the slim StatsForecast + feasible MLForecast path without smooth by default; "
    "auto is a legacy alias for light; "
    "baseline runs only simple benchmarks; statsforecast runs classical/open-source statistical models; "
    "mlforecast runs MLForecast only and raises if unavailable; all runs every eligible open-source family "
    "and raises on eligible MLForecast failure."
)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="nixtla-scaffold", description="Simple finance forecasting scaffold.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup_cmd = sub.add_parser("setup", help="Create an agent-ready forecast workspace and intake brief")
    setup_cmd.add_argument("--workspace", required=True, help="Workspace folder to scaffold")
    setup_cmd.add_argument("--interactive", action="store_true", help="Ask setup questions interactively")
    setup_cmd.add_argument("--name", default="forecast", help="Forecast/project name")
    setup_cmd.add_argument("--data-source", choices=DATA_SOURCES, default="unknown")
    setup_cmd.add_argument("--input", dest="setup_input", default=None, help="Existing source/export file for generated commands")
    setup_cmd.add_argument("--series-count", choices=SERIES_COUNTS, default="unknown")
    setup_cmd.add_argument("--target-name", default="y", help="Metric/target column or name")
    setup_cmd.add_argument("--target-semantics", default="", help="Business definition of the target")
    setup_cmd.add_argument("--units", default="", help="Target units, e.g. USD, seats, minutes")
    setup_cmd.add_argument("--grain", default="", help="Business/time grain, e.g. monthly by product")
    setup_cmd.add_argument("--decision", default="", help="Decision this forecast supports")
    setup_cmd.add_argument("--audience", default="", help="Primary decision audience")
    setup_cmd.add_argument("--refresh-cadence", default="", help="Expected refresh cadence")
    setup_cmd.add_argument("--time-col", default="ds")
    setup_cmd.add_argument("--id-col", default="unique_id")
    setup_cmd.add_argument("--id-value", default=None, help="Series ID to inject for one-metric query results")
    setup_cmd.add_argument("--preset", choices=PRESET_NAMES, default="standard", help="Starter forecast preset for generated commands/config")
    setup_cmd.add_argument("--horizon", type=int, default=12)
    setup_cmd.add_argument("--freq", default=None)
    setup_cmd.add_argument("--intervals", choices=INTERVAL_MODES, default="auto")
    setup_cmd.add_argument("--model-families", nargs="+", choices=MODEL_FAMILIES, default=["standard"])
    setup_cmd.add_argument("--exploration-mode", action=argparse.BooleanOptionalAction, default=True)
    setup_cmd.add_argument("--source-discovery", action=argparse.BooleanOptionalAction, default=True)
    setup_cmd.add_argument("--mcp-regressor-search", action=argparse.BooleanOptionalAction, default=False)
    setup_cmd.add_argument("--research-budget", choices=["time-boxed", "balanced", "deep", "custom"], default="balanced")
    setup_cmd.add_argument("--research-max-iterations", type=int, default=None)
    setup_cmd.add_argument("--research-max-variants-per-iteration", type=int, default=None)
    setup_cmd.add_argument("--research-max-wall-clock-minutes", type=int, default=None)
    setup_cmd.add_argument("--research-max-source-queries", type=int, default=None)
    setup_cmd.add_argument("--research-max-compute-units", type=float, default=None)
    setup_cmd.add_argument("--outputs", nargs="+", default=["all"], help="Outputs to produce: all csv excel html base64_html streamlit diagnostics model_card")
    setup_cmd.add_argument("--hierarchy-cols", nargs="*", default=[])
    setup_cmd.add_argument("--query-file", default=None)
    setup_cmd.add_argument("--notes", default="")

    profile_cmd = sub.add_parser("profile", help="Profile a forecast dataset")
    _add_input_args(profile_cmd)
    profile_cmd.add_argument("--freq", default=None, help="Optional frequency hint, e.g. D, B, W-SUN, ME, MS, QE, QS")
    profile_cmd.add_argument("--season-length", type=int, default=None, help="Optional season length override")
    profile_cmd.add_argument("--output", default=None, help="Optional JSON output path")

    forecast_cmd = sub.add_parser("forecast", help="Run an explainable forecast")
    _add_input_args(forecast_cmd)
    forecast_cmd.add_argument("--preset", choices=PRESET_NAMES, default=None, help="Opinionated defaults: quick, accuracy-first, standard, strict, or hierarchy; finance is a legacy alias for standard")
    forecast_cmd.add_argument("--horizon", type=int, default=12)
    forecast_cmd.add_argument("--freq", default=None)
    forecast_cmd.add_argument("--season-length", type=int, default=None)
    forecast_cmd.add_argument("--levels", nargs="+", type=int, default=[80, 95], help="Prediction interval levels, e.g. --levels 80 95")
    forecast_cmd.add_argument("--unit-label", default=None, help="Optional unit/currency label for headline values, e.g. $, USD, seats, ARR")
    forecast_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default="ffill")
    forecast_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="light", help=MODEL_POLICY_HELP)
    _add_model_allowlist_args(forecast_cmd)
    forecast_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none", help="Optional target transform for modeling; outputs are inverse-transformed for reporting")
    forecast_cmd.add_argument("--normalization-factor-col", default=None, help="Positive factor column used to normalize y before modeling, e.g. price_factor, fx_rate, inflation_index")
    forecast_cmd.add_argument("--normalization-label", default="", help="Readable label for the normalization assumption")
    forecast_cmd.add_argument(
        "--hierarchy-reconciliation",
        choices=["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"],
        default="none",
        help="Optional reconciliation for hierarchy nodes; use both to save bottom-up and top-down comparison paths",
    )
    forecast_cmd.add_argument(
        "--train-known-future-regressors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to training MLForecast candidates with declared model_candidate regressors that pass leakage and future-value audits",
    )
    forecast_cmd.add_argument(
        "--mlforecast-feature-policy",
        choices=["basic", "rolling"],
        default="basic",
        help="MLForecast feature policy: basic keeps current lag/date features; rolling adds a small audited rolling-transform set",
    )
    forecast_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=False, help="Fail if rolling backtest metrics cannot be produced")
    forecast_cmd.add_argument(
        "--strict-cv-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only score/select models on rolling-origin windows that match --horizon; short histories may skip backtests",
    )
    forecast_cmd.add_argument(
        "--weighted-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include inverse-error weighted ensemble forecasts and audit/model_weights.csv diagnostics",
    )
    _add_finn_inspired_args(forecast_cmd)
    _add_context_args(forecast_cmd)
    forecast_cmd.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit verbose model-engine output; diagnostics artifacts are always written for LLM handoff",
    )
    forecast_cmd.add_argument(
        "--event",
        action="append",
        default=[],
        help="Driver/event scenario JSON, e.g. {\"name\":\"Launch\",\"start\":\"2026-03-31\",\"effect\":\"multiplicative\",\"magnitude\":0.1}",
    )
    forecast_cmd.add_argument(
        "--event-file",
        action="append",
        default=[],
        help="JSON/YAML/CSV file of scenario assumptions. Structured files may contain an 'events' list.",
    )
    forecast_cmd.add_argument(
        "--regressor",
        action="append",
        default=[],
        help="Known-future regressor JSON declaration, e.g. {\"name\":\"Seats plan\",\"value_col\":\"seats_plan\",\"availability\":\"plan\",\"mode\":\"model_candidate\",\"future_file\":\"future_seats.csv\"}",
    )
    forecast_cmd.add_argument(
        "--regressor-file",
        action="append",
        default=[],
        help="JSON/YAML/CSV file of known-future regressor declarations. Structured files may contain a 'regressors' list.",
    )
    forecast_cmd.add_argument("--custom-model", default=None, help="Python import path for a custom forecast callable, e.g. package.module:function")
    forecast_cmd.add_argument("--custom-script", default=None, help="Executable custom model script; .py scripts are invoked with the current Python")
    forecast_cmd.add_argument("--custom-model-name", default=None, help="Readable custom model name; outputs use a Custom_ prefix")
    forecast_cmd.add_argument("--custom-timeout-seconds", type=int, default=120, help="Timeout for each custom script invocation")
    forecast_cmd.add_argument("--custom-arg", action="append", default=[], help="Extra argument passed through to --custom-script; repeat as needed")
    forecast_cmd.add_argument("--output", default="runs/latest")
    _add_ledger_registration_args(forecast_cmd)

    compare_models_cmd = sub.add_parser("compare-models", help="Write a clean advisory leaderboard for the existing model tournament")
    _add_input_args(compare_models_cmd)
    compare_models_cmd.add_argument("--preset", choices=PRESET_NAMES, default=None)
    compare_models_cmd.add_argument("--horizon", type=int, default=12)
    compare_models_cmd.add_argument("--freq", default=None)
    compare_models_cmd.add_argument("--season-length", type=int, default=None)
    compare_models_cmd.add_argument("--levels", nargs="+", type=int, default=[80, 95])
    compare_models_cmd.add_argument("--unit-label", default=None)
    compare_models_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default="ffill")
    compare_models_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="light", help=MODEL_POLICY_HELP)
    _add_model_allowlist_args(compare_models_cmd)
    compare_models_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none")
    compare_models_cmd.add_argument("--normalization-factor-col", default=None)
    compare_models_cmd.add_argument("--normalization-label", default="")
    compare_models_cmd.add_argument(
        "--hierarchy-reconciliation",
        choices=["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"],
        default="none",
    )
    compare_models_cmd.add_argument("--train-known-future-regressors", action=argparse.BooleanOptionalAction, default=False)
    compare_models_cmd.add_argument("--mlforecast-feature-policy", choices=["basic", "rolling"], default="basic")
    compare_models_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=False)
    compare_models_cmd.add_argument("--strict-cv-horizon", action=argparse.BooleanOptionalAction, default=False)
    compare_models_cmd.add_argument("--weighted-ensemble", action=argparse.BooleanOptionalAction, default=True)
    _add_finn_inspired_args(compare_models_cmd)
    compare_models_cmd.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    compare_models_cmd.add_argument("--event", action="append", default=[])
    compare_models_cmd.add_argument("--event-file", action="append", default=[])
    compare_models_cmd.add_argument("--regressor", action="append", default=[])
    compare_models_cmd.add_argument("--regressor-file", action="append", default=[])
    compare_models_cmd.add_argument("--output", default="runs/compare_models")

    experiment_cmd = sub.add_parser("experiment", help="Run bounded advisory forecast variants and recommend the next agent iteration")
    _add_input_args(experiment_cmd)
    experiment_cmd.add_argument("--preset", choices=PRESET_NAMES, default=None)
    experiment_cmd.add_argument("--horizon", type=int, default=12)
    experiment_cmd.add_argument("--freq", default=None)
    experiment_cmd.add_argument("--season-length", type=int, default=None)
    experiment_cmd.add_argument("--levels", nargs="+", type=int, default=[80, 95])
    experiment_cmd.add_argument("--unit-label", default=None)
    experiment_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default="ffill")
    experiment_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="light", help=MODEL_POLICY_HELP)
    _add_model_allowlist_args(experiment_cmd)
    experiment_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none")
    experiment_cmd.add_argument("--normalization-factor-col", default=None)
    experiment_cmd.add_argument("--normalization-label", default="")
    experiment_cmd.add_argument(
        "--hierarchy-reconciliation",
        choices=["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"],
        default="none",
    )
    experiment_cmd.add_argument("--train-known-future-regressors", action=argparse.BooleanOptionalAction, default=False)
    experiment_cmd.add_argument("--mlforecast-feature-policy", choices=["basic", "rolling"], default="basic")
    experiment_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=False)
    experiment_cmd.add_argument("--strict-cv-horizon", action=argparse.BooleanOptionalAction, default=False)
    experiment_cmd.add_argument("--weighted-ensemble", action=argparse.BooleanOptionalAction, default=True)
    _add_finn_inspired_args(experiment_cmd)
    _add_context_args(experiment_cmd)
    experiment_cmd.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    experiment_cmd.add_argument("--event", action="append", default=[])
    experiment_cmd.add_argument("--event-file", action="append", default=[])
    experiment_cmd.add_argument("--regressor", action="append", default=[])
    experiment_cmd.add_argument("--regressor-file", action="append", default=[])
    experiment_cmd.add_argument(
        "--variants",
        nargs="+",
        choices=sorted((*EXPERIMENT_VARIANTS, "all")),
        default=None,
        help="Named advisory variants to test. Defaults to a small data-aware set.",
    )
    experiment_cmd.add_argument("--max-variants", type=int, default=4, help="Safety cap for requested variants")
    experiment_cmd.add_argument("--hypothesis", default=None, help="Falsifiable hypothesis statement to record and test")
    experiment_cmd.add_argument("--output", default="runs/experiment")

    optimize_cmd = sub.add_parser("optimize", help="Run bounded evidence-led experiments with untouched promotion confirmation")
    _add_input_args(optimize_cmd)
    optimize_cmd.add_argument("--preset", choices=PRESET_NAMES, default=None)
    optimize_cmd.add_argument("--horizon", type=int, default=12)
    optimize_cmd.add_argument("--freq", default=None)
    optimize_cmd.add_argument("--season-length", type=int, default=None)
    optimize_cmd.add_argument("--levels", nargs="+", type=int, default=[80, 95])
    optimize_cmd.add_argument("--unit-label", default=None)
    optimize_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default="ffill")
    optimize_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="light", help=MODEL_POLICY_HELP)
    _add_model_allowlist_args(optimize_cmd)
    optimize_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none")
    optimize_cmd.add_argument("--normalization-factor-col", default=None)
    optimize_cmd.add_argument("--normalization-label", default="")
    optimize_cmd.add_argument("--hierarchy-reconciliation", choices=["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"], default="none")
    optimize_cmd.add_argument("--train-known-future-regressors", action=argparse.BooleanOptionalAction, default=False)
    optimize_cmd.add_argument("--mlforecast-feature-policy", choices=["basic", "rolling"], default="basic")
    optimize_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=False)
    optimize_cmd.add_argument("--strict-cv-horizon", action=argparse.BooleanOptionalAction, default=False)
    optimize_cmd.add_argument("--weighted-ensemble", action=argparse.BooleanOptionalAction, default=True)
    _add_finn_inspired_args(optimize_cmd)
    _add_context_args(optimize_cmd)
    optimize_cmd.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    optimize_cmd.add_argument("--event", action="append", default=[])
    optimize_cmd.add_argument("--event-file", action="append", default=[])
    optimize_cmd.add_argument("--regressor", action="append", default=[])
    optimize_cmd.add_argument("--regressor-file", action="append", default=[])
    optimize_cmd.add_argument("--variants", nargs="+", choices=sorted((*EXPERIMENT_VARIANTS, "all")), default=None)
    optimize_cmd.add_argument("--max-variants", type=int, default=4)
    optimize_cmd.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Defaults to the context research budget, or one iteration without context",
    )
    optimize_cmd.add_argument("--patience", type=int, default=2)
    optimize_cmd.add_argument("--output", default="runs/optimizer")

    refresh_cmd = sub.add_parser("refresh", help="Refresh a forecast by reusing a previous run's persisted setup")
    refresh_cmd.add_argument("--previous-run", required=True, help="Prior run directory with manifest.json")
    refresh_cmd.add_argument("--input", required=True, help="Updated CSV, XLSX, or XLSM file")
    refresh_cmd.add_argument("--sheet", default=None, help="Excel sheet name/index")
    refresh_cmd.add_argument("--id-col", default=None, help="Override series identifier column; defaults to previous run spec")
    refresh_cmd.add_argument("--time-col", default=None, help="Override date/time column; defaults to previous run spec")
    refresh_cmd.add_argument("--target-col", default=None, help="Override target column; defaults to previous run spec")
    refresh_cmd.add_argument("--horizon", type=int, default=None)
    refresh_cmd.add_argument("--freq", default=None)
    refresh_cmd.add_argument("--season-length", type=int, default=None)
    refresh_cmd.add_argument("--levels", nargs="+", type=int, default=None)
    refresh_cmd.add_argument("--unit-label", default=None)
    refresh_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default=None)
    refresh_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default=None, help=MODEL_POLICY_HELP)
    _add_model_allowlist_args(refresh_cmd)
    refresh_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default=None)
    refresh_cmd.add_argument("--normalization-factor-col", default=None)
    refresh_cmd.add_argument("--normalization-label", default=None)
    refresh_cmd.add_argument("--hierarchy-reconciliation", choices=["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"], default=None)
    refresh_cmd.add_argument("--train-known-future-regressors", action=argparse.BooleanOptionalAction, default=None)
    refresh_cmd.add_argument("--mlforecast-feature-policy", choices=["basic", "rolling"], default=None)
    refresh_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=None)
    refresh_cmd.add_argument("--strict-cv-horizon", action=argparse.BooleanOptionalAction, default=None)
    refresh_cmd.add_argument("--weighted-ensemble", action=argparse.BooleanOptionalAction, default=None)
    _add_finn_inspired_args(refresh_cmd)
    refresh_cmd.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    refresh_cmd.add_argument("--event", action="append", default=[], help="Replacement driver/event scenario JSON; reused from previous run unless any --event/--event-file is provided")
    refresh_cmd.add_argument("--event-file", action="append", default=[], help="Replacement event file; reused from previous run unless any --event/--event-file is provided")
    refresh_cmd.add_argument("--regressor", action="append", default=[], help="Replacement regressor declaration; reused from previous run unless any --regressor/--regressor-file is provided")
    refresh_cmd.add_argument("--regressor-file", action="append", default=[], help="Replacement regressor file; reused from previous run unless any --regressor/--regressor-file is provided")
    refresh_cmd.add_argument("--output", default="runs/refresh_latest")
    _add_ledger_registration_args(refresh_cmd)

    ingest_cmd = sub.add_parser("ingest", help="Convert a Kusto/DAX/MCP query result export into canonical forecast input")
    ingest_cmd.add_argument("--input", required=True, help="Query result file: CSV, XLSX/XLSM, JSON, JSONL, or MCP columnar JSON")
    ingest_cmd.add_argument("--sheet", default=None, help="Excel sheet name/index")
    ingest_cmd.add_argument("--source", default="mcp", help="Source kind for metadata/query extension, e.g. kusto, dax, sql, excel")
    ingest_cmd.add_argument("--query-file", default=None, help="Optional .kql/.dax/.sql query file to copy next to the canonical CSV")
    ingest_cmd.add_argument("--query", default=None, help="Optional query text to store next to the canonical CSV")
    ingest_cmd.add_argument("--id-col", default="unique_id", help="Series identifier column if present in the query result")
    ingest_cmd.add_argument("--id-value", default=None, help="Series identifier to inject when the query returns one metric and no ID column")
    ingest_cmd.add_argument("--time-col", default="ds", help="Date/time column in the query result")
    ingest_cmd.add_argument("--target-col", default="y", help="Numeric target column in the query result")
    ingest_cmd.add_argument("--output", required=True, help="Canonical CSV output path")
    ingest_cmd.add_argument("--forecast-output", default=None, help="Optional run directory; if provided, forecast immediately after ingest")
    ingest_cmd.add_argument("--preset", choices=PRESET_NAMES, default=None, help="Opinionated defaults when forecasting after ingest")
    ingest_cmd.add_argument("--horizon", type=int, default=12)
    ingest_cmd.add_argument("--freq", default=None)
    ingest_cmd.add_argument("--season-length", type=int, default=None)
    ingest_cmd.add_argument("--levels", nargs="+", type=int, default=[80, 95], help="Prediction interval levels when forecasting after ingest")
    ingest_cmd.add_argument("--unit-label", default=None, help="Optional unit/currency label for headline values when forecasting after ingest")
    ingest_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default="ffill")
    ingest_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="light", help=MODEL_POLICY_HELP)
    _add_model_allowlist_args(ingest_cmd)
    ingest_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none")
    ingest_cmd.add_argument("--normalization-factor-col", default=None)
    ingest_cmd.add_argument("--normalization-label", default="")
    ingest_cmd.add_argument("--hierarchy-reconciliation", choices=["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"], default="none")
    ingest_cmd.add_argument("--train-known-future-regressors", action=argparse.BooleanOptionalAction, default=False)
    ingest_cmd.add_argument("--mlforecast-feature-policy", choices=["basic", "rolling"], default="basic")
    ingest_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=False, help="Fail if rolling backtest metrics cannot be produced")
    ingest_cmd.add_argument("--strict-cv-horizon", action=argparse.BooleanOptionalAction, default=False, help="Require CV folds to match --horizon when forecasting after ingest")
    ingest_cmd.add_argument("--weighted-ensemble", action=argparse.BooleanOptionalAction, default=True)
    _add_finn_inspired_args(ingest_cmd)
    _add_context_args(ingest_cmd)
    ingest_cmd.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    ingest_cmd.add_argument("--event", action="append", default=[], help="Driver/event scenario JSON when forecasting after ingest")
    ingest_cmd.add_argument("--event-file", action="append", default=[], help="JSON/YAML/CSV event file when forecasting after ingest")
    ingest_cmd.add_argument("--regressor", action="append", default=[], help="Known-future regressor JSON declaration when forecasting after ingest")
    ingest_cmd.add_argument("--regressor-file", action="append", default=[], help="JSON/YAML/CSV regressor declaration file when forecasting after ingest")
    ingest_cmd.add_argument("--custom-model", default=None, help="Python import path for a custom forecast callable when forecasting after ingest")
    ingest_cmd.add_argument("--custom-script", default=None, help="Executable custom model script when forecasting after ingest")
    ingest_cmd.add_argument("--custom-model-name", default=None, help="Readable custom model name when forecasting after ingest")
    ingest_cmd.add_argument("--custom-timeout-seconds", type=int, default=120)
    ingest_cmd.add_argument("--custom-arg", action="append", default=[])
    _add_ledger_registration_args(ingest_cmd)

    hierarchy_cmd = sub.add_parser("hierarchy", help="Aggregate hierarchy columns into canonical forecast nodes")
    hierarchy_cmd.add_argument("--input", required=True, help="CSV, XLSX, or XLSM file")
    hierarchy_cmd.add_argument("--sheet", default=None, help="Excel sheet name/index")
    hierarchy_cmd.add_argument("--time-col", default="ds", help="Date/time column, e.g. Month or Week")
    hierarchy_cmd.add_argument("--target-col", default="y", help="Numeric value to forecast, e.g. Revenue or ARR")
    hierarchy_cmd.add_argument(
        "--hierarchy-cols",
        nargs="+",
        required=True,
        help="Ordered hierarchy columns; first is top level under Total, e.g. Region Product SKU",
    )
    hierarchy_cmd.add_argument("--output", required=True, help="CSV path for canonical hierarchy nodes")

    explain_cmd = sub.add_parser("explain", help="Print the saved model card from a run directory")
    explain_cmd.add_argument("--run", required=True, help="Run directory produced by forecast")

    report_cmd = sub.add_parser("report", help="Regenerate HTML/base64/Streamlit report artifacts for a run directory")
    report_cmd.add_argument("--run", required=True, help="Run directory produced by forecast")

    compare_cmd = sub.add_parser("compare", help="Compare a forecast run to an external finance forecast")
    compare_cmd.add_argument("--run", required=True, help="Run directory produced by forecast")
    compare_cmd.add_argument("--external", required=True, help="External forecast file: CSV, XLSX, or XLSM")
    compare_cmd.add_argument("--sheet", default=None, help="Excel sheet name/index for the external forecast")
    compare_cmd.add_argument("--format", choices=["auto", "long", "wide"], default="auto", help="External forecast shape; use wide for date-column finance workbooks")
    compare_cmd.add_argument("--model-name", default=None, help="External model label; inferred from the file stem when omitted")
    compare_cmd.add_argument("--source-id", default=None, help="External source identifier; inferred from the file stem when omitted")
    compare_cmd.add_argument("--id-col", default="unique_id", help="Series identifier column in the external forecast")
    compare_cmd.add_argument("--time-col", default="ds", help="Date column in the external forecast")
    compare_cmd.add_argument("--target-col", default="yhat", help="External forecast value column")
    compare_cmd.add_argument("--model-col", default="model", help="External model column")
    compare_cmd.add_argument("--scaffold-model", default=None, help="Optional scaffold model from appendix/forecast_long.csv; defaults to selected forecast.csv")
    compare_cmd.add_argument("--output", default=None, help="Output folder for comparison artifacts; defaults to <run>\\comparison")

    score_external_cmd = sub.add_parser("score-external", help="Score cutoff-labeled external forecasts against actuals")
    score_external_cmd.add_argument("--external", required=True, help="External forecast file with cutoff/forecast_origin labels")
    score_external_cmd.add_argument("--actuals", required=True, help="Actuals file with unique_id/ds/y history")
    score_external_cmd.add_argument("--output", required=True, help="Output folder for external scoring artifacts")
    score_external_cmd.add_argument("--sheet", default=None, help="Excel sheet name/index for the external forecast")
    score_external_cmd.add_argument("--actuals-sheet", default=None, help="Excel sheet name/index for the actuals")
    score_external_cmd.add_argument("--format", choices=["auto", "long", "wide"], default="auto", help="External forecast shape; scoring requires cutoff labels")
    score_external_cmd.add_argument("--model-name", default=None, help="External model label; inferred from the file stem when omitted")
    score_external_cmd.add_argument("--source-id", default=None, help="External source identifier; inferred from the file stem when omitted")
    score_external_cmd.add_argument("--id-col", default="unique_id", help="Series identifier column in the external forecast")
    score_external_cmd.add_argument("--time-col", default="ds", help="Target date column in the external forecast")
    score_external_cmd.add_argument("--target-col", default="yhat", help="External forecast value column")
    score_external_cmd.add_argument("--model-col", default="model", help="External model column")
    score_external_cmd.add_argument("--actual-id-col", default="unique_id", help="Series identifier column in the actuals file")
    score_external_cmd.add_argument("--actual-time-col", default="ds", help="Date column in the actuals file")
    score_external_cmd.add_argument("--actual-target-col", default="y", help="Actual value column")
    score_external_cmd.add_argument("--season-length", type=int, default=1, help="Seasonal period for leakage-safe MASE/RMSSE scales")
    score_external_cmd.add_argument("--horizon", type=int, default=None, help="Optional business horizon to record in scoring metrics")
    score_external_cmd.add_argument("--cutoff-contract", default=None, help="Optional native cutoff_contract.csv required for exact comparability")

    byo_cmd = sub.add_parser("byo-model", help="Import, compare, and score Excel-owned finance model outputs")
    byo_sub = byo_cmd.add_subparsers(dest="byo_command", required=True)

    byo_ingest = byo_sub.add_parser("ingest", help="Convert one or more BYO workbook sheets to canonical external forecasts")
    _add_byo_model_args(byo_ingest)
    byo_ingest.add_argument("--output", required=True, help="Output folder for BYO model artifacts")

    byo_compare = byo_sub.add_parser("compare", help="Compare BYO workbook forecasts against an existing forecast run")
    byo_compare.add_argument("--run", required=True, help="Forecast run directory produced by forecast")
    _add_byo_model_args(byo_compare)
    byo_compare.add_argument("--scaffold-model", default=None, help="Optional scaffold model from appendix/forecast_long.csv; defaults to selected forecast.csv")
    byo_compare.add_argument(
        "--main-model-preference",
        default=None,
        help="Optional display-only BYO model/scenario label to store in the manifest; does not overwrite forecast.csv",
    )
    byo_compare.add_argument("--output", default=None, help="Output folder; defaults to <run>\\byo_model")

    byo_score = byo_sub.add_parser("score", help="Score cutoff-labeled BYO snapshots against actuals")
    _add_byo_model_args(byo_score)
    byo_score.add_argument("--actuals", required=True, help="Actuals file with unique_id/ds/y or matching group columns")
    byo_score.add_argument("--actuals-sheet", default=None, help="Excel sheet name/index for actuals")
    byo_score.add_argument("--actual-id-col", default="unique_id", help="Series identifier column in the actuals file")
    byo_score.add_argument("--actual-time-col", default="ds", help="Date column in the actuals file")
    byo_score.add_argument("--actual-target-col", default="y", help="Actual value column")
    byo_score.add_argument("--season-length", type=int, default=1, help="Seasonal period for leakage-safe MASE/RMSSE scales")
    byo_score.add_argument("--horizon", type=int, default=None, help="Optional business horizon to record in scoring metrics")
    byo_score.add_argument("--output", required=True, help="Output folder for BYO scoring artifacts")

    finn_cmd = sub.add_parser("finn", help="Optional FINN/finnts bridge for advisory external forecasts")
    finn_sub = finn_cmd.add_subparsers(dest="finn_command", required=True)
    finn_check = finn_sub.add_parser("check", help="Check Rscript and the finnts R package")
    finn_check.add_argument("--rscript", default="Rscript")

    finn_run = finn_sub.add_parser("run", help="Run a user-supplied FINN R runner or write a runner template")
    finn_run.add_argument("--input", required=True, help="Canonical scaffold history input for FINN")
    finn_run.add_argument("--output", required=True, help="Output folder for FINN bridge artifacts")
    finn_run.add_argument("--runner", default=None, help="Optional R runner script; omit to write finn_runner_template.R only")
    finn_run.add_argument("--rscript", default="Rscript")
    finn_run.add_argument("--raw-output", default="finn_raw_forecast.csv")
    finn_run.add_argument("--seed", type=int, default=123)
    finn_run.add_argument("--model-name", default="FINN")
    finn_run.add_argument("--source-id", default="finn")
    finn_run.add_argument("--extra-arg", action="append", default=[])

    finn_pipeline = finn_sub.add_parser("pipeline", help="Run spec-driven FINN challenger orchestration against an existing scaffold run")
    finn_pipeline.add_argument("--run", required=True, help="Scaffold forecast run directory containing manifest.json")
    finn_pipeline.add_argument("--models", nargs="+", default=None, help="Override FINN models_to_run for this retrofit run")
    finn_pipeline.add_argument("--back-test-scenarios", type=int, default=None)
    finn_pipeline.add_argument("--back-test-spacing", type=int, default=None)
    finn_pipeline.add_argument("--rscript", default="Rscript")
    finn_pipeline.add_argument("--on-error", choices=["skip", "fail"], default="skip")
    finn_pipeline.add_argument("--timeout", type=int, default=3600)

    finn_ingest = finn_sub.add_parser("ingest", help="Canonicalize FINN forecast output into scaffold external format")
    _add_finn_external_args(finn_ingest, require_output=True)

    finn_compare = finn_sub.add_parser("compare", help="Compare a scaffold run to FINN advisory forecasts")
    finn_compare.add_argument("--run", required=True, help="Scaffold forecast run directory")
    finn_compare.add_argument("--scaffold-model", default=None)
    _add_finn_external_args(finn_compare, require_output=False)

    finn_score = finn_sub.add_parser("score", help="Score cutoff-labeled FINN forecasts against actuals")
    finn_score.add_argument("--run", default=None, help="Optional scaffold run directory; when supplied, default output is <run>\\finn and artifacts are registered for reporting")
    finn_score.add_argument("--actuals", required=True)
    finn_score.add_argument("--actuals-sheet", default=None)
    finn_score.add_argument("--actual-id-col", default="unique_id")
    finn_score.add_argument("--actual-time-col", default="ds")
    finn_score.add_argument("--actual-target-col", default="y")
    finn_score.add_argument("--season-length", type=int, default=1)
    finn_score.add_argument("--horizon", type=int, default=None)
    _add_finn_external_args(finn_score, require_output=False)

    lab_cmd = sub.add_parser("scenario-lab", help="Run synthetic forecast scenarios and score accuracy/ease/validity")
    lab_cmd.add_argument("--count", type=int, default=100)
    lab_cmd.add_argument("--output", default="runs/scenario_lab")
    lab_cmd.add_argument("--model-policy", choices=["standard", "light", "auto", "baseline", "statsforecast"], default="light")
    lab_cmd.add_argument("--seed", type=int, default=42)

    qa_cmd = sub.add_parser("workbench-qa", help="Run golden forecast workbench QA scenarios")
    qa_cmd.add_argument("--output", default="runs/workbench_qa", help="Folder for QA summary and generated runs")
    qa_cmd.add_argument("--scenarios", nargs="+", choices=WORKBENCH_QA_SCENARIOS, default=list(GOLDEN_SCENARIOS))
    qa_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="baseline")
    qa_cmd.add_argument("--app-test", action=argparse.BooleanOptionalAction, default=True, help="Run Streamlit AppTest for generated dashboards")
    qa_cmd.add_argument("--app-test-timeout", type=int, default=90, help="Streamlit AppTest timeout in seconds for each generated dashboard")

    release_cmd = sub.add_parser("release-gates", help="Run local release-readiness gates")
    release_cmd.add_argument("--output", default="runs/release_gates", help="Folder for release-gate outputs")
    release_cmd.add_argument(
        "--extended",
        action="store_true",
            help="Run a stricter local release profile: at least 20 light-policy scenarios, all-family workbench QA, and required ml/hierarchy extras",
    )
    release_cmd.add_argument("--build", action=argparse.BooleanOptionalAction, default=True, help="Build wheel/sdist and inspect package contents")
    release_cmd.add_argument("--install-smoke", action=argparse.BooleanOptionalAction, default=True, help="Install the built wheel in an isolated venv and smoke public APIs")
    release_cmd.add_argument("--scenario-count", type=int, default=8, help="Deterministic scenario-lab count for numeric gates")
    release_cmd.add_argument("--scenario-model-policy", choices=["standard", "light", "auto", "baseline", "statsforecast"], default="baseline")
    release_cmd.add_argument("--workbench-qa", action=argparse.BooleanOptionalAction, default=True, help="Run generated workbench QA scenarios")
    release_cmd.add_argument("--workbench-model-policy", choices=MODEL_POLICY_CHOICES, default="baseline")
    release_cmd.add_argument("--app-test", action=argparse.BooleanOptionalAction, default=True, help="Run Streamlit AppTest inside workbench QA")
    release_cmd.add_argument("--app-test-timeout", type=int, default=90, help="Streamlit AppTest timeout in seconds for workbench QA dashboards")
    release_cmd.add_argument("--live-streamlit", action=argparse.BooleanOptionalAction, default=True, help="Launch one generated dashboard and check HTTP health")
    release_cmd.add_argument("--live-streamlit-timeout", type=int, default=45)
    release_cmd.add_argument(
        "--require-optional",
        nargs="*",
        default=[],
        metavar="EXTRA",
        help=f"Optional dependency groups that must be importable; known groups: {', '.join(sorted(OPTIONAL_EXTRAS))}",
    )
    release_cmd.add_argument("--json", action="store_true", help="Print the full machine-readable release-gate payload instead of the compact summary")

    ledger_cmd = sub.add_parser("ledger", help="Manage a refreshable forecast ledger with versions, official locks, actuals, adjustments, and Power BI exports")
    ledger_sub = ledger_cmd.add_subparsers(dest="ledger_command", required=True)

    ledger_init = ledger_sub.add_parser("init", help="Create or migrate a forecast ledger")
    ledger_init.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Ledger folder; defaults to runs\\forecast_ledger")

    ledger_register = ledger_sub.add_parser("register", help="Register an existing forecast run directory as a ledger version")
    ledger_register.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ledger_register.add_argument("--run", required=True, help="Run directory produced by forecast")
    ledger_register.add_argument("--forecast-key", required=True, help="Business forecast key, e.g. Product ARR")
    ledger_register.add_argument("--version-label", default="", help="Friendly label such as March refresh")
    ledger_register.add_argument("--created-by", default="")
    ledger_register.add_argument("--notes", default="")
    ledger_register.add_argument("--source-metadata", default=None, help="Optional *.source.json from ingest/query refresh")
    ledger_register.add_argument("--source-kind", default="", help="Optional source kind when no source metadata file exists")

    ledger_lock = ledger_sub.add_parser("lock", help="Mark a registered version as an official submitted forecast")
    ledger_lock.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ledger_lock.add_argument("--version-id", required=True)
    ledger_lock.add_argument("--lock-label", required=True, help="Submitted view label, e.g. March lock")
    ledger_lock.add_argument("--audience", default="leadership")
    ledger_lock.add_argument("--planning-cycle", default="")
    ledger_lock.add_argument("--communication-date", default="")
    ledger_lock.add_argument("--submitted-to", default="")
    ledger_lock.add_argument("--reason", default="")
    ledger_lock.add_argument("--locked-by", default="")
    ledger_lock.add_argument("--notes", default="")

    ledger_actuals = ledger_sub.add_parser("actuals", help="Append a revised actuals refresh and rescore registered forecast versions")
    ledger_actuals.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ledger_actuals.add_argument("--input", required=True)
    ledger_actuals.add_argument("--sheet", default=None)
    ledger_actuals.add_argument("--forecast-key", required=True)
    ledger_actuals.add_argument("--id-col", default="unique_id")
    ledger_actuals.add_argument("--time-col", default="ds")
    ledger_actuals.add_argument("--target-col", default="y")
    ledger_actuals.add_argument("--source-kind", default="")
    ledger_actuals.add_argument("--source-id", default="")
    ledger_actuals.add_argument("--revision-label", default="")
    ledger_actuals.add_argument("--known-as-of", default="")

    ledger_adjustments = ledger_sub.add_parser("adjustments", help="Append anomaly/business-model/regime-change adjustment contracts")
    ledger_adjustments.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ledger_adjustments.add_argument("--input", required=True)
    ledger_adjustments.add_argument("--sheet", default=None)
    ledger_adjustments.add_argument("--forecast-key", required=True)

    ledger_compare = ledger_sub.add_parser("compare", help="Compare a selected official lock/version against the latest or chosen version")
    ledger_compare.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ledger_compare.add_argument("--forecast-key", required=True)
    ledger_compare.add_argument("--against-lock", default=None, help="Official lock label to compare against, e.g. March lock")
    ledger_compare.add_argument("--against-version-id", default=None)
    ledger_compare.add_argument("--latest-version-id", default=None)
    ledger_compare.add_argument("--watch-pct", type=float, default=None, help="Optional watch threshold as decimal, e.g. 0.05")
    ledger_compare.add_argument("--call-up-pct", type=float, default=None, help="Optional call-up threshold as decimal, e.g. 0.10")
    ledger_compare.add_argument("--call-down-pct", type=float, default=None, help="Optional call-down threshold as decimal, e.g. 0.10")

    ledger_export = ledger_sub.add_parser("export", help="Export stable CSV mirrors for Power BI / semantic model ingestion")
    ledger_export.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ledger_export.add_argument("--output", default=None)

    status_cmd = sub.add_parser("status", help="Summarize local forecast runs without opening Streamlit")
    status_cmd.add_argument("--runs", default="runs", help="Root folder to scan for forecast run manifests")
    status_cmd.add_argument("--run", default=None, help="Single forecast run folder to summarize")
    status_cmd.add_argument("--output", default=None, help="Optional output folder for run_status_summary.csv/json")

    doctor_cmd = sub.add_parser("doctor", help="Check a local forecast run for missing artifacts and operational next actions")
    doctor_cmd.add_argument("--run", required=True, help="Forecast run folder to inspect")
    doctor_cmd.add_argument("--output", default=None, help="Optional output folder for doctor_checks.csv/json")

    drift_cmd = sub.add_parser("drift", help="Roll up local forecast drift from refresh deltas and ledger exports")
    drift_cmd.add_argument("--ledger", default=None, help="Forecast ledger folder with exports")
    drift_cmd.add_argument("--previous-run", default=None, help="Previous run path for refresh-pair context")
    drift_cmd.add_argument("--refreshed-run", default=None, help="Refreshed run path containing appendix\\refresh_delta.csv")
    drift_cmd.add_argument("--output", required=True, help="Output folder for drift_summary.csv/json/md")

    operate_cmd = sub.add_parser("operate", help="Run a hard-capped linear local operating loop from a small YAML config")
    operate_cmd.add_argument("--config", required=True, help="YAML config with a top-level steps list")
    operate_cmd.add_argument("--output", required=True, help="Output folder for operate_manifest.json")

    pipeline_cmd = sub.add_parser("pipeline", help="Run or refresh a script-backed source pipeline into one canonical forecast input")
    pipeline_sub = pipeline_cmd.add_subparsers(dest="pipeline_command", required=True)

    pipeline_run = pipeline_sub.add_parser("run", help="Run extracts/transforms from pipeline YAML and optionally forecast")
    pipeline_run.add_argument("--config", required=True, help="Pipeline YAML config")
    pipeline_run.add_argument("--output", default=None, help="Pipeline output folder; defaults to runs\\pipeline_<name>_<timestamp>")
    pipeline_run.add_argument(
        "--no-forecast",
        dest="forecast",
        action="store_false",
        default=True,
        help="Prepare the canonical input and provenance only; skip forecast execution",
    )

    pipeline_refresh = pipeline_sub.add_parser("refresh", help="Rerun pipeline YAML and refresh from a previous forecast run manifest")
    pipeline_refresh.add_argument("--config", required=True, help="Pipeline YAML config")
    pipeline_refresh.add_argument("--previous-run", required=True, help="Prior forecast run directory with manifest.json")
    pipeline_refresh.add_argument("--output", default=None, help="Pipeline output folder; defaults to runs\\pipeline_<name>_<timestamp>")

    skill_cmd = sub.add_parser("skill", help="Check or install the canonical nixtla-forecast agent skill")
    skill_sub = skill_cmd.add_subparsers(dest="skill_command", required=True)
    skill_check = skill_sub.add_parser("check", help="Report canonical and installed skill hashes")
    skill_check.add_argument("--target", default=None, help="Skill directory or SKILL.md path; defaults to ~/.copilot/skills/nixtla-forecast")
    skill_sync = skill_sub.add_parser("sync", help="Install the canonical skill with a timestamped backup")
    skill_sync.add_argument("--target", default=None, help="Skill directory or SKILL.md path; defaults to ~/.copilot/skills/nixtla-forecast")
    skill_sync.add_argument("--yes", action="store_true", required=True, help="Confirm replacing the installed skill when it differs")

    guide_cmd = sub.add_parser("guide", help="Search Nixtla/FPPy best-practice guidance")
    guide_cmd.add_argument("query", nargs="?", default=None, help="Optional search term, e.g. intervals or hierarchy")

    args = parser.parse_args(raw_argv)
    args._provided_flags = _provided_flags(raw_argv)

    try:
        return _run(args)
    except (OSError, UnicodeDecodeError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        try:
            diagnostics_paths = write_failure_diagnostics(args, exc)
        except OSError as diagnostics_exc:
            print(f"Diagnostics write failed: {diagnostics_exc}", file=sys.stderr)
        else:
            if diagnostics_paths is not None:
                print(f"Failure diagnostics written to {diagnostics_paths[0]}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.command == "setup":
        answers = _setup_answers_from_args(args)
        if args.interactive:
            answers = interactive_answers(answers)
        artifact = create_forecast_setup(args.workspace, answers)
        print(json.dumps(artifact.to_dict(), indent=2, default=str))
        return 0

    if args.command == "profile":
        spec = _spec_from_args(args)
        df = load_forecast_dataset(args.input, sheet=_coerce_sheet(args.sheet), spec=spec)
        profile = profile_dataset(df, spec)
        payload = json.dumps(profile.to_dict(), indent=2, default=str)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    if args.command == "forecast":
        spec = _spec_from_args(args)
        run = run_forecast(args.input, spec, sheet=_coerce_sheet(args.sheet))
        output_dir = run.to_directory(args.output)
        challenger_summary = _maybe_run_challengers(output_dir)
        ledger_result = _maybe_register_ledger(args, output_dir)
        print(f"Forecast written to {output_dir}")
        accuracy_line = _accuracy_gate_summary_line(output_dir)
        if accuracy_line:
            print(accuracy_line)
        policy_line = _model_policy_summary_line(run)
        if policy_line:
            print(policy_line)
        if challenger_summary:
            print(challenger_summary)
        if ledger_result:
            print(f"Ledger version registered: {ledger_result['forecast_version_id']}")
        print()
        print(run.explanation())
        return 0

    if args.command == "compare-models":
        spec = _spec_from_args(args)
        leaderboard = compare_models(args.input, spec, sheet=_coerce_sheet(args.sheet), output_dir=args.output)
        print(f"Compare-models run written to {Path(args.output)}")
        print(leaderboard.head(20).to_string(index=False))
        return 0

    if args.command == "experiment":
        spec = _spec_from_args(args)
        result = run_experiment(
            args.input,
            spec,
            output_dir=args.output,
            variants=args.variants,
            max_variants=args.max_variants,
            hypothesis=args.hypothesis,
            sheet=_coerce_sheet(args.sheet),
        )
        print(f"Experiment written to {result.output_dir}")
        print(f"Variants: {len(result.summary)}")
        print("Recommendation: review experiment_recommendation.md")
        return 0

    if args.command == "optimize":
        spec = _spec_from_args(args)
        result = run_optimizer(
            args.input,
            spec,
            output_dir=args.output,
            variants=args.variants,
            max_iterations=args.max_iterations,
            max_variants=args.max_variants,
            patience=args.patience,
            sheet=_coerce_sheet(args.sheet),
        )
        print(f"Research optimizer written to {result.output_dir}")
        print(f"Stop reason: {result.manifest.get('stopped_reason', 'unknown')}")
        print(f"Promotion recommended: {result.manifest.get('promotion_recommended', False)}")
        print("Review promotion_decision.json and stop_receipt.json")
        return 0

    if args.command == "refresh":
        spec = _refresh_spec_from_args(args)
        run = run_forecast(args.input, spec, sheet=_coerce_sheet(args.sheet))
        output_dir = run.to_directory(args.output)
        refresh_result = write_refresh_artifacts(args.previous_run, output_dir)
        challenger_summary = _maybe_run_challengers(output_dir)
        ledger_result = _maybe_register_ledger(args, output_dir)
        print(f"Refresh forecast written to {output_dir}")
        accuracy_line = _accuracy_gate_summary_line(output_dir)
        if accuracy_line:
            print(accuracy_line)
        print(f"Refresh delta rows: {refresh_result['delta_rows']}")
        if challenger_summary:
            print(challenger_summary)
        if ledger_result:
            print(f"Ledger version registered: {ledger_result['forecast_version_id']}")
        print()
        print(run.explanation())
        return 0

    if args.command == "ingest":
        if not args.forecast_output and _custom_forecast_flags_requested(args):
            raise ValueError("ingest custom model flags require --forecast-output because custom models only run during forecasting")
        spec = _spec_from_args(args)
        metadata = ingest_query_result(
            args.input,
            args.output,
            sheet=_coerce_sheet(args.sheet),
            source_kind=args.source,
            query_file=args.query_file,
            query_text=args.query,
            id_col=args.id_col,
            id_value=args.id_value,
            time_col=args.time_col,
            target_col=args.target_col,
        )
        if args.forecast_output:
            run = run_forecast(args.output, replace(spec, id_col="unique_id", time_col="ds", target_col="y"))
            output_dir = run.to_directory(args.forecast_output)
            metadata["forecast_output"] = str(output_dir)
            accuracy_line = _accuracy_gate_summary_line(output_dir)
            if accuracy_line:
                metadata["accuracy_gate"] = accuracy_line
            ledger_result = _maybe_register_ledger(args, output_dir, source_metadata_path=metadata.get("metadata_file"))
            if ledger_result:
                metadata["ledger"] = ledger_result
        print(json.dumps(metadata, indent=2, default=str))
        return 0

    if args.command == "hierarchy":
        frame = aggregate_hierarchy_frame(
            args.input,
            sheet=_coerce_sheet(args.sheet),
            hierarchy_cols=args.hierarchy_cols,
            time_col=args.time_col,
            target_col=args.target_col,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(json.dumps(hierarchy_summary(frame), indent=2, default=str))
        return 0

    if args.command == "explain":
        model_card = Path(args.run) / "model_card.md"
        if not model_card.exists():
            raise FileNotFoundError(model_card)
        print(model_card.read_text(encoding="utf-8"))
        return 0

    if args.command == "report":
        paths = write_report_artifacts_from_directory(args.run)
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return 0

    if args.command == "compare":
        result = write_forecast_comparison(
            args.run,
            args.external,
            output_dir=args.output,
            sheet=_coerce_sheet(args.sheet),
            external_format=args.format,
            external_model_name=args.model_name,
            external_source_id=args.source_id,
            external_id_col=args.id_col,
            external_time_col=args.time_col,
            external_value_col=args.target_col,
            external_model_col=args.model_col,
            scaffold_model=args.scaffold_model,
        )
        print(json.dumps(result.manifest, indent=2, default=str))
        return 0

    if args.command == "score-external":
        result = write_external_forecast_scores(
            args.external,
            args.actuals,
            args.output,
            sheet=_coerce_sheet(args.sheet),
            actuals_sheet=_coerce_sheet(args.actuals_sheet),
            external_format=args.format,
            external_model_name=args.model_name,
            external_source_id=args.source_id,
            external_id_col=args.id_col,
            external_time_col=args.time_col,
            external_value_col=args.target_col,
            external_model_col=args.model_col,
            actual_id_col=args.actual_id_col,
            actual_time_col=args.actual_time_col,
            actual_value_col=args.actual_target_col,
            season_length=args.season_length,
            requested_horizon=args.horizon,
            cutoff_contract=args.cutoff_contract,
        )
        print(json.dumps(result.manifest, indent=2, default=str))
        return 0

    if args.command == "byo-model":
        result = _run_byo_model_command(args)
        print(json.dumps(result.manifest, indent=2, default=str))
        return 0

    if args.command == "finn":
        payload = _run_finn_command(args)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "scenario-lab":
        payload = run_scenario_lab(count=args.count, output_dir=args.output, model_policy=args.model_policy, seed=args.seed)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "workbench-qa":
        payload = run_workbench_qa(
            output_dir=args.output,
            scenarios=tuple(args.scenarios),
            model_policy=args.model_policy,
            app_test=args.app_test,
            app_test_timeout_seconds=args.app_test_timeout,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "release-gates":
        payload = run_release_gates(
            output_dir=args.output,
            extended=args.extended,
            build=args.build,
            install_smoke=args.install_smoke,
            scenario_count=args.scenario_count,
            scenario_model_policy=args.scenario_model_policy,
            workbench_qa=args.workbench_qa,
            workbench_model_policy=args.workbench_model_policy,
            workbench_app_test=args.app_test,
            workbench_app_test_timeout_seconds=args.app_test_timeout,
            live_streamlit=args.live_streamlit,
            live_streamlit_timeout=args.live_streamlit_timeout,
            require_optional=tuple(args.require_optional),
        )
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(format_release_gate_console_summary(payload), end="")
        return 0 if payload["summary"]["status"] == "passed" else 1

    if args.command == "ledger":
        result = _run_ledger(args)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "status":
        payload = build_status_payload(run=args.run, runs=args.runs)
        if args.output:
            payload["paths"] = write_status_outputs(payload, args.output)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "doctor":
        payload = build_doctor_payload(args.run)
        if args.output:
            payload["paths"] = write_doctor_outputs(payload, args.output)
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["overall_status"] != "fail" else 1

    if args.command == "drift":
        paths = write_drift_report(args.output, ledger=args.ledger, previous_run=args.previous_run, refreshed_run=args.refreshed_run)
        print(json.dumps(paths, indent=2, default=str))
        return 0

    if args.command == "operate":
        payload = run_operating_loop(args.config, args.output)
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["status"] != "failed" else 1

    if args.command == "pipeline":
        payload = _run_pipeline_command(args)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "skill":
        if args.skill_command == "check":
            payload = check_agent_skill(args.target)
            print(json.dumps(payload, indent=2, default=str))
            return 0 if payload["in_sync"] else 1
        if args.skill_command == "sync":
            payload = sync_agent_skill(args.target, confirmed=args.yes)
            print(json.dumps(payload, indent=2, default=str))
            return 0
        raise ValueError(f"unknown skill command: {args.skill_command}")

    if args.command == "guide":
        if args.query in {"skill", "agent-skill", "nixtla-forecast"}:
            print(load_agent_skill())
            return 0
        if args.query == "presets":
            print(json.dumps(preset_catalog(), indent=2, default=str))
            return 0
        print(format_knowledge(search_knowledge(args.query)))
        return 0

    raise ValueError(f"unknown command: {args.command}")


def _run_pipeline_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.pipeline_command == "run":
        return run_pipeline(args.config, args.output, forecast=args.forecast)
    if args.pipeline_command == "refresh":
        return refresh_pipeline(args.config, args.previous_run, args.output)
    raise ValueError(f"unknown pipeline command: {args.pipeline_command}")


def _run_byo_model_command(args: argparse.Namespace) -> Any:
    kwargs = _byo_model_kwargs_from_args(args)
    if args.byo_command == "ingest":
        return write_byo_model_ingest(args.file, args.output, sheets=_coerce_sheets(args.sheet), **kwargs)
    if args.byo_command == "compare":
        return write_byo_model_comparison(
            args.run,
            args.file,
            output_dir=args.output,
            sheets=_coerce_sheets(args.sheet),
            scaffold_model=args.scaffold_model,
            main_model_preference=args.main_model_preference,
            **kwargs,
        )
    if args.byo_command == "score":
        return write_byo_model_scores(
            args.file,
            args.actuals,
            args.output,
            sheets=_coerce_sheets(args.sheet),
            actuals_sheet=_coerce_sheet(args.actuals_sheet),
            actual_id_col=args.actual_id_col,
            actual_time_col=args.actual_time_col,
            actual_value_col=args.actual_target_col,
            season_length=args.season_length,
            requested_horizon=args.horizon,
            **kwargs,
        )
    raise ValueError(f"unknown byo-model command: {args.byo_command}")


def _run_finn_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.finn_command == "check":
        return check_finn_environment(rscript=args.rscript).manifest
    if args.finn_command == "pipeline":
        manifest = json.loads((Path(args.run) / "manifest.json").read_text(encoding="utf-8"))
        spec_challengers = forecast_spec_from_dict(manifest.get("spec", {})).challengers
        finn = next((challenger for challenger in spec_challengers if challenger.engine == "finn"), None)
        if finn is None:
            finn = ChallengerSpec(engine="finn")
        finn = replace(
            finn,
            enabled=True,
            models=tuple(args.models) if args.models else finn.models,
            back_test_scenarios=args.back_test_scenarios if args.back_test_scenarios is not None else finn.back_test_scenarios,
            back_test_spacing=args.back_test_spacing if args.back_test_spacing is not None else finn.back_test_spacing,
            rscript=args.rscript if args.rscript != "Rscript" else finn.rscript,
            on_error=args.on_error,
            timeout_seconds=args.timeout,
        )
        return run_challengers(args.run, challengers=(finn,))
    if args.finn_command == "run":
        result = run_finn_bridge(
            args.input,
            args.output,
            runner=args.runner,
            rscript=args.rscript,
            raw_output=args.raw_output,
            seed=args.seed,
            model_name=args.model_name,
            source_id=args.source_id,
            extra_args=tuple(args.extra_arg or ()),
        )
        return result.manifest
    kwargs = _finn_external_kwargs_from_args(args)
    if args.finn_command == "ingest":
        result = canonicalize_finn_forecasts(args.input, output_dir=args.output, sheet=_coerce_sheet(args.sheet), **kwargs)
        return result.manifest
    if args.finn_command == "compare":
        result = compare_finn_forecasts(
            args.run,
            args.input,
            args.output,
            sheet=_coerce_sheet(args.sheet),
            scaffold_model=args.scaffold_model,
            **kwargs,
        )
        return result.manifest
    if args.finn_command == "score":
        result = score_finn_forecasts(
            args.input,
            args.actuals,
            args.output,
            run_dir=args.run,
            actuals_sheet=_coerce_sheet(args.actuals_sheet),
            actual_id_col=args.actual_id_col,
            actual_time_col=args.actual_time_col,
            actual_value_col=args.actual_target_col,
            season_length=args.season_length,
            requested_horizon=args.horizon,
            sheet=_coerce_sheet(args.sheet),
            **kwargs,
        )
        return result.manifest
    raise ValueError(f"unknown finn command: {args.finn_command}")


def _finn_external_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": args.format,
        "model_name": args.model_name,
        "source_id": args.source_id,
        "id_col": args.id_col,
        "time_col": args.time_col,
        "value_col": args.target_col,
        "model_col": args.model_col,
        "forecast_origin_col": args.cutoff_col,
    }


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="CSV, XLSX, or XLSM file")
    parser.add_argument("--sheet", default=None, help="Excel sheet name/index")
    parser.add_argument("--id-col", default="unique_id", help="Series identifier column, e.g. Product or Account")
    parser.add_argument("--time-col", default="ds", help="Date/time column, e.g. Month or Week")
    parser.add_argument("--target-col", default="y", help="Numeric value to forecast, e.g. Revenue or ARR")


def _add_byo_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file", required=True, help="Excel/CSV file with BYO finance model forecast outputs")
    parser.add_argument("--sheet", nargs="+", default=None, help="One or more Excel sheets; omit to read all workbook sheets")
    parser.add_argument("--format", choices=["auto", "long", "wide"], default="auto", help="BYO forecast shape; use wide for date-column finance workbooks")
    parser.add_argument("--model-name", default=None, help="BYO model label; inferred from the file stem when omitted")
    parser.add_argument("--source-id", default=None, help="BYO source identifier; inferred from the file stem when omitted")
    parser.add_argument("--id-col", default="unique_id", help="Series identifier column; generated from --group-cols when absent")
    parser.add_argument("--time-col", default="ds", help="Date column for long-form BYO forecasts")
    parser.add_argument("--target-col", default="yhat", help="Forecast value column")
    parser.add_argument("--model-col", default="model", help="BYO model column")
    parser.add_argument("--cutoff-col", default="cutoff", help="Forecast-origin/cutoff column for historical snapshots")
    parser.add_argument("--scenario-col", default="scenario_name", help="Scenario/version column; sheet name is used when absent")
    parser.add_argument("--version-col", default="version", help="Optional version column to mirror into model_version")
    parser.add_argument("--group-cols", nargs="*", default=[], help="Ordered rollup columns, e.g. ProductGroup ProductLine Product")
    parser.add_argument("--rollups", action=argparse.BooleanOptionalAction, default=True, help="Create explicit Total/prefix rollup rows from --group-cols")
    parser.add_argument("--total-label", default="Total", help="Unique ID label for the derived total rollup")


def _add_finn_external_args(parser: argparse.ArgumentParser, *, require_output: bool) -> None:
    parser.add_argument("--input", required=True, help="FINN forecast output file")
    parser.add_argument("--output", required=require_output, default=None, help="Output folder for FINN bridge artifacts")
    parser.add_argument("--sheet", default=None, help="Excel sheet name/index for FINN forecast output")
    parser.add_argument("--format", choices=["auto", "long", "wide"], default="auto", help="FINN forecast shape")
    parser.add_argument("--model-name", default="FINN", help="FINN model label when the file lacks a model column")
    parser.add_argument("--source-id", default="finn", help="External source identifier for FINN outputs")
    parser.add_argument("--id-col", default="unique_id", help="Series identifier column in FINN output")
    parser.add_argument("--time-col", default="ds", help="Date column in FINN output")
    parser.add_argument("--target-col", default="yhat", help="Forecast value column in FINN output")
    parser.add_argument("--model-col", default="model", help="Model column in FINN output")
    parser.add_argument("--cutoff-col", default="cutoff", help="Forecast-origin column for historical FINN snapshots")


def _add_model_allowlist_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        dest="model_allowlist",
        action="append",
        default=[],
        help='Favorite model to include in the tournament; repeat for multiple models, e.g. --model arima --model "arima mstl"',
    )
    parser.add_argument(
        "--model-allowlist",
        dest="model_allowlist_grouped",
        nargs="+",
        action="append",
        default=[],
        help='Space-separated favorite model allowlist, e.g. --model-allowlist arima "arima mstl"',
    )


def _add_finn_inspired_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ensemble-policy",
        action="append",
        choices=["legacy_weighted", "top_k_average", "family_diverse_average"],
        default=None,
        help="Advisory ensemble lab policy to audit; repeat for multiple policies",
    )
    parser.add_argument("--ensemble-max-models", type=int, default=None, help="Maximum base models used by advisory ensemble policies")
    parser.add_argument("--fiscal-year-start", type=int, default=None, help="Fiscal year start month for feature recipe receipts")
    parser.add_argument("--fourier-period", action="append", type=int, default=None, help="Fourier period for feature recipe receipts; repeat as needed")
    parser.add_argument("--lag-period", action="append", type=int, default=None, help="Lag period for feature recipe receipts; repeat as needed")
    parser.add_argument(
        "--rolling-window-period",
        action="append",
        type=int,
        default=None,
        help="Rolling-window period for feature recipe receipts; repeat as needed",
    )
    parser.add_argument("--feature-recipe", action="append", default=None, help="Named feature recipe to record in run manifests")
    parser.add_argument("--feature-selection", action=argparse.BooleanOptionalAction, default=None, help="Record whether automated feature selection is intended")
    parser.add_argument("--pca", action=argparse.BooleanOptionalAction, default=None, help="Record whether PCA is intended for feature recipes")
    parser.add_argument("--weekly-to-daily", action=argparse.BooleanOptionalAction, default=None, help="Record weekly-to-daily allocation intent")
    parser.add_argument("--clean-missing-values", action=argparse.BooleanOptionalAction, default=None, help="Record missing-value cleaning policy intent")
    parser.add_argument("--clean-outliers", action=argparse.BooleanOptionalAction, default=None, help="Record outlier cleaning policy intent")
    parser.add_argument("--negative-forecast", action=argparse.BooleanOptionalAction, default=None, help="Allow negative forecasts when the metric can go below zero")
    parser.add_argument("--combo-cleanup-date", default=None, help="Optional combo cleanup cutoff date to record in cleaning policy receipts")
    parser.add_argument("--parallel-processing", choices=["none", "local_machine", "spark"], default=None, help="Execution intent metadata only; no cloud orchestration is started")
    parser.add_argument("--inner-parallel", action=argparse.BooleanOptionalAction, default=None, help="Record FINN-style inner parallelism intent")
    parser.add_argument("--num-cores", type=int, default=None, help="Requested local worker count metadata")
    parser.add_argument("--finn", action=argparse.BooleanOptionalAction, default=False, help="Run FINN/finnts as a spec-driven advisory challenger after the canonical forecast")
    parser.add_argument("--finn-models", nargs="+", default=None, help="FINN models_to_run, e.g. --finn-models ets snaive arima")
    parser.add_argument("--finn-back-test-scenarios", type=int, default=None, help="FINN back_test_scenarios override")
    parser.add_argument("--finn-back-test-spacing", type=int, default=None, help="FINN back_test_spacing override")
    parser.add_argument("--finn-rscript", default="Rscript", help="Rscript executable for the FINN challenger; auto-discovers Windows installs")
    parser.add_argument("--finn-on-error", choices=["skip", "fail"], default="skip", help="skip records a soft-fail status; fail stops the pipeline")
    parser.add_argument("--finn-timeout", type=int, default=3600, help="FINN challenger timeout in seconds")


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context-file", default=None, help="Accuracy-first ForecastContext JSON or YAML produced during intake/discovery")
    parser.add_argument("--research-budget", choices=["time-boxed", "balanced", "deep", "custom"], default=None)
    parser.add_argument("--research-max-iterations", type=int, default=None)
    parser.add_argument("--research-max-variants-per-iteration", type=int, default=None)
    parser.add_argument("--research-max-wall-clock-minutes", type=int, default=None)
    parser.add_argument("--research-max-source-queries", type=int, default=None)
    parser.add_argument("--research-max-compute-units", type=float, default=None)
    parser.add_argument(
        "--source-discovery",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record bounded connected-source discovery or an explicit opt-out in context_receipt.json",
    )


def _add_ledger_registration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ledger", default=None, help="Optional ledger folder; use runs\\forecast_ledger for the standard Power BI-friendly tracker")
    parser.add_argument("--forecast-key", default=None, help="Business key used when registering the run in a forecast ledger")
    parser.add_argument("--version-label", default="", help="Version label used when registering the run, e.g. March refresh")
    parser.add_argument("--ledger-created-by", default="", help="Optional owner/user recorded on the ledger version")
    parser.add_argument("--ledger-notes", default="", help="Optional notes recorded on the ledger version")
    parser.add_argument("--lock-official", action="store_true", help="After registering, also create an official lock/submitted forecast label")
    parser.add_argument("--lock-label", default="", help="Official lock label, e.g. March lock")
    parser.add_argument("--lock-audience", default="leadership")
    parser.add_argument("--planning-cycle", default="")
    parser.add_argument("--communication-date", default="")
    parser.add_argument("--submitted-to", default="")
    parser.add_argument("--lock-reason", default="")
    parser.add_argument("--locked-by", default="")


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _accuracy_gate_summary_line(run_dir: str | Path) -> str:
    gate_path = Path(run_dir) / "appendix" / "accuracy_gate.json"
    if not gate_path.exists():
        return ""
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    status = str(payload.get("status") or "unknown")
    failed = sorted(
        {
            str(gate_name)
            for series in payload.get("series", [])
            if isinstance(series, dict)
            for gate_name in series.get("failed_gates", [])
        }
    )
    suffix = f"; failed gates: {', '.join(failed)}" if failed else ""
    return f"Accuracy claim status: {status}{suffix}"


def _run_ledger(args: argparse.Namespace) -> dict[str, Any]:
    if args.ledger_command == "init":
        return init_ledger(args.ledger).to_dict()
    if args.ledger_command == "register":
        result = register_run(
            args.ledger,
            args.run,
            forecast_key=args.forecast_key,
            version_label=args.version_label,
            created_by=args.created_by,
            notes=args.notes,
            source_metadata_path=args.source_metadata,
            source_kind=args.source_kind,
        ).to_dict()
        _refresh_report_after_ledger_update(result)
        return result
    if args.ledger_command == "lock":
        result = lock_version(
            args.ledger,
            version_id=args.version_id,
            lock_label=args.lock_label,
            audience=args.audience,
            planning_cycle=args.planning_cycle,
            communication_date=args.communication_date,
            submitted_to=args.submitted_to,
            reason=args.reason,
            locked_by=args.locked_by,
            notes=args.notes,
        ).to_dict()
        _refresh_report_after_ledger_update(result)
        return result
    if args.ledger_command == "actuals":
        return ingest_actuals(
            args.ledger,
            args.input,
            forecast_key=args.forecast_key,
            id_col=args.id_col,
            time_col=args.time_col,
            target_col=args.target_col,
            sheet=_coerce_sheet(args.sheet),
            source_kind=args.source_kind,
            source_id=args.source_id,
            revision_label=args.revision_label,
            known_as_of=args.known_as_of,
        ).to_dict()
    if args.ledger_command == "adjustments":
        return ingest_adjustments(
            args.ledger,
            args.input,
            forecast_key=args.forecast_key,
            sheet=_coerce_sheet(args.sheet),
        ).to_dict()
    if args.ledger_command == "compare":
        return compare_versions(
            args.ledger,
            forecast_key=args.forecast_key,
            against_lock=args.against_lock,
            against_version_id=args.against_version_id,
            latest_version_id=args.latest_version_id,
            watch_pct=args.watch_pct,
            call_up_pct=args.call_up_pct,
            call_down_pct=args.call_down_pct,
        ).to_dict()
    if args.ledger_command == "export":
        return export_ledger(args.ledger, output=args.output).to_dict()
    raise ValueError(f"unknown ledger command: {args.ledger_command}")


def _maybe_run_challengers(run_dir: str | Path) -> str:
    """Run spec-declared challengers against a finished run; soft-fail lanes never break the pipeline."""

    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    challengers = forecast_spec_from_dict(manifest.get("spec", {})).challengers
    if not any(challenger.enabled for challenger in challengers):
        return ""
    payload = run_challengers(run_dir, challengers=challengers)
    parts = []
    for status in payload["challengers"]:
        label = f"{status['engine']}={status['status']}"
        if status["status"] != "completed" and status.get("reason"):
            label += f" ({status['reason']})"
        parts.append(label)
    return f"Challengers: {'; '.join(parts)} (advisory only; see challenger_status.json)"


def _maybe_register_ledger(
    args: argparse.Namespace,
    output_dir: str | Path,
    *,
    source_metadata_path: str | Path | None = None,
) -> dict[str, Any] | None:
    ledger_path = getattr(args, "ledger", None)
    if not ledger_path:
        return None
    forecast_key = getattr(args, "forecast_key", None)
    if not forecast_key:
        raise ValueError("--ledger requires --forecast-key so versions are grouped by business forecast")
    result = register_run(
        ledger_path,
        output_dir,
        forecast_key=forecast_key,
        version_label=getattr(args, "version_label", ""),
        created_by=getattr(args, "ledger_created_by", ""),
        notes=getattr(args, "ledger_notes", ""),
        source_metadata_path=source_metadata_path,
    ).to_dict()
    if getattr(args, "lock_official", False):
        lock_label = getattr(args, "lock_label", "") or getattr(args, "version_label", "") or "official"
        lock = lock_version(
            ledger_path,
            version_id=result["forecast_version_id"],
            lock_label=lock_label,
            audience=getattr(args, "lock_audience", "leadership"),
            planning_cycle=getattr(args, "planning_cycle", ""),
            communication_date=getattr(args, "communication_date", ""),
            submitted_to=getattr(args, "submitted_to", ""),
            reason=getattr(args, "lock_reason", ""),
            locked_by=getattr(args, "locked_by", ""),
        ).to_dict()
        result["official_lock"] = lock
    _refresh_report_after_ledger_update(result)
    return result


def _model_policy_summary_line(run: Any) -> str | None:
    resolution = getattr(run, "model_policy_resolution", {}) or {}
    if not isinstance(resolution, dict):
        return None
    families = resolution.get("families", [])
    if not isinstance(families, list):
        return None
    parts: list[str] = []
    for row in families:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family", "unknown"))
        ran = bool(row.get("ran", False))
        if ran:
            models = row.get("contributed_models", [])
            model_count = len(models) if isinstance(models, list) else 0
            parts.append(f"{family}=ran({model_count})")
        elif row.get("requested"):
            reason = str(row.get("reason_if_not_ran") or "not_available")
            parts.append(f"{family}=skipped({_compact_policy_reason(reason)})")
    if not parts:
        return None
    return "Model families: " + "; ".join(parts)


def _compact_policy_reason(reason: str) -> str:
    return reason.split(":", 1)[0].strip() or "not_available"


def _refresh_report_after_ledger_update(result: dict[str, Any] | None) -> None:
    if not result:
        return
    run_dir = result.get("run_dir")
    if not run_dir:
        return
    run_path = Path(str(run_dir))
    if (run_path / "manifest.json").exists():
        write_report_artifacts_from_directory(run_path)


def _provided_flags(argv: list[str]) -> set[str]:
    return {token.split("=", 1)[0] for token in argv if token.startswith("--")}


def _arg_or_preset(
    args: argparse.Namespace,
    attr: str,
    flag: str,
    preset_value: Any,
    provided_flags: set[str],
) -> Any:
    if getattr(args, "preset", None) and not _flag_was_provided(provided_flags, flag):
        return preset_value
    return getattr(args, attr, preset_value)


def _flag_was_provided(provided_flags: set[str], flag: str) -> bool:
    if flag in provided_flags:
        return True
    if flag.startswith("--"):
        no_flag = "--no-" + flag.removeprefix("--")
        return no_flag in provided_flags
    return False


def _coerce_sheet(sheet: str | None) -> str | int | None:
    if sheet is None:
        return None
    return int(sheet) if sheet.isdigit() else sheet


def _coerce_sheets(sheets: list[str] | None) -> tuple[str | int, ...] | None:
    if not sheets:
        return None
    if len(sheets) == 1 and sheets[0].strip().lower() == "all":
        return None
    return tuple(_coerce_sheet(sheet) for sheet in sheets if sheet is not None)


def _byo_model_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": args.format,
        "model_name": args.model_name,
        "source_id": args.source_id,
        "id_col": args.id_col,
        "time_col": args.time_col,
        "value_col": args.target_col,
        "model_col": args.model_col,
        "forecast_origin_col": args.cutoff_col,
        "scenario_col": args.scenario_col,
        "version_col": args.version_col,
        "group_cols": tuple(args.group_cols or ()),
        "include_rollups": args.rollups,
        "total_label": args.total_label,
    }


def _spec_from_args(args: argparse.Namespace) -> ForecastSpec:
    preset_name = getattr(args, "preset", None)
    provided_flags = getattr(args, "_provided_flags", set())
    base = forecast_spec_preset(preset_name) if preset_name else ForecastSpec()
    horizon = _arg_or_preset(args, "horizon", "--horizon", base.horizon, provided_flags)
    freq = _arg_or_preset(args, "freq", "--freq", base.freq, provided_flags)
    season_length = _arg_or_preset(args, "season_length", "--season-length", base.season_length, provided_flags)
    levels = tuple(_arg_or_preset(args, "levels", "--levels", list(base.levels), provided_flags))
    fill_method = _arg_or_preset(args, "fill_method", "--fill-method", base.fill_method, provided_flags)
    model_policy = _arg_or_preset(args, "model_policy", "--model-policy", base.model_policy, provided_flags)
    hierarchy_reconciliation = _arg_or_preset(
        args,
        "hierarchy_reconciliation",
        "--hierarchy-reconciliation",
        base.hierarchy_reconciliation,
        provided_flags,
    )
    require_backtest = _arg_or_preset(args, "require_backtest", "--require-backtest", base.require_backtest, provided_flags)
    strict_cv_horizon = _arg_or_preset(args, "strict_cv_horizon", "--strict-cv-horizon", base.strict_cv_horizon, provided_flags)
    weighted_ensemble = _arg_or_preset(args, "weighted_ensemble", "--weighted-ensemble", base.weighted_ensemble, provided_flags)
    verbose = _arg_or_preset(args, "verbose", "--verbose", base.verbose, provided_flags)
    train_known_future_regressors = _arg_or_preset(
        args,
        "train_known_future_regressors",
        "--train-known-future-regressors",
        base.train_known_future_regressors,
        provided_flags,
    )
    mlforecast_feature_policy = _arg_or_preset(
        args,
        "mlforecast_feature_policy",
        "--mlforecast-feature-policy",
        base.mlforecast_feature_policy,
        provided_flags,
    )
    feature_recipe = FeatureRecipeSpec(
        fiscal_year_start=_scalar_arg_or_preset(args, "fiscal_year_start", "--fiscal-year-start", base.feature_recipe.fiscal_year_start, provided_flags),
        fourier_periods=_list_arg_or_preset(args, "fourier_period", "--fourier-period", base.feature_recipe.fourier_periods, provided_flags),
        lag_periods=_list_arg_or_preset(args, "lag_period", "--lag-period", base.feature_recipe.lag_periods, provided_flags),
        rolling_window_periods=_list_arg_or_preset(
            args,
            "rolling_window_period",
            "--rolling-window-period",
            base.feature_recipe.rolling_window_periods,
            provided_flags,
        ),
        recipes_to_run=_list_arg_or_preset(args, "feature_recipe", "--feature-recipe", base.feature_recipe.recipes_to_run, provided_flags),
        pca=_scalar_arg_or_preset(args, "pca", "--pca", base.feature_recipe.pca, provided_flags),
        feature_selection=_scalar_arg_or_preset(args, "feature_selection", "--feature-selection", base.feature_recipe.feature_selection, provided_flags),
        weekly_to_daily=_scalar_arg_or_preset(args, "weekly_to_daily", "--weekly-to-daily", base.feature_recipe.weekly_to_daily, provided_flags),
    )
    cleaning = CleaningSpec(
        clean_missing_values=_scalar_arg_or_preset(
            args,
            "clean_missing_values",
            "--clean-missing-values",
            base.cleaning.clean_missing_values,
            provided_flags,
        ),
        clean_outliers=_scalar_arg_or_preset(args, "clean_outliers", "--clean-outliers", base.cleaning.clean_outliers, provided_flags),
        negative_forecast=_scalar_arg_or_preset(args, "negative_forecast", "--negative-forecast", base.cleaning.negative_forecast, provided_flags),
        combo_cleanup_date=_scalar_arg_or_preset(args, "combo_cleanup_date", "--combo-cleanup-date", base.cleaning.combo_cleanup_date, provided_flags),
    )
    ensemble = EnsembleSpec(
        policies=_list_arg_or_preset(args, "ensemble_policy", "--ensemble-policy", base.ensemble.policies, provided_flags),
        max_models=_scalar_arg_or_preset(args, "ensemble_max_models", "--ensemble-max-models", base.ensemble.max_models, provided_flags),
        scoring=base.ensemble.scoring,
        deployment=base.ensemble.deployment,
    )
    parallel = ParallelSpec(
        processing=_scalar_arg_or_preset(args, "parallel_processing", "--parallel-processing", base.parallel.processing, provided_flags),
        inner_parallel=_scalar_arg_or_preset(args, "inner_parallel", "--inner-parallel", base.parallel.inner_parallel, provided_flags),
        num_cores=_scalar_arg_or_preset(args, "num_cores", "--num-cores", base.parallel.num_cores, provided_flags),
    )
    challengers = base.challengers
    if getattr(args, "finn", False):
        finn_challenger = ChallengerSpec(
            engine="finn",
            enabled=True,
            on_error=getattr(args, "finn_on_error", "skip"),
            models=tuple(getattr(args, "finn_models", None) or ()),
            back_test_scenarios=getattr(args, "finn_back_test_scenarios", None),
            back_test_spacing=getattr(args, "finn_back_test_spacing", None),
            rscript=getattr(args, "finn_rscript", "Rscript"),
            timeout_seconds=getattr(args, "finn_timeout", 3600),
        )
        challengers = tuple(challenger for challenger in challengers if challenger.engine != "finn") + (finn_challenger,)
    context = _context_from_args(args, base.context)
    return ForecastSpec(
        horizon=horizon,
        freq=freq,
        season_length=season_length,
        levels=levels,
        fill_method=fill_method,
        model_policy=model_policy,
        model_allowlist=_parse_model_allowlist(args),
        id_col=getattr(args, "id_col", "unique_id"),
        time_col=getattr(args, "time_col", "ds"),
        target_col=getattr(args, "target_col", "y"),
        unit_label=getattr(args, "unit_label", None),
        events=_parse_events(getattr(args, "event", []), getattr(args, "event_file", [])),
        regressors=_parse_regressors(getattr(args, "regressor", []), getattr(args, "regressor_file", [])),
        custom_models=_parse_custom_models(args),
        transform=TransformSpec(
            target=getattr(args, "target_transform", "none"),
            normalization_factor_col=getattr(args, "normalization_factor_col", None),
            normalization_label=getattr(args, "normalization_label", ""),
        ),
        feature_recipe=feature_recipe,
        cleaning=cleaning,
        ensemble=ensemble,
        parallel=parallel,
        challengers=challengers,
        hierarchy_reconciliation=hierarchy_reconciliation,
        train_known_future_regressors=train_known_future_regressors,
        mlforecast_feature_policy=mlforecast_feature_policy,
        require_backtest=require_backtest,
        strict_cv_horizon=strict_cv_horizon,
        weighted_ensemble=weighted_ensemble,
        verbose=verbose,
        context=context,
    )


def _context_from_args(args: argparse.Namespace, base: ForecastContext | None) -> ForecastContext | None:
    context = base
    context_file = getattr(args, "context_file", None)
    if context_file:
        context = load_forecast_context(context_file)

    budget_fields = {
        "max_iterations": getattr(args, "research_max_iterations", None),
        "max_variants_per_iteration": getattr(args, "research_max_variants_per_iteration", None),
        "max_wall_clock_minutes": getattr(args, "research_max_wall_clock_minutes", None),
        "max_source_queries": getattr(args, "research_max_source_queries", None),
        "max_compute_units": getattr(args, "research_max_compute_units", None),
    }
    requested_profile = getattr(args, "research_budget", None)
    if requested_profile is not None or any(value is not None for value in budget_fields.values()):
        if requested_profile is None and context is not None:
            requested_profile = context.research_budget.profile
            budget_fields = {
                name: value if value is not None else getattr(context.research_budget, name)
                for name, value in budget_fields.items()
            }
        elif requested_profile is None:
            requested_profile = "custom"
        budget = ResearchBudget(profile=requested_profile, **budget_fields)
        context = replace(context or ForecastContext(), research_budget=budget)

    source_discovery = getattr(args, "source_discovery", None)
    if source_discovery is not None:
        context = replace(context or ForecastContext(), source_discovery_enabled=source_discovery)
    return context


def _refresh_spec_from_args(args: argparse.Namespace) -> ForecastSpec:
    manifest_path = Path(args.previous_run) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = forecast_spec_from_dict(manifest.get("spec", {}))
    provided_flags = getattr(args, "_provided_flags", set())
    model_allowlist = base.model_allowlist
    if _flag_was_provided(provided_flags, "--model") or _flag_was_provided(provided_flags, "--model-allowlist"):
        model_allowlist = _parse_model_allowlist(args)
    events = base.events
    if _flag_was_provided(provided_flags, "--event") or _flag_was_provided(provided_flags, "--event-file"):
        events = _parse_events(getattr(args, "event", []), getattr(args, "event_file", []))
    regressors = base.regressors
    if _flag_was_provided(provided_flags, "--regressor") or _flag_was_provided(provided_flags, "--regressor-file"):
        regressors = _parse_regressors(getattr(args, "regressor", []), getattr(args, "regressor_file", []))
    transform = TransformSpec(
        target=_override(base.transform.target, args, "target_transform"),
        normalization_factor_col=_override(base.transform.normalization_factor_col, args, "normalization_factor_col"),
        normalization_label=_override(base.transform.normalization_label, args, "normalization_label"),
    )
    feature_recipe = FeatureRecipeSpec(
        fiscal_year_start=_override(base.feature_recipe.fiscal_year_start, args, "fiscal_year_start"),
        fourier_periods=_override_tuple(base.feature_recipe.fourier_periods, args, "fourier_period"),
        lag_periods=_override_tuple(base.feature_recipe.lag_periods, args, "lag_period"),
        rolling_window_periods=_override_tuple(base.feature_recipe.rolling_window_periods, args, "rolling_window_period"),
        recipes_to_run=_override_tuple(base.feature_recipe.recipes_to_run, args, "feature_recipe"),
        pca=_override(base.feature_recipe.pca, args, "pca"),
        feature_selection=_override(base.feature_recipe.feature_selection, args, "feature_selection"),
        weekly_to_daily=_override(base.feature_recipe.weekly_to_daily, args, "weekly_to_daily"),
    )
    cleaning = CleaningSpec(
        clean_missing_values=_override(base.cleaning.clean_missing_values, args, "clean_missing_values"),
        clean_outliers=_override(base.cleaning.clean_outliers, args, "clean_outliers"),
        negative_forecast=_override(base.cleaning.negative_forecast, args, "negative_forecast"),
        combo_cleanup_date=_override(base.cleaning.combo_cleanup_date, args, "combo_cleanup_date"),
    )
    ensemble = EnsembleSpec(
        policies=_override_tuple(base.ensemble.policies, args, "ensemble_policy"),
        max_models=_override(base.ensemble.max_models, args, "ensemble_max_models"),
        scoring=base.ensemble.scoring,
        deployment=base.ensemble.deployment,
    )
    parallel = ParallelSpec(
        processing=_override(base.parallel.processing, args, "parallel_processing"),
        inner_parallel=_override(base.parallel.inner_parallel, args, "inner_parallel"),
        num_cores=_override(base.parallel.num_cores, args, "num_cores"),
    )
    return replace(
        base,
        horizon=_override(base.horizon, args, "horizon"),
        freq=_override(base.freq, args, "freq"),
        season_length=_override(base.season_length, args, "season_length"),
        levels=tuple(_override(list(base.levels), args, "levels")),
        fill_method=_override(base.fill_method, args, "fill_method"),
        model_policy=_override(base.model_policy, args, "model_policy"),
        model_allowlist=model_allowlist,
        id_col=_override(base.id_col, args, "id_col"),
        time_col=_override(base.time_col, args, "time_col"),
        target_col=_override(base.target_col, args, "target_col"),
        unit_label=_override(base.unit_label, args, "unit_label"),
        events=events,
        regressors=regressors,
        transform=transform,
        feature_recipe=feature_recipe,
        cleaning=cleaning,
        ensemble=ensemble,
        parallel=parallel,
        hierarchy_reconciliation=_override(base.hierarchy_reconciliation, args, "hierarchy_reconciliation"),
        train_known_future_regressors=_override(base.train_known_future_regressors, args, "train_known_future_regressors"),
        mlforecast_feature_policy=_override(base.mlforecast_feature_policy, args, "mlforecast_feature_policy"),
        require_backtest=_override(base.require_backtest, args, "require_backtest"),
        strict_cv_horizon=_override(base.strict_cv_horizon, args, "strict_cv_horizon"),
        weighted_ensemble=_override(base.weighted_ensemble, args, "weighted_ensemble"),
        verbose=_override(base.verbose, args, "verbose"),
    )


def _override(default: Any, args: argparse.Namespace, attr: str) -> Any:
    value = getattr(args, attr, None)
    return default if value is None else value


def _override_tuple(default: tuple[Any, ...], args: argparse.Namespace, attr: str) -> tuple[Any, ...]:
    value = getattr(args, attr, None)
    return default if value is None else tuple(value)


def _scalar_arg_or_preset(args: argparse.Namespace, attr: str, flag: str, preset_value: Any, provided_flags: set[str]) -> Any:
    if getattr(args, "preset", None) and not _flag_was_provided(provided_flags, flag):
        return preset_value
    value = getattr(args, attr, None)
    return preset_value if value is None else value


def _list_arg_or_preset(args: argparse.Namespace, attr: str, flag: str, preset_value: tuple[Any, ...], provided_flags: set[str]) -> tuple[Any, ...]:
    if getattr(args, "preset", None) and not _flag_was_provided(provided_flags, flag):
        return tuple(preset_value)
    value = getattr(args, attr, None)
    return tuple(preset_value) if value is None else tuple(value)


def _parse_model_allowlist(args: argparse.Namespace) -> tuple[str, ...]:
    models = list(getattr(args, "model_allowlist", []) or [])
    for group in getattr(args, "model_allowlist_grouped", []) or []:
        models.extend(group)
    return tuple(models)


def _parse_custom_models(args: argparse.Namespace) -> tuple[CustomModelSpec, ...]:
    callable_path = getattr(args, "custom_model", None)
    script_path = getattr(args, "custom_script", None)
    custom_name = getattr(args, "custom_model_name", None)
    extra_args = tuple(getattr(args, "custom_arg", []) or [])
    if not callable_path and not script_path:
        if custom_name or extra_args:
            raise ValueError("--custom-model-name and --custom-arg require --custom-model or --custom-script")
        return ()
    if callable_path and script_path:
        raise ValueError("use only one of --custom-model or --custom-script")
    name = custom_name or _default_custom_model_name(callable_path=callable_path, script_path=script_path)
    return (
        CustomModelSpec(
            name=name,
            callable_path=callable_path,
            script_path=script_path,
            timeout_seconds=int(getattr(args, "custom_timeout_seconds", 120)),
            extra_args=extra_args,
        ),
    )


def _custom_forecast_flags_requested(args: argparse.Namespace) -> bool:
    provided_flags = getattr(args, "_provided_flags", set())
    return bool(
        getattr(args, "custom_model", None)
        or getattr(args, "custom_script", None)
        or getattr(args, "custom_model_name", None)
        or getattr(args, "custom_arg", [])
        or _flag_was_provided(provided_flags, "--custom-timeout-seconds")
    )


def _default_custom_model_name(*, callable_path: str | None, script_path: str | None) -> str:
    if script_path:
        return Path(script_path).stem
    if callable_path:
        return callable_path.split(":", 1)[-1].rsplit(".", 1)[-1]
    return "custom_model"


def _setup_answers_from_args(args: argparse.Namespace) -> SetupAnswers:
    return SetupAnswers(
        name=args.name,
        data_source=args.data_source,
        input_path=args.setup_input,
        series_count=args.series_count,
        target_name=args.target_name,
        time_col=args.time_col,
        id_col=args.id_col,
        id_value=args.id_value,
        preset=args.preset,
        horizon=args.horizon,
        freq=args.freq,
        intervals=args.intervals,
        model_families=tuple(args.model_families),
        exploration_mode=args.exploration_mode,
        source_discovery=args.source_discovery,
        mcp_regressor_search=args.mcp_regressor_search,
        research_budget=ResearchBudget(
            profile=args.research_budget,
            max_iterations=args.research_max_iterations,
            max_variants_per_iteration=args.research_max_variants_per_iteration,
            max_wall_clock_minutes=args.research_max_wall_clock_minutes,
            max_source_queries=args.research_max_source_queries,
            max_compute_units=args.research_max_compute_units,
        ),
        decision=args.decision,
        audience=args.audience,
        target_semantics=args.target_semantics,
        units=args.units,
        grain=args.grain,
        refresh_cadence=args.refresh_cadence,
        outputs=tuple(args.outputs),
        hierarchy_cols=tuple(args.hierarchy_cols),
        query_file=args.query_file,
        notes=args.notes,
    )


def _parse_events(values: list[str], files: list[str] | None = None):
    return parse_driver_events(values, files or [])


def _parse_regressors(values: list[str], files: list[str] | None = None):
    return parse_known_future_regressors(values, files or [])


if __name__ == "__main__":
    raise SystemExit(main())
