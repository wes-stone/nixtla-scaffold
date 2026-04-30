from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from nixtla_scaffold.comparisons import write_forecast_comparison
from nixtla_scaffold.connectors import ingest_query_result
from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.diagnostics import write_failure_diagnostics
from nixtla_scaffold.drivers import parse_driver_events, parse_known_future_regressors
from nixtla_scaffold.external_scoring import write_external_forecast_scores
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.hierarchy import aggregate_hierarchy_frame, hierarchy_summary
from nixtla_scaffold.knowledge import format_knowledge, search_knowledge
from nixtla_scaffold.presets import PRESET_NAMES, forecast_spec_preset, preset_catalog
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.release_gates import OPTIONAL_EXTRAS, format_release_gate_console_summary, run_release_gates
from nixtla_scaffold.reports import write_report_artifacts_from_directory
from nixtla_scaffold.scenario_lab import run_scenario_lab
from nixtla_scaffold.schema import CustomModelSpec, ForecastSpec, TransformSpec
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

MODEL_POLICY_CHOICES = ["auto", "baseline", "statsforecast", "mlforecast", "all"]
MODEL_POLICY_HELP = (
    "Model family policy: auto runs StatsForecast plus MLForecast when eligible; "
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
    setup_cmd.add_argument("--series-count", choices=SERIES_COUNTS, default="unknown")
    setup_cmd.add_argument("--target-name", default="y", help="Metric/target column or name")
    setup_cmd.add_argument("--time-col", default="ds")
    setup_cmd.add_argument("--id-col", default="unique_id")
    setup_cmd.add_argument("--id-value", default=None, help="Series ID to inject for one-metric query results")
    setup_cmd.add_argument("--preset", choices=PRESET_NAMES, default="finance", help="Starter forecast preset for generated commands/config")
    setup_cmd.add_argument("--horizon", type=int, default=12)
    setup_cmd.add_argument("--freq", default=None)
    setup_cmd.add_argument("--intervals", choices=INTERVAL_MODES, default="auto")
    setup_cmd.add_argument("--model-families", nargs="+", choices=MODEL_FAMILIES, default=["auto"])
    setup_cmd.add_argument("--exploration-mode", action=argparse.BooleanOptionalAction, default=True)
    setup_cmd.add_argument("--mcp-regressor-search", action=argparse.BooleanOptionalAction, default=False)
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
    forecast_cmd.add_argument("--preset", choices=PRESET_NAMES, default=None, help="Opinionated defaults: quick, finance, strict, or hierarchy")
    forecast_cmd.add_argument("--horizon", type=int, default=12)
    forecast_cmd.add_argument("--freq", default=None)
    forecast_cmd.add_argument("--season-length", type=int, default=None)
    forecast_cmd.add_argument("--levels", nargs="+", type=int, default=[80, 95], help="Prediction interval levels, e.g. --levels 80 95")
    forecast_cmd.add_argument("--unit-label", default=None, help="Optional unit/currency label for headline values, e.g. $, USD, seats, ARR")
    forecast_cmd.add_argument("--fill-method", choices=["ffill", "zero", "interpolate", "drop"], default="ffill")
    forecast_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="auto", help=MODEL_POLICY_HELP)
    forecast_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none", help="Optional target transform for modeling; outputs are inverse-transformed for reporting")
    forecast_cmd.add_argument("--normalization-factor-col", default=None, help="Positive factor column used to normalize y before modeling, e.g. price_factor, fx_rate, inflation_index")
    forecast_cmd.add_argument("--normalization-label", default="", help="Readable label for the normalization assumption")
    forecast_cmd.add_argument(
        "--hierarchy-reconciliation",
        choices=["none", "bottom_up", "mint_ols", "mint_wls_struct"],
        default="none",
        help="Optional reconciliation for hierarchy nodes; keeps parent/child forecasts coherent for planning",
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
    ingest_cmd.add_argument("--model-policy", choices=MODEL_POLICY_CHOICES, default="auto", help=MODEL_POLICY_HELP)
    ingest_cmd.add_argument("--target-transform", choices=["none", "log", "log1p"], default="none")
    ingest_cmd.add_argument("--normalization-factor-col", default=None)
    ingest_cmd.add_argument("--normalization-label", default="")
    ingest_cmd.add_argument("--hierarchy-reconciliation", choices=["none", "bottom_up", "mint_ols", "mint_wls_struct"], default="none")
    ingest_cmd.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=False, help="Fail if rolling backtest metrics cannot be produced")
    ingest_cmd.add_argument("--strict-cv-horizon", action=argparse.BooleanOptionalAction, default=False, help="Require CV folds to match --horizon when forecasting after ingest")
    ingest_cmd.add_argument("--weighted-ensemble", action=argparse.BooleanOptionalAction, default=True)
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
    compare_cmd.add_argument("--scaffold-model", default=None, help="Optional scaffold model from forecast_long.csv; defaults to selected forecast.csv")
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

    lab_cmd = sub.add_parser("scenario-lab", help="Run synthetic forecast scenarios and score accuracy/ease/validity")
    lab_cmd.add_argument("--count", type=int, default=100)
    lab_cmd.add_argument("--output", default="runs/scenario_lab")
    lab_cmd.add_argument("--model-policy", choices=["auto", "baseline", "statsforecast"], default="auto")
    lab_cmd.add_argument("--seed", type=int, default=42)

    qa_cmd = sub.add_parser("workbench-qa", help="Run golden forecast workbench QA scenarios")
    qa_cmd.add_argument("--output", default="runs/workbench_qa", help="Folder for QA summary and generated runs")
    qa_cmd.add_argument("--scenarios", nargs="+", choices=WORKBENCH_QA_SCENARIOS, default=list(GOLDEN_SCENARIOS))
    qa_cmd.add_argument("--model-policy", choices=["auto", "baseline", "statsforecast", "mlforecast", "all"], default="baseline")
    qa_cmd.add_argument("--app-test", action=argparse.BooleanOptionalAction, default=True, help="Run Streamlit AppTest for generated dashboards")
    qa_cmd.add_argument("--app-test-timeout", type=int, default=90, help="Streamlit AppTest timeout in seconds for each generated dashboard")

    release_cmd = sub.add_parser("release-gates", help="Run local release-readiness gates")
    release_cmd.add_argument("--output", default="runs/release_gates", help="Folder for release-gate outputs")
    release_cmd.add_argument(
        "--extended",
        action="store_true",
        help="Run a stricter local release profile: at least 20 auto-policy scenarios, all-family workbench QA, and required ml/hierarchy extras",
    )
    release_cmd.add_argument("--build", action=argparse.BooleanOptionalAction, default=True, help="Build wheel/sdist and inspect package contents")
    release_cmd.add_argument("--install-smoke", action=argparse.BooleanOptionalAction, default=True, help="Install the built wheel in an isolated venv and smoke public APIs")
    release_cmd.add_argument("--scenario-count", type=int, default=8, help="Deterministic scenario-lab count for numeric gates")
    release_cmd.add_argument("--scenario-model-policy", choices=["auto", "baseline", "statsforecast"], default="baseline")
    release_cmd.add_argument("--workbench-qa", action=argparse.BooleanOptionalAction, default=True, help="Run generated workbench QA scenarios")
    release_cmd.add_argument("--workbench-model-policy", choices=["auto", "baseline", "statsforecast", "mlforecast", "all"], default="baseline")
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
        print(f"Forecast written to {output_dir}")
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
        )
        print(json.dumps(result.manifest, indent=2, default=str))
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

    if args.command == "guide":
        if args.query == "presets":
            print(json.dumps(preset_catalog(), indent=2, default=str))
            return 0
        print(format_knowledge(search_knowledge(args.query)))
        return 0

    raise ValueError(f"unknown command: {args.command}")


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="CSV, XLSX, or XLSM file")
    parser.add_argument("--sheet", default=None, help="Excel sheet name/index")
    parser.add_argument("--id-col", default="unique_id", help="Series identifier column, e.g. Product or Account")
    parser.add_argument("--time-col", default="ds", help="Date/time column, e.g. Month or Week")
    parser.add_argument("--target-col", default="y", help="Numeric value to forecast, e.g. Revenue or ARR")


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


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
    return ForecastSpec(
        horizon=horizon,
        freq=freq,
        season_length=season_length,
        levels=levels,
        fill_method=fill_method,
        model_policy=model_policy,
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
        hierarchy_reconciliation=hierarchy_reconciliation,
        require_backtest=require_backtest,
        strict_cv_horizon=strict_cv_horizon,
        weighted_ensemble=weighted_ensemble,
        verbose=verbose,
    )


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
        mcp_regressor_search=args.mcp_regressor_search,
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

