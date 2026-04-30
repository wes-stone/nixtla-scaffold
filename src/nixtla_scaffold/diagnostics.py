from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def build_run_diagnostics(run: Any) -> dict[str, Any]:
    """Build an LLM-readable runbook for debugging and handoff."""

    from nixtla_scaffold.headline import build_executive_headline
    from nixtla_scaffold.outputs import build_hierarchy_backtest_comparison, build_residual_test_summary, build_trust_summary

    manifest = run.manifest()
    trust_summary = build_trust_summary(run)
    residual_tests = build_residual_test_summary(run)
    hierarchy_backtest = build_hierarchy_backtest_comparison(run)
    executive_headline = build_executive_headline(run).to_dict()
    driver_audit_distribution = (
        run.driver_availability_audit["audit_status"].value_counts().sort_index().to_dict()
        if not run.driver_availability_audit.empty and "audit_status" in run.driver_availability_audit.columns
        else {}
    )
    trust_distribution = (
        trust_summary["trust_level"].value_counts().sort_index().to_dict()
        if not trust_summary.empty and "trust_level" in trust_summary.columns
        else {}
    )
    horizon_trust_distribution = (
        trust_summary["horizon_trust_state"].value_counts().sort_index().to_dict()
        if not trust_summary.empty and "horizon_trust_state" in trust_summary.columns
        else {}
    )
    residual_test_distribution = (
        residual_tests["overall_status"].value_counts().sort_index().to_dict()
        if not residual_tests.empty and "overall_status" in residual_tests.columns
        else {}
    )
    model_candidates = [
        col
        for col in run.all_models.columns
        if col not in {"unique_id", "ds"} and "-lo-" not in col and "-hi-" not in col
    ]
    return {
        "status": "success",
        "generated_at_utc": _utc_now(),
        "executive_headline": executive_headline,
        "llm_triage_summary": {
            "engine": run.engine,
            "series_count": run.profile.series_count,
            "rows": run.profile.rows,
            "frequency": run.profile.freq,
            "season_length": run.profile.season_length,
            "horizon": run.spec.horizon,
            "target_transform": run.spec.transform.target,
            "normalization_factor_col": run.spec.transform.normalization_factor_col,
            "transformation_audit_rows": int(len(run.transformation_audit)),
            "scenario_event_count": len(run.spec.events),
            "known_future_regressor_count": len(run.spec.regressors),
            "driver_availability_audit_rows": int(len(run.driver_availability_audit)),
            "driver_audit_distribution": driver_audit_distribution,
            "custom_model_count": len(run.spec.custom_models),
            "custom_model_contract_rows": int(len(run.custom_model_contracts)),
            "custom_model_invocation_rows": int(len(run.custom_model_invocations)),
            "hierarchy_reconciliation": run.spec.hierarchy_reconciliation,
            "hierarchy_reconciliation_rows": int(len(run.hierarchy_reconciliation)),
            "weighted_ensemble_enabled": run.spec.weighted_ensemble,
            "weighted_ensemble_weight_rows": int(len(run.model_weights)),
            "model_explainability_rows": int(len(run.model_explainability)),
            "residual_test_rows": int(len(residual_tests)),
            "residual_test_distribution": residual_test_distribution,
            "hierarchy_backtest_comparison_rows": int(len(hierarchy_backtest)),
            "warning_count": len(run.warnings),
            "trust_distribution": trust_distribution,
            "horizon_trust_distribution": horizon_trust_distribution,
            "average_trust_score": (
                float(trust_summary["trust_score_0_100"].mean())
                if not trust_summary.empty and "trust_score_0_100" in trust_summary.columns
                else None
            ),
        },
        "spec": run.spec.to_dict(),
        "profile": run.profile.to_dict(),
        "warnings": run.warnings,
        "model_policy_resolution": run.model_policy_resolution,
        "model_candidates": model_candidates,
        "model_selection": _records(run.model_selection),
        "model_weights": _records(run.model_weights),
        "model_explainability": _records(run.model_explainability),
        "custom_model_contracts": _records(run.custom_model_contracts),
        "custom_model_invocations": _records(run.custom_model_invocations),
        "residual_tests": _records(residual_tests),
        "driver_availability_audit": _records(run.driver_availability_audit),
        "trust_summary": _records(trust_summary),
        "reproducibility": manifest["reproducibility"],
        "best_practice_receipts": run.best_practice_receipts(),
        "outputs": manifest["outputs"],
        "next_diagnostic_steps": [
            "Quote diagnostics.json executive_headline.paragraph verbatim when summarizing this run; do not rewrite it into a stronger claim.",
            "Open model_card.md first for the human-readable forecast narrative.",
            "Open llm_context.json when asking an LLM to walk through the run; it bundles headline, trust, horizon, interval, residual, seasonality, hierarchy, driver, and artifact-index context.",
            "Open diagnostics.json for machine-readable run context and warnings.",
            "Open trust_summary.csv first for per-series High/Medium/Low readiness, caveats, and next actions.",
            "Filter forecast.csv to planning_eligible=True when users ask for stakeholder-ready horizon rows.",
            "Check horizon_trust_state, validated_through_horizon, and full_horizon_claim_allowed before calling a forecast planning-ready for the full requested horizon.",
            "Open audit/target_transform_audit.csv when target transforms or finance normalization are enabled; confirm whether outputs are raw units or normalized units.",
            "Open scenario_assumptions.csv and scenario_forecast.csv when event overlays are present; keep baseline yhat separate from yhat_scenario.",
            "Open known_future_regressors.csv and driver_availability_audit.csv when regressors are declared; model_candidate rows must pass future availability and leakage checks before future exogenous modeling work.",
            "Check profile.json for inferred frequency, season length, short histories, and repaired gaps.",
            "Check series_summary.csv and model_audit.csv before trusting a selected model.",
            "Check model_win_rates.csv to see which models beat SeasonalNaive/Naive across series.",
            "Open backtest_long.csv and model_window_metrics.csv to inspect each rolling-origin cutoff with actuals, errors, and model forecasts.",
            "Open residual_diagnostics.csv for horizon-step error diagnostics and interval_diagnostics.csv for empirical prediction-interval coverage when available.",
            "Open residual_tests.csv for heuristic residual bias, one-step autocorrelation, outlier, and early/late structural-break checks over rolling-origin residuals. Treat failures as diagnostic signals, not formal model adequacy certification; small samples are directional only.",
            "Open model_explainability.csv when MLForecast models run to inspect lag/date feature importance or coefficient magnitudes.",
            "Open audit/backtest_windows.csv and audit/backtest_predictions.csv for the raw wide cross-validation audit trail.",
            "Open audit/seasonality_diagnostics.csv and audit/seasonality_decomposition.csv to check cycle counts, credibility labels, trend/seasonal/remainder evidence, and warnings.",
            "If hierarchy metadata exists, open hierarchy_contribution.csv to see which child nodes drive each parent and how each child is allocated to parent-child incoherence.",
            "If hierarchy reconciliation is enabled, open hierarchy_reconciliation.csv plus audit/hierarchy_backtest_comparison.csv, audit/hierarchy_coherence_pre.csv, and audit/hierarchy_coherence_post.csv to compare planning coherence and node-level accuracy before and after reconciliation.",
            "Open interpretation.md for a human-readable backtesting and seasonality summary.",
            "Check audit/model_weights.csv when WeightedEnsemble is selected or present.",
            "Open report.html for the portable fixed-axis filmstrip report, or decode report_base64.txt for text-only handoff.",
            "Run streamlit run streamlit_app.py inside the output folder for the interactive CV window player.",
            "Check forecast_long.csv for all future model predictions and forecast.csv for selected yhat, intervals, interval_status, scenario columns, and event_names.",
            "If hierarchy_depth exists, check hierarchy_coherence.csv before comparing parent and child forecasts.",
        ],
    }


def build_llm_context(run: Any) -> dict[str, Any]:
    """Build a consolidated artifact for asking an LLM to review a forecast run."""

    from nixtla_scaffold.drivers import (
        build_driver_experiment_summary_frame,
        build_known_future_regressors_frame,
        build_scenario_assumptions_frame,
        build_scenario_forecast_frame,
    )
    from nixtla_scaffold.headline import build_executive_headline
    from nixtla_scaffold.interpretation import seasonality_diagnostics_frame, seasonality_summary_frame
    from nixtla_scaffold.outputs import (
        build_forecast_long,
        build_hierarchy_backtest_comparison,
        build_hierarchy_contribution_frame,
        build_interval_diagnostics,
        build_model_audit,
        build_model_win_rates,
        build_model_window_metrics,
        build_residual_diagnostics,
        build_residual_test_summary,
        build_selected_forecast,
        build_series_summary,
        build_trust_summary,
    )

    manifest = run.manifest()
    selected_forecast = build_selected_forecast(run)
    forecast_long = build_forecast_long(run)
    trust_summary = build_trust_summary(run)
    model_selection = run.model_selection.copy()
    series_summary = build_series_summary(run)
    residual_tests = build_residual_test_summary(run)
    interval_diagnostics = build_interval_diagnostics(run)
    seasonality_diagnostics = seasonality_diagnostics_frame(run)
    driver_experiment_summary = build_driver_experiment_summary_frame(run)
    hierarchy_contribution = build_hierarchy_contribution_frame(run)
    hierarchy_backtest = build_hierarchy_backtest_comparison(run)
    series_ids = _series_ids(selected_forecast, trust_summary, model_selection)

    return {
        "schema_version": "nixtla_scaffold.llm_context.v1",
        "generated_at_utc": _utc_now(),
        "purpose": "Paste or attach this JSON when asking an LLM to explain, audit, or walk through a forecast run.",
        "prompt_starter": (
            "You are reviewing a finance forecasting run. Quote executive_headline.paragraph verbatim, "
            "start with trust_summary and horizon validation, separate statistical baseline from scenarios/plans, "
            "and do not call any forecast planning-ready unless horizon, trust, interval, residual, hierarchy, "
            "and data-quality caveats support that claim."
        ),
        "guardrails": [
            "Forecasts are statistical baselines, not plans or guarantees.",
            "planning_eligible=True means horizon-validation only; still review trust, intervals, residuals, hierarchy, and data-quality caveats.",
            "Prediction intervals are planning-ready uncertainty bands only when interval_status is calibrated; future-only or adjusted bands need extra review.",
            "Residual tests are heuristic screening signals, not formal model adequacy certification.",
            "Hierarchy reconciliation enforces coherence and may improve or worsen node-level accuracy.",
            "Known-future regressors are audited for leakage/future coverage in this release; arbitrary external regressors are not auto-trained.",
        ],
        "executive_headline": build_executive_headline(run).to_dict(),
        "run_summary": build_run_diagnostics(run)["llm_triage_summary"],
        "spec": run.spec.to_dict(),
        "profile": run.profile.to_dict(),
        "warnings": run.warnings,
        "series_reviews": [
            _series_llm_review(
                uid,
                selected_forecast=selected_forecast,
                forecast_long=forecast_long,
                trust_summary=trust_summary,
                model_selection=model_selection,
                series_summary=series_summary,
                residual_tests=residual_tests,
                interval_diagnostics=interval_diagnostics,
                seasonality_diagnostics=seasonality_diagnostics,
                model_weights=run.model_weights,
                model_explainability=run.model_explainability,
            )
            for uid in series_ids
        ],
        "portfolio_tables": {
            "trust_summary": _records(trust_summary),
            "model_selection": _records(model_selection),
            "series_summary": _records(series_summary),
            "model_audit_top_rows": _records(build_model_audit(run).head(200)),
            "model_win_rates": _records(build_model_win_rates(run)),
            "model_window_metrics_top_rows": _records(build_model_window_metrics(run).head(200)),
            "residual_diagnostics_top_rows": _records(build_residual_diagnostics(run).head(200)),
            "residual_tests": _records(residual_tests),
            "interval_diagnostics": _records(interval_diagnostics),
            "seasonality_summary": _records(seasonality_summary_frame(run)),
            "seasonality_diagnostics": _records(seasonality_diagnostics),
            "driver_experiment_summary": _records(driver_experiment_summary),
            "custom_model_contracts": _records(run.custom_model_contracts),
            "custom_model_invocations": _records(run.custom_model_invocations.head(200)),
            "hierarchy_contribution_top_rows": _records(hierarchy_contribution.head(200)),
            "hierarchy_backtest_comparison_top_rows": _records(hierarchy_backtest.head(200)),
        },
        "assumptions_and_drivers": {
            "scenario_assumptions": _records(build_scenario_assumptions_frame(run.spec.events)),
            "scenario_forecast_top_rows": _records(build_scenario_forecast_frame(selected_forecast).head(200)),
            "known_future_regressors": _records(build_known_future_regressors_frame(run.spec.regressors)),
            "driver_availability_audit": _records(run.driver_availability_audit),
            "driver_experiment_summary": _records(driver_experiment_summary),
        },
        "artifact_index": manifest["outputs"],
        "recommended_questions": [
            "Which series are High/Medium/Low trust, and what should I do first?",
            "Which forecast rows are horizon-validated versus directional?",
            "Do the selected models beat simple Naive or SeasonalNaive benchmarks?",
            "Are intervals calibrated enough for stakeholder ranges?",
            "Do residual tests show bias, autocorrelation, outliers, or structural breaks?",
            "Is seasonality credible, or is the series too short?",
            "If hierarchy reconciliation is enabled, what coherence/accuracy tradeoff did it create?",
            "What events, pricing changes, launches, contracts, or known-future drivers should be added before planning use?",
        ],
    }


def format_run_diagnostics_markdown(run: Any) -> str:
    diagnostics = build_run_diagnostics(run)
    summary = diagnostics["llm_triage_summary"]
    executive = diagnostics.get("executive_headline", {})
    lines = [
        "# Forecast diagnostics",
        "",
        "## Executive headline",
        "",
        executive.get("paragraph") or "Executive headline unavailable.",
        "",
        "## LLM triage summary",
        "",
        f"- Status: `{diagnostics['status']}`",
        f"- Engine: `{summary['engine']}`",
        f"- Rows / series: {summary['rows']} / {summary['series_count']}",
        f"- Frequency / season length: `{summary['frequency']}` / {summary['season_length']}",
        f"- Horizon: {summary['horizon']}",
        f"- Target transform: `{summary['target_transform']}`",
        f"- Normalization factor column: `{summary['normalization_factor_col'] or 'none'}`",
        f"- Transformation audit rows: {summary['transformation_audit_rows']}",
        f"- Scenario events: {summary['scenario_event_count']}",
        f"- Known-future regressors: {summary['known_future_regressor_count']}",
        f"- Driver availability audit rows: {summary['driver_availability_audit_rows']}",
        f"- Driver audit distribution: {summary['driver_audit_distribution'] or 'N/A'}",
        f"- Custom models: {summary['custom_model_count']}",
        f"- Custom model contract rows: {summary['custom_model_contract_rows']}",
        f"- Custom model invocation rows: {summary['custom_model_invocation_rows']}",
        f"- Hierarchy reconciliation: `{summary['hierarchy_reconciliation']}`",
        f"- Hierarchy reconciliation rows: {summary['hierarchy_reconciliation_rows']}",
        f"- Weighted ensemble enabled: {summary['weighted_ensemble_enabled']}",
        f"- Model weight rows: {summary['weighted_ensemble_weight_rows']}",
        f"- Model explainability rows: {summary['model_explainability_rows']}",
        f"- Residual test rows: {summary['residual_test_rows']}",
        f"- Residual test distribution: {summary['residual_test_distribution'] or 'N/A'}",
        f"- Hierarchy backtest comparison rows: {summary['hierarchy_backtest_comparison_rows']}",
        f"- Warning count: {summary['warning_count']}",
        f"- Trust distribution: {summary['trust_distribution'] or 'N/A'}",
        f"- Average trust score: {summary['average_trust_score'] if summary['average_trust_score'] is not None else 'N/A'}",
        "",
        "## Model policy resolution",
        "",
    ]
    resolution = diagnostics.get("model_policy_resolution", {})
    families = resolution.get("families", []) if isinstance(resolution, dict) else []
    if families:
        for family in families:
            reason = family.get("reason_if_not_ran") or "ran"
            contributed = ", ".join(family.get("contributed_models") or []) or "none"
            lines.append(
                f"- `{family.get('family')}` requested={family.get('requested')} "
                f"eligible={family.get('eligible')} ran={family.get('ran')} reason={reason}; "
                f"models: {contributed}"
            )
    else:
        lines.append("- No model policy resolution was recorded.")
    lines.extend([
        "",
        "## Warnings",
        "",
    ])
    if run.warnings:
        lines.extend(f"- {warning}" for warning in run.warnings)
    else:
        lines.append("- None.")
    lines.extend(["", "## Selected models", ""])
    if run.model_selection.empty:
        lines.append("- No model selection rows were produced.")
    else:
        for row in run.model_selection.to_dict("records"):
            if "rmse" in row and pd.notna(row["rmse"]):
                metric = f", RMSE={row['rmse']:.4g}"
            elif "wape" in row and pd.notna(row["wape"]):
                metric = f", WAPE={row['wape']:.2%}"
            else:
                metric = ""
            cv_contract = ""
            if pd.notna(row.get("selection_horizon")) and pd.notna(row.get("requested_horizon")):
                cv_contract = f", CV h={int(row['selection_horizon'])}/{int(row['requested_horizon'])}"
            lines.append(f"- `{row['unique_id']}` -> `{row['selected_model']}`{metric}{cv_contract}: {row['selection_reason']}")
    lines.extend(["", "## Trust and next actions", ""])
    trust_rows = diagnostics.get("trust_summary", [])
    if not trust_rows:
        lines.append("- No trust summary rows were produced.")
    else:
        for row in trust_rows:
            score = row.get("trust_score_0_100")
            actions = row.get("next_actions") or "No next actions recorded."
            caveats = row.get("caveats") or "No major caveats."
            lines.append(
                f"- `{row.get('unique_id')}`: {row.get('trust_level')} trust"
                f" ({score}/100), model `{row.get('selected_model')}`. Caveats: {caveats} Next: {actions}"
            )
    lines.extend(["", "## Weighted ensemble weights", ""])
    if run.model_weights.empty:
        lines.append("- No model weights were produced.")
    else:
        for row in run.model_weights.to_dict("records"):
            lines.append(
                f"- `{row['unique_id']}` `{row['model']}` weight={row['weight']:.2%} "
                f"from {row['score_metric']}={row['score_value']:.4g}"
            )
    lines.extend(["", "## Next diagnostic steps", ""])
    lines.extend(f"- {step}" for step in diagnostics["next_diagnostic_steps"])
    return "\n".join(lines) + "\n"


def build_failure_diagnostics(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    message = str(exc)
    command = getattr(args, "command", None)
    return {
        "status": "failure",
        "generated_at_utc": _utc_now(),
        "command": command,
        "error_type": type(exc).__name__,
        "error": message,
        "args": _safe_args(args),
        "likely_causes": _failure_hints(message, command=command),
        "next_diagnostic_steps": _failure_next_steps(command),
    }


def format_failure_diagnostics_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Nixtla scaffold failure diagnostics",
        "",
        f"- Status: `{payload['status']}`",
        f"- Command: `{payload.get('command')}`",
        f"- Error type: `{payload['error_type']}`",
        f"- Error: {payload['error']}",
        "",
        "## Likely causes",
        "",
    ]
    lines.extend(f"- {hint}" for hint in payload["likely_causes"])
    lines.extend(["", "## Next diagnostic steps", ""])
    lines.extend(f"- {step}" for step in payload["next_diagnostic_steps"])
    return "\n".join(lines) + "\n"


def write_failure_diagnostics(args: argparse.Namespace, exc: Exception) -> tuple[Path, Path] | None:
    output_dir = _failure_output_dir(args)
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_failure_diagnostics(args, exc)
    json_path = output_dir / "failure_diagnostics.json"
    md_path = output_dir / "failure_diagnostics.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(format_failure_diagnostics_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _failure_output_dir(args: argparse.Namespace) -> Path | None:
    if getattr(args, "command", None) == "ingest" and getattr(args, "forecast_output", None):
        return Path(args.forecast_output)
    output = getattr(args, "output", None)
    if output is None:
        return None
    path = Path(output)
    if path.exists() and path.is_file():
        return path.parent
    return path.parent if path.suffix else path


def _failure_next_steps(command: str | None) -> list[str]:
    if command == "score-external":
        return [
            "Confirm external snapshots and actuals use the same unique_id values and target dates; scoring joins on unique_id + ds.",
            "Check that actuals contain realized values for the external target dates, not only history before the forecast cutoff.",
            "If the actuals file uses custom column names, pass --actual-id-col, --actual-time-col, and --actual-target-col.",
            "Ensure --season-length and --horizon are positive integers when supplied.",
            "Attach this failure_diagnostics.json plus the external and actuals schemas when asking another LLM for help.",
        ]
    return [
        "Re-run profile on the same input to inspect inferred frequency, missing dates, duplicates, and nulls.",
        "If frequency inference failed, pass --freq explicitly (for example ME, MS, W-SUN, D, B, QE, QS).",
        "If strict backtesting failed, add history, reduce --horizon, or remove --require-backtest.",
        "If column validation failed, pass --id-col, --time-col, and --target-col to match the source file.",
        "Attach this failure_diagnostics.json plus the command and input schema when asking another LLM for help.",
    ]


def _failure_hints(message: str, *, command: str | None = None) -> list[str]:
    lowered = message.lower()
    hints: list[str] = []
    if command == "score-external" or "external forecast scoring" in lowered:
        if "matching actuals" in lowered:
            hints.append("External forecast rows must match actuals on unique_id + ds; check series IDs, target-date coverage, and actuals column mappings.")
        if "requires cutoff" in lowered or "future-only forecasts" in lowered or "cutoff before ds" in lowered:
            hints.append("External scoring requires historical forecast snapshots with cutoff/forecast_origin labels and cutoff < ds.")
        if "season_length" in lowered or "requested_horizon" in lowered or "positive integer" in lowered:
            hints.append("External scoring parameters must be positive integers; use --season-length >= 1 and --horizon >= 1 when provided.")
        if "duplicate" in lowered:
            hints.append("Actuals and external snapshots must be unique at their scoring grain; aggregate or deduplicate before scoring.")
    if "duplicate" in lowered:
        hints.append("Duplicate unique_id/ds rows must be aggregated or deduplicated before forecasting.")
    if "frequency" in lowered or "freq" in lowered:
        hints.append("The date grain could not be inferred safely; pass --freq explicitly and confirm future dates align.")
    if "require_backtest" in lowered or "backtest" in lowered:
        hints.append("Strict backtesting requires enough observations for every series at the requested horizon.")
    if "missing" in lowered and "column" in lowered:
        hints.append("The source file likely needs --id-col, --time-col, or --target-col mappings.")
    if "xls" in lowered:
        hints.append("Legacy .xls files are unsupported; save as .xlsx, .xlsm, or CSV.")
    if "no usable" in lowered or "null" in lowered or "numeric" in lowered:
        hints.append("The target column may be blank, non-numeric, or fully removed by the fill policy.")
    if not hints:
        hints.append("The error did not match a known pattern; start with profile.json/failure args and validate the input contract.")
    return hints


def _safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "func"}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: _safe_json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _series_ids(*frames: pd.DataFrame) -> list[str]:
    ids: set[str] = set()
    for frame in frames:
        if not frame.empty and "unique_id" in frame.columns:
            ids.update(frame["unique_id"].dropna().astype(str).tolist())
    return sorted(ids)


def _series_llm_review(
    unique_id: str,
    *,
    selected_forecast: pd.DataFrame,
    forecast_long: pd.DataFrame,
    trust_summary: pd.DataFrame,
    model_selection: pd.DataFrame,
    series_summary: pd.DataFrame,
    residual_tests: pd.DataFrame,
    interval_diagnostics: pd.DataFrame,
    seasonality_diagnostics: pd.DataFrame,
    model_weights: pd.DataFrame,
    model_explainability: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "unique_id": unique_id,
        "trust": _first_record(trust_summary, unique_id),
        "model_selection": _first_record(model_selection, unique_id),
        "series_summary": _first_record(series_summary, unique_id),
        "forecast_rows": _records(_filter_series(selected_forecast, unique_id)),
        "selected_model_forecast_long_rows": _records(_filter_series(forecast_long, unique_id, selected_only=True)),
        "residual_tests": _records(_filter_series(residual_tests, unique_id)),
        "interval_diagnostics": _records(_filter_series(interval_diagnostics, unique_id)),
        "seasonality_diagnostics": _records(_filter_series(seasonality_diagnostics, unique_id)),
        "model_weights": _records(_filter_series(model_weights, unique_id)),
        "model_explainability_top_rows": _records(_filter_series(model_explainability, unique_id).head(50)),
    }


def _first_record(frame: pd.DataFrame, unique_id: str) -> dict[str, Any]:
    filtered = _filter_series(frame, unique_id)
    if filtered.empty:
        return {}
    return {key: _safe_json_value(value) for key, value in filtered.iloc[0].to_dict().items()}


def _filter_series(frame: pd.DataFrame, unique_id: str, *, selected_only: bool = False) -> pd.DataFrame:
    if frame.empty or "unique_id" not in frame.columns:
        return pd.DataFrame()
    filtered = frame[frame["unique_id"].astype(str) == str(unique_id)].copy()
    if selected_only and "is_selected_model" in filtered.columns:
        filtered = filtered[filtered["is_selected_model"].fillna(False).astype(bool)]
    return filtered


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
