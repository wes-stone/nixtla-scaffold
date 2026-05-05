from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from nixtla_scaffold.best_practices import best_practice_receipts_frame
from nixtla_scaffold.diagnostics import build_llm_context, format_run_diagnostics_markdown
from nixtla_scaffold.drivers import (
    build_driver_experiment_summary_frame,
    build_known_future_regressors_frame,
    build_scenario_assumptions_frame,
    build_scenario_forecast_frame,
)
from nixtla_scaffold.hierarchy import hierarchy_coherence, reconcile_hierarchy_forecast
from nixtla_scaffold.interpretation import (
    backtest_windows_frame,
    format_interpretation_markdown,
    seasonality_decomposition_frame,
    seasonality_diagnostics_frame,
    seasonality_profile_frame,
    seasonality_summary_frame,
)
from nixtla_scaffold.model_families import model_family
from nixtla_scaffold.reports import write_report_artifacts
from nixtla_scaffold.schema import ForecastRun


def write_run(run: ForecastRun, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = out / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    forecast_long = build_forecast_long(run)
    selected_forecast = build_selected_forecast(run, forecast_long)

    run.history.to_csv(out / "history.csv", index=False)
    selected_forecast.to_csv(out / "forecast.csv", index=False)
    forecast_long.to_csv(out / "forecast_long.csv", index=False)
    build_backtest_long(run).to_csv(out / "backtest_long.csv", index=False)
    build_series_summary(run).to_csv(out / "series_summary.csv", index=False)
    build_model_audit(run).to_csv(out / "model_audit.csv", index=False)
    build_model_win_rates(run).to_csv(out / "model_win_rates.csv", index=False)
    build_model_window_metrics(run).to_csv(out / "model_window_metrics.csv", index=False)
    build_residual_diagnostics(run).to_csv(out / "residual_diagnostics.csv", index=False)
    build_residual_test_summary(run).to_csv(out / "residual_tests.csv", index=False)
    build_interval_diagnostics(run).to_csv(out / "interval_diagnostics.csv", index=False)
    build_trust_summary(run).to_csv(out / "trust_summary.csv", index=False)
    run.model_explainability.to_csv(out / "model_explainability.csv", index=False)
    run.all_models.to_csv(audit / "all_models.csv", index=False)
    run.model_selection.to_csv(audit / "model_selection.csv", index=False)
    run.backtest_metrics.to_csv(audit / "backtest_metrics.csv", index=False)
    run.backtest_predictions.to_csv(audit / "backtest_predictions.csv", index=False)
    backtest_windows_frame(run).to_csv(audit / "backtest_windows.csv", index=False)
    run.model_weights.to_csv(audit / "model_weights.csv", index=False)
    run.transformation_audit.to_csv(audit / "target_transform_audit.csv", index=False)
    if run.spec.events:
        build_scenario_assumptions_frame(run.spec.events).to_csv(out / "scenario_assumptions.csv", index=False)
        build_scenario_forecast_frame(selected_forecast).to_csv(out / "scenario_forecast.csv", index=False)
    if run.spec.regressors or not run.driver_availability_audit.empty:
        build_known_future_regressors_frame(run.spec.regressors).to_csv(out / "known_future_regressors.csv", index=False)
        run.driver_availability_audit.to_csv(out / "driver_availability_audit.csv", index=False)
    if run.spec.events or run.spec.regressors or not run.driver_availability_audit.empty:
        build_driver_experiment_summary_frame(run).to_csv(out / "driver_experiment_summary.csv", index=False)
    if run.spec.custom_models or not run.custom_model_contracts.empty:
        run.custom_model_contracts.to_csv(out / "custom_model_contracts.csv", index=False)
    if run.spec.custom_models or not run.custom_model_invocations.empty:
        run.custom_model_invocations.to_csv(audit / "custom_model_invocations.csv", index=False)
    seasonality_profile_frame(run).to_csv(audit / "seasonality_profile.csv", index=False)
    seasonality_summary_frame(run).to_csv(audit / "seasonality_summary.csv", index=False)
    seasonality_diagnostics_frame(run).to_csv(audit / "seasonality_diagnostics.csv", index=False)
    seasonality_decomposition_frame(run).to_csv(audit / "seasonality_decomposition.csv", index=False)
    best_practice_receipts_frame(run).to_csv(out / "best_practice_receipts.csv", index=False)
    if not run.unreconciled_forecast.empty:
        run.unreconciled_forecast.to_csv(audit / "hierarchy_unreconciled_forecast.csv", index=False)
        hierarchy_coherence(run.unreconciled_forecast).to_csv(audit / "hierarchy_coherence_pre.csv", index=False)
        hierarchy_coherence(run.forecast).to_csv(audit / "hierarchy_coherence_post.csv", index=False)
    if not run.hierarchy_reconciliation.empty:
        run.hierarchy_reconciliation.to_csv(out / "hierarchy_reconciliation.csv", index=False)
    coherence = hierarchy_coherence(run.forecast)
    if not coherence.empty:
        coherence.to_csv(out / "hierarchy_coherence.csv", index=False)
    hierarchy_contribution = build_hierarchy_contribution_frame(run)
    if not hierarchy_contribution.empty:
        hierarchy_contribution.to_csv(out / "hierarchy_contribution.csv", index=False)
    if "hierarchy_depth" in run.forecast.columns:
        hierarchy_backtest = build_hierarchy_backtest_comparison(run)
        hierarchy_backtest.to_csv(audit / "hierarchy_backtest_comparison.csv", index=False)
    (out / "profile.json").write_text(_json(run.profile.to_dict()), encoding="utf-8")
    (out / "manifest.json").write_text(_json(run.manifest()), encoding="utf-8")
    (out / "diagnostics.json").write_text(_json(run.diagnostics()), encoding="utf-8")
    (out / "llm_context.json").write_text(_json(build_llm_context(run)), encoding="utf-8")
    (out / "diagnostics.md").write_text(format_run_diagnostics_markdown(run), encoding="utf-8")
    (audit / "interpretation.json").write_text(_json(run.interpretation()), encoding="utf-8")
    (out / "interpretation.md").write_text(format_interpretation_markdown(run), encoding="utf-8")
    (out / "model_card.md").write_text(run.explanation(), encoding="utf-8")
    write_report_artifacts(run, out)
    write_workbook(run, out / "forecast.xlsx")
    write_review_outputs(run, out, selected_forecast=selected_forecast, forecast_long=forecast_long)
    return out


def write_workbook(run: ForecastRun, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecast_long = build_forecast_long(run)
    selected_forecast = build_selected_forecast(run, forecast_long)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        selected_forecast.to_excel(writer, sheet_name="Forecast", index=False)
        run.history.to_excel(writer, sheet_name="History", index=False)
        run.all_models.to_excel(writer, sheet_name="All Models", index=False)
        run.model_selection.to_excel(writer, sheet_name="Model Selection", index=False)
        run.backtest_metrics.to_excel(writer, sheet_name="Backtest Metrics", index=False)
        run.backtest_predictions.to_excel(writer, sheet_name="Backtest Predictions", index=False)
        backtest_windows_frame(run).to_excel(writer, sheet_name="Backtest Windows", index=False)
        run.model_weights.to_excel(writer, sheet_name="Model Weights", index=False)
        run.transformation_audit.to_excel(writer, sheet_name="Target Transform Audit", index=False)
        if run.spec.events:
            build_scenario_assumptions_frame(run.spec.events).to_excel(writer, sheet_name="Scenario Assumptions", index=False)
            build_scenario_forecast_frame(selected_forecast).to_excel(writer, sheet_name="Scenario Forecast", index=False)
        if run.spec.regressors or not run.driver_availability_audit.empty:
            build_known_future_regressors_frame(run.spec.regressors).to_excel(writer, sheet_name="Known Future Regressors", index=False)
            run.driver_availability_audit.to_excel(writer, sheet_name="Driver Audit", index=False)
        if run.spec.events or run.spec.regressors or not run.driver_availability_audit.empty:
            build_driver_experiment_summary_frame(run).to_excel(writer, sheet_name="Driver Experiments", index=False)
        if run.spec.custom_models or not run.custom_model_contracts.empty:
            run.custom_model_contracts.to_excel(writer, sheet_name="Custom Model Contracts", index=False)
        if run.spec.custom_models or not run.custom_model_invocations.empty:
            run.custom_model_invocations.to_excel(writer, sheet_name="Custom Invocations", index=False)
        forecast_long.to_excel(writer, sheet_name="Forecast Long", index=False)
        build_backtest_long(run).to_excel(writer, sheet_name="Backtest Long", index=False)
        build_series_summary(run).to_excel(writer, sheet_name="Series Summary", index=False)
        build_model_audit(run).to_excel(writer, sheet_name="Model Audit", index=False)
        build_model_win_rates(run).to_excel(writer, sheet_name="Model Win Rates", index=False)
        build_model_window_metrics(run).to_excel(writer, sheet_name="Window Metrics", index=False)
        build_residual_diagnostics(run).to_excel(writer, sheet_name="Residual Diagnostics", index=False)
        build_residual_test_summary(run).to_excel(writer, sheet_name="Residual Tests", index=False)
        build_interval_diagnostics(run).to_excel(writer, sheet_name="Interval Diagnostics", index=False)
        build_trust_summary(run).to_excel(writer, sheet_name="Trust Summary", index=False)
        run.model_explainability.to_excel(writer, sheet_name="Model Explainability", index=False)
        seasonality_profile_frame(run).to_excel(writer, sheet_name="Seasonality Profile", index=False)
        seasonality_summary_frame(run).to_excel(writer, sheet_name="Seasonality Summary", index=False)
        seasonality_diagnostics_frame(run).to_excel(writer, sheet_name="Seasonality Diagnostics", index=False)
        seasonality_decomposition_frame(run).to_excel(writer, sheet_name="Seasonal Decomposition", index=False)
        best_practice_receipts_frame(run).to_excel(writer, sheet_name="Best Practices", index=False)
        coherence = hierarchy_coherence(run.forecast)
        if not coherence.empty:
            coherence.to_excel(writer, sheet_name="Hierarchy Coherence", index=False)
        if not run.hierarchy_reconciliation.empty:
            run.hierarchy_reconciliation.to_excel(writer, sheet_name="Hierarchy Reconciliation", index=False)
        if not run.unreconciled_forecast.empty:
            run.unreconciled_forecast.to_excel(writer, sheet_name="Unreconciled Forecast", index=False)
            hierarchy_coherence(run.unreconciled_forecast).to_excel(writer, sheet_name="Hierarchy Coherence Pre", index=False)
            hierarchy_coherence(run.forecast).to_excel(writer, sheet_name="Hierarchy Coherence Post", index=False)
        hierarchy_contribution = build_hierarchy_contribution_frame(run)
        if not hierarchy_contribution.empty:
            hierarchy_contribution.to_excel(writer, sheet_name="Hierarchy Contribution", index=False)
        if "hierarchy_depth" in run.forecast.columns:
            hierarchy_backtest = build_hierarchy_backtest_comparison(run)
            hierarchy_backtest.to_excel(writer, sheet_name="Hierarchy Backtest", index=False)
        pd.DataFrame([run.profile.to_dict() | {"series": ""}]).to_excel(writer, sheet_name="Profile", index=False)
        pd.DataFrame({"model_card": run.explanation().splitlines()}).to_excel(writer, sheet_name="Model Card", index=False)
        pd.DataFrame({"diagnostics": format_run_diagnostics_markdown(run).splitlines()}).to_excel(
            writer, sheet_name="Diagnostics", index=False
        )
        pd.DataFrame({"interpretation": format_interpretation_markdown(run).splitlines()}).to_excel(
            writer, sheet_name="Interpretation", index=False
        )
    return path


def write_review_outputs(
    run: ForecastRun,
    output_dir: str | Path,
    *,
    selected_forecast: pd.DataFrame | None = None,
    forecast_long: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the curated output layer beside the full agent artifacts."""

    out = Path(output_dir)
    review_dir = out / "output"
    appendix_dir = review_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    selected = selected_forecast.copy() if selected_forecast is not None else build_selected_forecast(run, forecast_long)
    trust_summary = build_trust_summary(run)
    model_audit = build_model_audit(run)
    forecast_review = build_review_forecast(selected)
    decision_summary = build_review_decision_summary(trust_summary)
    model_leaderboard = build_review_model_leaderboard(model_audit)
    forecast_brief = build_review_forecast_brief(run, selected, trust_summary)
    artifact_guide = build_review_artifact_guide()

    forecast_path = review_dir / "forecast_for_review.csv"
    decision_path = review_dir / "decision_summary.csv"
    leaderboard_path = appendix_dir / "model_leaderboard.csv"
    brief_path = appendix_dir / "forecast_brief.csv"
    guide_path = appendix_dir / "artifact_guide.csv"
    workbook_path = review_dir / "forecast_review.xlsx"
    index_path = review_dir / "index.html"
    open_first_path = out / "OPEN_ME_FIRST.html"

    forecast_review.to_csv(forecast_path, index=False)
    decision_summary.to_csv(decision_path, index=False)
    model_leaderboard.to_csv(leaderboard_path, index=False)
    forecast_brief.to_csv(brief_path, index=False)
    artifact_guide.to_csv(guide_path, index=False)
    write_review_workbook(
        workbook_path,
        forecast_brief=forecast_brief,
        forecast_review=forecast_review,
        decision_summary=decision_summary,
        model_leaderboard=model_leaderboard,
        artifact_guide=artifact_guide,
    )

    open_first_path.write_text(
        _review_index_html(
            title="Open this forecast first",
            forecast_brief=forecast_brief,
            decision_summary=decision_summary,
            root_prefix="",
            output_prefix="output/",
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        _review_index_html(
            title="Forecast output review",
            forecast_brief=forecast_brief,
            decision_summary=decision_summary,
            root_prefix="../",
            output_prefix="",
        ),
        encoding="utf-8",
    )
    return {
        "output_open_first": open_first_path,
        "output_index": index_path,
        "output_workbook": workbook_path,
        "output_forecast": forecast_path,
        "output_decision_summary": decision_path,
        "output_model_leaderboard": leaderboard_path,
        "output_forecast_brief": brief_path,
        "output_artifact_guide": guide_path,
    }


def write_review_outputs_from_directory(run_dir: str | Path) -> dict[str, Path]:
    """Write curated output artifacts from an existing run directory."""

    out = Path(run_dir)
    review_dir = out / "output"
    appendix_dir = review_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    selected = _read_review_frame(out / "forecast.csv")
    trust_summary = _read_review_frame(out / "trust_summary.csv")
    model_audit = _read_review_frame(out / "model_audit.csv")
    manifest = _read_review_json(out / "manifest.json")
    diagnostics = _read_review_json(out / "diagnostics.json")

    forecast_review = build_review_forecast(selected)
    decision_summary = build_review_decision_summary(trust_summary)
    model_leaderboard = build_review_model_leaderboard(model_audit)
    forecast_brief = build_review_forecast_brief_from_directory(manifest, diagnostics, selected, trust_summary)
    artifact_guide = build_review_artifact_guide()

    forecast_path = review_dir / "forecast_for_review.csv"
    decision_path = review_dir / "decision_summary.csv"
    leaderboard_path = appendix_dir / "model_leaderboard.csv"
    brief_path = appendix_dir / "forecast_brief.csv"
    guide_path = appendix_dir / "artifact_guide.csv"
    workbook_path = review_dir / "forecast_review.xlsx"
    index_path = review_dir / "index.html"
    open_first_path = out / "OPEN_ME_FIRST.html"

    forecast_review.to_csv(forecast_path, index=False)
    decision_summary.to_csv(decision_path, index=False)
    model_leaderboard.to_csv(leaderboard_path, index=False)
    forecast_brief.to_csv(brief_path, index=False)
    artifact_guide.to_csv(guide_path, index=False)
    write_review_workbook(
        workbook_path,
        forecast_brief=forecast_brief,
        forecast_review=forecast_review,
        decision_summary=decision_summary,
        model_leaderboard=model_leaderboard,
        artifact_guide=artifact_guide,
    )
    open_first_path.write_text(
        _review_index_html(
            title="Open this forecast first",
            forecast_brief=forecast_brief,
            decision_summary=decision_summary,
            root_prefix="",
            output_prefix="output/",
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        _review_index_html(
            title="Forecast output review",
            forecast_brief=forecast_brief,
            decision_summary=decision_summary,
            root_prefix="../",
            output_prefix="",
        ),
        encoding="utf-8",
    )
    paths = {
        "output_open_first": open_first_path,
        "output_index": index_path,
        "output_workbook": workbook_path,
        "output_forecast": forecast_path,
        "output_decision_summary": decision_path,
        "output_model_leaderboard": leaderboard_path,
        "output_forecast_brief": brief_path,
        "output_artifact_guide": guide_path,
    }
    _update_manifest_review_outputs(out / "manifest.json")
    return paths


def write_review_workbook(
    output_path: str | Path,
    *,
    forecast_brief: pd.DataFrame,
    forecast_review: pd.DataFrame,
    decision_summary: pd.DataFrame,
    model_leaderboard: pd.DataFrame,
    artifact_guide: pd.DataFrame,
) -> Path:
    """Write a compact workbook intended for standard forecast review, not full audit replay."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    watchouts = decision_summary[
        [col for col in ["unique_id", "trust_level", "caveats", "next_actions"] if col in decision_summary.columns]
    ].copy()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        forecast_brief.to_excel(writer, sheet_name="Start Here", index=False)
        forecast_review.to_excel(writer, sheet_name="Forecast", index=False)
        decision_summary.to_excel(writer, sheet_name="Decision Summary", index=False)
        model_leaderboard.to_excel(writer, sheet_name="Model Leaderboard", index=False)
        watchouts.to_excel(writer, sheet_name="Watchouts", index=False)
        artifact_guide.to_excel(writer, sheet_name="File Guide", index=False)
    return path


def build_review_forecast(selected_forecast: pd.DataFrame) -> pd.DataFrame:
    """Selected forecast columns that a finance reader can scan without model-feed noise."""

    columns = [
        "unique_id",
        "ds",
        "model",
        "selected_model",
        "horizon_step",
        "yhat",
        "yhat_lo_80",
        "yhat_hi_80",
        "yhat_lo_95",
        "yhat_hi_95",
        "yhat_scenario",
        "event_adjustment",
        "event_names",
        "planning_eligible",
        "planning_eligibility_reason",
        "row_horizon_status",
        "horizon_trust_state",
        "validated_through_horizon",
        "interval_status",
        "interval_method",
    ]
    return _select_existing_columns(selected_forecast, columns)


def build_review_decision_summary(trust_summary: pd.DataFrame) -> pd.DataFrame:
    """Condensed decision table; detailed trust evidence remains in trust_summary.csv."""

    columns = [
        "unique_id",
        "trust_level",
        "trust_score_0_100",
        "selected_model",
        "primary_metric",
        "primary_metric_value",
        "requested_horizon",
        "selection_horizon",
        "validated_through_horizon",
        "full_horizon_claim_allowed",
        "horizon_trust_state",
        "interval_status",
        "seasonality_status",
        "hierarchy_status",
        "event_status",
        "caveats",
        "next_actions",
    ]
    return _select_existing_columns(trust_summary, columns)


def build_review_model_leaderboard(model_audit: pd.DataFrame, *, top_n_per_series: int = 8) -> pd.DataFrame:
    """Small model leaderboard; the full model audit remains in model_audit.csv."""

    columns = [
        "unique_id",
        "model",
        "family",
        "rmse",
        "mae",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "weight",
        "is_selected_model",
        "selection_horizon",
        "requested_horizon",
        "cv_windows",
    ]
    leaderboard = _select_existing_columns(model_audit, columns)
    if leaderboard.empty or "unique_id" not in leaderboard.columns:
        return leaderboard
    sort_cols = [col for col in ["unique_id", "is_selected_model", "rmse", "mae", "wape", "model"] if col in leaderboard.columns]
    ascending = [True, False, True, True, True, True][: len(sort_cols)]
    leaderboard = leaderboard.sort_values(sort_cols, ascending=ascending)
    return leaderboard.groupby("unique_id", sort=True, group_keys=False).head(top_n_per_series).reset_index(drop=True)


def build_review_forecast_brief(run: ForecastRun, selected_forecast: pd.DataFrame, trust_summary: pd.DataFrame) -> pd.DataFrame:
    """One-page run brief for the review workbook and open-first HTML."""

    from nixtla_scaffold.headline import build_executive_headline

    headline = build_executive_headline(run)
    forecast = selected_forecast.copy()
    if not forecast.empty and "ds" in forecast.columns:
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    trust_counts = (
        trust_summary["trust_level"].fillna("Unknown").astype(str).value_counts().to_dict()
        if not trust_summary.empty and "trust_level" in trust_summary.columns
        else {}
    )
    rows = [
        ("Start here", "Executive headline", headline.paragraph),
        ("Start here", "Output workbook", "output/forecast_review.xlsx"),
        ("Start here", "Static report", "report.html"),
        ("Start here", "Interactive app", "streamlit_app.py"),
        ("Run context", "Series count", _series_count(selected_forecast)),
        ("Run context", "Forecast horizon", run.spec.horizon),
        ("Run context", "Frequency", run.spec.freq or run.profile.freq),
        ("Run context", "Season length", run.profile.season_length),
        ("Run context", "Model policy", run.spec.model_policy),
        ("Run context", "Engine", run.engine),
        ("Run context", "Forecast start", _date_min(forecast.get("ds"))),
        ("Run context", "Forecast end", _date_max(forecast.get("ds"))),
        ("Readiness", "Trust distribution", "; ".join(f"{key}: {value}" for key, value in sorted(trust_counts.items()))),
        ("Readiness", "Full-horizon claim allowed series", headline.full_horizon_claim_allowed_count),
        ("Readiness", "Top caveat", headline.top_caveat),
        ("Readiness", "Next action", headline.next_action),
        ("Guardrail", "Baseline vs scenario", "Keep statistical yhat, scenario yhat, and plan/target separate."),
        ("Guardrail", "Planning eligibility", "planning_eligible is a horizon-validation flag only, not global approval."),
        ("Guardrail", "LLM handoff", "Attach llm_context.json when asking an agent to inspect the full run."),
    ]
    if run.warnings:
        rows.append(("Warnings", "Warning count", len(run.warnings)))
        rows.extend(("Warnings", f"Warning {idx}", warning) for idx, warning in enumerate(run.warnings[:5], start=1))
    return pd.DataFrame(rows, columns=["section", "item", "value"])


def build_review_forecast_brief_from_directory(
    manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    selected_forecast: pd.DataFrame,
    trust_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the review run brief when only persisted run artifacts are available."""

    forecast = selected_forecast.copy()
    if not forecast.empty and "ds" in forecast.columns:
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    spec = manifest.get("spec", {})
    profile = manifest.get("profile", {})
    executive = diagnostics.get("executive_headline", {})
    trust_counts = (
        trust_summary["trust_level"].fillna("Unknown").astype(str).value_counts().to_dict()
        if not trust_summary.empty and "trust_level" in trust_summary.columns
        else {}
    )
    rows = [
        ("Start here", "Executive headline", executive.get("paragraph") or "Executive headline unavailable for this run."),
        ("Start here", "Output workbook", "output/forecast_review.xlsx"),
        ("Start here", "Static report", "report.html"),
        ("Start here", "Interactive app", "streamlit_app.py"),
        ("Run context", "Series count", _series_count(selected_forecast) or profile.get("series_count", "")),
        ("Run context", "Forecast horizon", spec.get("horizon", "")),
        ("Run context", "Frequency", spec.get("freq") or profile.get("freq", "")),
        ("Run context", "Season length", profile.get("season_length", "")),
        ("Run context", "Model policy", spec.get("model_policy", "")),
        ("Run context", "Engine", manifest.get("engine", "")),
        ("Run context", "Forecast start", _date_min(forecast.get("ds"))),
        ("Run context", "Forecast end", _date_max(forecast.get("ds"))),
        ("Readiness", "Trust distribution", "; ".join(f"{key}: {value}" for key, value in sorted(trust_counts.items()))),
        ("Readiness", "Full-horizon claim allowed series", executive.get("full_horizon_claim_allowed_count", "")),
        ("Readiness", "Top caveat", executive.get("top_caveat", "")),
        ("Readiness", "Next action", executive.get("next_action", "")),
        ("Guardrail", "Baseline vs scenario", "Keep statistical yhat, scenario yhat, and plan/target separate."),
        ("Guardrail", "Planning eligibility", "planning_eligible is a horizon-validation flag only, not global approval."),
        ("Guardrail", "LLM handoff", "Attach llm_context.json when asking an agent to inspect the full run."),
    ]
    warnings = manifest.get("warnings", [])
    if warnings:
        rows.append(("Warnings", "Warning count", len(warnings)))
        rows.extend(("Warnings", f"Warning {idx}", warning) for idx, warning in enumerate(warnings[:5], start=1))
    return pd.DataFrame(rows, columns=["section", "item", "value"])


def build_review_artifact_guide() -> pd.DataFrame:
    """Opinionated file map for curated output, appendix, agent, and audit artifacts."""

    rows = [
        ("output", 1, "OPEN_ME_FIRST.html", "Start here. Simple file map and forecast headline."),
        ("output", 2, "output/forecast_review.xlsx", "Compact workbook with the forecast, decision summary, leaderboard, watchouts, and file guide."),
        ("output", 3, "report.html", "Portable static review with charts, decision evidence, and ledger preview when available."),
        ("output", 4, "streamlit_app.py", "Interactive local workbench for deeper visual review."),
        ("output", 5, "output/forecast_for_review.csv", "Selected forecast rows only, stripped down for finance review."),
        ("output", 6, "output/decision_summary.csv", "Condensed trust/readiness table with caveats and next actions."),
        ("appendix", 1, "output/appendix/model_leaderboard.csv", "Small top-model view behind the workbook leaderboard."),
        ("appendix", 2, "output/appendix/forecast_brief.csv", "One-page run brief used by OPEN_ME_FIRST.html and the workbook."),
        ("appendix", 3, "output/appendix/artifact_guide.csv", "This file map."),
        ("agent", 1, "llm_context.json", "Full LLM handoff packet with guardrails and all major context."),
        ("agent", 2, "forecast_long.csv", "All future model/date rows for agents, dashboards, and model feeds."),
        ("agent", 3, "backtest_long.csv", "All rolling-origin validation rows for programmatic inspection."),
        ("audit", 1, "audit/all_models.csv", "Every candidate model forecast for transparency."),
        ("audit", 2, "audit/backtest_metrics.csv", "Raw model CV metrics behind selection."),
        ("audit", 3, "audit/seasonality_diagnostics.csv", "Cycle-count and seasonality credibility checks."),
        ("audit", 4, "manifest.json", "Reproducibility, model policy resolution, and complete output index."),
    ]
    return pd.DataFrame(rows, columns=["audience", "priority", "artifact", "purpose"])


def _select_existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        existing = [col for col in columns if frame is not None and col in frame.columns]
        return pd.DataFrame(columns=existing or columns)
    return frame[[col for col in columns if col in frame.columns]].copy()


def _read_review_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_review_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _update_manifest_review_outputs(path: Path) -> None:
    if not path.exists():
        return
    manifest = _read_review_json(path)
    outputs = manifest.setdefault("outputs", {})
    for key in list(outputs):
        if key.startswith("human_"):
            outputs.pop(key)
    outputs.update(
        {
            "output_open_first": "OPEN_ME_FIRST.html",
            "output_index": "output/index.html",
            "output_workbook": "output/forecast_review.xlsx",
            "output_forecast": "output/forecast_for_review.csv",
            "output_decision_summary": "output/decision_summary.csv",
            "output_model_leaderboard": "output/appendix/model_leaderboard.csv",
            "output_forecast_brief": "output/appendix/forecast_brief.csv",
            "output_artifact_guide": "output/appendix/artifact_guide.csv",
        }
    )
    path.write_text(_json(manifest), encoding="utf-8")


def _series_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "unique_id" in frame.columns:
        return int(frame["unique_id"].astype(str).nunique())
    return 1


def _date_min(series: pd.Series | None) -> str:
    if series is None:
        return ""
    values = pd.to_datetime(series, errors="coerce").dropna()
    return values.min().date().isoformat() if not values.empty else ""


def _date_max(series: pd.Series | None) -> str:
    if series is None:
        return ""
    values = pd.to_datetime(series, errors="coerce").dropna()
    return values.max().date().isoformat() if not values.empty else ""


def _review_index_html(
    *,
    title: str,
    forecast_brief: pd.DataFrame,
    decision_summary: pd.DataFrame,
    root_prefix: str,
    output_prefix: str,
) -> str:
    headline = _brief_value(forecast_brief, "Executive headline")
    cards = [
        (
            f"{output_prefix}forecast_review.xlsx",
            "Forecast review workbook",
            "The clean output workbook: start here for the forecast, decision summary, leaderboard, watchouts, and file guide.",
        ),
        (
            f"{root_prefix}report.html",
            "Static report",
            "The polished visual review with charts, model evidence, intervals, seasonality, and ledger preview when present.",
        ),
        (
            f"{output_prefix}forecast_for_review.csv",
            "Forecast for review",
            "Selected forecast rows only, with scenario columns and horizon/interval guardrails.",
        ),
        (
            f"{output_prefix}decision_summary.csv",
            "Decision summary",
            "Condensed trust, caveats, and next actions by series.",
        ),
        (
            f"{root_prefix}streamlit_app.py",
            "Interactive app",
            "Run locally with `uv run streamlit run streamlit_app.py` when you need interactive drilldown.",
        ),
        (
            f"{root_prefix}llm_context.json",
            "LLM context",
            "Attach this to an agent when you want the full machine-readable run context.",
        ),
        (
            f"{output_prefix}appendix/artifact_guide.csv",
            "Appendix file guide",
            "Supporting output map for appendix, agent, and audit artifacts.",
        ),
    ]
    card_html = "\n".join(_review_link_card(*card) for card in cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #f6f8fb;
      color: #172033;
    }}
    body {{ margin: 0; padding: 28px; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    .hero, .panel, .card {{
      background: #fff;
      border: 1px solid #dde5f0;
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(23, 32, 51, 0.06);
    }}
    .hero {{ padding: 28px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ line-height: 1.55; }}
    .headline {{
      margin-top: 16px;
      padding: 16px;
      border-left: 4px solid #3867d6;
      background: #f3f6ff;
      border-radius: 12px;
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .card {{ display: block; padding: 18px; color: inherit; text-decoration: none; }}
    .card:hover {{ border-color: #3867d6; transform: translateY(-1px); }}
    .card strong {{ display: block; font-size: 17px; margin-bottom: 8px; }}
    .card span {{ color: #53627a; }}
    .panel {{ padding: 22px; margin-top: 18px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e8edf5; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ color: #53627a; font-weight: 700; }}
    .note {{ color: #53627a; }}
    code {{ background: #edf2f7; border-radius: 6px; padding: 2px 5px; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{escape(title)}</h1>
      <p class="note">This folder keeps a small curated output layer on top of the full audit trail. Use these files for review; the larger CSV/JSON outputs remain for agents, dashboards, and reproducibility.</p>
      <div class="headline">{escape(headline)}</div>
    </section>
    <section class="grid">
      {card_html}
    </section>
    <section class="panel">
      <h2>Run brief</h2>
      {_html_table(forecast_brief, ["section", "item", "value"], limit=18)}
    </section>
    <section class="panel">
      <h2>Decision summary preview</h2>
      {_html_table(decision_summary, ["unique_id", "trust_level", "trust_score_0_100", "selected_model", "horizon_trust_state", "interval_status", "caveats", "next_actions"], limit=10)}
    </section>
  </main>
</body>
</html>
"""


def _review_link_card(href: str, title: str, description: str) -> str:
    return f'<a class="card" href="{escape(href)}"><strong>{escape(title)}</strong><span>{escape(description)}</span></a>'


def _brief_value(brief: pd.DataFrame, item: str) -> str:
    if brief.empty or not {"item", "value"}.issubset(brief.columns):
        return ""
    matches = brief.loc[brief["item"].astype(str) == item, "value"]
    return str(matches.iloc[0]) if not matches.empty else ""


def _html_table(frame: pd.DataFrame, columns: list[str], *, limit: int) -> str:
    available = [col for col in columns if col in frame.columns]
    if frame.empty or not available:
        return '<p class="note">No rows available.</p>'
    rows = []
    for record in frame[available].head(limit).to_dict("records"):
        cells = "".join(f"<td>{escape(_review_value(record.get(col)))}</td>" for col in available)
        rows.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{escape(col)}</th>" for col in available)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _review_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str) + "\n"


INTERVAL_DIAGNOSTIC_COLUMNS = [
    "unique_id",
    "model",
    "level",
    "horizon_step",
    "target_coverage",
    "empirical_coverage",
    "coverage_gap",
    "coverage_status",
    "interval_status",
    "interval_method",
    "requested_horizon",
    "selection_horizon",
    "cv_windows",
    "cv_step_size",
    "cv_horizon_matches_requested",
    "avg_width",
    "normalized_width",
    "miss_above_rate",
    "miss_below_rate",
    "observations",
]

RESIDUAL_TEST_COLUMNS = [
    "unique_id",
    "model",
    "observations",
    "residual_scope",
    "mean_error",
    "residual_std",
    "bias_t_stat",
    "bias_status",
    "white_noise_residual_scope",
    "white_noise_observations",
    "max_abs_acf",
    "acf_threshold_95",
    "significant_acf_lags",
    "significant_acf_lag_count",
    "white_noise_status",
    "outlier_count",
    "outlier_rate",
    "outlier_status",
    "structural_break_score",
    "late_vs_early_rmse_ratio",
    "structural_break_status",
    "overall_status",
    "interpretation",
]

HIERARCHY_CONTRIBUTION_COLUMNS = [
    "value_col",
    "ds",
    "parent_unique_id",
    "child_unique_id",
    "child_hierarchy_level",
    "child_hierarchy_depth",
    "parent_value",
    "child_value",
    "child_sum",
    "child_share_of_parent",
    "child_share_of_children",
    "gap_allocation_weight",
    "parent_child_gap",
    "gap_pct",
    "gap_contribution",
    "gap_contribution_formula",
]

HIERARCHY_BACKTEST_COMPARISON_COLUMNS = [
    "unique_id",
    "model",
    "cutoff",
    "ds",
    "horizon_step",
    "hierarchy_level",
    "hierarchy_depth",
    "reconciliation_method",
    "applied_method",
    "comparison_status",
    "comparison_note",
    "y_actual",
    "yhat_unreconciled",
    "yhat_reconciled",
    "error_unreconciled",
    "error_reconciled",
    "abs_error_unreconciled",
    "abs_error_reconciled",
    "abs_error_delta",
    "squared_error_unreconciled",
    "squared_error_reconciled",
    "squared_error_delta",
]


SELECTED_FORECAST_INTERVAL_COLUMNS = [
    "horizon_step",
    "interval_status",
    "interval_method",
    "interval_evidence",
    "requested_horizon",
    "selection_horizon",
    "cv_windows",
    "cv_step_size",
    "cv_horizon_matches_requested",
    "row_horizon_status",
    "horizon_trust_state",
    "horizon_trust_label",
    "validated_through_horizon",
    "is_beyond_validated_horizon",
    "planning_eligible",
    "planning_eligibility_scope",
    "planning_eligibility_reason",
    "horizon_warning",
    "forecast_horizon_status",
    "horizon_validated_in_cv",
    "validation_evidence",
]


def build_selected_forecast(run: ForecastRun, forecast_long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Selected forecast rows enriched with interval provenance for analyst-facing CSVs."""

    if run.forecast.empty:
        return run.forecast.copy()
    forecast = run.forecast.copy()
    for column in SELECTED_FORECAST_INTERVAL_COLUMNS:
        if column not in forecast.columns:
            forecast[column] = None

    long = forecast_long if forecast_long is not None else build_forecast_long(run)
    required = {"unique_id", "ds", "is_selected_model"}
    if long.empty or not required.issubset(long.columns):
        return _order_model_feed_columns(forecast)

    selected = long[long["is_selected_model"].fillna(False).astype(bool)].copy()
    available_cols = [col for col in SELECTED_FORECAST_INTERVAL_COLUMNS if col in selected.columns]
    if selected.empty or not available_cols:
        return _order_model_feed_columns(forecast)

    forecast["_merge_unique_id"] = forecast["unique_id"].astype(str)
    forecast["_merge_ds"] = pd.to_datetime(forecast["ds"])
    selected["_merge_unique_id"] = selected["unique_id"].astype(str)
    selected["_merge_ds"] = pd.to_datetime(selected["ds"])
    selected = selected[["_merge_unique_id", "_merge_ds", *available_cols]].drop_duplicates(
        ["_merge_unique_id", "_merge_ds"],
        keep="first",
    )
    forecast = forecast.drop(columns=available_cols, errors="ignore")
    enriched = forecast.merge(selected, on=["_merge_unique_id", "_merge_ds"], how="left")
    return _order_model_feed_columns(enriched.drop(columns=["_merge_unique_id", "_merge_ds"]))


def build_forecast_long(run: ForecastRun) -> pd.DataFrame:
    """Future predictions in a feeder-friendly long format, one row per model/date."""

    if run.all_models.empty:
        return pd.DataFrame(
            columns=[
                "record_type",
                "unique_id",
                "ds",
                "model",
                "family",
                "horizon_step",
                "yhat",
                "is_selected_model",
                "selected_model",
                "weight",
                "interval_status",
                "interval_method",
                "interval_evidence",
                "requested_horizon",
                "selection_horizon",
                "cv_windows",
                "cv_step_size",
                "cv_horizon_matches_requested",
                "row_horizon_status",
                "horizon_trust_state",
                "horizon_trust_label",
                "validated_through_horizon",
                "is_beyond_validated_horizon",
                "planning_eligible",
                "planning_eligibility_scope",
                "planning_eligibility_reason",
                "horizon_warning",
                "forecast_horizon_status",
                "horizon_validated_in_cv",
                "validation_evidence",
            ]
        )
    rows: list[dict[str, Any]] = []
    selected = _selected_model_map(run)
    weights = _weight_map(run)
    intervals = build_interval_diagnostics(run)
    metadata = _cv_metadata_map(run)
    final_lookup = _final_forecast_lookup(run)
    all_models = run.all_models.copy()
    all_models["ds"] = pd.to_datetime(all_models["ds"])
    model_cols = _model_columns(all_models)
    levels = run.effective_levels()
    for uid_value, grp in all_models.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
        uid = str(uid_value)
        selected_model = selected.get(uid, "")
        for horizon_step, row in enumerate(grp.to_dict("records"), start=1):
            ds_value = pd.Timestamp(row["ds"])
            final_row = final_lookup.get((uid, ds_value))
            for model in model_cols:
                is_selected = model == selected_model
                final_selected = final_row if is_selected else None
                value = final_selected.get("yhat") if final_selected is not None else row.get(model)
                if _is_missing(value):
                    continue
                horizon_contract = _horizon_row_contract(
                    horizon_step,
                    metadata.get((uid, model), {}),
                    requested_horizon=run.spec.horizon,
                )
                interval_values: dict[int, tuple[Any, Any]] = {}
                for level in levels:
                    if final_selected is not None:
                        lo_value = final_selected.get(f"yhat_lo_{level}")
                        hi_value = final_selected.get(f"yhat_hi_{level}")
                    else:
                        lo_value = row.get(f"{model}-lo-{level}")
                        hi_value = row.get(f"{model}-hi-{level}")
                    if not _is_missing(lo_value) and not _is_missing(hi_value):
                        interval_values[level] = (lo_value, hi_value)
                meta = metadata.get((uid, model), {})
                adjusted = (
                    is_selected
                    and final_selected is not None
                    and _final_selected_forecast_adjusted(row, final_selected, model, levels)
                )
                interval_status = _future_interval_status(
                    uid=uid,
                    model=model,
                    horizon_step=horizon_step,
                    has_interval=bool(interval_values),
                    intervals=intervals,
                    metadata=meta,
                    adjusted=adjusted,
                )
                out: dict[str, Any] = {
                    "record_type": "forecast",
                    "unique_id": uid,
                    "ds": ds_value,
                    "model": model,
                    "family": model_family(model),
                    "horizon_step": horizon_step,
                    "yhat": value,
                    "is_selected_model": is_selected,
                    "selected_model": selected_model,
                    "weight": weights.get((uid, model)),
                    "interval_status": interval_status,
                    "interval_method": _interval_method(model, has_interval=bool(interval_values)),
                    "interval_evidence": _interval_evidence(interval_status, meta),
                    "requested_horizon": meta.get("requested_horizon"),
                    "selection_horizon": meta.get("selection_horizon"),
                    "cv_windows": meta.get("cv_windows"),
                    "cv_step_size": meta.get("cv_step_size"),
                    "cv_horizon_matches_requested": meta.get("cv_horizon_matches_requested"),
                    **horizon_contract,
                }
                for level, (lo_value, hi_value) in interval_values.items():
                    out[f"yhat_lo_{level}"] = lo_value
                    out[f"yhat_hi_{level}"] = hi_value
                if final_selected is not None:
                    for column, column_value in final_selected.items():
                        if column.startswith("yhat_scenario") or column in {"event_adjustment", "event_names"}:
                            out[column] = column_value
                rows.append(out)
    return _order_model_feed_columns(_sort_for_feeder(pd.DataFrame(rows), ["unique_id", "ds", "is_selected_model", "weight"]))


def build_backtest_long(run: ForecastRun) -> pd.DataFrame:
    """Rolling-origin predictions in a feeder-friendly long format with actuals."""

    if run.backtest_predictions.empty:
        return pd.DataFrame(
            columns=[
                "record_type",
                "unique_id",
                "cutoff",
                "ds",
                "model",
                "y_actual",
                "yhat",
                "horizon_step",
                "h",
                "error",
                "abs_error",
                "squared_error",
                "mase_scale",
                "rmsse_scale",
                "pct_error",
                "is_selected_model",
                "selected_model",
                "weight",
            ]
        )
    rows: list[dict[str, Any]] = []
    selected = _selected_model_map(run)
    weights = _weight_map(run)
    scales = _scale_maps(run.history, run.profile.season_length)
    model_cols = _model_columns(run.backtest_predictions, extra_exclude={"cutoff", "y"})
    frame = run.backtest_predictions.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["cutoff"] = pd.to_datetime(frame["cutoff"])
    for (uid, cutoff), grp in frame.sort_values(["unique_id", "cutoff", "ds"]).groupby(["unique_id", "cutoff"], sort=True):
        selected_model = selected.get(str(uid), "")
        for step, (_, row) in enumerate(grp.iterrows(), start=1):
            for model in model_cols:
                value = row.get(model)
                if _is_missing(value):
                    continue
                actual = row.get("y")
                actual_f = _safe_float(actual)
                yhat_f = _safe_float(value)
                error = actual_f - yhat_f if actual_f is not None and yhat_f is not None else None
                pct_error = error / actual_f if error is not None and actual_f not in (0.0, None) else None
                out: dict[str, Any] = {
                    "record_type": "backtest",
                    "unique_id": str(uid),
                    "cutoff": cutoff,
                    "ds": row["ds"],
                    "model": model,
                    "y_actual": actual,
                    "yhat": value,
                    "horizon_step": step,
                    "h": step,
                    "error": error,
                    "abs_error": abs(error) if error is not None else None,
                    "squared_error": error ** 2 if error is not None else None,
                    "mase_scale": scales.get(str(uid), {}).get("mase_scale"),
                    "rmsse_scale": scales.get(str(uid), {}).get("rmsse_scale"),
                    "pct_error": pct_error,
                    "is_selected_model": model == selected_model,
                    "selected_model": selected_model,
                    "weight": weights.get((str(uid), model)),
                }
                for level in _interval_levels_for_model(row, model):
                    lo_col = f"{model}-lo-{level}"
                    hi_col = f"{model}-hi-{level}"
                    lo_value = row.get(lo_col)
                    hi_value = row.get(hi_col)
                    out[f"yhat_lo_{level}"] = lo_value
                    out[f"yhat_hi_{level}"] = hi_value
                    covered = _covered(actual, lo_value, hi_value)
                    out[f"covered_{level}"] = covered
                    out[f"miss_direction_{level}"] = _miss_direction(actual, lo_value, hi_value)
                rows.append(
                    out
                )
    return _order_model_feed_columns(_sort_for_feeder(pd.DataFrame(rows), ["unique_id", "cutoff", "ds", "is_selected_model", "weight"]))


def build_series_summary(run: ForecastRun) -> pd.DataFrame:
    """One row per series for analysts and downstream model feeds."""

    if run.model_selection.empty:
        return pd.DataFrame()
    summary = run.model_selection.copy()
    seasonality = seasonality_summary_frame(run)
    if not seasonality.empty:
        summary = summary.merge(seasonality, on="unique_id", how="left")
    top_models = _top_models_frame(run)
    if not top_models.empty:
        summary = summary.merge(top_models, on="unique_id", how="left")
    history_stats = (
        run.history.groupby("unique_id", sort=True)
        .agg(history_rows=("y", "count"), history_start=("ds", "min"), history_end=("ds", "max"), last_actual=("y", "last"))
        .reset_index()
    )
    summary = summary.merge(history_stats, on="unique_id", how="left")
    columns = [
        "unique_id",
        "selected_model",
        "rmse",
        "mae",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "observations",
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
        "history_rows",
        "history_start",
        "history_end",
        "last_actual",
        "top_weighted_models",
        "top_weighted_families",
        "top_weighted_weights",
        "seasonality_strength",
        "peak_position",
        "trough_position",
        "selection_reason",
    ]
    return summary[[col for col in columns if col in summary.columns]]


def build_model_audit(run: ForecastRun) -> pd.DataFrame:
    """Model leaderboard enriched with weights and selected/challenger flags."""

    if run.backtest_metrics.empty:
        return pd.DataFrame()
    audit = run.backtest_metrics.copy()
    audit["family"] = audit["model"].map(model_family)
    selected = _selected_model_map(run)
    if not run.model_weights.empty:
        audit = audit.merge(run.model_weights[["unique_id", "model", "weight"]], on=["unique_id", "model"], how="left")
    audit["selected_model"] = audit["unique_id"].astype(str).map(selected)
    audit["is_selected_model"] = audit["model"].astype(str) == audit["selected_model"].astype(str)
    sort_cols = [col for col in ["unique_id", "rmse", "mase", "mae", "wape"] if col in audit.columns]
    if sort_cols:
        audit = audit.sort_values(sort_cols)
    return audit.reset_index(drop=True)


def build_model_win_rates(run: ForecastRun) -> pd.DataFrame:
    """Benchmark win rates by model, using SeasonalNaive when available."""

    if run.backtest_metrics.empty:
        return pd.DataFrame(
            columns=[
                "benchmark_model",
                "metric",
                "model",
                "eligible_series",
                "wins_vs_benchmark",
                "win_rate_vs_benchmark",
                "avg_skill_vs_benchmark",
            ]
        )
    metrics = run.backtest_metrics.copy()
    metric = "mase" if "mase" in metrics.columns and metrics["mase"].notna().any() else "rmse"
    benchmark_model = "SeasonalNaive" if (metrics["model"].astype(str) == "SeasonalNaive").any() else "Naive"
    benchmark = metrics[metrics["model"].astype(str) == benchmark_model][["unique_id", metric]].rename(columns={metric: "benchmark_metric"})
    if benchmark.empty:
        return pd.DataFrame(
            columns=[
                "benchmark_model",
                "metric",
                "model",
                "eligible_series",
                "wins_vs_benchmark",
                "win_rate_vs_benchmark",
                "avg_skill_vs_benchmark",
            ]
        )
    rows: list[dict[str, Any]] = []
    merged = metrics.merge(benchmark, on="unique_id", how="inner")
    merged = merged[merged[metric].notna() & merged["benchmark_metric"].notna() & (merged["benchmark_metric"] > 0)]
    for model, grp in merged.groupby("model", sort=True):
        wins = grp[metric] <= grp["benchmark_metric"]
        skill = 1.0 - pd.to_numeric(grp[metric], errors="coerce") / pd.to_numeric(grp["benchmark_metric"], errors="coerce")
        rows.append(
            {
                "benchmark_model": benchmark_model,
                "metric": metric,
                "model": model,
                "eligible_series": int(len(grp)),
                "wins_vs_benchmark": int(wins.sum()),
                "win_rate_vs_benchmark": float(wins.mean()) if len(wins) else None,
                "avg_skill_vs_benchmark": float(skill.mean()) if skill.notna().any() else None,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "benchmark_model",
                "metric",
                "model",
                "eligible_series",
                "wins_vs_benchmark",
                "win_rate_vs_benchmark",
                "avg_skill_vs_benchmark",
            ]
        )
    return pd.DataFrame(rows).sort_values(["win_rate_vs_benchmark", "avg_skill_vs_benchmark", "model"], ascending=[False, False, True]).reset_index(drop=True)


def build_model_window_metrics(run: ForecastRun) -> pd.DataFrame:
    """Error metrics by series, cutoff, and model."""

    long = build_backtest_long(run)
    if long.empty:
        return pd.DataFrame(columns=["unique_id", "cutoff", "model", "rmse", "mae", "wape", "mase", "rmsse", "bias", "observations"])
    rows: list[dict[str, Any]] = []
    for (uid, cutoff, model), grp in long.groupby(["unique_id", "cutoff", "model"], sort=True):
        rows.append(_metric_row(uid, model, grp, cutoff=cutoff))
    return pd.DataFrame(rows).sort_values(["unique_id", "cutoff", "rmse", "model"]).reset_index(drop=True)


def build_residual_diagnostics(run: ForecastRun) -> pd.DataFrame:
    """Error diagnostics by series, model, and horizon step."""

    long = build_backtest_long(run)
    if long.empty:
        return pd.DataFrame(
            columns=[
                "unique_id",
                "model",
                "horizon_step",
                "rmse",
                "mae",
                "wape",
                "mase",
                "rmsse",
                "bias",
                "mean_error",
                "median_error",
                "error_std",
                "observations",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (uid, model, h), grp in long.groupby(["unique_id", "model", "horizon_step"], sort=True):
        row = _metric_row(uid, model, grp)
        errors = pd.to_numeric(grp["error"], errors="coerce")
        row.update(
            {
                "horizon_step": int(h),
                "mean_error": float(errors.mean()) if errors.notna().any() else None,
                "median_error": float(errors.median()) if errors.notna().any() else None,
                "error_std": float(errors.std(ddof=0)) if errors.notna().any() else None,
            }
        )
        rows.append(row)
    columns = [
        "unique_id",
        "model",
        "horizon_step",
        "rmse",
        "mae",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "mean_error",
        "median_error",
        "error_std",
        "observations",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["unique_id", "horizon_step", "rmse", "model"]).reset_index(drop=True)


def build_residual_test_summary(run: ForecastRun) -> pd.DataFrame:
    """Heuristic residual health checks by series and model."""

    long = build_backtest_long(run)
    if long.empty or "error" not in long.columns:
        return pd.DataFrame(columns=RESIDUAL_TEST_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (uid, model), grp in long.groupby(["unique_id", "model"], sort=True):
        errors = pd.to_numeric(grp["error"], errors="coerce").dropna()
        observations = int(len(errors))
        if observations == 0:
            continue

        mean_error = float(errors.mean())
        residual_std = float(errors.std(ddof=1)) if observations > 1 else None
        bias_t_stat = _bias_t_stat(errors)
        bias_status = _bias_status(bias_t_stat, observations)
        acf_errors, acf_scope = _one_step_residuals_for_acf(grp)
        acf_observations = int(len(acf_errors))
        acf = _residual_acf(acf_errors)
        threshold = 1.96 / math.sqrt(acf_observations) if acf_observations > 0 else None
        significant_lags = (
            acf.loc[acf["significant"], "lag"].astype(int).tolist()
            if not acf.empty and "significant" in acf.columns
            else []
        )
        max_abs_acf = float(acf["acf"].abs().max()) if not acf.empty else None
        white_noise_status = _white_noise_status(acf_observations, significant_lags)
        outlier_count, outlier_rate = _residual_outlier_stats(errors)
        outlier_status = _outlier_status(observations, outlier_rate)
        break_score, rmse_ratio = _structural_break_scores(errors)
        structural_break_status = _structural_break_status(observations, break_score, rmse_ratio)
        overall = _residual_overall_status(
            [bias_status, white_noise_status, outlier_status, structural_break_status],
            observations=observations,
        )
        rows.append(
            {
                "unique_id": uid,
                "model": model,
                "observations": observations,
                "residual_scope": "all_backtest_horizons",
                "mean_error": mean_error,
                "residual_std": residual_std,
                "bias_t_stat": bias_t_stat,
                "bias_status": bias_status,
                "white_noise_residual_scope": acf_scope,
                "white_noise_observations": acf_observations,
                "max_abs_acf": max_abs_acf,
                "acf_threshold_95": threshold,
                "significant_acf_lags": ",".join(str(lag) for lag in significant_lags),
                "significant_acf_lag_count": len(significant_lags),
                "white_noise_status": white_noise_status,
                "outlier_count": outlier_count,
                "outlier_rate": outlier_rate,
                "outlier_status": outlier_status,
                "structural_break_score": break_score,
                "late_vs_early_rmse_ratio": rmse_ratio,
                "structural_break_status": structural_break_status,
                "overall_status": overall,
                "interpretation": _residual_test_interpretation(
                    bias_status=bias_status,
                    white_noise_status=white_noise_status,
                    outlier_status=outlier_status,
                    structural_break_status=structural_break_status,
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=RESIDUAL_TEST_COLUMNS)
    return pd.DataFrame(rows)[RESIDUAL_TEST_COLUMNS].sort_values(
        ["unique_id", "overall_status", "model"]
    ).reset_index(drop=True)


def build_hierarchy_contribution_frame(run: ForecastRun) -> pd.DataFrame:
    """Parent/child contribution view for hierarchy storytelling.

    Gap contribution allocates parent-child incoherence across immediate
    children in proportion to each child's share of sibling forecasts. It is an
    explanatory allocation heuristic, not a reconciliation algorithm.
    """

    forecast = run.forecast.copy()
    required = {"unique_id", "ds", "hierarchy_depth"}
    if forecast.empty or not required.issubset(forecast.columns):
        return pd.DataFrame(columns=HIERARCHY_CONTRIBUTION_COLUMNS)

    forecast["hierarchy_depth"] = pd.to_numeric(forecast["hierarchy_depth"], errors="coerce")
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    root_ids = forecast.loc[forecast["hierarchy_depth"] == 0, "unique_id"].dropna().astype(str).unique()
    root_id = root_ids[0] if len(root_ids) else "Total"
    value_cols = [col for col in ("yhat", "yhat_scenario") if col in forecast.columns]
    rows: list[dict[str, Any]] = []

    for value_col in value_cols:
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(forecast[value_col], errors="coerce")):
            continue
        child = forecast[forecast["hierarchy_depth"] > 0].copy()
        if child.empty:
            continue
        child["parent_unique_id"] = child.apply(lambda row: _hierarchy_parent_id(row, root_id), axis=1)
        child[value_col] = pd.to_numeric(child[value_col], errors="coerce")
        child_sum = child.groupby(["parent_unique_id", "ds"], as_index=False)[value_col].sum(min_count=1)
        child_sum = child_sum.rename(columns={value_col: "child_sum"})
        parent = forecast[["unique_id", "ds", value_col]].rename(
            columns={"unique_id": "parent_unique_id", value_col: "parent_value"}
        )
        merged = child.merge(child_sum, on=["parent_unique_id", "ds"], how="left").merge(
            parent,
            on=["parent_unique_id", "ds"],
            how="left",
        )
        parent_value = pd.to_numeric(merged["parent_value"], errors="coerce")
        child_value = pd.to_numeric(merged[value_col], errors="coerce")
        child_sum_value = pd.to_numeric(merged["child_sum"], errors="coerce")
        gap = parent_value - child_sum_value
        share_parent = child_value / parent_value.where(parent_value != 0)
        share_children = child_value / child_sum_value.where(child_sum_value != 0)
        allocation_weight = share_children
        gap_pct = gap / parent_value.abs().where(parent_value.abs() > 0)
        for idx, row in merged.iterrows():
            rows.append(
                {
                    "value_col": value_col,
                    "ds": row["ds"],
                    "parent_unique_id": row["parent_unique_id"],
                    "child_unique_id": row["unique_id"],
                    "child_hierarchy_level": row.get("hierarchy_level"),
                    "child_hierarchy_depth": row.get("hierarchy_depth"),
                    "parent_value": parent_value.loc[idx],
                    "child_value": child_value.loc[idx],
                    "child_sum": child_sum_value.loc[idx],
                    "child_share_of_parent": share_parent.loc[idx],
                    "child_share_of_children": share_children.loc[idx],
                    "gap_allocation_weight": allocation_weight.loc[idx],
                    "parent_child_gap": gap.loc[idx],
                    "gap_pct": gap_pct.loc[idx],
                    "gap_contribution": gap.loc[idx] * allocation_weight.loc[idx] if pd.notna(allocation_weight.loc[idx]) else None,
                    "gap_contribution_formula": "parent_child_gap * child_value / immediate_child_sum",
                }
            )
    if not rows:
        return pd.DataFrame(columns=HIERARCHY_CONTRIBUTION_COLUMNS)
    return pd.DataFrame(rows)[HIERARCHY_CONTRIBUTION_COLUMNS].sort_values(
        ["value_col", "parent_unique_id", "ds", "child_unique_id"]
    ).reset_index(drop=True)


def build_hierarchy_backtest_comparison(run: ForecastRun) -> pd.DataFrame:
    """Compare selected hierarchy backtests before and after reconciliation."""

    metadata_cols = [col for col in ["unique_id", "hierarchy_level", "hierarchy_depth"] if col in run.history.columns]
    if "hierarchy_depth" not in metadata_cols:
        return pd.DataFrame(columns=HIERARCHY_BACKTEST_COMPARISON_COLUMNS)
    long = build_backtest_long(run)
    if long.empty or "error" not in long.columns:
        return pd.DataFrame(columns=HIERARCHY_BACKTEST_COMPARISON_COLUMNS)
    if "is_selected_model" in long.columns:
        selected = long[long["is_selected_model"].fillna(False).astype(bool)].copy()
    else:
        selected = long.copy()
    if selected.empty:
        selected = long.copy()
    metadata = run.history[metadata_cols].drop_duplicates("unique_id")
    selected = selected.merge(metadata, on="unique_id", how="left")
    selected["hierarchy_depth"] = pd.to_numeric(selected["hierarchy_depth"], errors="coerce")
    selected = selected.dropna(subset=["hierarchy_depth"])
    if selected.empty:
        return pd.DataFrame(columns=HIERARCHY_BACKTEST_COMPARISON_COLUMNS)

    if run.spec.hierarchy_reconciliation == "none":
        return pd.DataFrame(columns=HIERARCHY_BACKTEST_COMPARISON_COLUMNS)
    root_ids = selected.loc[selected["hierarchy_depth"] == 0, "unique_id"].dropna().astype(str).unique()
    if len(root_ids) != 1:
        return pd.DataFrame(columns=HIERARCHY_BACKTEST_COMPARISON_COLUMNS)
    method = run.spec.hierarchy_reconciliation
    requested_method = method
    rows: list[pd.DataFrame] = []
    for (cutoff, model), grp in selected.groupby(["cutoff", "model"], sort=True):
        frame = grp[["unique_id", "ds", "yhat", "hierarchy_level", "hierarchy_depth"]].copy()
        try:
            reconciled, summary, reconciliation_warnings = reconcile_hierarchy_forecast(frame, method=method)
        except Exception as exc:
            rows.append(
                _hierarchy_backtest_failure_frame(
                    grp,
                    cutoff=cutoff,
                    requested_method=requested_method,
                    message=f"hierarchy backtest reconciliation failed: {exc}",
                )
            )
            continue
        applied_method = (
            str(summary["applied_method"].dropna().iloc[0])
            if not summary.empty and "applied_method" in summary.columns and summary["applied_method"].notna().any()
            else method
        )
        comparison_note = "; ".join(reconciliation_warnings)
        merged = grp.merge(
            reconciled[["unique_id", "ds", "yhat"]].rename(columns={"yhat": "yhat_reconciled"}),
            on=["unique_id", "ds"],
            how="left",
        )
        y_actual = pd.to_numeric(merged["y_actual"], errors="coerce")
        yhat_unrec = pd.to_numeric(merged["yhat"], errors="coerce")
        yhat_rec = pd.to_numeric(merged["yhat_reconciled"], errors="coerce")
        error_unrec = y_actual - yhat_unrec
        error_rec = y_actual - yhat_rec
        out = pd.DataFrame(
            {
                "unique_id": merged["unique_id"],
                "model": merged["model"],
                "cutoff": cutoff,
                "ds": merged["ds"],
                "horizon_step": merged.get("horizon_step"),
                "hierarchy_level": merged.get("hierarchy_level"),
                "hierarchy_depth": merged.get("hierarchy_depth"),
                "reconciliation_method": requested_method,
                "applied_method": applied_method,
                "comparison_status": "compared",
                "comparison_note": comparison_note,
                "y_actual": y_actual,
                "yhat_unreconciled": yhat_unrec,
                "yhat_reconciled": yhat_rec,
                "error_unreconciled": error_unrec,
                "error_reconciled": error_rec,
                "abs_error_unreconciled": error_unrec.abs(),
                "abs_error_reconciled": error_rec.abs(),
                "abs_error_delta": error_rec.abs() - error_unrec.abs(),
                "squared_error_unreconciled": error_unrec.pow(2),
                "squared_error_reconciled": error_rec.pow(2),
                "squared_error_delta": error_rec.pow(2) - error_unrec.pow(2),
            }
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=HIERARCHY_BACKTEST_COMPARISON_COLUMNS)
    return pd.concat(rows, ignore_index=True)[HIERARCHY_BACKTEST_COMPARISON_COLUMNS].sort_values(
        ["cutoff", "ds", "hierarchy_depth", "unique_id"]
    ).reset_index(drop=True)


def build_interval_diagnostics(run: ForecastRun) -> pd.DataFrame:
    """Empirical interval coverage and width diagnostics."""

    long = build_backtest_long(run)
    if long.empty:
        return pd.DataFrame(columns=INTERVAL_DIAGNOSTIC_COLUMNS)
    levels = _interval_levels_from_columns(long.columns)
    metadata = _cv_metadata_map(run)
    rows: list[dict[str, Any]] = []
    for level in levels:
        lo_col = f"yhat_lo_{level}"
        hi_col = f"yhat_hi_{level}"
        covered_col = f"covered_{level}"
        miss_col = f"miss_direction_{level}"
        valid = long[long[lo_col].notna() & long[hi_col].notna()].copy()
        if valid.empty:
            continue
        for (uid, model, h), grp in valid.groupby(["unique_id", "model", "horizon_step"], sort=True):
            width = pd.to_numeric(grp[hi_col], errors="coerce") - pd.to_numeric(grp[lo_col], errors="coerce")
            actual = pd.to_numeric(grp["y_actual"], errors="coerce").abs()
            normalized_width = width / actual.mask(actual == 0)
            covered = grp[covered_col].dropna().astype(bool)
            miss = grp[miss_col].astype(str)
            empirical = float(covered.mean()) if len(covered) else None
            target = float(level) / 100.0
            coverage_gap = empirical - target if empirical is not None else None
            observations = int(len(grp))
            coverage_status = _coverage_status(coverage_gap, observations)
            meta = metadata.get((str(uid), str(model)), {})
            rows.append(
                {
                    "unique_id": uid,
                    "model": model,
                    "level": int(level),
                    "horizon_step": int(h),
                    "target_coverage": target,
                    "empirical_coverage": empirical,
                    "coverage_gap": coverage_gap,
                    "coverage_status": coverage_status,
                    "interval_status": _coverage_status_to_interval_status(coverage_status),
                    "interval_method": _interval_method(str(model), has_interval=True),
                    "requested_horizon": meta.get("requested_horizon"),
                    "selection_horizon": meta.get("selection_horizon"),
                    "cv_windows": meta.get("cv_windows"),
                    "cv_step_size": meta.get("cv_step_size"),
                    "cv_horizon_matches_requested": meta.get("cv_horizon_matches_requested"),
                    "avg_width": float(width.mean()) if width.notna().any() else None,
                    "normalized_width": float(normalized_width.mean()) if normalized_width.notna().any() else None,
                    "miss_above_rate": float((miss == "above").mean()) if len(miss) else None,
                    "miss_below_rate": float((miss == "below").mean()) if len(miss) else None,
                    "observations": observations,
                }
            )
    if not rows:
        return pd.DataFrame(columns=INTERVAL_DIAGNOSTIC_COLUMNS)
    return pd.DataFrame(rows)[INTERVAL_DIAGNOSTIC_COLUMNS].sort_values(
        ["unique_id", "model", "level", "horizon_step"]
    ).reset_index(drop=True)


TRUST_SUMMARY_COLUMNS = [
    "unique_id",
    "trust_level",
    "trust_score_0_100",
    "selected_model",
    "primary_metric",
    "primary_metric_value",
    "backtest_observations",
    "requested_horizon",
    "selection_horizon",
    "cv_windows",
    "cv_step_size",
    "cv_horizon_matches_requested",
    "horizon_trust_state",
    "validated_through_horizon",
    "unvalidated_horizon_start",
    "unvalidated_steps",
    "full_horizon_claim_allowed",
    "planning_ready_horizon",
    "horizon_trust_score_cap",
    "horizon_gate_result",
    "history_rows",
    "history_readiness",
    "cv_horizon_status",
    "naive_skill",
    "residual_stability",
    "interval_coverage_gap",
    "interval_status",
    "interval_method",
    "seasonality_status",
    "hierarchy_status",
    "event_status",
    "data_quality_issues",
    "score_deductions",
    "caveats",
    "next_actions",
]


def build_trust_summary(run: ForecastRun) -> pd.DataFrame:
    """Decision-ready trust and next-action summary, one row per series."""

    series_ids = _series_ids_for_trust(run)
    if not series_ids:
        return pd.DataFrame(columns=TRUST_SUMMARY_COLUMNS)

    selection = run.model_selection.copy()
    if not selection.empty:
        selection["unique_id"] = selection["unique_id"].astype(str)
    metrics = run.backtest_metrics.copy()
    if not metrics.empty:
        metrics["unique_id"] = metrics["unique_id"].astype(str)
        metrics["model"] = metrics["model"].astype(str)
    residuals = build_residual_diagnostics(run)
    intervals = build_interval_diagnostics(run)
    seasonality = seasonality_diagnostics_frame(run)
    coherence = hierarchy_coherence(run.forecast)
    profile_map = {str(profile.unique_id): profile for profile in run.profile.series}
    history_rows = run.history.groupby("unique_id", sort=True).size().astype(int).to_dict()
    history_mean_abs = (
        run.history.assign(abs_y=pd.to_numeric(run.history["y"], errors="coerce").abs())
        .groupby("unique_id", sort=True)["abs_y"]
        .mean()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for uid in series_ids:
        selected_row = _selection_row(selection, uid)
        selected_model = str(selected_row.get("selected_model") or "")
        primary_metric, primary_value = _primary_metric(selected_row)
        observed = _int_or_none(selected_row.get("observations"))
        requested_horizon = _int_or_none(selected_row.get("requested_horizon"))
        selection_horizon = _int_or_none(selected_row.get("selection_horizon"))
        cv_windows = _int_or_none(selected_row.get("cv_windows"))
        cv_step_size = _int_or_none(selected_row.get("cv_step_size"))
        cv_horizon_matches = _bool_or_none(selected_row.get("cv_horizon_matches_requested"))
        history_count = int(history_rows.get(uid, 0))
        profile = profile_map.get(uid)
        history_readiness = _history_readiness(history_count, run.spec.horizon, run.profile.season_length, profile.readiness if profile else "")
        cv_horizon_status = _cv_horizon_status(selected_row)
        naive_skill = _naive_skill(uid, selected_model, primary_metric, primary_value, metrics)
        horizon_gate = _horizon_gate(
            selected_row,
            requested_horizon=run.spec.horizon,
            observed=observed,
            cv_windows=cv_windows,
            naive_skill=naive_skill,
        )
        residual_stability = _assess_residual_stability(uid, selected_model, residuals)
        interval_gap = _interval_coverage_gap(uid, selected_model, intervals)
        interval_status = _interval_status(uid, selected_model, intervals, run)
        interval_method = _interval_method(selected_model, has_interval=_selected_future_has_intervals(run, uid))
        seasonality_status = _seasonality_status(uid, seasonality)
        hierarchy_status = _hierarchy_status(uid, coherence)
        event_status = "scenario_events_applied" if run.spec.events else "statistical_baseline_only"
        data_quality_issues = _data_quality_issues(profile)
        caveats = _build_trust_caveats(
            uid=uid,
            selected_model=selected_model,
            primary_metric=primary_metric,
            primary_value=primary_value,
            observed=observed,
            cv_windows=cv_windows,
            history_readiness=history_readiness,
            history_rows=history_count,
            season_length=run.profile.season_length,
            cv_horizon_status=cv_horizon_status,
            requested_horizon=requested_horizon,
            selection_horizon=selection_horizon,
            naive_skill=naive_skill,
            residual_stability=residual_stability,
            interval_gap=interval_gap,
            interval_status=interval_status,
            seasonality_status=seasonality_status,
            hierarchy_status=hierarchy_status,
            event_status=event_status,
            data_quality_issues=data_quality_issues,
            horizon_gate=horizon_gate,
        )
        score, level, deductions = _compute_trust_score(
            backtest_count=observed,
            cv_windows=cv_windows,
            primary_metric=primary_metric,
            primary_value=primary_value,
            history_readiness=history_readiness,
            cv_horizon_status=cv_horizon_status,
            naive_skill=naive_skill,
            residual_stability=residual_stability,
            interval_gap=interval_gap,
            interval_status=interval_status,
            seasonality_status=seasonality_status,
            hierarchy_status=hierarchy_status,
            data_quality_issues=data_quality_issues,
            horizon_score_cap=horizon_gate["horizon_trust_score_cap"],
            horizon_gate_result=horizon_gate["horizon_gate_result"],
        )
        actions = _recommend_trust_actions(
            level=level,
            caveats=caveats,
            history_readiness=history_readiness,
            cv_horizon_status=cv_horizon_status,
            naive_skill=naive_skill,
            residual_stability=residual_stability,
            interval_gap=interval_gap,
            interval_status=interval_status,
            seasonality_status=seasonality_status,
            hierarchy_status=hierarchy_status,
            event_status=event_status,
            horizon_gate=horizon_gate,
        )
        rows.append(
            {
                "unique_id": uid,
                "trust_level": level,
                "trust_score_0_100": score,
                "selected_model": selected_model,
                "primary_metric": primary_metric,
                "primary_metric_value": primary_value,
                "backtest_observations": observed or 0,
                "requested_horizon": requested_horizon or run.spec.horizon,
                "selection_horizon": selection_horizon,
                "cv_windows": cv_windows or 0,
                "cv_step_size": cv_step_size,
                "cv_horizon_matches_requested": cv_horizon_matches,
                "horizon_trust_state": horizon_gate["horizon_trust_state"],
                "validated_through_horizon": horizon_gate["validated_through_horizon"],
                "unvalidated_horizon_start": horizon_gate["unvalidated_horizon_start"],
                "unvalidated_steps": horizon_gate["unvalidated_steps"],
                "full_horizon_claim_allowed": horizon_gate["full_horizon_claim_allowed"],
                "planning_ready_horizon": horizon_gate["planning_ready_horizon"],
                "horizon_trust_score_cap": horizon_gate["horizon_trust_score_cap"],
                "horizon_gate_result": horizon_gate["horizon_gate_result"],
                "history_rows": history_count,
                "history_readiness": history_readiness,
                "cv_horizon_status": cv_horizon_status,
                "naive_skill": naive_skill,
                "residual_stability": residual_stability,
                "interval_coverage_gap": interval_gap,
                "interval_status": interval_status,
                "interval_method": interval_method,
                "seasonality_status": seasonality_status,
                "hierarchy_status": hierarchy_status,
                "event_status": event_status,
                "data_quality_issues": "; ".join(data_quality_issues),
                "score_deductions": "; ".join(deductions),
                "caveats": "; ".join(caveats),
                "next_actions": "; ".join(actions[:3]),
            }
        )
    return pd.DataFrame(rows, columns=TRUST_SUMMARY_COLUMNS).sort_values(
        ["trust_score_0_100", "unique_id"],
        ascending=[True, True],
    ).reset_index(drop=True)


def _coverage_status(coverage_gap: float | None, observations: int) -> str:
    if coverage_gap is None:
        return "unavailable"
    if observations < 5:
        return "insufficient_observations"
    if coverage_gap < -0.10:
        return "fail_undercoverage"
    if abs(coverage_gap) > 0.10:
        return "warn_miscalibrated"
    return "pass"


def _one_step_residuals_for_acf(group: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return residuals for autocorrelation screening.

    Prefer one-step rolling-origin residuals because they are the clearest
    serial-correlation diagnostic for one-ahead forecast errors. The
    all-horizon fallback is weaker and should be read as directional.
    """

    if "horizon_step" in group.columns:
        one_step = group[pd.to_numeric(group["horizon_step"], errors="coerce") == 1]
        errors = pd.to_numeric(one_step.get("error"), errors="coerce").dropna()
        if not errors.empty:
            return errors, "horizon_step_1"
    return pd.to_numeric(group.get("error"), errors="coerce").dropna(), "all_horizons_fallback"


def _bias_t_stat(errors: pd.Series) -> float | None:
    observations = int(len(errors))
    if observations < 2:
        return None
    std = float(errors.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return 0.0
    return float(errors.mean() / (std / math.sqrt(observations)))


def _bias_status(t_stat: float | None, observations: int) -> str:
    """Classify directional residual bias.

    The 2.0/3.0 cutoffs are empirical screening thresholds for short
    rolling-origin samples, not formal hypothesis-test alpha levels.
    """

    if observations < 5 or t_stat is None:
        return "insufficient"
    abs_t = abs(t_stat)
    if abs_t >= 3.0:
        return "fail"
    if abs_t >= 2.0:
        return "warn"
    return "pass"


def _residual_acf(errors: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    values = errors.dropna().to_numpy(dtype="float64")
    n = len(values)
    if n < 3:
        return pd.DataFrame(columns=["lag", "acf", "threshold", "significant"])
    centered = values - float(values.mean())
    denom = float((centered ** 2).sum())
    if denom <= 0:
        return pd.DataFrame(columns=["lag", "acf", "threshold", "significant"])
    threshold = 1.96 / math.sqrt(n)
    rows = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        acf = float((centered[lag:] * centered[:-lag]).sum() / denom)
        rows.append({"lag": lag, "acf": acf, "threshold": threshold, "significant": abs(acf) > threshold})
    return pd.DataFrame(rows)


def _white_noise_status(observations: int, significant_lags: list[int]) -> str:
    """Classify residual autocorrelation as a screening heuristic.

    Three or more individually significant ACF lags flags likely serial
    dependence; one or two lags warn. This is not a Ljung-Box replacement.
    """

    if observations < 8:
        return "insufficient"
    if len(significant_lags) >= 3:
        return "fail"
    if significant_lags:
        return "warn"
    return "pass"


def _residual_outlier_stats(errors: pd.Series) -> tuple[int, float | None]:
    observations = int(len(errors))
    if observations < 2:
        return 0, None
    std = float(errors.std(ddof=0))
    if not math.isfinite(std) or std <= 0:
        return 0, 0.0
    standardized = (errors - float(errors.mean())) / std
    count = int((standardized.abs() >= 3.0).sum())
    return count, float(count / observations)


def _outlier_status(observations: int, outlier_rate: float | None) -> str:
    if observations < 8 or outlier_rate is None:
        return "insufficient"
    if outlier_rate >= 0.10:
        return "fail"
    if outlier_rate > 0:
        return "warn"
    return "pass"


def _structural_break_scores(errors: pd.Series) -> tuple[float | None, float | None]:
    observations = int(len(errors))
    if observations < 8:
        return None, None
    midpoint = observations // 2
    early = errors.iloc[:midpoint]
    late = errors.iloc[midpoint:]
    if early.empty or late.empty:
        return None, None
    pooled_std = float(errors.std(ddof=0))
    mean_shift = abs(float(late.mean()) - float(early.mean()))
    break_score = mean_shift / pooled_std if math.isfinite(pooled_std) and pooled_std > 0 else 0.0
    early_rmse = math.sqrt(float((early ** 2).mean()))
    late_rmse = math.sqrt(float((late ** 2).mean()))
    if not math.isfinite(early_rmse) or early_rmse <= 0:
        rmse_ratio = None
    else:
        rmse_ratio = float(late_rmse / early_rmse)
    return float(break_score), rmse_ratio


def _structural_break_status(observations: int, break_score: float | None, rmse_ratio: float | None) -> str:
    """Classify early/late residual shifts as empirical screening signals.

    Break-score and RMSE-ratio cutoffs are pragmatic finance QA thresholds for
    short rolling-origin samples, not formal structural-break tests.
    """

    if observations < 8 or break_score is None:
        return "insufficient"
    ratio = rmse_ratio if rmse_ratio is not None else 1.0
    if break_score >= 2.0 or ratio >= 2.5:
        return "fail"
    if break_score >= 1.2 or ratio >= 1.75:
        return "warn"
    return "pass"


def _residual_overall_status(statuses: list[str], *, observations: int) -> str:
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    if observations < 8 or all(status == "insufficient" for status in statuses):
        return "insufficient"
    return "pass"


def _residual_test_interpretation(
    *,
    bias_status: str,
    white_noise_status: str,
    outlier_status: str,
    structural_break_status: str,
) -> str:
    issues: list[str] = []
    if bias_status in {"warn", "fail"}:
        issues.append("directional bias")
    if white_noise_status in {"warn", "fail"}:
        issues.append("residual autocorrelation")
    if outlier_status in {"warn", "fail"}:
        issues.append("large residual outliers")
    if structural_break_status in {"warn", "fail"}:
        issues.append("early/late residual shift")
    if issues:
        return "Review " + ", ".join(issues) + " before treating the model as stable."
    if "insufficient" in {bias_status, white_noise_status, outlier_status, structural_break_status}:
        return "Residual sample is small; these heuristic checks are directional screening tools, not formal model adequacy certification."
    return "Residual checks did not flag bias, autocorrelation, outliers, or early/late shift."


def _hierarchy_parent_id(row: pd.Series, root_id: str) -> str:
    """Return the immediate parent id for aggregate_hierarchy_frame-style nodes."""

    depth = int(row.get("hierarchy_depth", 0))
    if depth <= 1:
        return root_id
    unique_id = str(row["unique_id"])
    if "|" not in unique_id:
        raise ValueError(
            "hierarchy contribution requires pipe-delimited unique_id ancestors "
            "for nodes with hierarchy_depth > 1; build hierarchy nodes with aggregate_hierarchy_frame"
        )
    return unique_id.rsplit("|", 1)[0]


def _hierarchy_backtest_failure_frame(
    group: pd.DataFrame,
    *,
    cutoff: Any,
    requested_method: str,
    message: str,
) -> pd.DataFrame:
    y_actual = pd.to_numeric(group.get("y_actual"), errors="coerce")
    yhat_unrec = pd.to_numeric(group.get("yhat"), errors="coerce")
    error_unrec = y_actual - yhat_unrec
    return pd.DataFrame(
        {
            "unique_id": group.get("unique_id"),
            "model": group.get("model"),
            "cutoff": cutoff,
            "ds": group.get("ds"),
            "horizon_step": group.get("horizon_step"),
            "hierarchy_level": group.get("hierarchy_level"),
            "hierarchy_depth": group.get("hierarchy_depth"),
            "reconciliation_method": requested_method,
            "applied_method": None,
            "comparison_status": "reconciliation_failed",
            "comparison_note": message,
            "y_actual": y_actual,
            "yhat_unreconciled": yhat_unrec,
            "yhat_reconciled": None,
            "error_unreconciled": error_unrec,
            "error_reconciled": None,
            "abs_error_unreconciled": error_unrec.abs(),
            "abs_error_reconciled": None,
            "abs_error_delta": None,
            "squared_error_unreconciled": error_unrec.pow(2),
            "squared_error_reconciled": None,
            "squared_error_delta": None,
        }
    )


def _series_ids_for_trust(run: ForecastRun) -> list[str]:
    ids: set[str] = set()
    for frame in (run.model_selection, run.forecast, run.history):
        if "unique_id" in frame.columns:
            ids.update(frame["unique_id"].dropna().astype(str).tolist())
    return sorted(ids)


def _selection_row(selection: pd.DataFrame, uid: str) -> dict[str, Any]:
    if selection.empty or "unique_id" not in selection.columns:
        return {"unique_id": uid}
    rows = selection[selection["unique_id"].astype(str) == uid]
    if rows.empty:
        return {"unique_id": uid}
    return rows.iloc[0].to_dict()


def _primary_metric(row: dict[str, Any]) -> tuple[str, float | None]:
    for metric in ("mase", "rmsse", "wape", "rmse", "mae"):
        value = _safe_float(row.get(metric))
        if value is not None:
            return metric, value
    return "unavailable", None


def _int_or_none(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)


def _history_readiness(history_rows: int, horizon: int, season_length: int, profile_readiness: str) -> str:
    if profile_readiness in {"insufficient", "limited"} or history_rows < max(2, horizon):
        return "short"
    if history_rows < max(2 * horizon, horizon + 2):
        return "short"
    if history_rows < max(5 * horizon, 2 * max(1, season_length)):
        return "minimal"
    if profile_readiness == "rich":
        return "rich"
    return "adequate"


def _cv_horizon_status(row: dict[str, Any]) -> str:
    selected = _safe_float(row.get("selection_horizon"))
    requested = _safe_float(row.get("requested_horizon"))
    if selected is None or requested is None:
        return "unavailable"
    if int(selected) == int(requested):
        return "matches_requested"
    if selected < requested:
        return "shorter_than_requested"
    return "longer_than_requested"


def _horizon_gate(
    row: dict[str, Any],
    *,
    requested_horizon: int,
    observed: int | None,
    cv_windows: int | None,
    naive_skill: float | None,
) -> dict[str, Any]:
    requested = _int_or_none(row.get("requested_horizon")) or int(requested_horizon)
    selected = _int_or_none(row.get("selection_horizon"))
    windows = cv_windows if cv_windows is not None else _int_or_none(row.get("cv_windows"))
    has_evidence = bool(observed) and bool(windows) and selected is not None and selected > 0

    if not has_evidence:
        state = "no_rolling_origin_evidence"
        validated = 0
        unvalidated_start = 1
        unvalidated_steps = requested
        full_claim = False
        cap = 35
        result = "fail_no_rolling_origin_validation"
    elif selected >= requested:
        state = "full_horizon_validated"
        validated = requested
        unvalidated_start = None
        unvalidated_steps = 0
        full_claim = (windows or 0) >= 2
        cap = None
        result = "passed"
    else:
        state = "partial_horizon_validated"
        validated = selected
        unvalidated_start = selected + 1
        unvalidated_steps = max(0, requested - selected)
        full_claim = False
        ratio = selected / requested if requested else 0.0
        if ratio < 0.5:
            cap = 45
            result = "fail_less_than_half_horizon_validated"
        else:
            cap = 60
            result = "warning_partial_horizon_validation"

    if has_evidence and (windows or 0) < 2:
        cap = min(cap, 65) if cap is not None else 65
        if result == "passed":
            result = "warning_single_cv_window"

    if naive_skill is not None and naive_skill < 0:
        cap = min(cap, 35) if cap is not None else 35
    elif naive_skill is not None and naive_skill < 0.05:
        cap = min(cap, 65) if cap is not None else 65

    return {
        "horizon_trust_state": state,
        "validated_through_horizon": validated,
        "unvalidated_horizon_start": unvalidated_start,
        "unvalidated_steps": unvalidated_steps,
        "full_horizon_claim_allowed": full_claim,
        "planning_ready_horizon": validated,
        "horizon_trust_score_cap": cap,
        "horizon_gate_result": result,
    }


def _horizon_row_contract(
    horizon_step: int,
    metadata: dict[str, Any],
    *,
    requested_horizon: int,
) -> dict[str, Any]:
    requested = _int_or_none(metadata.get("requested_horizon")) or int(requested_horizon)
    selected = _int_or_none(metadata.get("selection_horizon"))
    windows = _int_or_none(metadata.get("cv_windows"))
    if selected is None or selected <= 0 or not windows:
        state = "no_rolling_origin_evidence"
        validated = 0
        is_beyond = True
        eligible = False
        label = "No out-of-sample validation"
        warning = "no rolling-origin validation evidence; do not claim a validated champion"
    elif horizon_step > selected:
        state = "beyond_validated_horizon"
        validated = selected
        is_beyond = True
        eligible = False
        label = f"Directional beyond CV horizon {selected} of {requested}"
        warning = f"forecast step {horizon_step} is beyond CV horizon {selected} of requested {requested}; directional only"
    elif selected >= requested:
        state = "full_horizon_validated"
        validated = requested
        is_beyond = False
        eligible = windows >= 2
        if eligible:
            label = f"Validated through requested horizon {requested}"
            warning = f"validated through requested horizon {requested} with {windows} rolling-origin window(s)"
        else:
            label = f"Full horizon evaluated on only {windows} rolling-origin window(s)"
            warning = (
                f"validated through requested horizon {requested}, but only {windows} rolling-origin window(s) "
                "are available; planning-ready champion claim is limited"
            )
    else:
        state = "partial_horizon_validated"
        validated = selected
        is_beyond = False
        eligible = windows >= 2
        label = f"Validated only through step {selected} of {requested}"
        warning = f"validated through CV horizon {selected} of requested {requested}; later steps are directional"

    if eligible:
        if state == "full_horizon_validated":
            eligibility_reason = (
                f"Passes horizon-validation gate: requested horizon {requested} validated "
                f"with {windows} rolling-origin window(s)"
            )
        else:
            eligibility_reason = (
                f"Passes row horizon gate: step {horizon_step} is within validated CV horizon "
                f"{selected} of requested {requested} with {windows} rolling-origin window(s); "
                "later steps remain directional"
            )
    else:
        eligibility_reason = warning

    return {
        "row_horizon_status": state,
        "horizon_trust_state": state,
        "horizon_trust_label": label,
        "validated_through_horizon": validated,
        "is_beyond_validated_horizon": is_beyond,
        "planning_eligible": eligible,
        "planning_eligibility_scope": "horizon_validation_only",
        "planning_eligibility_reason": eligibility_reason,
        "horizon_warning": warning,
        "forecast_horizon_status": state,
        "horizon_validated_in_cv": (not is_beyond and state != "no_rolling_origin_evidence"),
        "validation_evidence": warning,
    }


def _naive_skill(
    uid: str,
    selected_model: str,
    metric: str,
    selected_value: float | None,
    metrics: pd.DataFrame,
) -> float | None:
    if selected_value is None or metric == "unavailable" or metrics.empty or metric not in metrics.columns:
        return None
    series_metrics = metrics[metrics["unique_id"].astype(str) == uid].copy()
    if series_metrics.empty:
        return None
    benchmark_name = "SeasonalNaive" if (series_metrics["model"].astype(str) == "SeasonalNaive").any() else "Naive"
    benchmark = series_metrics[series_metrics["model"].astype(str) == benchmark_name]
    if benchmark.empty:
        return None
    benchmark_value = _safe_float(benchmark.iloc[0].get(metric))
    if benchmark_value is None or benchmark_value <= 0:
        return None
    if selected_model == benchmark_name:
        selected_value = benchmark_value
    return 1.0 - (selected_value / benchmark_value)


def _assess_residual_stability(uid: str, selected_model: str, residuals: pd.DataFrame) -> str:
    if residuals.empty or "unique_id" not in residuals.columns:
        return "unavailable"
    frame = residuals[residuals["unique_id"].astype(str) == uid].copy()
    if selected_model and "model" in frame.columns:
        selected = frame[frame["model"].astype(str) == selected_model]
        if not selected.empty:
            frame = selected
    if frame.empty:
        return "unavailable"
    observations = pd.to_numeric(frame.get("observations"), errors="coerce")
    if observations.notna().sum() == 0 or observations.sum() < 2:
        return "insufficient"
    rmse = pd.to_numeric(frame.get("rmse"), errors="coerce").dropna()
    mean_error = pd.to_numeric(frame.get("mean_error"), errors="coerce").dropna()
    bias = pd.to_numeric(frame.get("bias"), errors="coerce").dropna()
    positive_rmse = rmse[rmse > 0]
    if len(positive_rmse) >= 2 and float(positive_rmse.max() / positive_rmse.min()) > 3:
        return "volatile"
    max_bias = float(bias.abs().max()) if not bias.empty else 0.0
    if max_bias > 0.15:
        return "drift"
    if not mean_error.empty and not positive_rmse.empty:
        ratio = float(mean_error.abs().max() / positive_rmse.max())
        if ratio > 0.5:
            return "drift"
    return "stable"


def _interval_coverage_gap(uid: str, selected_model: str, intervals: pd.DataFrame) -> float | None:
    if selected_model == "WeightedEnsemble":
        return None
    if intervals.empty or "unique_id" not in intervals.columns or "coverage_gap" not in intervals.columns:
        return None
    frame = intervals[intervals["unique_id"].astype(str) == uid].copy()
    if selected_model and "model" in frame.columns:
        selected = frame[frame["model"].astype(str) == selected_model]
        if selected.empty:
            return None
        frame = selected
    gaps = pd.to_numeric(frame.get("coverage_gap"), errors="coerce").abs().dropna()
    if gaps.empty:
        return None
    return float(gaps.max())


def _interval_status(uid: str, selected_model: str, intervals: pd.DataFrame, run: ForecastRun) -> str:
    if selected_model == "WeightedEnsemble":
        return "point_only_ensemble"
    if _selected_future_has_intervals(run, uid) and _selected_forecast_adjusted(run, uid, selected_model):
        return "adjusted_not_recalibrated"
    if intervals.empty or "unique_id" not in intervals.columns:
        return "future_only" if _selected_future_has_intervals(run, uid) else "unavailable"
    selected_meta = _selection_row(run.model_selection.copy(), uid)
    requested_horizon = _int_or_none(selected_meta.get("requested_horizon")) or run.spec.horizon
    selection_horizon = _int_or_none(selected_meta.get("selection_horizon"))
    frame = intervals[intervals["unique_id"].astype(str) == uid].copy()
    if selected_model and "model" in frame.columns:
        selected = frame[frame["model"].astype(str) == selected_model]
        if selected.empty:
            return "future_only" if _selected_future_has_intervals(run, uid) else "unavailable"
        frame = selected
    if frame.empty or "coverage_status" not in frame.columns:
        return "future_only" if _selected_future_has_intervals(run, uid) else "unavailable"
    if selection_horizon is not None and selection_horizon < requested_horizon and _selected_future_has_intervals(run, uid):
        return "future_only"
    statuses = set(frame.get("interval_status", pd.Series(dtype=str)).dropna().astype(str))
    if not statuses:
        statuses = {_coverage_status_to_interval_status(status) for status in frame["coverage_status"].dropna().astype(str)}
    if "calibration_fail" in statuses:
        return "calibration_fail"
    if "calibration_warning" in statuses:
        return "calibration_warning"
    if statuses == {"insufficient_observations"}:
        return "insufficient_observations"
    if "calibrated" in statuses:
        return "calibrated"
    return "future_only" if _selected_future_has_intervals(run, uid) else "unavailable"


def _seasonality_status(uid: str, seasonality: pd.DataFrame) -> str:
    if seasonality.empty or "unique_id" not in seasonality.columns or "credibility_label" not in seasonality.columns:
        return "unavailable"
    rows = seasonality[seasonality["unique_id"].astype(str) == uid]
    if rows.empty:
        return "unavailable"
    return str(rows.iloc[0].get("credibility_label") or "unavailable")


def _hierarchy_status(uid: str, coherence: pd.DataFrame) -> str:
    if coherence.empty or "parent_unique_id" not in coherence.columns:
        return "no_hierarchy"
    frame = coherence[coherence["parent_unique_id"].astype(str) == uid].copy()
    if frame.empty:
        return "leaf_or_unchecked"
    gaps = pd.to_numeric(frame.get("gap_pct"), errors="coerce").abs().dropna()
    if gaps.empty:
        return "unchecked"
    max_gap = float(gaps.max())
    if max_gap <= 0.01:
        return "coherent"
    if max_gap <= 0.05:
        return "minor_gap"
    return "incoherent"


def _data_quality_issues(profile: Any | None) -> list[str]:
    if profile is None:
        return []
    issues = list(profile.warnings)
    if getattr(profile, "missing_timestamps", 0):
        issues.append(f"{profile.missing_timestamps} missing timestamp(s)")
    if getattr(profile, "null_y", 0):
        issues.append(f"{profile.null_y} null y value(s)")
    if getattr(profile, "negative_y", 0):
        issues.append(f"{profile.negative_y} negative y value(s)")
    return list(dict.fromkeys(issues))


def _build_trust_caveats(
    *,
    uid: str,
    selected_model: str,
    primary_metric: str,
    primary_value: float | None,
    observed: int | None,
    cv_windows: int | None,
    history_readiness: str,
    history_rows: int,
    season_length: int,
    cv_horizon_status: str,
    requested_horizon: int | None,
    selection_horizon: int | None,
    naive_skill: float | None,
    residual_stability: str,
    interval_gap: float | None,
    interval_status: str,
    seasonality_status: str,
    hierarchy_status: str,
    event_status: str,
    data_quality_issues: list[str],
    horizon_gate: dict[str, Any],
) -> list[str]:
    caveats: list[str] = []
    if history_readiness == "short":
        caveats.append(f"history is short relative to horizon ({history_rows} rows)")
    elif history_readiness == "minimal":
        caveats.append(f"history is minimal relative to horizon ({history_rows} rows)")
    if season_length > 1 and history_rows < 2 * season_length:
        caveats.append("fewer than two full seasonal cycles; seasonal claims need review")
    if not observed:
        caveats.append("no backtest observations available")
    elif (cv_windows or 0) < 2:
        caveats.append("very few rolling-origin windows")
    if cv_horizon_status == "shorter_than_requested":
        if requested_horizon and selection_horizon:
            caveats.append(f"validated through CV horizon {selection_horizon} of requested {requested_horizon}")
        else:
            caveats.append("model was selected on a shorter CV horizon than requested")
    if horizon_gate.get("horizon_trust_state") == "no_rolling_origin_evidence":
        caveats.append("no rolling-origin validation evidence; do not call this a validated champion forecast")
    elif not horizon_gate.get("full_horizon_claim_allowed", False):
        validated = horizon_gate.get("validated_through_horizon")
        requested = requested_horizon
        if horizon_gate.get("horizon_trust_state") == "partial_horizon_validated" and validated and requested:
            caveats.append(f"full-horizon champion claim is not allowed; validation supports horizon {validated} of {requested}")
        elif horizon_gate.get("horizon_gate_result") == "warning_single_cv_window":
            caveats.append("full requested horizon was evaluated on only one rolling-origin window; champion claim remains limited")
    if primary_metric == "unavailable" or primary_value is None:
        caveats.append("selected model has no usable validation metric")
    if naive_skill is not None and naive_skill < 0:
        caveats.append("selected model underperformed the naive benchmark")
    elif naive_skill is not None and naive_skill < 0.05:
        caveats.append("selected model barely beat the naive benchmark")
    if residual_stability == "drift":
        caveats.append("residuals show directional bias")
    elif residual_stability == "volatile":
        caveats.append("residual errors vary sharply by horizon")
    elif residual_stability in {"unavailable", "insufficient"}:
        caveats.append("residual diagnostics are limited")
    if interval_status == "point_only_ensemble":
        caveats.append("selected WeightedEnsemble is point-only; prediction intervals are not in scope")
    elif interval_status == "future_only":
        caveats.append("future interval bands are not empirically calibrated in rolling-origin CV")
    elif interval_status == "adjusted_not_recalibrated":
        caveats.append("interval bands were adjusted after calibration and were not recalibrated")
    elif interval_status == "unavailable":
        caveats.append("prediction interval calibration is unavailable")
    elif interval_status in {"calibration_fail", "calibration_warning"} and interval_gap is not None:
        caveats.append(f"prediction intervals are miscalibrated (max gap {interval_gap:.0%})")
    if seasonality_status == "low":
        caveats.append("seasonality evidence is weak or has too few full cycles")
    if hierarchy_status == "incoherent":
        caveats.append("hierarchy parent/child forecasts are materially incoherent")
    elif hierarchy_status == "minor_gap":
        caveats.append("hierarchy parent/child forecasts have small coherence gaps")
    if event_status == "statistical_baseline_only":
        caveats.append("no known future events or drivers were applied")
    if _is_simple_benchmark(selected_model):
        caveats.append("selected model is a simple benchmark")
    caveats.extend(data_quality_issues[:3])
    return list(dict.fromkeys(caveats))


def _compute_trust_score(
    *,
    backtest_count: int | None,
    cv_windows: int | None,
    primary_metric: str,
    primary_value: float | None,
    history_readiness: str,
    cv_horizon_status: str,
    naive_skill: float | None,
    residual_stability: str,
    interval_gap: float | None,
    interval_status: str,
    seasonality_status: str,
    hierarchy_status: str,
    data_quality_issues: list[str],
    horizon_score_cap: int | None,
    horizon_gate_result: str,
) -> tuple[int, str, list[str]]:
    score = 100
    deductions: list[str] = []

    def deduct(points: int, reason: str) -> None:
        nonlocal score
        if points <= 0:
            return
        score -= points
        deductions.append(f"-{points} {reason}")

    if not backtest_count or not cv_windows:
        deduct(25, "no rolling-origin validation")
    elif cv_windows == 1:
        deduct(15, "only one rolling-origin window")
    elif cv_windows == 2:
        deduct(6, "limited rolling-origin windows")

    deduct(_metric_penalty(primary_metric, primary_value), f"{primary_metric} validation quality")

    if history_readiness == "short":
        deduct(25, "short history")
    elif history_readiness == "minimal":
        deduct(12, "minimal history")

    if cv_horizon_status == "shorter_than_requested":
        deduct(15, "CV horizon shorter than forecast horizon")
    elif cv_horizon_status == "unavailable":
        deduct(4, "CV horizon contract unavailable")

    if naive_skill is None:
        deduct(4, "naive skill unavailable")
    elif naive_skill < 0:
        deduct(15, "underperformed naive benchmark")
    elif naive_skill < 0.05:
        deduct(8, "thin naive benchmark win")

    if residual_stability == "volatile":
        deduct(15, "volatile residual diagnostics")
    elif residual_stability == "drift":
        deduct(10, "residual bias")
    elif residual_stability == "insufficient":
        deduct(5, "insufficient residual diagnostics")
    elif residual_stability == "unavailable":
        deduct(8, "residual diagnostics unavailable")

    if interval_status == "calibration_fail":
        deduct(15, "interval undercoverage")
    elif interval_status == "calibration_warning" or (interval_gap is not None and interval_gap > 0.10):
        deduct(8, "interval calibration warning")
    elif interval_status == "insufficient_observations":
        deduct(6, "limited interval calibration observations")
    elif interval_status == "point_only_ensemble":
        deduct(8, "ensemble is point-only")
    elif interval_status == "future_only":
        deduct(12, "future intervals lack CV calibration")
    elif interval_status == "adjusted_not_recalibrated":
        deduct(10, "intervals adjusted after calibration")
    elif interval_status == "unavailable":
        deduct(8, "interval calibration unavailable")

    if seasonality_status == "low":
        deduct(4, "weak seasonality evidence")
    elif seasonality_status == "unavailable":
        deduct(2, "seasonality evidence unavailable")

    if hierarchy_status == "incoherent":
        deduct(10, "hierarchy incoherence")
    elif hierarchy_status == "minor_gap":
        deduct(4, "minor hierarchy coherence gap")

    deduct(min(12, len(data_quality_issues) * 3), "data quality caveats")

    if horizon_score_cap is not None and score > horizon_score_cap:
        score = horizon_score_cap
        deductions.append(f"cap {horizon_score_cap} horizon validation gate ({horizon_gate_result})")

    score = max(0, min(100, int(round(score))))
    level = "High" if score >= 75 else "Medium" if score >= 40 else "Low"
    return score, level, deductions


def _metric_penalty(metric: str, value: float | None) -> int:
    if value is None or metric == "unavailable":
        return 10
    if metric in {"mase", "rmsse"}:
        if value <= 1.0:
            return 0
        if value <= 1.5:
            return 8
        if value <= 2.0:
            return 15
        return 25
    if metric == "wape":
        if value <= 0.10:
            return 0
        if value <= 0.20:
            return 5
        if value <= 0.35:
            return 12
        if value <= 0.50:
            return 18
        return 25
    return 5


def _recommend_trust_actions(
    *,
    level: str,
    caveats: list[str],
    history_readiness: str,
    cv_horizon_status: str,
    naive_skill: float | None,
    residual_stability: str,
    interval_gap: float | None,
    interval_status: str,
    seasonality_status: str,
    hierarchy_status: str,
    event_status: str,
    horizon_gate: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if level == "High":
        actions.append("Use as a production pilot baseline; monitor accuracy as actuals land.")
    elif level == "Medium":
        actions.append("Validate with a domain expert before using for planning decisions.")
    else:
        actions.append("Do not deploy for planning yet; resolve the largest trust blockers first.")

    if history_readiness in {"short", "minimal"}:
        actions.append("Acquire more history or shorten the forecast horizon before stakeholder use.")
    if cv_horizon_status == "shorter_than_requested":
        actions.append("Rerun with a shorter horizon or strict full-horizon CV before claiming a champion.")
    if horizon_gate.get("horizon_trust_state") == "no_rolling_origin_evidence":
        actions.append("Add history or shorten the horizon; do not use this as a validated planning forecast yet.")
    elif horizon_gate.get("horizon_trust_state") == "partial_horizon_validated":
        validated = horizon_gate.get("validated_through_horizon")
        actions.append(f"Use only horizons through step {validated} as validation-backed; treat later steps as directional.")
    elif not horizon_gate.get("full_horizon_claim_allowed", False):
        actions.append("Treat the full-horizon validation as limited because it has fewer than two rolling-origin windows.")
    if naive_skill is not None and naive_skill < 0.05:
        actions.append("Treat the champion as weak; compare against plan/prior year and simpler benchmarks.")
    if residual_stability in {"drift", "volatile"}:
        actions.append("Investigate residual bias, structural breaks, outliers, or missing drivers.")
    if interval_status == "point_only_ensemble":
        actions.append("Use component-model intervals or select a single calibrated model if planning ranges matter.")
    if interval_status == "future_only":
        actions.append("Treat interval bands as directional only until rolling-origin interval coverage is available.")
    if interval_status == "adjusted_not_recalibrated":
        actions.append("Review adjusted interval bands with a domain expert before using them as planning ranges.")
    if interval_status in {"calibration_fail", "calibration_warning"} or (interval_gap is not None and interval_gap > 0.10):
        actions.append("Review prediction interval calibration and widen planning ranges if needed.")
    if seasonality_status == "low":
        actions.append("Review decomposition and cycle counts before trusting seasonal model behavior.")
    if hierarchy_status in {"incoherent", "minor_gap"}:
        actions.append("Use hierarchy reconciliation or parent/child override review before planning.")
    if event_status == "statistical_baseline_only":
        actions.append("Add known future launches, pricing changes, contracts, or other driver scenarios.")
    if any("seasonal" in caveat for caveat in caveats):
        actions.append("Review seasonality evidence before trusting seasonal model behavior.")
    return list(dict.fromkeys(actions))


def _is_simple_benchmark(model: str) -> bool:
    return model in {
        "Naive",
        "SeasonalNaive",
        "HistoricAverage",
        "WindowAverage",
        "SeasonalWindowAverage",
        "ZeroForecast",
    }


def _selected_model_map(run: ForecastRun) -> dict[str, str]:
    if run.model_selection.empty or "selected_model" not in run.model_selection.columns:
        return {}
    return {
        str(row["unique_id"]): str(row["selected_model"])
        for row in run.model_selection[["unique_id", "selected_model"]].to_dict("records")
    }


def _weight_map(run: ForecastRun) -> dict[tuple[str, str], float]:
    if run.model_weights.empty:
        return {}
    return {
        (str(row["unique_id"]), str(row["model"])): row["weight"]
        for row in run.model_weights[["unique_id", "model", "weight"]].to_dict("records")
    }


def _top_models_frame(run: ForecastRun) -> pd.DataFrame:
    if run.model_weights.empty:
        return pd.DataFrame(columns=["unique_id", "top_weighted_models", "top_weighted_families", "top_weighted_weights"])
    rows: list[dict[str, Any]] = []
    selected = _selected_model_map(run)
    frame = run.model_weights.copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    for uid, grp in frame.sort_values(["unique_id", "weight"], ascending=[True, False]).groupby("unique_id", sort=True):
        selected_model = selected.get(str(uid), "")
        top = grp[grp["model"].astype(str) != selected_model].head(3)
        rows.append(
            {
                "unique_id": uid,
                "top_weighted_models": "; ".join(top["model"].astype(str).tolist()),
                "top_weighted_families": "; ".join(top["model"].map(model_family).astype(str).tolist()),
                "top_weighted_weights": "; ".join(f"{weight:.4f}" for weight in top["weight"].fillna(0).tolist()),
            }
        )
    return pd.DataFrame(rows)


def _model_columns(frame: pd.DataFrame, *, extra_exclude: set[str] | None = None) -> list[str]:
    excluded = {"unique_id", "ds"} | (extra_exclude or set())
    return [
        col
        for col in frame.columns
        if col not in excluded and "-lo-" not in col and "-hi-" not in col
    ]


def _metric_row(uid: Any, model: Any, grp: pd.DataFrame, *, cutoff: Any | None = None) -> dict[str, Any]:
    actual = pd.to_numeric(grp["y_actual"], errors="coerce")
    error = pd.to_numeric(grp["error"], errors="coerce")
    abs_error = error.abs()
    squared_error = pd.to_numeric(grp["squared_error"], errors="coerce")
    denom = float(actual.abs().sum())
    rmse = float(squared_error.mean() ** 0.5) if squared_error.notna().any() else None
    mae = float(abs_error.mean()) if abs_error.notna().any() else None
    mase_scale = _first_finite_positive(grp.get("mase_scale"))
    rmsse_scale = _first_finite_positive(grp.get("rmsse_scale"))
    row: dict[str, Any] = {
        "unique_id": uid,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "wape": float(abs_error.sum() / denom) if denom else None,
        "mase": mae / mase_scale if mae is not None and mase_scale is not None else None,
        "rmsse": rmse / rmsse_scale if rmse is not None and rmsse_scale is not None else None,
        "bias": float((-error).sum() / denom) if denom else None,
        "observations": int(error.notna().sum()),
    }
    if cutoff is not None:
        row["cutoff"] = cutoff
    return row


def _scale_maps(history: pd.DataFrame, season_length: int) -> dict[str, dict[str, float | None]]:
    scales: dict[str, dict[str, float | None]] = {}
    lag = max(1, int(season_length or 1))
    for uid, grp in history.groupby("unique_id", sort=True):
        y = pd.to_numeric(grp.sort_values("ds")["y"], errors="coerce").dropna().to_numpy(dtype="float64")
        active_lag = lag if lag > 1 and len(y) > lag else 1
        if len(y) <= active_lag:
            scales[str(uid)] = {"mase_scale": None, "rmsse_scale": None}
            continue
        diff = y[active_lag:] - y[:-active_lag]
        abs_scale = float(pd.Series(diff).abs().mean())
        squared_scale = float((pd.Series(diff) ** 2).mean() ** 0.5)
        scales[str(uid)] = {
            "mase_scale": abs_scale if abs_scale > 0 else None,
            "rmsse_scale": squared_scale if squared_scale > 0 else None,
        }
    return scales


def _first_finite_positive(values: Any) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric.notna() & (numeric > 0)]
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _interval_levels_for_model(row: pd.Series, model: str) -> list[int]:
    prefix = f"{model}-lo-"
    levels: list[int] = []
    for col in row.index:
        if isinstance(col, str) and col.startswith(prefix):
            try:
                level = int(col.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if f"{model}-hi-{level}" in row.index:
                levels.append(level)
    return sorted(set(levels))


def _interval_levels_from_columns(columns: Any) -> list[int]:
    levels: list[int] = []
    for col in columns:
        if isinstance(col, str) and col.startswith("yhat_lo_"):
            try:
                level = int(col.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if f"yhat_hi_{level}" in columns:
                levels.append(level)
    return sorted(set(levels))


def _future_interval_status(
    *,
    uid: str,
    model: str,
    horizon_step: int,
    has_interval: bool,
    intervals: pd.DataFrame,
    metadata: dict[str, Any],
    adjusted: bool,
) -> str:
    if model == "WeightedEnsemble":
        return "point_only_ensemble"
    if not has_interval:
        return "unavailable"
    if adjusted:
        return "adjusted_not_recalibrated"
    selection_horizon = _int_or_none(metadata.get("selection_horizon"))
    if selection_horizon is not None and horizon_step > selection_horizon:
        return "future_only"
    if intervals.empty or "unique_id" not in intervals.columns:
        return "future_only"
    frame = intervals[
        (intervals["unique_id"].astype(str) == uid)
        & (intervals["model"].astype(str) == model)
        & (pd.to_numeric(intervals["horizon_step"], errors="coerce") == horizon_step)
    ]
    if frame.empty:
        return "future_only"
    statuses = set(frame.get("interval_status", pd.Series(dtype=str)).dropna().astype(str))
    if "calibration_fail" in statuses:
        return "calibration_fail"
    if "calibration_warning" in statuses:
        return "calibration_warning"
    if statuses == {"insufficient_observations"}:
        return "insufficient_observations"
    if "calibrated" in statuses:
        return "calibrated"
    return "future_only"


def _coverage_status_to_interval_status(status: str) -> str:
    if str(status).startswith("fail"):
        return "calibration_fail"
    if str(status).startswith("warn"):
        return "calibration_warning"
    if status == "insufficient_observations":
        return "insufficient_observations"
    if status == "pass":
        return "calibrated"
    return "unavailable"


def _interval_method(model: str, *, has_interval: bool | None = None) -> str:
    family = model_family(model)
    if model == "WeightedEnsemble":
        return "point_forecast_only"
    if family == "statsforecast":
        return "statsforecast_conformal_distribution"
    if family == "mlforecast":
        return "mlforecast_conformal_distribution"
    if family == "baseline" and has_interval:
        return "statsforecast_conformal_distribution"
    if family == "ensemble":
        return "point_forecast_only"
    return "not_applicable"


def _interval_evidence(status: str, metadata: dict[str, Any]) -> str:
    selection_horizon = metadata.get("selection_horizon")
    requested_horizon = metadata.get("requested_horizon")
    cv_windows = metadata.get("cv_windows")
    if status == "calibrated":
        return f"rolling_origin_cv_h={selection_horizon}, windows={cv_windows}"
    if status in {"calibration_warning", "calibration_fail", "insufficient_observations"}:
        return f"rolling_origin_cv_h={selection_horizon}, windows={cv_windows}, status={status}"
    if status == "future_only":
        return f"future bands available but no matching CV interval evidence for requested horizon {requested_horizon}"
    if status == "adjusted_not_recalibrated":
        return "selected bands were adjusted after model calibration"
    if status == "point_only_ensemble":
        return "weighted ensemble point forecast; component intervals are not inherited"
    return "no interval bands available"


def _cv_metadata_map(run: ForecastRun) -> dict[tuple[str, str], dict[str, Any]]:
    if run.backtest_metrics.empty:
        return {}
    metadata_cols = [
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
    ]
    available_cols = [col for col in metadata_cols if col in run.backtest_metrics.columns]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in run.backtest_metrics[["unique_id", "model", *available_cols]].to_dict("records"):
        out[(str(row["unique_id"]), str(row["model"]))] = {col: row.get(col) for col in available_cols}
    return out


def _final_forecast_lookup(run: ForecastRun) -> dict[tuple[str, pd.Timestamp], dict[str, Any]]:
    if run.forecast.empty:
        return {}
    frame = run.forecast.copy()
    frame["unique_id"] = frame["unique_id"].astype(str)
    frame["ds"] = pd.to_datetime(frame["ds"])
    return {
        (str(row["unique_id"]), pd.Timestamp(row["ds"])): row
        for row in frame.to_dict("records")
    }


def _final_selected_forecast_adjusted(
    source_row: dict[str, Any],
    final_row: dict[str, Any],
    model: str,
    levels: list[int],
) -> bool:
    if _values_differ(source_row.get(model), final_row.get("yhat")):
        return True
    for level in levels:
        if _values_differ(source_row.get(f"{model}-lo-{level}"), final_row.get(f"yhat_lo_{level}")):
            return True
        if _values_differ(source_row.get(f"{model}-hi-{level}"), final_row.get(f"yhat_hi_{level}")):
            return True
    return False


def _selected_forecast_adjusted(run: ForecastRun, uid: str, selected_model: str) -> bool:
    if run.forecast.empty or run.all_models.empty or not selected_model:
        return False
    final_lookup = _final_forecast_lookup(run)
    all_models = run.all_models.copy()
    all_models["unique_id"] = all_models["unique_id"].astype(str)
    all_models["ds"] = pd.to_datetime(all_models["ds"])
    levels = run.effective_levels()
    for row in all_models[all_models["unique_id"] == uid].to_dict("records"):
        final_row = final_lookup.get((uid, pd.Timestamp(row["ds"])))
        if final_row is not None and _final_selected_forecast_adjusted(row, final_row, selected_model, levels):
            return True
    return False


def _selected_future_has_intervals(run: ForecastRun, uid: str) -> bool:
    if run.forecast.empty:
        return False
    frame = run.forecast[run.forecast["unique_id"].astype(str) == uid]
    for level in run.effective_levels():
        lo_col = f"yhat_lo_{level}"
        hi_col = f"yhat_hi_{level}"
        if lo_col in frame.columns and hi_col in frame.columns and frame[lo_col].notna().any() and frame[hi_col].notna().any():
            return True
    return False


def _values_differ(left: Any, right: Any) -> bool:
    left_f = _safe_float(left)
    right_f = _safe_float(right)
    if left_f is None and right_f is None:
        return False
    if left_f is None or right_f is None:
        return True
    return abs(left_f - right_f) > max(1e-6, 1e-6 * max(abs(left_f), abs(right_f), 1.0))


def _covered(actual: Any, lo: Any, hi: Any) -> bool | None:
    actual_f, lo_f, hi_f = _safe_float(actual), _safe_float(lo), _safe_float(hi)
    if actual_f is None or lo_f is None or hi_f is None:
        return None
    return lo_f <= actual_f <= hi_f


def _miss_direction(actual: Any, lo: Any, hi: Any) -> str | None:
    actual_f, lo_f, hi_f = _safe_float(actual), _safe_float(lo), _safe_float(hi)
    if actual_f is None or lo_f is None or hi_f is None:
        return None
    if actual_f < lo_f:
        return "below"
    if actual_f > hi_f:
        return "above"
    return "inside"


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_model_feed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = list(frame.columns)
    interval_levels = _interval_levels_from_columns(columns)
    yhat_group = ["yhat"]
    for level in interval_levels:
        yhat_group.extend([f"yhat_lo_{level}", f"yhat_hi_{level}"])
    coverage_group: list[str] = []
    for level in interval_levels:
        coverage_group.extend([f"covered_{level}", f"miss_direction_{level}"])
    front_candidates = [
        "record_type",
        "unique_id",
        "cutoff",
        "ds",
        "model",
        "family",
        "y_actual",
        "horizon_step",
        "h",
    ]
    front = [col for col in front_candidates if col in columns]
    grouped = [col for col in [*yhat_group, *coverage_group] if col in columns]
    used = set(front + grouped)
    remaining = [col for col in columns if col not in used]
    return frame[[*front, *grouped, *remaining]]


def _sort_for_feeder(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    ascending = [True] * len(columns)
    if "is_selected_model" in columns:
        ascending[columns.index("is_selected_model")] = False
    if "weight" in columns:
        ascending[columns.index("weight")] = False
    sort_cols = [col for col in columns if col in frame.columns]
    sort_ascending = [ascending[columns.index(col)] for col in sort_cols]
    return frame.sort_values(sort_cols, ascending=sort_ascending).reset_index(drop=True)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None

