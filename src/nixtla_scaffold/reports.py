from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd

from nixtla_scaffold.best_practices import best_practice_receipts_frame
from nixtla_scaffold.drivers import (
    build_driver_experiment_summary_frame,
    build_known_future_regressors_frame,
    build_scenario_assumptions_frame,
    build_scenario_forecast_frame,
)
from nixtla_scaffold.headline import build_executive_headline
from nixtla_scaffold.interpretation import (
    backtest_windows_frame,
    seasonality_decomposition_frame,
    seasonality_diagnostics_frame,
    seasonality_summary_frame,
)
from nixtla_scaffold.model_families import model_family
from nixtla_scaffold.schema import ForecastRun


INTERVAL_STATUS_LABELS = {
    "calibrated": "Calibrated in rolling-origin CV",
    "calibration_warning": "Calibration warning",
    "calibration_fail": "Undercoverage risk",
    "insufficient_observations": "Too few interval observations",
    "future_only": "Future bands only; no empirical CV coverage",
    "unavailable": "Prediction intervals unavailable",
    "adjusted_not_recalibrated": "Adjusted bands; not recalibrated",
    "point_only_ensemble": "WeightedEnsemble is point-only",
    # Backward-compatible labels for older run directories.
    "pass": "Calibrated in rolling-origin CV",
    "warn": "Calibration warning",
    "fail": "Undercoverage risk",
    "ensemble_intervals_not_in_scope": "WeightedEnsemble is point-only",
    "available": "Future interval bands available",
}
INTERVAL_METHOD_LABELS = {
    "statsforecast_conformal_distribution": "StatsForecast conformal",
    "mlforecast_conformal_distribution": "MLForecast conformal",
    "point_forecast_only": "Point forecast only",
    "not_applicable": "Not applicable",
}
CV_HORIZON_STATUS_LABELS = {
    "matches_requested": "CV horizon matches request",
    "shorter_than_requested": "CV horizon shorter than request",
    "longer_than_requested": "CV horizon longer than request",
    "unavailable": "CV horizon unavailable",
}
HORIZON_TRUST_LABELS = {
    "full_horizon_validated": "Validated through requested horizon",
    "partial_horizon_validated": "Validated only through part of requested horizon",
    "beyond_validated_horizon": "Directional beyond CV horizon",
    "no_rolling_origin_evidence": "No rolling-origin validation",
}
HORIZON_GATE_LABELS = {
    "passed": "Passed horizon gate",
    "warning_single_cv_window": "Only one CV window",
    "warning_partial_horizon_validation": "Partial-horizon validation",
    "fail_less_than_half_horizon_validated": "Less than half of requested horizon validated",
    "fail_no_rolling_origin_validation": "No rolling-origin validation",
}
INTERVAL_GLOSSARY_TEXT = (
    "Interval states distinguish calibrated CV evidence, future-only bands, adjusted-not-recalibrated bands, "
    "unavailable intervals, and point-only ensembles. WeightedEnsemble does not inherit component intervals; "
    "use component-model intervals or a single calibrated model when planning ranges matter. "
    "Model disagreement lines are not prediction intervals."
)
TRUST_RUBRIC_TEXT = (
    "High >=75, Medium 40-74, Low <40. Review score deductions and caveats "
    "before using a forecast for planning."
)


def write_report_artifacts(run: ForecastRun, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _payload_from_run(run)
    report_html = build_html_report(payload)
    report_path = out / "report.html"
    report_base64_path = out / "report_base64.txt"
    streamlit_path = out / "streamlit_app.py"
    report_path.write_text(report_html, encoding="utf-8")
    report_base64_path.write_text(base64.b64encode(report_html.encode("utf-8")).decode("ascii") + "\n", encoding="utf-8")
    streamlit_path.write_text(build_streamlit_app(), encoding="utf-8")
    return {
        "html_report": report_path,
        "html_report_base64": report_base64_path,
        "streamlit_app": streamlit_path,
    }


def write_report_artifacts_from_directory(run_dir: str | Path) -> dict[str, Path]:
    out = Path(run_dir)
    payload = _payload_from_directory(out)
    report_html = build_html_report(payload)
    report_path = out / "report.html"
    report_base64_path = out / "report_base64.txt"
    streamlit_path = out / "streamlit_app.py"
    report_path.write_text(report_html, encoding="utf-8")
    report_base64_path.write_text(base64.b64encode(report_html.encode("utf-8")).decode("ascii") + "\n", encoding="utf-8")
    streamlit_path.write_text(build_streamlit_app(), encoding="utf-8")
    return {
        "html_report": report_path,
        "html_report_base64": report_base64_path,
        "streamlit_app": streamlit_path,
    }


def build_html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    executive_headline = payload.get("executive_headline", {})
    executive_paragraph = executive_headline.get("paragraph") or "Executive headline unavailable for this run."
    model_selection = payload["model_selection"]
    backtest_metrics = payload["backtest_metrics"]
    model_win_rates = payload["model_win_rates"]
    residual_tests = payload.get("residual_tests", [])
    trust_summary = payload["trust_summary"]
    trust_summary_display = _trust_summary_display_rows(trust_summary)
    target_transform_audit = payload["target_transform_audit"]
    scenario_assumptions = payload.get("scenario_assumptions", [])
    scenario_forecast = payload.get("scenario_forecast", [])
    known_future_regressors = payload.get("known_future_regressors", [])
    driver_availability_audit = payload.get("driver_availability_audit", [])
    driver_experiment_summary = payload.get("driver_experiment_summary", [])
    hierarchy_contribution = payload.get("hierarchy_contribution", [])
    hierarchy_backtest_comparison = payload.get("hierarchy_backtest_comparison", [])
    seasonality = payload["seasonality_summary"]
    seasonality_diagnostics = payload["seasonality_diagnostics"]
    model_weights = payload["model_weights"]
    warnings = payload["warnings"]
    model_policy_resolution = payload.get("model_policy_resolution", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(summary["title"])} forecast review</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #59636e;
      --line: #d9dee5;
      --panel: #ffffff;
      --soft: #f6f8fb;
      --blue: #17324d;
      --copper: #b65f32;
      --green: #516246;
      --purple: #62518f;
      --gold: #9a6b21;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #f3f5f8;
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ width: min(1540px, calc(100vw - 32px)); margin: 0 auto; padding: 22px 0 42px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 8px 22px rgba(23, 32, 42, 0.06);
    }}
    .hero {{ padding: 22px 24px; margin-bottom: 14px; }}
    .eyebrow {{ color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }}
    h1 {{ margin: 6px 0 8px; font-size: clamp(1.8rem, 4vw, 3.2rem); letter-spacing: -0.04em; line-height: 1.02; }}
    h2 {{ margin: 0 0 12px; font-size: 1.25rem; letter-spacing: -0.02em; }}
    h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .card {{ border: 1px solid var(--line); border-radius: 10px; background: var(--soft); padding: 12px; }}
    .headline {{ margin-top: 14px; padding: 14px 16px; border-left: 4px solid var(--copper); background: #fff8f4; border-radius: 10px; color: var(--ink); }}
    .headline p {{ color: var(--ink); font-size: 0.98rem; }}
    .label {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}
    .value {{ margin-top: 5px; font-size: 1.12rem; font-weight: 750; }}
    .decision-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; margin-top: 14px; }}
    .series-decision-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    }}
    .decision-title {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .series-name {{ font-size: 1.05rem; font-weight: 800; letter-spacing: -0.02em; overflow-wrap: anywhere; }}
    .selected-model {{ color: var(--muted); font-size: 0.82rem; margin-top: 2px; }}
    .trust-badge {{ flex: 0 0 auto; border-radius: 999px; padding: 5px 10px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }}
    .trust-high {{ background: #edf5e8; color: #315224; border: 1px solid #cfe0c6; }}
    .trust-medium {{ background: #fff4d9; color: #765118; border: 1px solid #ead39d; }}
    .trust-low {{ background: #fae9e5; color: #84391e; border: 1px solid #ecc7bd; }}
    .trust-unknown {{ background: #eef1f5; color: #4d5965; border: 1px solid var(--line); }}
    .decision-chips {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }}
    .decision-chip {{ border: 1px solid #e5e9ef; border-radius: 10px; background: #fff; padding: 8px; min-width: 0; }}
    .chip-label {{ color: var(--muted); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 750; }}
    .chip-value {{ color: var(--ink); font-size: 0.9rem; font-weight: 760; margin-top: 2px; overflow-wrap: anywhere; }}
    .decision-detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .decision-field {{ border-top: 1px solid #edf0f4; padding-top: 9px; min-width: 0; }}
    .decision-field.wide {{ grid-column: 1 / -1; }}
    .field-label {{ color: var(--muted); font-size: 0.69rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 750; }}
    .field-value {{ color: var(--ink); margin-top: 3px; overflow-wrap: anywhere; }}
    .field-value.long {{ max-width: 92ch; line-height: 1.5; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 14px; margin-top: 14px; }}
    .panel {{ padding: 18px; overflow: hidden; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .chart-block {{ border: 1px solid var(--line); border-radius: 10px; background: #fff; overflow: hidden; }}
    .chart-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; padding: 10px 12px; border-bottom: 1px solid var(--line); background: var(--soft); }}
    .chart-head strong {{ font-size: 0.95rem; }}
    .chart-head span {{ color: var(--muted); font-size: 0.82rem; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    th {{ color: var(--muted); text-align: left; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    th, td {{ padding: 8px 8px; border-bottom: 1px solid #edf0f4; vertical-align: top; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ background: #eef1f5; padding: 2px 5px; border-radius: 4px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; color: var(--muted); font-size: 0.82rem; margin: 8px 0 0; }}
    .swatch {{ display: inline-block; width: 18px; height: 3px; border-radius: 999px; margin-right: 6px; vertical-align: middle; background: var(--ink); }}
    .footnote {{ color: var(--muted); font-size: 0.84rem; }}
    .warn {{ margin: 0; padding-left: 18px; color: #7a3a1c; }}
    .output-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .output-item {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--soft); }}
    .output-item strong {{ display: block; margin-bottom: 4px; }}
    @media (max-width: 980px) {{ .cards, .grid, .output-list, .decision-grid, .decision-chips, .decision-detail-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="eyebrow">Nixtla scaffold forecast review</div>
      <h1>{_esc(summary["title"])}</h1>
      <p>{_esc(summary["narrative"])}</p>
      <div class="headline">
        <div class="label">Executive forecast headline</div>
        <p>{_esc(executive_paragraph)}</p>
      </div>
      <div class="cards">
        {_metric_card("Engine", summary["engine"])}
        {_metric_card("Frequency", f"{summary['freq']} / season {summary['season_length']}")}
        {_metric_card("History rows", summary["rows"])}
        {_metric_card("Series", summary["series_count"])}
        {_metric_card("Horizon", summary["horizon"])}
      </div>
    </section>

    <section class="panel" style="margin-bottom:14px">
      <h2>Decision summary</h2>
      <p class="footnote">Start here before reading the model tournament. Trust is a deterministic review score that combines history depth, CV evidence, naive skill, residuals, interval calibration, hierarchy coherence, event coverage, and data-quality caveats.</p>
      <p class="footnote"><strong>Planning eligibility scope:</strong> <code>planning_eligible</code> is a horizon-validation flag only. It does not override Low trust, interval issues, residual warnings, hierarchy tradeoffs, or data-quality caveats.</p>
      <p class="footnote"><strong>How to read forecast.csv:</strong> Start with <code>row_horizon_status</code> and <code>planning_eligible</code>, then cross-check <code>trust_summary.csv</code>, <code>interval_status</code>, residual warnings, hierarchy tradeoffs, and data-quality caveats before stakeholder sharing.</p>
      <p class="footnote"><strong>Trust rubric:</strong> {_esc(TRUST_RUBRIC_TEXT)}</p>
      {_trust_summary_cards(trust_summary_display, limit=16)}
      <p class="footnote"><strong>Interval glossary:</strong> {_esc(INTERVAL_GLOSSARY_TEXT)}</p>
    </section>
    {_model_policy_resolution_section(model_policy_resolution)}
    {_target_transform_section(target_transform_audit)}
    {_driver_assumptions_section(scenario_assumptions, scenario_forecast, known_future_regressors, driver_availability_audit, driver_experiment_summary)}

    <section class="panel">
      <h2>Forecast model review</h2>
      <p class="footnote">Each series has two required views: all candidate models with future bands where written, then the champion plus the top weighted alternatives only. Check interval_status before treating shaded bands as planning ranges.</p>
      <div class="legend">
        <span><i class="swatch" style="background:var(--blue)"></i>history / actual</span>
        <span><i class="swatch" style="background:var(--copper)"></i>selected model</span>
        <span><i class="swatch" style="background:var(--green)"></i>top weighted alternatives</span>
        <span><i class="swatch" style="background:#b9c0c9"></i>other candidates</span>
        <span>shaded bands = future model bands; check interval_status before planning use</span>
      </div>
      <div class="chart-grid" style="margin-top:12px">
        {_forecast_review_charts(payload)}
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Model selection leaderboard</h2>
        {_table(model_selection, ["unique_id", "selected_model", "rmse", "mae", "mase", "rmsse", "wape", "bias", "selection_horizon", "requested_horizon", "cv_windows", "selection_reason"], limit=16)}
      </article>
      <article class="panel">
        <h2>Top weighted models</h2>
        {_table(_top_weights(model_weights), ["unique_id", "model", "family", "weight"], limit=20)}
        <h2 style="margin-top:18px">Win rate vs benchmark</h2>
        {_table(model_win_rates, ["benchmark_model", "metric", "model", "eligible_series", "wins_vs_benchmark", "win_rate_vs_benchmark", "avg_skill_vs_benchmark"], limit=12)}
        <h2 style="margin-top:18px">Seasonality check</h2>
        {_table(seasonality, ["unique_id", "seasonality_strength", "peak_position", "trough_position", "interpretation"], limit=10)}
        <h2 style="margin-top:18px">Seasonality credibility</h2>
        {_table(seasonality_diagnostics, ["unique_id", "cycle_count", "credibility_label", "seasonality_strength", "trend_strength", "warning"], limit=10)}
      </article>
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Rolling-origin fixed-axis filmstrip</h2>
      <p class="footnote">Each panel holds the date and value axes constant for the series so the cutoff progression is comparable. The full actual history stays visible, the training and holdout regions slide forward, and champion/challenger labels remain attached to forecast endpoints.</p>
      {_backtest_review_charts(payload)}
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Backtest metrics by model</h2>
        {_table(_sort_metrics(backtest_metrics), ["unique_id", "model", "rmse", "mae", "mase", "rmsse", "wape", "bias", "observations", "selection_horizon", "requested_horizon", "cv_windows"], limit=24)}
        <h2 style="margin-top:18px">Heuristic residual checks</h2>
        <p class="footnote">Bias, outlier, and early/late structural-break checks summarize all rolling-origin residuals; the white-noise/ACF check uses one-step residuals when available. Treat small samples as directional, not proof of model adequacy.</p>
        {_table(residual_tests, ["unique_id", "model", "overall_status", "bias_status", "white_noise_status", "white_noise_residual_scope", "outlier_status", "structural_break_status", "observations", "interpretation"], limit=24)}
      </article>
      <article class="panel">
        <h2>Core feeder outputs</h2>
        <div class="output-list">
          {_output_item("forecast_long.csv", "One row per future series/model/date with yhat, intervals, weight, and selected-model flag.")}
          {_output_item("backtest_long.csv", "One row per backtest cutoff/series/model/date with actuals, errors, interval bounds, and coverage flags.")}
          {_output_item("series_summary.csv", "One row per series with selected model, RMSE/MAE/MASE/RMSSE/WAPE, seasonality, and top alternatives.")}
          {_output_item("model_audit.csv", "Model leaderboard enriched with weights and selected/challenger flags.")}
          {_output_item("model_win_rates.csv", "Model win rates against SeasonalNaive or Naive benchmarks for cross-series review.")}
          {_output_item("model_window_metrics.csv", "Rolling-origin error metrics by cutoff and model for window-by-window review.")}
          {_output_item("residual_diagnostics.csv", "Residual/error diagnostics by model and horizon step.")}
          {_output_item("residual_tests.csv", "Heuristic residual bias, one-step autocorrelation, outlier, and early/late structural-break screening; not formal model adequacy certification.")}
          {_output_item("interval_diagnostics.csv", "Prediction interval empirical coverage, method, horizon metadata, and calibration state when interval backtests are available.")}
          {_output_item("trust_summary.csv", "Per-series High/Medium/Low trust rating with score drivers, caveats, and next recommended actions.")}
          {_output_item("scenario_assumptions.csv", "Post-model event/scenario assumptions when overlays are supplied.")}
          {_output_item("scenario_forecast.csv", "Baseline yhat beside yhat_scenario, event adjustment, and event names for scenario review.")}
          {_output_item("known_future_regressors.csv", "Declared known-future regressor contracts for leakage and future-availability audit.")}
          {_output_item("driver_availability_audit.csv", "Known-future regressor audit status, leakage risk, required future rows, and modeling decision.")}
          {_output_item("driver_experiment_summary.csv", "One summary table for event overlays and known-future regressor audit outcomes.")}
          {_output_item("audit/target_transform_audit.csv", "Raw, adjusted, transformed, and modeled target trail when finance normalization or log/log1p transforms are enabled.")}
          {_output_item("hierarchy_reconciliation.csv", "Method summary and pre/post coherence gaps when hierarchy reconciliation is enabled; node-level accuracy may decrease.")}
          {_output_item("hierarchy_contribution.csv", "Parent/child contribution and gap attribution table for hierarchy storytelling; allocation heuristic, not reconciliation output.")}
          {_output_item("audit/hierarchy_backtest_comparison.csv", "Selected hierarchy backtests before and after reconciliation for node-level accuracy/coherence tradeoff review.")}
          {_output_item("llm_context.json", "Single LLM feeder packet with executive headline, trust, horizon, interval, residual, seasonality, hierarchy, driver, and artifact-index context.")}
          {_output_item("forecast.xlsx", "Curated workbook for analysts who want one file with all major sheets.")}
          {_output_item("streamlit_app.py", "Interactive local dashboard: run with uv run streamlit run streamlit_app.py.")}
        </div>
        {_hierarchy_depth_section(hierarchy_contribution, hierarchy_backtest_comparison)}
      </article>
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Limitations and next actions</h2>
      {_limitations(summary, warnings)}
    </section>
  </main>
</body>
</html>
"""


def build_streamlit_app() -> str:
    return dedent(
        '''
        from __future__ import annotations

        from pathlib import Path
        import json
        import time

        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        import streamlit as st


        RUN_DIR = Path(__file__).parent
        C = {
            "hist": "#17324d",
            "champ": "#b65f32",
            "alt1": "#516246",
            "alt2": "#62518f",
            "alt3": "#9a6b21",
            "gold": "#9a6b21",
            "dim": "#b9c0c9",
            "i80": "rgba(182, 95, 50, 0.18)",
            "i95": "rgba(182, 95, 50, 0.08)",
            "train": "rgba(23, 50, 77, 0.07)",
            "test": "rgba(182, 95, 50, 0.10)",
        }
        ALT_COLORS = [C["alt1"], C["alt2"], C["alt3"]]
        MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


        def color_to_rgba(color: str, alpha: float) -> str:
            text = str(color)
            if text.startswith("#") and len(text) == 7:
                r = int(text[1:3], 16)
                g = int(text[3:5], 16)
                b = int(text[5:7], 16)
                return f"rgba({r}, {g}, {b}, {alpha})"
            return text


        def read_csv(name: str) -> pd.DataFrame:
            for path in (RUN_DIR / name, RUN_DIR / "audit" / name):
                if path.exists():
                    return pd.read_csv(path)
            return pd.DataFrame()


        def read_json(name: str) -> dict:
            path = RUN_DIR / name
            if not path.exists():
                return {}
            return json.loads(path.read_text(encoding="utf-8"))


        def prep_dates(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            out = frame.copy()
            if "ds" in out.columns:
                out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
            if "cutoff" in out.columns:
                out["cutoff"] = pd.to_datetime(out["cutoff"], errors="coerce")
            return out


        def model_columns(frame: pd.DataFrame, *, backtest: bool = False) -> list[str]:
            excluded = {"unique_id", "ds", "cutoff", "y"} if backtest else {"unique_id", "ds"}
            return [c for c in frame.columns if c not in excluded and "-lo-" not in c and "-hi-" not in c]


        MLFORECAST_MODELS = {
            "LinearRegression",
            "Ridge",
            "Ridge_Regularized",
            "BayesianRidge",
            "ElasticNet",
            "Huber",
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
            "HistGradientBoosting",
            "KNeighbors",
            "LightGBM",
            "LightGBM_Conservative",
            "LightGBM_Shallow",
            "LightGBM_Robust",
        }
        BASELINE_MODELS = {
            "Naive",
            "HistoricAverage",
            "RandomWalkWithDrift",
            "WindowAverage",
            "SeasonalNaive",
            "SeasonalWindowAverage",
            "ZeroForecast",
        }
        STATSFORECAST_MODELS = {
            "SES",
            "SeasonalExpSmoothing",
            "AutoETS",
            "AutoARIMA",
            "Holt",
            "HoltWinters",
            "AutoTheta",
            "Theta",
            "OptimizedTheta",
            "DynamicOptimizedTheta",
            "MSTL",
            "MSTL_AutoARIMA",
            "MFLES",
            "AutoMFLES",
            "CrostonClassic",
            "CrostonOptimized",
            "CrostonSBA",
            "ADIDA",
            "IMAPA",
            "TSB",
        }
        CHAMPION_SCOPE_OPTIONS = ["Best overall tournament", "Best StatsForecast/classical", "Best MLForecast"]
        FAMILY_MENU_ORDER = {
            "statsforecast": 0,
            "mlforecast": 1,
            "baseline": 2,
            "ensemble": 3,
            "custom": 4,
            "unknown": 5,
        }
        WINNER_METRIC_GUIDANCE = {
            "rmse": {
                "label": "RMSE - penalize large misses",
                "when": "Default for high-stakes forecasts where big misses are much worse than small misses. Squaring errors makes one bad month matter.",
                "direction": "lower",
            },
            "mae": {
                "label": "MAE - typical absolute miss",
                "when": "Use when you want the average miss in business units and do not want one extreme miss to dominate as much as RMSE.",
                "direction": "lower",
            },
            "wape": {
                "label": "WAPE - business percentage error",
                "when": "Useful for stakeholder-readable percent error on positive, non-sparse series. Be careful with zeros or very small actuals.",
                "direction": "lower",
            },
            "mase": {
                "label": "MASE - skill vs naive, scale-free",
                "when": "Use to compare models across differently-sized series. Values below 1 usually beat the naive benchmark.",
                "direction": "lower",
            },
            "rmsse": {
                "label": "RMSSE - scale-free, big-miss penalty",
                "when": "Use for cross-series comparison when large misses should be penalized more strongly than MASE.",
                "direction": "lower",
            },
            "abs_bias": {
                "label": "Absolute bias - avoid systematic over/under",
                "when": "Use when directional planning risk matters, such as consistently over-forecasting capacity or under-forecasting cost.",
                "direction": "lower",
            },
            "weight": {
                "label": "Ensemble weight - validation-weighted support",
                "when": "Use to inspect which models the inverse-error ensemble trusts most. Higher weight is better; not all models receive a weight.",
                "direction": "higher",
            },
        }


        def dedupe_models(models: list[str] | tuple[str, ...] | pd.Series | None) -> list[str]:
            out: list[str] = []
            if models is None:
                return out
            for model in list(models):
                name = str(model)
                if name and name != "nan" and name not in out:
                    out.append(name)
            return out


        def ordered_model_feed_columns(frame: pd.DataFrame) -> list[str]:
            priority = [
                "record_type",
                "unique_id",
                "ds",
                "cutoff",
                "model",
                "family",
                "horizon_step",
                "h",
                "y",
                "yhat",
                "yhat_lo_80",
                "yhat_hi_80",
                "yhat_lo_95",
                "yhat_hi_95",
                "error",
                "abs_error",
                "squared_error",
                "pct_error",
                "is_selected_model",
                "selected_model",
                "weight",
                "interval_status",
                "interval_method",
                "interval_evidence",
                "row_horizon_status",
                "horizon_trust_state",
                "validated_through_horizon",
                "planning_eligible",
                "planning_eligibility_scope",
                "planning_eligibility_reason",
            ]
            return [col for col in priority if col in frame.columns] + [col for col in frame.columns if col not in priority]


        FAMILY_VALUES = {"baseline", "statsforecast", "mlforecast", "custom", "ensemble", "unknown"}


        def normalize_family_label(value) -> str:
            raw = str(value).strip()
            lowered = raw.lower()
            if not raw or lowered in {"nan", "none"}:
                return ""
            aliases = {
                "baseline": "baseline",
                "statsforecast": "statsforecast",
                "statsforecast/classical": "statsforecast",
                "stats forecast": "statsforecast",
                "mlforecast": "mlforecast",
                "ml forecast": "mlforecast",
                "custom": "custom",
                "ensemble": "ensemble",
                "weightedensemble": "ensemble",
                "other": "unknown",
                "unknown": "unknown",
            }
            return aliases.get(lowered, lowered if lowered in FAMILY_VALUES else "")


        def model_family(model: str) -> str:
            name = str(model)
            if name == "WeightedEnsemble" or "Ensemble" in name:
                return "ensemble"
            if name.startswith("Custom_"):
                return "custom"
            if name in MLFORECAST_MODELS or name.startswith("LightGBM"):
                return "mlforecast"
            if name in BASELINE_MODELS:
                return "baseline"
            if name in STATSFORECAST_MODELS or name.startswith(("Auto", "Croston")):
                return "statsforecast"
            return "unknown"


        def display_family(family: str) -> str:
            return {
                "baseline": "Baseline",
                "statsforecast": "StatsForecast",
                "mlforecast": "MLForecast",
                "custom": "Custom",
                "ensemble": "Ensemble",
                "unknown": "Other",
            }.get(normalize_family_label(family) or "unknown", "Other")


        def model_type_label(model: str, family: str | None = None) -> str:
            family = normalize_family_label(family) or model_family(model)
            name = str(model)
            if family == "statsforecast" and name.startswith("Auto"):
                return "Automatic classical"
            if family == "statsforecast":
                return "Classical/statistical"
            if family == "mlforecast":
                if name.startswith("LightGBM"):
                    return "Gradient boosting learner"
                if name in {"RandomForest", "ExtraTrees"}:
                    return "Tree ensemble learner"
                if name in {"LinearRegression", "Ridge", "Ridge_Regularized", "BayesianRidge", "ElasticNet", "Huber"}:
                    return "Linear lag learner"
                return "ML lag/date learner"
            if family == "baseline":
                return "Benchmark baseline"
            if family == "ensemble":
                return "Validation-weighted ensemble"
            if family == "custom":
                return "Finance/custom challenger"
            return "Other candidate"


        def family_menu_sort_key(family: str | None) -> int:
            return FAMILY_MENU_ORDER.get(normalize_family_label(family) or "unknown", 99)


        INTERVAL_STATUS_LABELS = {
            "calibrated": "Calibrated in rolling-origin CV",
            "calibration_warning": "Calibration warning",
            "calibration_fail": "Undercoverage risk",
            "insufficient_observations": "Too few interval observations",
            "future_only": "Future bands only; no empirical CV coverage",
            "unavailable": "Prediction intervals unavailable",
            "adjusted_not_recalibrated": "Adjusted bands; not recalibrated",
            "point_only_ensemble": "WeightedEnsemble is point-only",
            "pass": "Calibrated in rolling-origin CV",
            "warn": "Calibration warning",
            "fail": "Undercoverage risk",
            "ensemble_intervals_not_in_scope": "WeightedEnsemble is point-only",
            "available": "Future interval bands available",
        }
        INTERVAL_METHOD_LABELS = {
            "statsforecast_conformal_distribution": "StatsForecast conformal",
            "mlforecast_conformal_distribution": "MLForecast conformal",
            "point_forecast_only": "Point forecast only",
            "not_applicable": "Not applicable",
        }
        CV_HORIZON_STATUS_LABELS = {
            "matches_requested": "CV horizon matches request",
            "shorter_than_requested": "CV horizon shorter than request",
            "longer_than_requested": "CV horizon longer than request",
            "unavailable": "CV horizon unavailable",
        }
        HORIZON_TRUST_LABELS = {
            "full_horizon_validated": "Full requested horizon evaluated",
            "partial_horizon_validated": "Validated only through part of requested horizon",
            "beyond_validated_horizon": "Directional beyond CV horizon",
            "no_rolling_origin_evidence": "No rolling-origin validation",
        }
        HORIZON_GATE_LABELS = {
            "passed": "Passed horizon gate",
            "warning_single_cv_window": "Only one CV window",
            "warning_partial_horizon_validation": "Partial-horizon validation",
            "fail_less_than_half_horizon_validated": "Less than half of requested horizon validated",
            "fail_no_rolling_origin_validation": "No rolling-origin validation",
        }
        INTERVAL_GLOSSARY_TEXT = "Interval states distinguish calibrated CV evidence, future-only bands, adjusted-not-recalibrated bands, unavailable intervals, and point-only ensembles. WeightedEnsemble does not inherit component intervals; use component-model intervals or a single calibrated model when planning ranges matter. Model disagreement lines are not prediction intervals."
        TRUST_RUBRIC_TEXT = "High >=75, Medium 40-74, Low <40. Review score deductions and caveats before using a forecast for planning."


        def interval_status_label(value) -> str:
            raw = str(value or "unavailable")
            return INTERVAL_STATUS_LABELS.get(raw, raw.replace("_", " "))


        def interval_method_label(value) -> str:
            raw = str(value or "not_applicable")
            return INTERVAL_METHOD_LABELS.get(raw, raw.replace("_", " "))


        def cv_horizon_status_label(value) -> str:
            raw = str(value or "unavailable")
            return CV_HORIZON_STATUS_LABELS.get(raw, raw.replace("_", " "))


        def horizon_trust_label(value) -> str:
            raw = str(value or "no_rolling_origin_evidence")
            return HORIZON_TRUST_LABELS.get(raw, raw.replace("_", " "))


        def horizon_gate_label(value) -> str:
            raw = str(value or "fail_no_rolling_origin_validation")
            return HORIZON_GATE_LABELS.get(raw, raw.replace("_", " "))


        def display_trust_summary(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            out = frame.copy()
            if "interval_status" in out.columns:
                out["interval_status"] = out["interval_status"].map(interval_status_label)
            if "interval_method" in out.columns:
                out["interval_method"] = out["interval_method"].map(interval_method_label)
            if "cv_horizon_status" in out.columns:
                out["cv_horizon_status"] = out["cv_horizon_status"].map(cv_horizon_status_label)
            if "horizon_trust_state" in out.columns:
                out["horizon_trust_state"] = out["horizon_trust_state"].map(horizon_trust_label)
            if "horizon_gate_result" in out.columns:
                out["horizon_gate_result"] = out["horizon_gate_result"].map(horizon_gate_label)
            return out


        def policy_reason_label(value) -> str:
            raw = str(value or "not_available")
            if raw.startswith("min_history_below_threshold"):
                return raw.replace("min_history_below_threshold", "minimum history gate not met")
            return raw.replace("_", " ")


        def scope_families(scope: str) -> list[str]:
            if scope == "Best MLForecast":
                return ["mlforecast"]
            if scope == "Best StatsForecast/classical":
                return ["statsforecast", "baseline"]
            return []


        def model_score_table(uid: str) -> pd.DataFrame:
            table = model_audit if not model_audit.empty else metrics
            if table.empty or "model" not in table.columns:
                return pd.DataFrame()
            frame = table.copy()
            if "unique_id" in frame.columns:
                frame = frame[frame["unique_id"].astype(str) == uid]
            if frame.empty:
                return frame
            frame["model"] = frame["model"].astype(str)
            if "family" in frame.columns:
                frame["family"] = frame.apply(
                    lambda row: normalize_family_label(row.get("family")) or model_family(row.get("model")),
                    axis=1,
                )
            else:
                frame["family"] = frame["model"].map(model_family)
            for col in ["rmse", "mae", "mase", "rmsse", "wape", "bias", "abs_bias", "weight"]:
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if "abs_bias" not in frame.columns and "bias" in frame.columns:
                frame["abs_bias"] = frame["bias"].abs()
            sort_cols = [col for col in ["rmse", "mae", "wape", "abs_bias", "model"] if col in frame.columns]
            return frame.sort_values(sort_cols) if sort_cols else frame


        def candidate_models(uid: str, families: list[str] | None = None) -> list[str]:
            score = model_score_table(uid)
            ranked = score["model"].astype(str).tolist() if not score.empty and "model" in score.columns else []
            extras: list[str] = []
            for frame, is_backtest in [(all_models, False), (backtest, True)]:
                if frame.empty:
                    continue
                one = frame[frame["unique_id"].astype(str) == uid] if "unique_id" in frame.columns else frame
                extras.extend(model_columns(one, backtest=is_backtest))
            for frame in [forecast_long, backtest_long]:
                if not frame.empty and "model" in frame.columns:
                    one = frame[frame["unique_id"].astype(str) == uid] if "unique_id" in frame.columns else frame
                    extras.extend(one["model"].astype(str).dropna().tolist())
            models = dedupe_models([*ranked, *sorted(set(extras))])
            if families:
                family_by_model = (
                    dict(zip(score["model"].astype(str), score["family"].astype(str), strict=False))
                    if not score.empty and {"model", "family"}.issubset(score.columns)
                    else {}
                )
                models = [model for model in models if family_by_model.get(model, model_family(model)) in families]
            return models


        def available_winner_metrics(uid: str) -> list[str]:
            score = model_score_table(uid)
            if score.empty:
                return ["rmse"]
            available = [
                metric
                for metric in WINNER_METRIC_GUIDANCE
                if metric in score.columns and pd.to_numeric(score[metric], errors="coerce").notna().any()
            ]
            return available or ["rmse"]


        def metric_label(metric: str) -> str:
            return WINNER_METRIC_GUIDANCE.get(metric, {"label": metric})["label"]


        def metric_direction(metric: str) -> str:
            return WINNER_METRIC_GUIDANCE.get(metric, {"direction": "lower"})["direction"]


        def best_model_for_scope(uid: str, scope: str, metric: str = "rmse") -> str:
            families = scope_families(scope)
            score = model_score_table(uid)
            if not score.empty:
                scoped = score[score["family"].isin(families)].copy() if families else score.copy()
                if metric not in scoped.columns or pd.to_numeric(scoped[metric], errors="coerce").notna().sum() == 0:
                    metric = "rmse" if "rmse" in scoped.columns else scoped.columns[0]
                if metric in scoped.columns:
                    scoped = scoped[pd.to_numeric(scoped[metric], errors="coerce").notna()]
                if not scoped.empty:
                    sort_cols = [col for col in [metric, "rmse", "mae", "wape", "model"] if col in scoped.columns]
                    ascending = [metric_direction(col) != "higher" for col in sort_cols]
                    return str(scoped.sort_values(sort_cols, ascending=ascending).iloc[0]["model"]) if sort_cols else str(scoped.iloc[0]["model"])
            scoped_candidates = candidate_models(uid, families)
            return scoped_candidates[0] if scoped_candidates else ""


        def model_investigation_table(
            uid: str,
            models: list[str],
            *,
            active_champion: str | None = None,
            metric: str = "rmse",
        ) -> pd.DataFrame:
            table = model_menu_table(
                uid,
                models,
                active_champion=active_champion,
                focus_models=models,
                metric=metric,
            )
            if table.empty:
                return pd.DataFrame({"model": models, "engine": [display_family(model_family(model)) for model in models]})
            preferred = [
                "rank",
                "model",
                "engine",
                "model_type",
                "role",
                metric,
                "rmse",
                "mae",
                "mase",
                "rmsse",
                "wape",
                "bias",
                "weight",
                "selection_horizon",
                "cv_windows",
                "intervals",
            ]
            cols = []
            for col in preferred:
                if col in table.columns and col not in cols:
                    cols.append(col)
            return table[cols]


        def policy_resolution_table() -> pd.DataFrame:
            if not isinstance(model_policy_resolution, dict):
                return pd.DataFrame()
            rows = []
            for row in model_policy_resolution.get("families", []) or []:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "family": row.get("family", ""),
                        "requested": bool(row.get("requested", False)),
                        "eligible": bool(row.get("eligible", False)),
                        "ran": bool(row.get("ran", False)),
                        "reason_if_not_ran": row.get("reason_if_not_ran", ""),
                        "contributed_models": ", ".join(str(model) for model in row.get("contributed_models", []) or []),
                    }
                )
            return pd.DataFrame(rows)


        def policy_message_for_scope(scope: str) -> str:
            family_key = "mlforecast" if scope == "Best MLForecast" else ("statsforecast" if scope == "Best StatsForecast/classical" else "")
            if not family_key:
                return ""
            table = policy_resolution_table()
            if table.empty or "family" not in table.columns:
                return "No model_policy_resolution metadata was written for this run. Open `manifest.json -> model_policy_resolution` for full details."
            family_rows = table[table["family"].astype(str) == family_key]
            if family_rows.empty:
                return f"{display_family(family_key)} was not recorded in model_policy_resolution. Open `manifest.json -> model_policy_resolution` for full details."
            row = family_rows.iloc[0]
            if bool(row.get("ran")):
                return f"{display_family(family_key)} ran, but no scored candidate is available for this series and metric. Open the Model policy resolution expander above or `manifest.json -> model_policy_resolution` for full details."
            reason = policy_reason_label(row.get("reason_if_not_ran") or "not_available")
            requested = "requested" if bool(row.get("requested")) else "not requested"
            eligible = "eligible" if bool(row.get("eligible")) else "not eligible"
            return f"{display_family(family_key)} did not run ({requested}, {eligible}): {reason}. Open the Model policy resolution expander above or `manifest.json -> model_policy_resolution` for full details."


        def family_summary(uid: str) -> pd.DataFrame:
            score = model_score_table(uid)
            if score.empty or "family" not in score.columns:
                return pd.DataFrame()
            metric = "rmse" if "rmse" in score.columns else None
            rows = []
            for family, group in score.groupby("family", dropna=False):
                row = {
                    "engine": display_family(family),
                    "candidate_count": int(group["model"].nunique()),
                    "_family_order": family_menu_sort_key(family),
                }
                if metric:
                    valid = group[group[metric].notna()].sort_values(metric)
                    if not valid.empty:
                        row["best_model"] = str(valid.iloc[0]["model"])
                        row[f"best_{metric}"] = float(valid.iloc[0][metric])
                rows.append(row)
            frame = pd.DataFrame(rows).sort_values(["_family_order", "engine"])
            return frame.drop(columns=["_family_order"], errors="ignore")


        def selected_model(uid: str) -> str:
            if selection.empty:
                return ""
            one = selection[selection["unique_id"].astype(str) == uid]
            return str(one.iloc[0]["selected_model"]) if not one.empty and "selected_model" in one.columns else ""


        def top_weighted(uid: str, n: int = 3, champion: str | None = None) -> list[str]:
            if weights.empty:
                return []
            one = weights[weights["unique_id"].astype(str) == uid].copy()
            if one.empty or "weight" not in one.columns or "model" not in one.columns:
                return []
            one["weight"] = pd.to_numeric(one["weight"], errors="coerce")
            sel = champion or selected_model(uid)
            return [m for m in one.sort_values("weight", ascending=False)["model"].astype(str).tolist() if m != sel][:n]


        def model_family_for(uid: str, model: str) -> str:
            score = model_score_table(uid)
            name = str(model)
            if not score.empty and {"model", "family"}.issubset(score.columns):
                rows = score[score["model"].astype(str) == name]
                if not rows.empty:
                    family = normalize_family_label(rows.iloc[0].get("family"))
                    if family:
                        return family
            return model_family(name)


        def model_rank_map(uid: str, metric: str = "rmse") -> dict[str, int]:
            score = model_score_table(uid)
            if score.empty or "model" not in score.columns:
                return {}
            metric = metric if metric in score.columns else ("rmse" if "rmse" in score.columns else "")
            ranked = score.copy()
            if metric:
                ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
                ranked = ranked[ranked[metric].notna()]
            if ranked.empty:
                return {}
            sort_cols = [col for col in [metric, "rmse", "mae", "wape", "model"] if col and col in ranked.columns]
            ascending = [metric_direction(col) != "higher" for col in sort_cols]
            if sort_cols:
                ranked = ranked.sort_values(sort_cols, ascending=ascending)
            out: dict[str, int] = {}
            for rank, model in enumerate(dedupe_models(ranked["model"].astype(str).tolist()), start=1):
                out[model] = rank
            return out


        def model_rank_label(uid: str, model: str, metric: str = "rmse") -> str:
            rank = model_rank_map(uid, metric).get(str(model))
            return f"#{rank:02d}" if rank is not None else "#--"


        def model_menu_options(uid: str, models: list[str] | tuple[str, ...] | pd.Series | None = None) -> list[str]:
            options = dedupe_models(models if models is not None else candidate_models(uid))
            return sorted(options, key=lambda model: (family_menu_sort_key(model_family_for(uid, model)), str(model).lower()))


        def model_menu_label(uid: str, model: str, metric: str = "rmse") -> str:
            family = display_family(model_family_for(uid, model))
            return f"{model_rank_label(uid, model, metric)} | {model} | {family}"


        def model_role_label(uid: str, model: str, active_champion: str | None = None, focus_models: list[str] | None = None) -> str:
            roles: list[str] = []
            name = str(model)
            if active_champion and name == str(active_champion):
                roles.append("active champion")
            if name == selected_model(uid):
                roles.append("forecast.csv selected")
            if name in top_weighted(uid, champion=active_champion):
                roles.append("top weighted")
            if focus_models and name in set(map(str, focus_models)):
                roles.append("investigated")
            if selected_interval_levels(uid, name):
                roles.append("interval bands")
            return ", ".join(dedupe_models(roles)) or "candidate"


        def model_interval_level_label(uid: str, model: str) -> str:
            levels = selected_interval_levels(uid, model)
            return ", ".join(f"{level}%" for level in levels) if levels else "none"


        def model_menu_table(
            uid: str,
            models: list[str] | tuple[str, ...] | pd.Series | None = None,
            *,
            active_champion: str | None = None,
            focus_models: list[str] | None = None,
            metric: str = "rmse",
        ) -> pd.DataFrame:
            ordered = model_menu_options(uid, models)
            if not ordered:
                return pd.DataFrame()
            score = model_score_table(uid)
            metric_lookup = {}
            if not score.empty and "model" in score.columns:
                metric_lookup = score.drop_duplicates("model").set_index("model").to_dict("index")
            rows = []
            for model in ordered:
                family = model_family_for(uid, model)
                metrics_for_model = metric_lookup.get(model, {})
                row = {
                    "rank": model_rank_label(uid, model, metric),
                    "model": model,
                    "engine": display_family(family),
                    "model_type": model_type_label(model, family),
                    "role": model_role_label(uid, model, active_champion=active_champion, focus_models=focus_models),
                    "intervals": model_interval_level_label(uid, model),
                }
                for col in [metric, "rmse", "mae", "wape", "mase", "rmsse", "bias", "weight", "selection_horizon", "cv_windows"]:
                    if col in metrics_for_model and col not in row:
                        row[col] = metrics_for_model.get(col)
                rows.append(row)
            table = pd.DataFrame(rows)
            return table.drop(columns=[col for col in ["_family_order"] if col in table.columns], errors="ignore")


        def model_picker_guide_view(table: pd.DataFrame, *, limit: int = 40) -> pd.DataFrame:
            if table.empty:
                return table
            preferred = [
                "rank",
                "model",
                "engine",
                "model_type",
                "role",
                "intervals",
                "rmse",
                "mae",
                "wape",
                "mase",
                "rmsse",
                "bias",
                "weight",
                "selection_horizon",
                "cv_windows",
            ]
            cols = [col for col in preferred if col in table.columns]
            view = table[cols].head(limit).copy()
            return view.rename(
                columns={
                    "rank": "Rank",
                    "model": "Model",
                    "engine": "Engine",
                    "model_type": "Type",
                    "role": "Role",
                    "intervals": "Intervals",
                    "rmse": "RMSE",
                    "mae": "MAE",
                    "wape": "WAPE",
                    "mase": "MASE",
                    "rmsse": "RMSSE",
                    "bias": "Bias",
                    "weight": "Weight",
                    "selection_horizon": "Selection horizon",
                    "cv_windows": "CV windows",
                }
            )


        def model_picker_guide_style(table: pd.DataFrame, *, limit: int = 40):
            view = model_picker_guide_view(table, limit=limit)
            if view.empty:
                return view
            return view.style.set_properties(
                subset=["Engine"],
                **{"font-style": "italic", "font-size": "0.85em", "color": "#64748b"},
            )


        def selected_interval_levels(uid: str, model: str | None = None) -> list[int]:
            model = model or selected_model(uid)
            one = forecast[forecast["unique_id"].astype(str) == uid].copy()
            levels: list[int] = []
            if model == selected_model(uid) and not one.empty:
                for level in (80, 95):
                    lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                    if lo_col in one.columns and hi_col in one.columns and one[[lo_col, hi_col]].notna().any().all():
                        levels.append(level)
            fl = forecast_long[forecast_long["unique_id"].astype(str) == uid].copy() if not forecast_long.empty else pd.DataFrame()
            if model and not fl.empty and "model" in fl.columns:
                fl = fl[fl["model"].astype(str) == str(model)]
                for level in (80, 95):
                    lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                    if level not in levels and lo_col in fl.columns and hi_col in fl.columns and fl[[lo_col, hi_col]].notna().any().all():
                        levels.append(level)
            am = all_models[all_models["unique_id"].astype(str) == uid].copy() if not all_models.empty else pd.DataFrame()
            if model and not am.empty:
                for level in (80, 95):
                    lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
                    if level not in levels and lo_col in am.columns and hi_col in am.columns and am[[lo_col, hi_col]].notna().any().all():
                        levels.append(level)
            return levels


        def active_model_forecast_long(uid: str, model: str | None) -> pd.DataFrame:
            if forecast_long.empty or not model or "model" not in forecast_long.columns:
                return pd.DataFrame()
            return forecast_long[
                (forecast_long["unique_id"].astype(str) == str(uid))
                & (forecast_long["model"].astype(str) == str(model))
            ].copy()


        def future_model_frame(uid: str, model: str | None) -> pd.DataFrame:
            model = model or selected_model(uid)
            rows = active_model_forecast_long(uid, model)
            if not rows.empty and "yhat" in rows.columns:
                cols = [
                    col
                    for col in [
                        "unique_id",
                        "model",
                        "ds",
                        "horizon_step",
                        "yhat",
                        "yhat_lo_80",
                        "yhat_hi_80",
                        "yhat_lo_95",
                        "yhat_hi_95",
                        "interval_status",
                        "interval_method",
                        "row_horizon_status",
                        "planning_eligible",
                    ]
                    if col in rows.columns
                ]
                out = rows[cols].copy()
                out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
                return out.sort_values("ds")
            if model == selected_model(uid):
                out = forecast[forecast["unique_id"].astype(str) == uid].copy()
                if not out.empty:
                    out["model"] = model
                    return out.sort_values("ds")
            am = all_models[all_models["unique_id"].astype(str) == uid].copy() if not all_models.empty else pd.DataFrame()
            if am.empty or not model or model not in am.columns:
                return pd.DataFrame()
            out = pd.DataFrame({"unique_id": am["unique_id"], "ds": am["ds"], "model": model, "yhat": pd.to_numeric(am[model], errors="coerce")})
            for level in (80, 95):
                lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
                if lo_col in am.columns and hi_col in am.columns:
                    out[f"yhat_lo_{level}"] = pd.to_numeric(am[lo_col], errors="coerce")
                    out[f"yhat_hi_{level}"] = pd.to_numeric(am[hi_col], errors="coerce")
            return out.sort_values("ds")


        def active_model_interval_status(uid: str, model: str | None) -> str:
            rows = active_model_forecast_long(uid, model)
            if not rows.empty and "interval_status" in rows.columns:
                statuses = [str(value) for value in rows["interval_status"].dropna().astype(str).unique()]
                if "calibration_fail" in statuses:
                    return "calibration_fail"
                if "calibration_warning" in statuses:
                    return "calibration_warning"
                if "future_only" in statuses:
                    return "future_only"
                if "adjusted_not_recalibrated" in statuses:
                    return "adjusted_not_recalibrated"
                if "point_only_ensemble" in statuses:
                    return "point_only_ensemble"
                if "calibrated" in statuses:
                    return "calibrated"
            return "unavailable"


        def active_model_interval_method(uid: str, model: str | None) -> str:
            rows = active_model_forecast_long(uid, model)
            if not rows.empty and "interval_method" in rows.columns:
                methods = [str(value) for value in rows["interval_method"].dropna().astype(str).unique()]
                if methods:
                    return ", ".join(methods[:3])
            trust_row = trust_summary[trust_summary["unique_id"].astype(str) == uid] if not trust_summary.empty else pd.DataFrame()
            if not trust_row.empty and "interval_method" in trust_row.columns:
                method = trust_row.iloc[0].get("interval_method")
                if pd.notna(method):
                    return str(method)
            return "unavailable"


        def active_model_horizon_summary(uid: str, model: str | None) -> str:
            rows = active_model_forecast_long(uid, model)
            if not rows.empty and {"requested_horizon", "validated_through_horizon", "horizon_trust_state"}.issubset(rows.columns):
                first = rows.iloc[0]
                state = str(first.get("horizon_trust_state") or "no_rolling_origin_evidence")
                requested = first.get("requested_horizon", "")
                validated = first.get("validated_through_horizon", "")
                planning_eligible = str(first.get("planning_eligible")).strip().lower() in {"true", "1", "yes"}
                cv_windows = first.get("cv_windows", "")
                cv_windows_display = str(int(float(cv_windows))) if str(cv_windows).replace(".", "", 1).isdigit() else cv_windows
                beyond = 0
                if "is_beyond_validated_horizon" in rows.columns:
                    beyond = int(rows["is_beyond_validated_horizon"].fillna(False).astype(bool).sum())
                if state == "full_horizon_validated":
                    if not planning_eligible:
                        return f"Requested horizon {requested} was evaluated, but only {cv_windows_display} rolling-origin window(s) support it; planning-ready champion claim is limited."
                    return f"Validated through requested horizon {validated} of {requested}."
                if state == "partial_horizon_validated" or beyond:
                    return f"Validated through horizon {validated} of requested {requested}; {beyond} step(s) are directional beyond CV horizon."
                return "No rolling-origin validation evidence for this active champion."
            score = model_score_table(uid)
            if not score.empty and model and "model" in score.columns:
                row = score[score["model"].astype(str) == str(model)]
                if not row.empty:
                    item = row.iloc[0]
                    return f"Validated through horizon {item.get('selection_horizon', 'N/A')} of requested {item.get('requested_horizon', 'N/A')}."
            return "Horizon validation contract unavailable for this active champion."


        def horizon_message_severity(message: str) -> str:
            lowered = str(message).lower()
            if "no rolling-origin" in lowered or "unavailable" in lowered:
                return "error"
            if "directional" in lowered or "limited" in lowered or "only " in lowered:
                return "warning"
            return "caption"


        def other_candidate_count(uid: str, champion: str | None = None, focus_models: list[str] | None = None) -> int:
            am = all_models[all_models["unique_id"].astype(str) == uid] if not all_models.empty else pd.DataFrame()
            if am.empty:
                return 0
            champ = champion or selected_model(uid)
            shown = set(dedupe_models([champ, *top_weighted(uid, champion=champ), *(focus_models or [])]))
            return len([col for col in model_columns(am) if col not in shown])


        def add_intervals(
            fig: go.Figure,
            f: pd.DataFrame,
            *,
            model: str = "Forecast",
            color: str | None = None,
            opacity_scale: float = 1.0,
            showlegend: bool = False,
        ) -> None:
            band_color = color or C["champ"]
            for level, alpha in [(95, 0.08), (80, 0.16)]:
                lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                if lo_col in f.columns and hi_col in f.columns and f[[lo_col, hi_col]].notna().any().all():
                    fig.add_trace(go.Scatter(x=f["ds"], y=f[hi_col], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", legendgroup=model))
                    fig.add_trace(
                        go.Scatter(
                            x=f["ds"],
                            y=f[lo_col],
                            fill="tonexty",
                            fillcolor=color_to_rgba(band_color, alpha * opacity_scale),
                            mode="lines",
                            line=dict(width=0),
                            name=f"{model} {level}% interval",
                            hoverinfo="skip",
                            showlegend=showlegend,
                            legendgroup=model,
                        )
                    )


        def add_model_intervals(
            fig: go.Figure,
            frame: pd.DataFrame,
            model: str,
            *,
            enabled: bool = True,
            color: str | None = None,
            opacity_scale: float = 1.0,
            showlegend: bool = False,
        ) -> None:
            if not enabled or not model or frame.empty:
                return
            band_color = color or C["champ"]
            for level, alpha in [(95, 0.08), (80, 0.16)]:
                lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
                if lo_col in frame.columns and hi_col in frame.columns and frame[[lo_col, hi_col]].notna().any().all():
                    fig.add_trace(go.Scatter(x=frame["ds"], y=pd.to_numeric(frame[hi_col], errors="coerce"), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", legendgroup=model))
                    fig.add_trace(
                        go.Scatter(
                            x=frame["ds"],
                            y=pd.to_numeric(frame[lo_col], errors="coerce"),
                            fill="tonexty",
                            fillcolor=color_to_rgba(band_color, alpha * opacity_scale),
                            mode="lines",
                            line=dict(width=0),
                            name=f"{model} {level}% interval",
                            hoverinfo="skip",
                            showlegend=showlegend,
                            legendgroup=model,
                        )
                    )


        def add_endpoint_label(fig: go.Figure, x, y, text: str, color: str, *, yshift: int = 0) -> None:
            if pd.isna(x) or pd.isna(y):
                return
            fig.add_annotation(
                x=x,
                y=y,
                text=text,
                showarrow=False,
                xanchor="left",
                xshift=8,
                yshift=yshift,
                font=dict(size=11, color=color),
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor="rgba(0,0,0,0.12)",
                borderpad=2,
            )


        def add_cutoff_marker(fig: go.Figure, cutoff) -> None:
            cutoff_ts = pd.Timestamp(cutoff).to_pydatetime()
            fig.add_shape(
                type="line",
                x0=cutoff_ts,
                x1=cutoff_ts,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="#808891", width=1, dash="dash"),
            )
            fig.add_annotation(
                x=cutoff_ts,
                y=1.02,
                xref="x",
                yref="paper",
                text="cutoff",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="#808891"),
            )


        def add_period_shading(fig: go.Figure, x0, x1, color: str, label: str) -> None:
            if pd.isna(x0) or pd.isna(x1):
                return
            x0_ts = pd.Timestamp(x0).to_pydatetime()
            x1_ts = pd.Timestamp(x1).to_pydatetime()
            fig.add_shape(
                type="rect",
                x0=x0_ts,
                x1=x1_ts,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor=color,
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=x0_ts,
                y=1.02,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="#6b7480"),
            )


        def padded_range(values: list[float]) -> list[float] | None:
            clean = [float(v) for v in values if pd.notna(v) and np.isfinite(float(v))]
            if not clean:
                return None
            y_min, y_max = min(clean), max(clean)
            span = y_max - y_min if y_max != y_min else max(abs(y_max), 1.0)
            return [y_min - span * 0.08, y_max + span * 0.08]


        def backtest_axis_ranges(uid: str, *, fixed_x: bool, fixed_y: bool, models: list[str]) -> tuple[list | None, list | None]:
            h = history[history["unique_id"].astype(str) == uid].sort_values("ds")
            bt_all = backtest[backtest["unique_id"].astype(str) == uid].copy()
            x_range = None
            y_range = None
            if fixed_x:
                dates = pd.concat([h["ds"], bt_all["ds"]], ignore_index=True).dropna()
                if not dates.empty:
                    x_range = [dates.min(), dates.max()]
            if fixed_y:
                values: list[float] = []
                if "y" in h:
                    values += pd.to_numeric(h["y"], errors="coerce").dropna().tolist()
                if "y" in bt_all:
                    values += pd.to_numeric(bt_all["y"], errors="coerce").dropna().tolist()
                for col in models:
                    if col in bt_all:
                        values += pd.to_numeric(bt_all[col], errors="coerce").dropna().tolist()
                    for level in (95, 80):
                        for bound in (f"{col}-lo-{level}", f"{col}-hi-{level}"):
                            if bound in bt_all:
                                values += pd.to_numeric(bt_all[bound], errors="coerce").dropna().tolist()
                y_range = padded_range(values)
            return x_range, y_range


        def add_backtest_intervals(fig: go.Figure, bt: pd.DataFrame, model: str, *, enabled: bool) -> None:
            if not enabled or not model:
                return
            for level, color in [(95, C["i95"]), (80, C["i80"])]:
                lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
                if lo_col in bt.columns and hi_col in bt.columns and bt[[lo_col, hi_col]].notna().any().all():
                    fig.add_trace(go.Scatter(x=bt["ds"], y=pd.to_numeric(bt[hi_col], errors="coerce"), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(
                        go.Scatter(
                            x=bt["ds"],
                            y=pd.to_numeric(bt[lo_col], errors="coerce"),
                            fill="tonexty",
                            fillcolor=color,
                            mode="lines",
                            line=dict(width=0),
                            name=f"{model} {level}% interval",
                            hoverinfo="skip",
                        )
                    )


        def forecast_chart(
            uid: str,
            *,
            show_all_models: bool,
            champion: str | None = None,
            focus_models: list[str] | None = None,
        ) -> go.Figure:
            h = history[history["unique_id"].astype(str) == uid].sort_values("ds")
            f = forecast[forecast["unique_id"].astype(str) == uid].sort_values("ds")
            am = all_models[all_models["unique_id"].astype(str) == uid].sort_values("ds") if not all_models.empty else pd.DataFrame()
            sel = champion or selected_model(uid)
            explicit_focus = focus_models is not None
            focus = [model for model in dedupe_models(focus_models) if model != sel]
            top = [model for model in top_weighted(uid, champion=sel) if model not in focus]
            highlighted = focus if explicit_focus else top
            cols = model_columns(all_models)
            display_models = cols if show_all_models else [model for model in dedupe_models([sel, *highlighted]) if model]
            display_models = [model for model in dedupe_models(display_models) if not future_model_frame(uid, model).empty]
            show_interval_legend = not show_all_models and len(display_models) <= 4

            def display_color(model: str) -> str:
                if model == sel:
                    return C["champ"]
                if model in highlighted:
                    return ALT_COLORS[highlighted.index(model) % len(ALT_COLORS)]
                return C["dim"]

            def interval_opacity(model: str) -> float:
                if model == sel:
                    return 1.0
                if model in highlighted:
                    return 0.82
                return 0.35

            fig = go.Figure()
            for model in display_models:
                frame = future_model_frame(uid, model)
                color = display_color(model)
                opacity = interval_opacity(model)
                add_intervals(fig, frame, model=model, color=color, opacity_scale=opacity, showlegend=show_interval_legend)
            if not h.empty:
                fig.add_trace(go.Scatter(x=h["ds"], y=h["y"], name="History", line=dict(color=C["hist"], width=2.6)))
            if show_all_models:
                for col in display_models:
                    if col == sel or col in highlighted:
                        continue
                    frame = future_model_frame(uid, col)
                    if frame.empty or "yhat" not in frame.columns:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=frame["ds"],
                            y=pd.to_numeric(frame["yhat"], errors="coerce"),
                            name=col,
                            line=dict(color=C["dim"], width=1, dash="dot"),
                            opacity=0.32,
                            showlegend=False,
                        )
                    )
            for idx, model in enumerate(highlighted):
                frame = future_model_frame(uid, model)
                if not frame.empty and "yhat" in frame.columns:
                    y = pd.to_numeric(frame["yhat"], errors="coerce")
                    fig.add_trace(
                        go.Scatter(
                            x=frame["ds"],
                            y=y,
                            name=model,
                            line=dict(color=ALT_COLORS[idx % len(ALT_COLORS)], width=2.2, dash="dash"),
                        )
                    )
                    add_endpoint_label(fig, frame["ds"].iloc[-1], y.iloc[-1], model, ALT_COLORS[idx % len(ALT_COLORS)], yshift=(idx - 1) * 10)
            selected_frame = future_model_frame(uid, sel)
            if sel and not selected_frame.empty and "yhat" in selected_frame.columns:
                y = pd.to_numeric(selected_frame["yhat"], errors="coerce")
                fig.add_trace(
                    go.Scatter(
                        x=selected_frame["ds"],
                        y=y,
                        name=f"{sel} (active champion)",
                        line=dict(color=C["champ"], width=3.2),
                    )
                )
                add_endpoint_label(fig, selected_frame["ds"].iloc[-1], y.iloc[-1], f"{sel} (champion)", C["champ"], yshift=14)
            elif not f.empty and "yhat" in f.columns:
                fig.add_trace(go.Scatter(x=f["ds"], y=f["yhat"], name="Forecast", line=dict(color=C["champ"], width=3.2)))
                add_endpoint_label(fig, f["ds"].iloc[-1], f["yhat"].iloc[-1], "Forecast", C["champ"])
            fig.update_layout(
                height=430,
                margin=dict(l=40, r=190, t=25, b=40),
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.18),
                yaxis_title="value",
            )
            return fig


        def backtest_chart(
            uid: str,
            cutoff,
            *,
            show_all_models: bool,
            fixed_x: bool = True,
            fixed_y: bool = True,
            show_intervals: bool = True,
            champion: str | None = None,
            focus_models: list[str] | None = None,
        ) -> go.Figure:
            h = history[history["unique_id"].astype(str) == uid].sort_values("ds")
            bt_all = backtest[backtest["unique_id"].astype(str) == uid].copy()
            bt = bt_all[bt_all["cutoff"] == cutoff].sort_values("ds")
            sel = champion or selected_model(uid)
            explicit_focus = focus_models is not None
            focus = [model for model in dedupe_models(focus_models) if model != sel]
            top = [model for model in top_weighted(uid, champion=sel) if model not in focus]
            highlighted = focus if explicit_focus else top
            cols = model_columns(backtest, backtest=True)
            display_models = cols if show_all_models else [m for m in [sel, *highlighted] if m]
            x_range, y_range = backtest_axis_ranges(uid, fixed_x=fixed_x, fixed_y=fixed_y, models=display_models)
            train = h[h["ds"] <= cutoff]
            fig = go.Figure()
            if not h.empty:
                fig.add_trace(
                    go.Scatter(
                        x=h["ds"],
                        y=pd.to_numeric(h["y"], errors="coerce"),
                        name="Full actual history",
                        line=dict(color=C["hist"], width=1.3),
                        opacity=0.28,
                    )
                )
            if not train.empty:
                add_period_shading(fig, h["ds"].min(), cutoff, C["train"], "train")
                fig.add_trace(go.Scatter(x=train["ds"], y=pd.to_numeric(train["y"], errors="coerce"), name="Training actuals", line=dict(color=C["hist"], width=2.3)))
            if not bt.empty:
                add_period_shading(fig, bt["ds"].min(), bt["ds"].max(), C["test"], "holdout")
                fig.add_trace(
                    go.Scatter(
                        x=bt["ds"],
                        y=pd.to_numeric(bt["y"], errors="coerce"),
                        name="Actual holdout",
                        mode="lines+markers",
                        line=dict(color=C["hist"], width=3.0),
                        marker=dict(size=5),
                    )
                )
            if show_all_models and not bt.empty:
                for col in cols:
                    if col == sel or col in highlighted or col not in bt.columns:
                        continue
                    y = pd.to_numeric(bt[col], errors="coerce")
                    fig.add_trace(
                        go.Scatter(
                            x=bt["ds"],
                            y=y,
                            name=col,
                            line=dict(color=C["dim"], width=1, dash="dot"),
                            opacity=0.28,
                            showlegend=False,
                        )
                    )
            for idx, model in enumerate(highlighted):
                if model in bt.columns:
                    y = pd.to_numeric(bt[model], errors="coerce")
                    fig.add_trace(
                        go.Scatter(
                            x=bt["ds"],
                            y=y,
                            name=model,
                            line=dict(color=ALT_COLORS[idx % len(ALT_COLORS)], width=2.3, dash="dash"),
                        )
                    )
                    add_endpoint_label(fig, bt["ds"].iloc[-1], y.iloc[-1], model, ALT_COLORS[idx % len(ALT_COLORS)], yshift=(idx - 1) * 11)
            if sel and sel in bt.columns:
                add_backtest_intervals(fig, bt, sel, enabled=show_intervals)
                y = pd.to_numeric(bt[sel], errors="coerce")
                fig.add_trace(
                    go.Scatter(
                        x=bt["ds"],
                        y=y,
                        name=f"{sel} (active champion)",
                        line=dict(color=C["champ"], width=3.4),
                    )
                )
                add_endpoint_label(fig, bt["ds"].iloc[-1], y.iloc[-1], f"{sel} (champion)", C["champ"], yshift=16)
            add_cutoff_marker(fig, cutoff)
            fig.update_layout(
                height=500,
                margin=dict(l=45, r=210, t=30, b=45),
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.16),
                yaxis_title="value",
                xaxis_title="date",
            )
            if x_range is not None:
                fig.update_xaxes(range=x_range)
            if y_range is not None:
                fig.update_yaxes(range=y_range)
            return fig


        def hierarchy_parent_options() -> list[str]:
            if not hierarchy_coherence.empty and "parent_unique_id" in hierarchy_coherence.columns:
                return sorted(hierarchy_coherence["parent_unique_id"].astype(str).dropna().unique())
            if "hierarchy_depth" not in forecast.columns:
                return []
            depth = pd.to_numeric(forecast["hierarchy_depth"], errors="coerce")
            return sorted(forecast.loc[depth < depth.max(), "unique_id"].astype(str).dropna().unique())


        def hierarchy_children(parent_id: str) -> list[str]:
            if "hierarchy_depth" not in forecast.columns:
                return []
            meta = forecast[["unique_id", "hierarchy_depth"]].drop_duplicates().copy()
            meta["hierarchy_depth"] = pd.to_numeric(meta["hierarchy_depth"], errors="coerce")
            root_rows = meta[meta["hierarchy_depth"] == 0]
            root_id = str(root_rows.iloc[0]["unique_id"]) if not root_rows.empty else "Total"
            parent_rows = meta[meta["unique_id"].astype(str) == parent_id]
            if parent_rows.empty:
                return []
            parent_depth = int(parent_rows.iloc[0]["hierarchy_depth"])
            if parent_id == root_id:
                child_mask = meta["hierarchy_depth"] == 1
            else:
                child_mask = meta["unique_id"].astype(str).str.startswith(parent_id + "|") & (meta["hierarchy_depth"] == parent_depth + 1)
            return sorted(meta.loc[child_mask, "unique_id"].astype(str).tolist())


        def hierarchy_rollup_chart(parent_id: str) -> go.Figure:
            fig = go.Figure()
            if not hierarchy_coherence.empty and "parent_unique_id" in hierarchy_coherence.columns:
                c = hierarchy_coherence[hierarchy_coherence["parent_unique_id"].astype(str) == parent_id].sort_values("ds")
                if not c.empty:
                    fig.add_trace(go.Scatter(x=c["ds"], y=pd.to_numeric(c["parent_value"], errors="coerce"), name="Parent selected forecast", line=dict(color=C["champ"], width=3)))
                    fig.add_trace(go.Scatter(x=c["ds"], y=pd.to_numeric(c["child_sum"], errors="coerce"), name="Immediate child sum", line=dict(color=C["hist"], width=2.5, dash="dash")))
                    fig.add_trace(go.Bar(x=c["ds"], y=pd.to_numeric(c["gap"], errors="coerce"), name="Parent minus children gap", marker_color="rgba(98,81,143,0.35)", yaxis="y2"))
                    fig.update_layout(yaxis2=dict(title="gap", overlaying="y", side="right", showgrid=False))
            fig.update_layout(height=430, margin=dict(l=45, r=70, t=30, b=45), hovermode="x unified", yaxis_title="forecast value")
            return fig


        def hierarchy_children_chart(parent_id: str) -> go.Figure:
            fig = go.Figure()
            child_ids = hierarchy_children(parent_id)
            parent = forecast[forecast["unique_id"].astype(str) == parent_id].sort_values("ds")
            if not parent.empty:
                fig.add_trace(go.Scatter(x=parent["ds"], y=pd.to_numeric(parent["yhat"], errors="coerce"), name=f"{parent_id} parent", line=dict(color=C["champ"], width=3)))
            for idx, child_id in enumerate(child_ids):
                child = forecast[forecast["unique_id"].astype(str) == child_id].sort_values("ds")
                if child.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=child["ds"],
                        y=pd.to_numeric(child["yhat"], errors="coerce"),
                        name=child_id,
                        line=dict(color=ALT_COLORS[idx % len(ALT_COLORS)], width=2, dash="dot"),
                    )
                )
            fig.update_layout(height=430, margin=dict(l=45, r=160, t=30, b=45), hovermode="x unified", yaxis_title="selected forecast")
            return fig


        def feature_importance_chart(models: list[str] | None = None) -> go.Figure:
            fig = go.Figure()
            if model_explainability.empty or "importance" not in model_explainability.columns:
                return fig
            frame = model_explainability.copy()
            selected_models = dedupe_models(models)
            if selected_models and "model" in frame.columns:
                frame = frame[frame["model"].astype(str).isin(selected_models)]
            if frame.empty:
                return fig
            frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
            top = frame.sort_values(["model", "importance"], ascending=[True, False]).groupby("model").head(12)
            fig.add_trace(
                go.Bar(
                    x=top["importance"],
                    y=top["model"].astype(str) + " | " + top["feature"].astype(str),
                    orientation="h",
                    marker_color=C["purple"] if "purple" in C else "#62518f",
                    text=top["interpretation"] if "interpretation" in top.columns else None,
                    hovertemplate="%{y}<br>importance=%{x}<br>%{text}<extra></extra>",
                )
            )
            fig.update_layout(height=max(320, 26 * len(top)), margin=dict(l=220, r=30, t=25, b=35), xaxis_title="feature importance / absolute coefficient")
            return fig


        def benchmark_win_rate_chart() -> go.Figure:
            fig = go.Figure()
            if model_win_rates.empty or "win_rate_vs_benchmark" not in model_win_rates.columns:
                return fig
            frame = model_win_rates.copy()
            frame["win_rate_vs_benchmark"] = pd.to_numeric(frame["win_rate_vs_benchmark"], errors="coerce")
            if "avg_skill_vs_benchmark" in frame.columns:
                frame["avg_skill_vs_benchmark"] = pd.to_numeric(frame["avg_skill_vs_benchmark"], errors="coerce")
            else:
                frame["avg_skill_vs_benchmark"] = np.nan
            frame = frame.sort_values(["win_rate_vs_benchmark", "avg_skill_vs_benchmark", "model"], ascending=[True, True, False]).tail(18)
            fig.add_trace(
                go.Bar(
                    x=frame["win_rate_vs_benchmark"],
                    y=frame["model"].astype(str),
                    orientation="h",
                    marker_color=C["alt1"],
                    text=frame["metric"].astype(str) + " vs " + frame["benchmark_model"].astype(str),
                    hovertemplate="%{y}<br>win rate=%{x:.1%}<br>%{text}<extra></extra>",
                )
            )
            fig.update_layout(height=max(320, 26 * len(frame)), margin=dict(l=170, r=30, t=25, b=35), xaxis_title="win rate vs benchmark", xaxis_tickformat=".0%")
            return fig


        def residual_horizon_chart(uid: str, models: list[str] | None = None) -> go.Figure:
            fig = go.Figure()
            if residual_diagnostics.empty or "horizon_step" not in residual_diagnostics.columns:
                return fig
            frame = residual_diagnostics[residual_diagnostics["unique_id"].astype(str) == uid].copy()
            selected_models = dedupe_models(models) or [selected_model(uid)]
            if selected_models and "model" in frame.columns:
                selected = frame[frame["model"].astype(str).isin(selected_models)]
                if not selected.empty:
                    frame = selected
            if frame.empty:
                return fig
            frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="coerce")
            if "model" in frame.columns and frame["model"].nunique() > 1:
                for idx, (model, group) in enumerate(frame.groupby("model")):
                    if "rmse" not in group.columns:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=group["horizon_step"],
                            y=pd.to_numeric(group["rmse"], errors="coerce"),
                            name=f"{model} RMSE",
                            mode="lines+markers",
                            line=dict(color=([C["champ"], *ALT_COLORS, C["dim"]][idx % 5]), width=2.4),
                        )
                    )
                fig.update_layout(height=390, margin=dict(l=45, r=70, t=25, b=45), hovermode="x unified", xaxis_title="horizon step", yaxis_title="RMSE")
                return fig
            metrics_to_plot = [
                ("rmse", C["champ"]),
                ("mae", C["hist"]),
                ("mase", C["alt1"]),
                ("rmsse", C["alt2"]),
            ]
            for metric, color in metrics_to_plot:
                if metric in frame.columns and pd.to_numeric(frame[metric], errors="coerce").notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=frame["horizon_step"],
                            y=pd.to_numeric(frame[metric], errors="coerce"),
                            name=metric.upper(),
                            mode="lines+markers",
                            line=dict(color=color, width=2.4),
                        )
                    )
            if "mean_error" in frame.columns and pd.to_numeric(frame["mean_error"], errors="coerce").notna().any():
                fig.add_trace(
                    go.Bar(
                        x=frame["horizon_step"],
                        y=pd.to_numeric(frame["mean_error"], errors="coerce"),
                        name="Mean residual",
                        marker_color="rgba(98,81,143,0.28)",
                        yaxis="y2",
                    )
                )
                fig.update_layout(yaxis2=dict(title="mean residual", overlaying="y", side="right", showgrid=False))
            fig.update_layout(height=390, margin=dict(l=45, r=70, t=25, b=45), hovermode="x unified", xaxis_title="horizon step", yaxis_title="error metric")
            return fig


        def selected_residual_rows(uid: str, models: list[str] | None = None) -> pd.DataFrame:
            if backtest_long.empty or "error" not in backtest_long.columns:
                return pd.DataFrame()
            frame = backtest_long[backtest_long["unique_id"].astype(str) == uid].copy()
            selected_models = dedupe_models(models) or [selected_model(uid)]
            if selected_models and "model" in frame.columns:
                selected = frame[frame["model"].astype(str).isin(selected_models)]
                if not selected.empty:
                    frame = selected
            if "ds" in frame.columns:
                frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
            if "cutoff" in frame.columns:
                frame["cutoff"] = pd.to_datetime(frame["cutoff"], errors="coerce")
            frame["error"] = pd.to_numeric(frame["error"], errors="coerce")
            return frame.dropna(subset=["error"]).sort_values([col for col in ["cutoff", "ds"] if col in frame.columns])


        def residual_time_chart(uid: str, models: list[str] | None = None) -> go.Figure:
            fig = go.Figure()
            frame = selected_residual_rows(uid, models)
            if frame.empty:
                return fig
            groups = frame.groupby("model") if "model" in frame.columns else [("Selected", frame)]
            colors = [C["champ"], *ALT_COLORS, C["dim"]]
            for idx, (model, group) in enumerate(groups):
                fig.add_trace(
                    go.Scatter(
                        x=group["ds"] if "ds" in group.columns else list(range(len(group))),
                        y=group["error"],
                        mode="markers+lines",
                        name=f"{model} residual",
                        marker=dict(color=colors[idx % len(colors)], size=7, opacity=0.78),
                        line=dict(color=colors[idx % len(colors)], width=1),
                        text=group["cutoff"].dt.strftime("%Y-%m-%d") if "cutoff" in group.columns else None,
                        hovertemplate="date=%{x}<br>actual - forecast=%{y:,.3f}<br>cutoff=%{text}<extra></extra>",
                    )
                )
            fig.add_shape(type="line", x0=frame["ds"].min(), x1=frame["ds"].max(), y0=0, y1=0, xref="x", yref="y", line=dict(color=C["hist"], width=1.2, dash="dash"))
            fig.update_layout(height=360, margin=dict(l=45, r=30, t=25, b=45), xaxis_title="holdout date", yaxis_title="actual - forecast", hovermode="x unified")
            return fig


        def residual_distribution_chart(uid: str, models: list[str] | None = None) -> go.Figure:
            fig = go.Figure()
            frame = selected_residual_rows(uid, models)
            if frame.empty or "error" not in frame.columns:
                return fig
            groups = frame.groupby("model") if "model" in frame.columns else [("Residuals", frame)]
            colors = [C["champ"], *ALT_COLORS, C["dim"]]
            for idx, (model, group) in enumerate(groups):
                errors = group["error"].dropna()
                if errors.empty:
                    continue
                fig.add_trace(go.Histogram(x=errors, nbinsx=min(30, max(8, int(len(errors) ** 0.5) + 2)), name=str(model), marker_color=colors[idx % len(colors)], opacity=0.58))
            fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, xref="x", yref="paper", line=dict(color=C["hist"], width=1.5, dash="dash"))
            fig.add_annotation(x=0, y=1.03, xref="x", yref="paper", text="zero error", showarrow=False, font=dict(size=10, color=C["hist"]))
            fig.update_layout(height=390, margin=dict(l=45, r=30, t=25, b=45), xaxis_title="actual - forecast", yaxis_title="count", bargap=0.05, barmode="overlay")
            return fig


        def residual_acf_values(uid: str, models: list[str] | None = None, max_lag: int = 24) -> pd.DataFrame:
            primary = dedupe_models(models)[:1]
            frame = selected_residual_rows(uid, primary)
            errors = frame["error"].dropna().to_numpy(dtype="float64") if "error" in frame.columns else np.array([])
            n = len(errors)
            if n < 3:
                return pd.DataFrame(columns=["lag", "acf", "threshold", "significant"])
            centered = errors - np.nanmean(errors)
            denom = float(np.nansum(centered ** 2))
            if denom <= 0:
                return pd.DataFrame(columns=["lag", "acf", "threshold", "significant"])
            threshold = 1.96 / np.sqrt(n)
            rows = []
            for lag in range(1, min(max_lag, n - 1) + 1):
                acf = float(np.nansum(centered[lag:] * centered[:-lag]) / denom)
                rows.append({"lag": lag, "acf": acf, "threshold": threshold, "significant": abs(acf) > threshold})
            return pd.DataFrame(rows)


        def residual_acf_chart(uid: str, models: list[str] | None = None) -> go.Figure:
            fig = go.Figure()
            acf = residual_acf_values(uid, models)
            if acf.empty:
                return fig
            colors = [C["champ"] if sig else C["dim"] for sig in acf["significant"]]
            fig.add_trace(go.Bar(x=acf["lag"], y=acf["acf"], marker_color=colors, name="Residual ACF"))
            threshold = float(acf["threshold"].iloc[0])
            fig.add_shape(type="line", x0=0.5, x1=float(acf["lag"].max()) + 0.5, y0=threshold, y1=threshold, xref="x", yref="y", line=dict(color=C["hist"], width=1, dash="dash"))
            fig.add_shape(type="line", x0=0.5, x1=float(acf["lag"].max()) + 0.5, y0=-threshold, y1=-threshold, xref="x", yref="y", line=dict(color=C["hist"], width=1, dash="dash"))
            fig.update_layout(height=360, margin=dict(l=45, r=30, t=25, b=45), xaxis_title="lag", yaxis_title="autocorrelation")
            return fig


        def residual_white_noise_verdict(uid: str, models: list[str] | None = None) -> str:
            acf = residual_acf_values(uid, models)
            primary = (dedupe_models(models) or [selected_model(uid) or "selected model"])[0]
            if acf.empty:
                return f"Residual white-noise check for {primary}: insufficient residual observations."
            significant = acf[acf["significant"]]
            if significant.empty:
                return f"Residual white-noise check for {primary}: no lag exceeded the approximate 95% autocorrelation threshold."
            lags = ", ".join(str(int(lag)) for lag in significant["lag"].head(6))
            return f"Residual white-noise check for {primary}: autocorrelation flagged at lag(s) {lags}; inspect seasonality, drivers, or model class."


        def residual_outlier_table(uid: str, models: list[str] | None = None) -> pd.DataFrame:
            frame = selected_residual_rows(uid, models)
            if frame.empty:
                return pd.DataFrame()
            std = float(frame["error"].std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                return pd.DataFrame()
            out = frame.copy()
            out["standardized_residual"] = out["error"] / std
            out = out[out["standardized_residual"].abs() >= 3.0]
            cols = [col for col in ["unique_id", "model", "cutoff", "ds", "y_actual", "yhat", "error", "standardized_residual"] if col in out.columns]
            return out[cols].sort_values("standardized_residual", key=lambda s: s.abs(), ascending=False)


        def seasonality_decomposition_chart(uid: str) -> go.Figure:
            fig = go.Figure()
            if seasonality_decomposition.empty or "unique_id" not in seasonality_decomposition.columns:
                return fig
            frame = seasonality_decomposition[seasonality_decomposition["unique_id"].astype(str) == uid].copy()
            if frame.empty:
                return fig
            frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
            for column, label, color in [
                ("observed", "observed", C["hist"]),
                ("trend", "trend", C["champ"]),
                ("seasonal", "seasonal component", C["alt1"]),
                ("remainder", "remainder", C["alt2"]),
            ]:
                if column not in frame.columns:
                    continue
                values = pd.to_numeric(frame[column], errors="coerce")
                fig.add_trace(go.Scatter(x=frame["ds"], y=values, mode="lines", name=label, line=dict(color=color, width=2 if column in {"observed", "trend"} else 1.5)))
            fig.update_layout(height=430, margin=dict(l=45, r=30, t=25, b=45), xaxis_title="date", yaxis_title="value / component", hovermode="x unified")
            return fig


        def seasonality_year_profile_source_frame(
            frame: pd.DataFrame,
            *,
            start_month: int,
            value_col: str,
            source: str,
            model: str | None = None,
        ) -> pd.DataFrame:
            if frame.empty or "ds" not in frame.columns or value_col not in frame.columns:
                return pd.DataFrame()
            frame = frame.copy()
            frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
            frame["value"] = pd.to_numeric(frame[value_col], errors="coerce")
            frame = frame.dropna(subset=["ds", "value"]).sort_values("ds")
            if frame.empty:
                return pd.DataFrame()
            start_month = int(start_month)
            frame["season_year_start"] = np.where(frame["ds"].dt.month >= start_month, frame["ds"].dt.year, frame["ds"].dt.year - 1)
            frame["season_year_label"] = frame["season_year_start"].map(
                lambda year: str(int(year)) if start_month == 1 else f"{int(year)}-{str((int(year) + 1) % 100).zfill(2)}"
            )
            diffs = frame["ds"].dropna().sort_values().diff().dt.days.dropna()
            median_gap = float(diffs.median()) if not diffs.empty else 31.0
            if median_gap >= 27:
                frame["season_position"] = ((frame["ds"].dt.month - start_month) % 12) + 1
                frame["season_label"] = frame["season_position"].map(lambda value: MONTH_LABELS[(start_month + int(value) - 2) % 12])
                frame["season_axis_title"] = "month in configured year"
            else:
                anchors = frame["season_year_start"].map(lambda year: pd.Timestamp(year=int(year), month=start_month, day=1))
                frame["season_position"] = (frame["ds"] - anchors).dt.days + 1
                frame["season_label"] = frame["season_position"].astype(int).astype(str)
                frame["season_axis_title"] = "day in configured year"
            frame["source"] = source
            frame["model"] = model or ""
            frame["value_kind"] = "forecast yhat" if source == "forecast" else "actual"
            if source == "forecast":
                label_model = model or "selected model"
                frame["line_label"] = frame["season_year_label"].map(lambda label: f"{label_model} forecast {label}")
            else:
                frame["line_label"] = frame["season_year_label"].astype(str)
            return frame[
                [
                    "unique_id",
                    "ds",
                    "value",
                    "source",
                    "model",
                    "value_kind",
                    "season_year_start",
                    "season_year_label",
                    "line_label",
                    "season_position",
                    "season_label",
                    "season_axis_title",
                ]
            ]


        def seasonality_year_profile_frame(
            uid: str,
            start_month: int,
            *,
            model: str | None = None,
            include_history: bool = True,
        ) -> pd.DataFrame:
            frames: list[pd.DataFrame] = []
            if include_history:
                history_frame = history[history["unique_id"].astype(str) == uid].copy()
                frames.append(
                    seasonality_year_profile_source_frame(
                        history_frame,
                        start_month=start_month,
                        value_col="y",
                        source="actual",
                    )
                )
            if model:
                forecast_frame = future_model_frame(uid, model)
                frames.append(
                    seasonality_year_profile_source_frame(
                        forecast_frame,
                        start_month=start_month,
                        value_col="yhat",
                        source="forecast",
                        model=model,
                    )
                )
            frames = [frame for frame in frames if not frame.empty]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True).sort_values(["source", "season_year_start", "season_position"])


        def seasonality_year_profile_chart(
            uid: str,
            start_month: int,
            *,
            model: str | None = None,
            include_history: bool = True,
        ) -> go.Figure:
            fig = go.Figure()
            frame = seasonality_year_profile_frame(uid, start_month, model=model, include_history=include_history)
            if frame.empty:
                return fig
            palette = [C["hist"], C["champ"], C["alt1"], C["alt2"], C["alt3"], "#3f7f93", "#8a5f2a", "#7a7f87"]
            actual_idx = 0
            forecast_idx = 0
            for line_label, group in frame.groupby("line_label", sort=False):
                group = group.sort_values("season_position")
                is_forecast = str(group["source"].iloc[0]) == "forecast"
                if is_forecast:
                    color = [C["champ"], C["alt1"], C["alt2"], C["alt3"]][forecast_idx % 4]
                    forecast_idx += 1
                    line = dict(color=color, width=3.2, dash="dash")
                    marker = dict(size=7, symbol="diamond")
                    opacity = 0.98
                else:
                    color = palette[actual_idx % len(palette)]
                    actual_idx += 1
                    line = dict(color=color, width=1.8)
                    marker = dict(size=5)
                    opacity = 0.58 if model else 0.78
                fig.add_trace(
                    go.Scatter(
                        x=group["season_position"],
                        y=group["value"],
                        mode="lines+markers",
                        name=str(line_label),
                        line=line,
                        marker=marker,
                        opacity=opacity,
                        text=group["ds"].dt.strftime("%Y-%m-%d"),
                        customdata=group[["value_kind", "model"]].to_numpy(),
                        hovertemplate="date=%{text}<br>period=%{x}<br>%{customdata[0]}=%{y:,.3f}<extra></extra>",
                    )
                )
            axis_title = str(frame["season_axis_title"].iloc[0])
            fig.update_layout(
                height=430,
                margin=dict(l=45, r=180, t=25, b=45),
                xaxis_title=axis_title,
                yaxis_title="actual / forecast value",
                hovermode="closest",
                legend_title="seasonal year / forecast model",
            )
            if axis_title.startswith("month"):
                tickvals = list(range(1, 13))
                ticktext = [MONTH_LABELS[(int(start_month) + idx - 1) % 12] for idx in range(12)]
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            return fig


        def interval_calibration_chart(uid: str, models: list[str] | None = None) -> go.Figure:
            fig = go.Figure()
            if interval_diagnostics.empty or "empirical_coverage" not in interval_diagnostics.columns:
                return fig
            frame = interval_diagnostics[interval_diagnostics["unique_id"].astype(str) == uid].copy()
            selected_models = dedupe_models(models) or [selected_model(uid)]
            if selected_models and "model" in frame.columns:
                selected = frame[frame["model"].astype(str).isin(selected_models)]
                if not selected.empty:
                    frame = selected
            if frame.empty:
                return fig
            frame["target_coverage"] = pd.to_numeric(frame["target_coverage"], errors="coerce")
            frame["empirical_coverage"] = pd.to_numeric(frame["empirical_coverage"], errors="coerce")
            frame = frame.dropna(subset=["target_coverage", "empirical_coverage"])
            if frame.empty:
                return fig
            colors = {
                "pass": C["alt1"],
                "warn_miscalibrated": C["gold"] if "gold" in C else "#9a6b21",
                "fail_undercoverage": C["champ"],
                "insufficient_observations": C["dim"],
                "unavailable": C["dim"],
            }
            marker_colors = [colors.get(str(status), C["dim"]) for status in frame.get("coverage_status", pd.Series(["unavailable"] * len(frame)))]
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="ideal calibration", mode="lines", line=dict(color=C["dim"], dash="dash")))
            fig.add_trace(
                go.Scatter(
                    x=frame["target_coverage"],
                    y=frame["empirical_coverage"],
                    mode="markers",
                    name="interval coverage",
                    marker=dict(size=10, color=marker_colors, line=dict(color="white", width=1)),
                    text=(
                        "model=" + frame.get("model", pd.Series([""] * len(frame))).astype(str)
                        + "<br>level=" + frame.get("level", pd.Series([""] * len(frame))).astype(str)
                        + "<br>h=" + frame.get("horizon_step", pd.Series([""] * len(frame))).astype(str)
                        + "<br>status=" + frame.get("interval_status", frame.get("coverage_status", pd.Series([""] * len(frame)))).astype(str)
                        + "<br>method=" + frame.get("interval_method", pd.Series([""] * len(frame))).astype(str)
                    ),
                    hovertemplate="target=%{x:.0%}<br>empirical=%{y:.0%}<br>%{text}<extra></extra>",
                )
            )
            fig.update_layout(height=390, margin=dict(l=45, r=30, t=25, b=45), xaxis_title="nominal coverage", yaxis_title="empirical coverage", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            return fig


        def interval_focus_frame(uid: str, model: str | None) -> pd.DataFrame:
            return future_model_frame(uid, model)


        def interval_bearing_models(uid: str) -> list[str]:
            extras: list[str] = []
            if not forecast_long.empty and "model" in forecast_long.columns:
                one = forecast_long[forecast_long["unique_id"].astype(str) == uid] if "unique_id" in forecast_long.columns else forecast_long
                extras.extend(one["model"].astype(str).dropna().tolist())
            am = all_models[all_models["unique_id"].astype(str) == uid].copy() if not all_models.empty else pd.DataFrame()
            if not am.empty:
                extras.extend(model_columns(am))
            ranked = candidate_models(uid)
            models = dedupe_models([selected_model(uid), *ranked, *extras])
            return [model for model in models if selected_interval_levels(uid, model)]


        def coerce_interval_models(uid: str, models) -> list[str]:
            if isinstance(models, str):
                requested = [models]
            elif models is None:
                requested = []
            else:
                requested = list(models)
            available = interval_bearing_models(uid)
            requested = dedupe_models(requested) or available
            return [model for model in requested if model in available]


        def interval_focus_models_frame(uid: str, models=None) -> pd.DataFrame:
            frames = []
            for model in coerce_interval_models(uid, models):
                frame = interval_focus_frame(uid, model)
                if frame.empty:
                    continue
                has_interval = False
                for level in (80, 95):
                    lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                    if lo_col in frame.columns and hi_col in frame.columns and frame[[lo_col, hi_col]].notna().any().all():
                        has_interval = True
                        break
                if has_interval:
                    frames.append(frame)
            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, ignore_index=True, sort=False)
            out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
            return out.sort_values(["model", "ds"])


        def prediction_interval_focus_chart(uid: str, models=None) -> go.Figure:
            fig = go.Figure()
            selected_models = coerce_interval_models(uid, models)
            f = interval_focus_models_frame(uid, selected_models)
            h = history[history["unique_id"].astype(str) == uid].sort_values("ds")
            if not h.empty:
                future_periods = f["ds"].nunique() if not f.empty and "ds" in f.columns else 1
                tail_n = max(36, min(len(h), 4 * max(future_periods, 1)))
                h_tail = h.tail(tail_n)
                fig.add_trace(go.Scatter(x=h_tail["ds"], y=pd.to_numeric(h_tail["y"], errors="coerce"), name="recent history", line=dict(color=C["hist"], width=2.2)))
            if f.empty or "yhat" not in f.columns:
                return fig
            palette = [C["champ"], *ALT_COLORS, C["gold"], C["dim"]]
            overall = selected_model(uid)
            for idx, model_name in enumerate(selected_models):
                mf = f[f["model"].astype(str) == str(model_name)].sort_values("ds")
                if mf.empty:
                    continue
                line_color = C["champ"] if model_name == overall else palette[(idx + 1) % len(palette)]
                for level, alpha in [(95, 0.07), (80, 0.14)]:
                    lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                    if lo_col in mf.columns and hi_col in mf.columns and mf[[lo_col, hi_col]].notna().any().all():
                        fig.add_trace(
                            go.Scatter(
                                x=mf["ds"],
                                y=pd.to_numeric(mf[hi_col], errors="coerce"),
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=model_name,
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=mf["ds"],
                                y=pd.to_numeric(mf[lo_col], errors="coerce"),
                                fill="tonexty",
                                fillcolor=color_to_rgba(line_color, alpha),
                                mode="lines",
                                line=dict(width=0),
                                name=f"{model_name} {level}% PI",
                                hoverinfo="skip",
                                legendgroup=model_name,
                            )
                        )
                fig.add_trace(
                    go.Scatter(
                        x=mf["ds"],
                        y=pd.to_numeric(mf["yhat"], errors="coerce"),
                        name=f"{model_name}{' (champion)' if model_name == overall else ''}",
                        line=dict(color=line_color, width=3.1 if model_name == overall else 2.2),
                        mode="lines+markers",
                        legendgroup=model_name,
                    )
                )
            fig.update_layout(height=520, margin=dict(l=45, r=210, t=25, b=45), hovermode="x unified", yaxis_title="value", xaxis_title="date")
            return fig


        def prediction_interval_width_chart(uid: str, models=None) -> go.Figure:
            fig = go.Figure()
            selected_models = coerce_interval_models(uid, models)
            f = interval_focus_models_frame(uid, selected_models)
            if f.empty:
                return fig
            palette = [C["champ"], *ALT_COLORS, C["gold"], C["dim"]]
            overall = selected_model(uid)
            for idx, model_name in enumerate(selected_models):
                mf = f[f["model"].astype(str) == str(model_name)].sort_values("ds")
                if mf.empty:
                    continue
                color = C["champ"] if model_name == overall else palette[(idx + 1) % len(palette)]
                for level, dash in [(80, "solid"), (95, "dash")]:
                    lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                    if lo_col not in mf.columns or hi_col not in mf.columns:
                        continue
                    width = pd.to_numeric(mf[hi_col], errors="coerce") - pd.to_numeric(mf[lo_col], errors="coerce")
                    if width.notna().any():
                        fig.add_trace(go.Scatter(x=mf["ds"], y=width, mode="lines+markers", name=f"{model_name} {level}% width", line=dict(color=color, width=2.4, dash=dash), legendgroup=model_name))
            fig.update_layout(height=380, margin=dict(l=45, r=170, t=25, b=45), hovermode="x unified", xaxis_title="date", yaxis_title="upper - lower")
            return fig


        def prediction_interval_summary_table(uid: str, models=None) -> pd.DataFrame:
            selected_models = coerce_interval_models(uid, models)
            f = interval_focus_models_frame(uid, selected_models)
            rows = []
            for model_name in selected_models:
                mf = f[f["model"].astype(str) == str(model_name)].sort_values("ds") if not f.empty and "model" in f.columns else pd.DataFrame()
                for level in selected_interval_levels(uid, model_name):
                    lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                    if mf.empty or lo_col not in mf.columns or hi_col not in mf.columns:
                        continue
                    yhat = pd.to_numeric(mf.get("yhat", pd.Series(dtype=float)), errors="coerce")
                    width = pd.to_numeric(mf[hi_col], errors="coerce") - pd.to_numeric(mf[lo_col], errors="coerce")
                    rel_width = width / yhat.abs().where(yhat.abs() > 0)
                    rows.append(
                        {
                            "model": model_name,
                            "level": level,
                            "future_rows_with_band": int(width.notna().sum()),
                            "avg_width": float(width.mean()) if width.notna().any() else np.nan,
                            "max_width": float(width.max()) if width.notna().any() else np.nan,
                            "avg_width_pct_of_yhat": float(rel_width.mean()) if rel_width.notna().any() else np.nan,
                            "interval_status": interval_status_label(active_model_interval_status(uid, model_name)),
                            "interval_method": interval_method_label(active_model_interval_method(uid, model_name)),
                        }
                    )
            return pd.DataFrame(rows)


        st.set_page_config(page_title="Forecast review workbench", layout="wide")
        st.markdown(
            """
            <style>
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 0.45rem;
                border-bottom: 1px solid #d9dee5;
                margin-top: 0.4rem;
                margin-bottom: 0.8rem;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] {
                background: #f6f8fb;
                border: 1px solid #d9dee5;
                border-bottom: 0;
                border-radius: 0.85rem 0.85rem 0 0;
                padding: 0.85rem 1.05rem;
                min-height: 3rem;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] p {
                font-size: 1.04rem;
                font-weight: 800;
                letter-spacing: -0.01em;
            }
            div[data-testid="stTabs"] [aria-selected="true"] {
                background: #fff8f4;
                color: #9a4622;
                box-shadow: inset 0 -3px 0 #b65f32;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        manifest = read_json("manifest.json")
        diagnostics = read_json("diagnostics.json")
        executive_headline = diagnostics.get("executive_headline", {})
        model_policy_resolution = manifest.get("model_policy_resolution", {})
        history = prep_dates(read_csv("history.csv"))
        forecast = prep_dates(read_csv("forecast.csv"))
        all_models = prep_dates(read_csv("all_models.csv"))
        selection = read_csv("model_selection.csv")
        metrics = read_csv("backtest_metrics.csv")
        weights = read_csv("model_weights.csv")
        backtest = prep_dates(read_csv("backtest_predictions.csv"))
        forecast_long = prep_dates(read_csv("forecast_long.csv"))
        backtest_long = prep_dates(read_csv("backtest_long.csv"))
        series_summary = read_csv("series_summary.csv")
        model_audit = read_csv("model_audit.csv")
        model_win_rates = read_csv("model_win_rates.csv")
        model_window_metrics = prep_dates(read_csv("model_window_metrics.csv"))
        residual_diagnostics = read_csv("residual_diagnostics.csv")
        residual_tests = read_csv("residual_tests.csv")
        interval_diagnostics = read_csv("interval_diagnostics.csv")
        trust_summary = read_csv("trust_summary.csv")
        target_transform_audit = read_csv("target_transform_audit.csv")
        seasonality_diagnostics = read_csv("seasonality_diagnostics.csv")
        seasonality_decomposition = prep_dates(read_csv("seasonality_decomposition.csv"))
        model_explainability = read_csv("model_explainability.csv")
        hierarchy_coherence = prep_dates(read_csv("hierarchy_coherence.csv"))
        hierarchy_reconciliation = read_csv("hierarchy_reconciliation.csv")
        hierarchy_coherence_pre = prep_dates(read_csv("hierarchy_coherence_pre.csv"))
        hierarchy_coherence_post = prep_dates(read_csv("hierarchy_coherence_post.csv"))
        hierarchy_contribution = prep_dates(read_csv("hierarchy_contribution.csv"))
        hierarchy_backtest_comparison = prep_dates(read_csv("hierarchy_backtest_comparison.csv"))
        scenario_assumptions = read_csv("scenario_assumptions.csv")
        scenario_forecast = prep_dates(read_csv("scenario_forecast.csv"))
        known_future_regressors = read_csv("known_future_regressors.csv")
        driver_availability_audit = read_csv("driver_availability_audit.csv")
        driver_experiment_summary = read_csv("driver_experiment_summary.csv")

        if forecast.empty:
            st.error("forecast.csv was not found in this run directory.")
            st.stop()

        uids = sorted(forecast["unique_id"].astype(str).dropna().unique())
        st.title("Forecast review workbench")
        st.caption("Use this to inspect decision readiness, model disagreement, intervals, rolling-origin windows, residuals, and feeder-ready outputs.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Series", len(uids))
        c2.metric("Horizon rows", int(forecast.groupby("unique_id").size().max()))
        c3.metric("Candidate models", len(model_columns(all_models)))
        if not selection.empty and "rmse" in selection.columns and selection["rmse"].notna().any():
            c4.metric("Best selected RMSE", f"{pd.to_numeric(selection['rmse'], errors='coerce').min():,.2f}")
        else:
            c4.metric("Best selected RMSE", "N/A")

        with st.sidebar:
            st.header("Controls")
            uid = st.selectbox("Series", uids)
            champion_scope = st.radio(
                "Champion lens",
                CHAMPION_SCOPE_OPTIONS,
                help="Switch the active champion between the tournament winner, the best StatsForecast/classical candidate, and the best MLForecast candidate.",
            )
            winner_metric = st.selectbox(
                "Winner metric",
                available_winner_metrics(uid),
                index=0,
                format_func=metric_label,
                help="Choose the metric used to pick the active champion inside the selected lens. This changes the dashboard champion for review; it does not rewrite forecast.csv.",
            )
            with st.expander("When should I use each metric?"):
                for metric in available_winner_metrics(uid):
                    guidance = WINNER_METRIC_GUIDANCE[metric]
                    direction = "higher is better" if guidance["direction"] == "higher" else "lower is better"
                    st.markdown(f"**{guidance['label']}** ({direction}) - {guidance['when']}")
            active_champion = best_model_for_scope(uid, champion_scope, winner_metric)
            if not active_champion:
                reason = policy_message_for_scope(champion_scope)
                message = f"No model was available for `{champion_scope}`"
                if reason:
                    message += f": {reason}"
                st.warning(message + " Falling back to the overall selected model.")
                active_champion = selected_model(uid)
            overall_selected = selected_model(uid)
            st.caption(f"Overall selected model: {model_menu_label(uid, overall_selected, winner_metric) if overall_selected else 'not available'}")
            st.caption(f"Active champion: {model_menu_label(uid, active_champion, winner_metric) if active_champion else 'not available'} (by {metric_label(winner_metric)})")
            weighted_labels = [model_menu_label(uid, model, winner_metric) for model in top_weighted(uid, champion=active_champion)]
            st.caption("Top weighted alternatives: " + (", ".join(weighted_labels) or "not available"))
            show_context_models = st.toggle(
                "Show all other models as faint context",
                value=True,
                help="Turn this off when the chart is too crowded; models chosen in the Model investigation tab remain highlighted.",
            )

        headline_series = {
            str(row.get("unique_id")): row.get("paragraph")
            for row in executive_headline.get("series", [])
            if row.get("unique_id") and row.get("paragraph")
        } if isinstance(executive_headline.get("series"), list) else {}
        headline_text = headline_series.get(str(uid)) or executive_headline.get("paragraph")
        if headline_text:
            st.subheader("Forecast headline")
            st.info(headline_text)
            st.text_area("Copy headline", value=headline_text, height=120, key=f"copy_headline_{str(uid)}")
            st.caption("This deterministic headline reuses trust_summary.csv, forecast.csv, interval_status, and horizon validation gates. Copy this text or Quote diagnostics.json executive_headline.paragraph verbatim for run-level summaries.")

        active_interval_status = "unavailable"
        if not trust_summary.empty:
            st.subheader("Decision summary")
            trust_row = trust_summary[trust_summary["unique_id"].astype(str) == uid]
            if not trust_row.empty:
                trust = trust_row.iloc[0]
                active_interval_status = str(trust.get("interval_status", "unavailable"))
                trust_cols = st.columns(6)
                trust_cols[0].metric("Trust level", str(trust.get("trust_level", "N/A")))
                trust_cols[1].metric("Trust score", str(trust.get("trust_score_0_100", "N/A")))
                trust_cols[2].metric("History", str(trust.get("history_readiness", "N/A")))
                trust_cols[3].metric("Unvalidated steps", str(trust.get("unvalidated_steps", "N/A")))
                trust_cols[4].metric("Horizon score cap", str(trust.get("horizon_trust_score_cap", "None")))
                trust_cols[5].metric("Intervals", interval_status_label(trust.get("interval_status", "N/A")))
                caveats = str(trust.get("caveats") or "No major caveats.")
                actions = str(trust.get("next_actions") or "No next actions recorded.")
                st.caption("planning_eligible is a horizon-validation flag only; still review trust, intervals, residuals, hierarchy, and data-quality caveats before stakeholder use.")
                st.markdown(f"**Caveats:** {caveats}")
                st.markdown(f"**Next actions:** {actions}")
                st.caption(f"Trust rubric: {TRUST_RUBRIC_TEXT}")
                st.caption(f"Interval glossary: {INTERVAL_GLOSSARY_TEXT}")
            with st.expander("All series trust summary"):
                st.dataframe(display_trust_summary(trust_summary), width="stretch", hide_index=True)
        horizon_message = active_model_horizon_summary(uid, active_champion)
        horizon_severity = horizon_message_severity(horizon_message)
        if horizon_severity == "error":
            st.error(horizon_message)
        elif horizon_severity == "warning":
            st.warning(horizon_message)
        else:
            st.caption(horizon_message)
        if not target_transform_audit.empty:
            with st.expander("Target transformation audit", expanded=False):
                st.caption("Raw, normalized, and modeled target values. Log/log1p forecasts are inverse-transformed for reporting; factor-normalized forecasts remain in normalized units unless future factors are supplied externally.")
                st.dataframe(target_transform_audit.head(2000), width="stretch", hide_index=True)
        policy_table = policy_resolution_table()
        if not policy_table.empty:
            with st.expander("Model policy resolution", expanded=False):
                st.caption(f"model_policy={model_policy_resolution.get('model_policy', 'unknown')}. `all` means all eligible open-source families; skipped or unavailable families are explicit.")
                st.dataframe(policy_table, width="stretch", hide_index=True)

        tab_forecast, tab_investigate, tab_backtest, tab_intervals, tab_audit, tab_seasonality, tab_hierarchy, tab_drivers, tab_outputs = st.tabs(
            ["Forecast review", "Model investigation", "CV window player", "Prediction intervals", "Model audit", "Seasonality", "Hierarchy", "Assumptions & Drivers", "Feeder outputs"]
        )

        model_options = model_menu_options(uid, candidate_models(uid))
        default_models = dedupe_models([active_champion, *top_weighted(uid, champion=active_champion)])
        default_models = [model for model in default_models if model in model_options][:4]

        with tab_investigate:
            st.subheader("Models to investigate")
            st.caption("Pick specific models here, then compare their future path, rolling-origin windows, errors, intervals, weights, and explainability without crowding the main forecast review. Menu labels use `#rank | model | engine`; the guide below is grouped by engine and alphabetized by model so StatsForecast/classical and MLForecast candidates are easy to distinguish.")
            picker_guide = model_menu_table(uid, model_options, active_champion=active_champion, focus_models=default_models, metric=winner_metric)
            if not picker_guide.empty:
                with st.expander("Model picker guide: rank, engine, and role", expanded=True):
                    st.caption("Rank comes from the current winner metric. Engine labels distinguish StatsForecast/classical, MLForecast, baseline, ensemble, custom, and other candidates; the native guide table italicizes the Engine column when Streamlit dataframe styling is available.")
                    st.dataframe(model_picker_guide_style(picker_guide), width="stretch", hide_index=True)
            focus_models = st.multiselect(
                "Models to investigate",
                model_options,
                default=default_models,
                format_func=lambda model: model_menu_label(uid, model, winner_metric),
                help="These models are highlighted in forecast, CV, residual, interval, and audit sections. Clear the list to only use the active champion.",
                key=f"models_to_investigate_{uid}",
            )
            if active_champion and active_champion not in focus_models:
                focus_models = [active_champion, *focus_models]
            investigation = model_investigation_table(uid, dedupe_models(focus_models), active_champion=active_champion, metric=winner_metric)
            inv_cols = st.columns(4)
            inv_cols[0].metric("Active champion", active_champion or "N/A")
            inv_cols[1].metric("Winner metric", metric_label(winner_metric))
            inv_cols[2].metric("Investigated models", len(dedupe_models(focus_models)))
            inv_cols[3].metric("Available candidates", len(model_options))
            if not investigation.empty:
                st.dataframe(investigation, width="stretch", hide_index=True)
            st.subheader("Focused future forecast")
            focused_interval_models = [model for model in dedupe_models(focus_models) if selected_interval_levels(uid, model)]
            if focused_interval_models:
                st.caption(
                    "Focused future forecast interval ownership: "
                    + "; ".join(
                        f"{model} ({', '.join(str(level) + '%' for level in selected_interval_levels(uid, model))})"
                        for model in focused_interval_models
                    )
                    + ". Ribbons use the same color as their model line; the legend names each interval band. Point forecasts and bands come from the same `forecast_long.csv` model feed."
                )
            else:
                st.caption("Focused future forecast has no interval-bearing selected models; lines show point forecasts only.")
            st.plotly_chart(
                forecast_chart(uid, show_all_models=False, champion=active_champion, focus_models=focus_models),
                width="stretch",
                key="investigation_forecast_chart",
            )
            if not backtest.empty and "cutoff" in backtest.columns:
                bt_uid = backtest[backtest["unique_id"].astype(str) == uid].copy()
                investigation_cutoffs = sorted(bt_uid["cutoff"].dropna().unique())
                if investigation_cutoffs:
                    cutoff = st.selectbox(
                        "Investigation CV cutoff",
                        investigation_cutoffs,
                        index=len(investigation_cutoffs) - 1,
                        format_func=lambda value: str(pd.Timestamp(value).date()),
                        key=f"model_investigation_cutoff_{uid}",
                    )
                    st.subheader("Focused rolling-origin window")
                    st.plotly_chart(
                        backtest_chart(
                            uid,
                            cutoff,
                            show_all_models=False,
                            fixed_x=True,
                            fixed_y=True,
                            show_intervals=True,
                            champion=active_champion,
                            focus_models=focus_models,
                        ),
                        width="stretch",
                        key="investigation_backtest_chart",
                    )
                    if not model_window_metrics.empty:
                        metrics_view = model_window_metrics[
                            (model_window_metrics["unique_id"].astype(str) == uid) & (model_window_metrics["cutoff"] == cutoff)
                        ]
                        if "model" in metrics_view.columns and focus_models:
                            metrics_view = metrics_view[metrics_view["model"].astype(str).isin(dedupe_models(focus_models))]
                        st.dataframe(metrics_view.sort_values("rmse").head(50), width="stretch", hide_index=True)
                    if not backtest_long.empty:
                        view = backtest_long[(backtest_long["unique_id"].astype(str) == uid) & (backtest_long["cutoff"] == cutoff)]
                        if "model" in view.columns and focus_models:
                            view = view[view["model"].astype(str).isin(dedupe_models(focus_models))]
                        st.dataframe(view.head(500), width="stretch", hide_index=True)
                else:
                    st.info("No rolling-origin windows are available for this series.")
            else:
                st.info("No backtest predictions are available for focused model investigation.")

        with tab_forecast:
            st.subheader("Tournament lens")
            lens_cols = st.columns(4)
            lens_cols[0].metric("Champion lens", champion_scope.replace("Best ", ""))
            lens_cols[1].metric("Active champion", active_champion or "N/A")
            lens_cols[2].metric("Winner metric", metric_label(winner_metric))
            lens_cols[3].metric("Investigated models", len(dedupe_models(focus_models)))
            family_view = family_summary(uid)
            if not family_view.empty:
                st.dataframe(family_view, width="stretch", hide_index=True)
            st.subheader("Candidate model context")
            hidden_count = other_candidate_count(uid, champion=active_champion, focus_models=focus_models)
            if hidden_count:
                st.caption(f"{hidden_count} other candidates are shown as faint unlabeled context lines. Labels are reserved for the active champion and investigated models.")
            chart_models = model_columns(all_models) if show_context_models else dedupe_models([active_champion, *focus_models])
            chart_interval_models = [model for model in chart_models if selected_interval_levels(uid, model)]
            chart_interval_levels = sorted({level for model in chart_interval_models for level in selected_interval_levels(uid, model)})
            if chart_interval_models:
                st.caption(
                    f"First-glance chart includes interval bands for {len(chart_interval_models)} candidate model(s): "
                    f"{', '.join(str(level) + '%' for level in chart_interval_levels)} where available. "
                    "Check the Prediction intervals tab for calibration and row-level review."
                )
            else:
                st.warning("No prediction interval bands were written for the models currently shown. The faint gray spread is model disagreement, not calibrated uncertainty.")
            st.plotly_chart(
                forecast_chart(uid, show_all_models=show_context_models, champion=active_champion, focus_models=focus_models),
                width="stretch",
                key="forecast_context_chart",
            )
            st.caption("Use the Model investigation tab for the dedicated model picker and focused comparison charts.")
            if not series_summary.empty:
                st.dataframe(series_summary[series_summary["unique_id"].astype(str) == uid], width="stretch", hide_index=True)

        with tab_backtest:
            if backtest.empty or "cutoff" not in backtest.columns:
                st.warning("No backtest predictions are available.")
            else:
                bt_uid = backtest[backtest["unique_id"].astype(str) == uid].copy()
                cutoffs = sorted(bt_uid["cutoff"].dropna().unique())
                if not cutoffs:
                    st.warning("No backtest windows for this series.")
                else:
                    state_key = f"cv_cutoff_idx_{uid}"
                    slider_key = f"{state_key}_slider"
                    if state_key not in st.session_state:
                        st.session_state[state_key] = len(cutoffs) - 1
                    st.session_state[state_key] = min(max(int(st.session_state[state_key]), 0), len(cutoffs) - 1)
                    if slider_key not in st.session_state or int(st.session_state[slider_key]) != int(st.session_state[state_key]):
                        st.session_state[slider_key] = int(st.session_state[state_key])

                    def sync_cutoff_slider() -> None:
                        st.session_state[state_key] = min(max(int(st.session_state[slider_key]), 0), len(cutoffs) - 1)

                    st.subheader("Rolling-origin CV window player")
                    st.caption("The x-axis and y-axis stay fixed by default so the training region, cutoff, and holdout window visibly slide through time.")
                    st.caption("Use Previous/Next, drag the cutoff slider, or turn on Auto-advance to play through the backtest windows. Auto-advance loops until you switch it off.")
                    control_a, control_b, control_c, control_d, control_e = st.columns([0.7, 2.4, 0.7, 1.1, 1.1])
                    with control_a:
                        if st.button("Previous", disabled=st.session_state[state_key] <= 0):
                            st.session_state[state_key] -= 1
                            st.rerun()
                    with control_b:
                        st.slider(
                            "Cutoff index",
                            min_value=0,
                            max_value=len(cutoffs) - 1,
                            key=slider_key,
                            on_change=sync_cutoff_slider,
                            help="Step through rolling-origin windows while axes remain stable.",
                        )
                    with control_c:
                        if st.button("Next", disabled=st.session_state[state_key] >= len(cutoffs) - 1):
                            st.session_state[state_key] += 1
                            st.rerun()
                    with control_d:
                        fixed_x = st.toggle("Fixed date axis", value=True)
                        fixed_y = st.toggle("Fixed value axis", value=True)
                    with control_e:
                        show_intervals = st.toggle("Show intervals", value=True)
                        autoplay = st.toggle("Auto-advance", value=False, key=f"{state_key}_autoplay")
                        autoplay_delay = st.slider("Delay seconds", 0.3, 3.0, 1.0, 0.1, key=f"{state_key}_delay")
                    cutoff = cutoffs[st.session_state[state_key]]
                    st.markdown(f"**Window {st.session_state[state_key] + 1} of {len(cutoffs)}** | cutoff `{pd.Timestamp(cutoff).date()}`")
                    st.subheader("All backtested candidate models")
                    st.plotly_chart(
                        backtest_chart(
                            uid,
                            cutoff,
                            show_all_models=show_context_models,
                            fixed_x=fixed_x,
                            fixed_y=fixed_y,
                            show_intervals=show_intervals,
                            champion=active_champion,
                            focus_models=focus_models,
                        ),
                        width="stretch",
                        key="backtest_context_chart",
                    )
                    if not model_window_metrics.empty:
                        st.subheader("Window metrics for current cutoff")
                        metrics_view = model_window_metrics[
                            (model_window_metrics["unique_id"].astype(str) == uid) & (model_window_metrics["cutoff"] == cutoff)
                        ]
                        if "model" in metrics_view.columns and focus_models:
                            metrics_view = metrics_view[metrics_view["model"].astype(str).isin(dedupe_models(focus_models))]
                        st.dataframe(metrics_view.sort_values("rmse").head(50), width="stretch", hide_index=True)
                    if not backtest_long.empty:
                        view = backtest_long[(backtest_long["unique_id"].astype(str) == uid) & (backtest_long["cutoff"] == cutoff)]
                        if "model" in view.columns and focus_models:
                            view = view[view["model"].astype(str).isin(dedupe_models(focus_models))]
                        st.dataframe(view.head(500), width="stretch", hide_index=True)
                    if autoplay and len(cutoffs) > 1:
                        st.caption(f"Auto-advance is active. Next window in {autoplay_delay:.1f}s.")
                        time.sleep(float(autoplay_delay))
                        st.session_state[state_key] = (st.session_state[state_key] + 1) % len(cutoffs)
                        st.rerun()

        with tab_intervals:
            st.subheader("Prediction interval focus")
            st.caption("Prediction intervals are the uncertainty ranges around each model's point forecast. All interval-bearing candidate models are selected by default so you can compare ranges directly; model spread itself is still not a prediction interval.")
            interval_options = model_menu_options(uid, interval_bearing_models(uid))
            interval_guide = model_menu_table(uid, interval_options, active_champion=active_champion, focus_models=interval_options, metric=winner_metric)
            if not interval_guide.empty:
                with st.expander("Interval model picker guide: rank and engine", expanded=False):
                    st.caption("Interval model menus also use `#rank | model | engine`; use the rank and Engine columns to separate StatsForecast/classical intervals from MLForecast intervals.")
                    st.dataframe(model_picker_guide_style(interval_guide), width="stretch", hide_index=True)
            interval_models = st.multiselect(
                "Models with interval bands",
                interval_options,
                default=interval_options,
                format_func=lambda model: model_menu_label(uid, model, winner_metric),
                help="These are candidate models that wrote future lower/upper interval bounds. Keep all selected for a full uncertainty review, or narrow the list if the chart gets crowded.",
                key=f"interval_models_{uid}",
            )
            interval_models = coerce_interval_models(uid, interval_models)
            interval_levels = sorted({level for model in interval_models for level in selected_interval_levels(uid, model)})
            interval_statuses = sorted({interval_status_label(active_model_interval_status(uid, model)) for model in interval_models})
            interval_methods = sorted({interval_method_label(active_model_interval_method(uid, model)) for model in interval_models})
            interval_future = interval_focus_models_frame(uid, interval_models)
            interval_summary = prediction_interval_summary_table(uid, interval_models)
            interval_cols = st.columns(5)
            interval_cols[0].metric("Models shown", len(interval_models))
            interval_cols[1].metric("Statuses", ", ".join(interval_statuses) or "N/A")
            interval_cols[2].metric("Methods", ", ".join(interval_methods) or "N/A")
            interval_cols[3].metric("Levels", ", ".join(str(level) + "%" for level in interval_levels) or "None")
            interval_cols[4].metric("Model/date rows", len(interval_future))
            st.caption(f"Interval glossary: {INTERVAL_GLOSSARY_TEXT}")
            if interval_levels and interval_models:
                st.plotly_chart(prediction_interval_focus_chart(uid, interval_models), width="stretch", key="prediction_interval_focus_chart")
                st.plotly_chart(prediction_interval_width_chart(uid, interval_models), width="stretch", key="prediction_interval_width_chart")
                if not interval_summary.empty:
                    st.subheader("Future interval width summary")
                    st.dataframe(interval_summary, width="stretch", hide_index=True)
            else:
                st.warning("No prediction interval bands are available for this series/model set. Use calibrated lower/upper bands when planning ranges matter; model spread is not a prediction interval.")
            st.subheader("Empirical calibration evidence")
            if not interval_diagnostics.empty:
                st.caption("This compares nominal interval coverage to rolling-origin empirical coverage. Coverage below target means ranges were too narrow historically.")
                st.plotly_chart(interval_calibration_chart(uid, interval_models), width="stretch", key="interval_focus_calibration_plot")
                interval_view = interval_diagnostics[interval_diagnostics["unique_id"].astype(str) == uid]
                if "model" in interval_view.columns and interval_models:
                    interval_view = interval_view[interval_view["model"].astype(str).isin(dedupe_models(interval_models))]
                st.dataframe(interval_view.head(1000), width="stretch", hide_index=True)
            else:
                st.info("No empirical interval diagnostics are available. Future interval bands, if present, should be treated as future-only or model-assumption ranges until rolling-origin coverage evidence exists.")
            st.subheader("Forecast rows with interval fields")
            interval_columns = ordered_model_feed_columns(interval_future)
            if interval_columns:
                st.dataframe(interval_future[interval_columns].head(1000), width="stretch", hide_index=True)
            else:
                st.info("No interval-bearing forecast rows were found for the selected interval models.")

        with tab_audit:
            left, right = st.columns([1.25, 0.75])
            with left:
                st.subheader("Model leaderboard")
                table = model_score_table(uid)
                if not table.empty:
                    sort_col = winner_metric if winner_metric in table.columns else ("rmse" if "rmse" in table.columns else table.columns[0])
                    st.dataframe(table.sort_values(sort_col, ascending=metric_direction(sort_col) != "higher").head(1000), width="stretch", hide_index=True)
                else:
                    st.info("No model audit table available.")
                st.subheader("Investigated model comparison")
                investigation = model_investigation_table(uid, dedupe_models(focus_models))
                if not investigation.empty:
                    st.dataframe(investigation, width="stretch", hide_index=True)
                else:
                    st.info("Pick models in the Model investigation tab to compare them here.")
                st.subheader("Win rate vs benchmark")
                if not model_win_rates.empty:
                    st.plotly_chart(benchmark_win_rate_chart(), width="stretch", key="benchmark_win_rate_chart")
                    st.dataframe(model_win_rates.head(1000), width="stretch", hide_index=True)
                else:
                    st.info("No benchmark win-rate table available.")
            with right:
                st.subheader("Weights")
                if not weights.empty:
                    st.dataframe(weights[weights["unique_id"].astype(str) == uid].sort_values("weight", ascending=False), width="stretch", hide_index=True)
                else:
                    st.info("No model weights available.")
            st.subheader("Heuristic residual checks")
            if not residual_tests.empty:
                st.caption("Bias, outlier, and early/late structural-break checks summarize all rolling-origin residuals; the white-noise/ACF check uses one-step residuals when available. Small samples are directional, not proof of model adequacy.")
                residual_test_view = residual_tests[residual_tests["unique_id"].astype(str) == uid]
                if "model" in residual_test_view.columns and focus_models:
                    residual_test_view = residual_test_view[residual_test_view["model"].astype(str).isin(dedupe_models(focus_models))]
                status_values = set(residual_test_view.get("overall_status", pd.Series(dtype=str)).dropna().astype(str))
                if "fail" in status_values:
                    st.error("Residual checks flagged a failure. Investigate bias, autocorrelation, outliers, or structural breaks before planning use.")
                elif "warn" in status_values:
                    st.warning("Residual checks flagged warnings. Treat the forecast as review-needed until the residual pattern is explained.")
                elif "insufficient" in status_values:
                    st.info("Residual check sample is small. Treat these diagnostics as directional.")
                st.dataframe(residual_test_view.head(1000), width="stretch", hide_index=True)
            else:
                st.info("No residual test summary available.")
            st.subheader("Residual diagnostics by horizon")
            if not residual_diagnostics.empty:
                st.caption("Residual views follow the models selected in the Model investigation tab. The ACF/white-noise panel uses the first investigated model so it remains readable.")
                st.plotly_chart(residual_horizon_chart(uid, focus_models), width="stretch", key="residual_horizon_plot")
                res_left, res_mid, res_right = st.columns(3)
                with res_left:
                    st.plotly_chart(residual_time_chart(uid, focus_models), width="stretch", key="residual_time_plot")
                with res_mid:
                    st.plotly_chart(residual_distribution_chart(uid, focus_models), width="stretch", key="residual_distribution_plot")
                with res_right:
                    st.plotly_chart(residual_acf_chart(uid, focus_models), width="stretch", key="residual_acf_plot")
                    st.caption(residual_white_noise_verdict(uid, focus_models))
                outliers = residual_outlier_table(uid, focus_models)
                if not outliers.empty:
                    st.warning("Residual outliers detected (|standardized residual| >= 3). Map these dates to known events before trusting the model.")
                    st.dataframe(outliers.head(100), width="stretch", hide_index=True)
                diagnostics_view = residual_diagnostics[residual_diagnostics["unique_id"].astype(str) == uid]
                if "model" in diagnostics_view.columns and focus_models:
                    diagnostics_view = diagnostics_view[diagnostics_view["model"].astype(str).isin(dedupe_models(focus_models))]
                st.dataframe(diagnostics_view.head(1000), width="stretch", hide_index=True)
            else:
                st.info("No residual diagnostics available.")
            st.subheader("Prediction interval calibration")
            if not interval_diagnostics.empty:
                st.caption("Interval diagnostics are empirical rolling-origin checks; future-only or adjusted-not-recalibrated bands need extra review before planning use.")
                st.plotly_chart(interval_calibration_chart(uid, focus_models), width="stretch", key="interval_calibration_plot")
                interval_view = interval_diagnostics[interval_diagnostics["unique_id"].astype(str) == uid]
                if "model" in interval_view.columns and focus_models:
                    interval_view = interval_view[interval_view["model"].astype(str).isin(dedupe_models(focus_models))]
                st.dataframe(interval_view.head(1000), width="stretch", hide_index=True)
            else:
                st.info("No empirical interval diagnostics available. Future interval bands, if present, should be treated as future-only until rolling-origin coverage evidence is available.")
            st.subheader("MLForecast interpretability")
            if not model_explainability.empty:
                st.plotly_chart(feature_importance_chart(focus_models), width="stretch", key="feature_importance_plot")
                explainability_view = model_explainability.copy()
                if "model" in explainability_view.columns and focus_models:
                    explainability_view = explainability_view[explainability_view["model"].astype(str).isin(dedupe_models(focus_models))]
                if explainability_view.empty:
                    st.info("No MLForecast feature importance rows match the currently investigated models.")
                else:
                    st.dataframe(explainability_view.sort_values(["model", "importance"], ascending=[True, False]), width="stretch", hide_index=True)
            else:
                st.info("No MLForecast feature importance available. Use model policy auto/all with enough history and installed MLForecast/LightGBM dependencies.")

        with tab_seasonality:
            st.subheader("Seasonality credibility")
            st.caption("This view separates pattern strength from evidence quality. A seasonal chart that looks plausible is not planning-ready unless enough full cycles exist.")
            if not seasonality_diagnostics.empty:
                seasonality_view = seasonality_diagnostics[seasonality_diagnostics["unique_id"].astype(str) == uid]
                if not seasonality_view.empty:
                    row = seasonality_view.iloc[0]
                    cols = st.columns(4)
                    cols[0].metric("Credibility", str(row.get("credibility_label", "N/A")))
                    cols[1].metric("Observed cycles", f"{pd.to_numeric(row.get('cycle_count'), errors='coerce'):.1f}")
                    cols[2].metric("Seasonal strength", f"{pd.to_numeric(row.get('seasonality_strength'), errors='coerce'):.2f}")
                    cols[3].metric("Trend strength", f"{pd.to_numeric(row.get('trend_strength'), errors='coerce'):.0%}")
                    warning = row.get("warning")
                    if isinstance(warning, str) and warning:
                        st.warning(warning)
                    st.markdown(f"**Interpretation:** {row.get('interpretation', 'No interpretation available.')}")
                st.dataframe(seasonality_diagnostics, width="stretch", hide_index=True)
            else:
                st.info("No seasonality diagnostics are available.")
            st.subheader("Seasonal year overlay")
            st.caption("Forecast overlay uses the active/best model by default, and each observed or forecast line is aligned onto a normal configured year. Choose any candidate model to compare its forecast seasonality against historical actual years.")
            seasonal_model_options = model_menu_options(uid, candidate_models(uid))
            seasonal_default_model = active_champion if active_champion in seasonal_model_options else selected_model(uid)
            seasonal_default_index = seasonal_model_options.index(seasonal_default_model) if seasonal_default_model in seasonal_model_options else 0
            overlay_controls = st.columns([1, 2, 1])
            with overlay_controls[0]:
                year_start_label = st.selectbox(
                    "Beginning of year month",
                    MONTH_LABELS,
                    index=0,
                    key=f"seasonality_year_start_month_{uid}",
                    help="Use January for calendar year, or choose a fiscal/planning-year start month.",
                )
            seasonal_overlay_model = None
            with overlay_controls[1]:
                if seasonal_model_options:
                    seasonal_overlay_model = st.selectbox(
                        "Forecast model overlay",
                        seasonal_model_options,
                        index=seasonal_default_index,
                        format_func=lambda model: model_menu_label(uid, model, winner_metric),
                        key=f"seasonality_overlay_model_{uid}",
                        help="Defaults to the active champion/best model. Pick any candidate model to overlay its future yhat seasonal shape.",
                    )
                else:
                    st.info("No candidate model forecast is available for the seasonal overlay.")
            with overlay_controls[2]:
                include_seasonal_history = st.toggle(
                    "Include actual years",
                    value=True,
                    key=f"seasonality_include_actual_years_{uid}",
                    help="Keep historical actual seasonal years as context behind the selected model forecast overlay.",
                )
            year_start_month = MONTH_LABELS.index(year_start_label) + 1
            seasonal_profile = seasonality_year_profile_frame(
                uid,
                year_start_month,
                model=seasonal_overlay_model,
                include_history=include_seasonal_history,
            )
            if not seasonal_profile.empty:
                has_forecast_overlay = "source" in seasonal_profile.columns and (seasonal_profile["source"].astype(str) == "forecast").any()
                if seasonal_overlay_model and has_forecast_overlay:
                    st.caption(f"Model overlay: {model_menu_label(uid, seasonal_overlay_model, winner_metric)}. Forecast rows use `forecast_long.csv` yhat when available, with `all_models.csv` only as a fallback.")
                elif seasonal_overlay_model:
                    st.warning("The selected model has no future forecast rows available for this seasonal overlay, so only actual-year context is shown.")
                st.plotly_chart(
                    seasonality_year_profile_chart(
                        uid,
                        year_start_month,
                        model=seasonal_overlay_model,
                        include_history=include_seasonal_history,
                    ),
                    width="stretch",
                    key="seasonality_year_profile_chart",
                )
                with st.expander("Seasonal year overlay data"):
                    st.dataframe(seasonal_profile.head(2000), width="stretch", hide_index=True)
            else:
                st.info("No actual or selected-model forecast rows are available for a seasonal year overlay.")
            st.subheader("Additive decomposition evidence")
            if not seasonality_decomposition.empty:
                st.plotly_chart(seasonality_decomposition_chart(uid), width="stretch", key="seasonality_decomposition_plot")
                st.dataframe(
                    seasonality_decomposition[seasonality_decomposition["unique_id"].astype(str) == uid].head(500),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No decomposition table is available. This usually means the series has fewer than two full seasonal cycles or no repeating seasonal period was inferred.")

        with tab_hierarchy:
            if "hierarchy_depth" not in forecast.columns:
                st.info("Hierarchy metadata is not present in this run. Use the `hierarchy` command before forecasting to enable roll-up/down diagnostics.")
            else:
                st.subheader("Hierarchy roll-up / roll-down")
                if not hierarchy_reconciliation.empty:
                    st.success("Hierarchy reconciliation is enabled. Parent/child forecasts are coherent after the applied reconciliation method; unreconciled forecasts and pre/post gap audits are saved under audit/.")
                    st.dataframe(hierarchy_reconciliation, width="stretch", hide_index=True)
                else:
                    st.caption("Current forecasts are independent per node unless reconciliation is enabled. This view compares selected parent forecasts with the sum of their immediate children.")
                level_summary = (
                    forecast[["unique_id", "hierarchy_level", "hierarchy_depth"]]
                    .drop_duplicates()
                    .groupby(["hierarchy_depth", "hierarchy_level"], as_index=False)
                    .agg(series_count=("unique_id", "nunique"))
                    .sort_values(["hierarchy_depth", "hierarchy_level"])
                )
                st.dataframe(level_summary, width="stretch", hide_index=True)
                parent_options = hierarchy_parent_options()
                if parent_options:
                    parent_id = st.selectbox("Parent node", parent_options)
                    st.plotly_chart(hierarchy_rollup_chart(parent_id), width="stretch", key="hierarchy_rollup_plot")
                    st.plotly_chart(hierarchy_children_chart(parent_id), width="stretch", key="hierarchy_children_plot")
                    if not hierarchy_contribution.empty and "parent_unique_id" in hierarchy_contribution.columns:
                        st.subheader("Child contribution to selected parent")
                        contribution_view = hierarchy_contribution[hierarchy_contribution["parent_unique_id"].astype(str) == parent_id].copy()
                        if "child_share_of_parent" in contribution_view.columns:
                            contribution_view["abs_child_share"] = pd.to_numeric(contribution_view["child_share_of_parent"], errors="coerce").abs()
                            contribution_view = contribution_view.sort_values("abs_child_share", ascending=False)
                        st.caption("Use this table to tell which child nodes drive parent totals and parent-minus-child gaps.")
                        st.dataframe(contribution_view.head(1000), width="stretch", hide_index=True)
                if not hierarchy_coherence.empty:
                    st.subheader("Coherence gaps")
                    display = hierarchy_coherence.copy()
                    if "gap_pct" in display.columns:
                        display["abs_gap_pct"] = pd.to_numeric(display["gap_pct"], errors="coerce").abs()
                        display = display.sort_values("abs_gap_pct", ascending=False)
                    st.dataframe(display.head(1000), width="stretch", hide_index=True)
                    if not hierarchy_coherence_pre.empty or not hierarchy_coherence_post.empty:
                        st.subheader("Pre/post reconciliation gap audit")
                        cols = st.columns(2)
                        with cols[0]:
                            st.caption("Before reconciliation")
                            st.dataframe(hierarchy_coherence_pre.head(1000), width="stretch", hide_index=True)
                        with cols[1]:
                            st.caption("After reconciliation")
                            st.dataframe(hierarchy_coherence_post.head(1000), width="stretch", hide_index=True)
                    if not hierarchy_backtest_comparison.empty:
                        st.subheader("Reconciled vs unreconciled backtest comparison")
                        st.caption("This compares selected-model rolling-origin errors before and after applying the reconciliation method. Reconciliation guarantees coherence, but it can improve or worsen node-level accuracy.")
                        comparison = hierarchy_backtest_comparison.copy()
                        if "abs_error_delta" in comparison.columns:
                            comparison["abs_error_delta"] = pd.to_numeric(comparison["abs_error_delta"], errors="coerce")
                            worsened = int((comparison["abs_error_delta"] > 0).sum())
                            improved = int((comparison["abs_error_delta"] < 0).sum())
                            st.info(f"Reconciliation improved {improved} selected backtest row(s) and worsened {worsened} row(s) by absolute error; review level-specific tradeoffs before planning.")
                        st.dataframe(comparison.head(1000), width="stretch", hide_index=True)
                else:
                    st.warning("Hierarchy metadata exists, but hierarchy_coherence.csv is empty.")

        with tab_drivers:
            st.subheader("Assumptions & Drivers")
            st.caption("Scenario/event overlays are post-model assumptions: baseline `yhat` stays separate from `yhat_scenario`. Known-future regressors are audited for leakage and future availability; this release does not automatically train arbitrary external regressors.")
            if (
                scenario_assumptions.empty
                and scenario_forecast.empty
                and known_future_regressors.empty
                and driver_availability_audit.empty
                and driver_experiment_summary.empty
            ):
                st.info("No scenario assumptions or known-future regressors were declared for this run.")
            if not scenario_assumptions.empty:
                st.subheader("Scenario assumptions")
                st.dataframe(scenario_assumptions, width="stretch", hide_index=True)
            if not scenario_forecast.empty:
                st.subheader("Scenario forecast preview")
                scenario_view = scenario_forecast.copy()
                if "unique_id" in scenario_view.columns:
                    scenario_view = scenario_view[scenario_view["unique_id"].astype(str) == uid]
                st.dataframe(scenario_view.head(1000), width="stretch", hide_index=True)
            if not known_future_regressors.empty:
                st.subheader("Known-future regressor declarations")
                st.dataframe(known_future_regressors, width="stretch", hide_index=True)
            if not driver_availability_audit.empty:
                st.subheader("Driver audit distribution")
                if "audit_status" in driver_availability_audit.columns:
                    distribution = (
                        driver_availability_audit["audit_status"]
                        .fillna("unknown")
                        .astype(str)
                        .value_counts()
                        .rename_axis("audit_status")
                        .reset_index(name="regressor_count")
                    )
                    st.dataframe(distribution, width="stretch", hide_index=True)
                st.subheader("Driver availability audit")
                st.dataframe(driver_availability_audit, width="stretch", hide_index=True)
            if not driver_experiment_summary.empty:
                st.subheader("Driver experiment summary")
                st.dataframe(driver_experiment_summary, width="stretch", hide_index=True)

        with tab_outputs:
            st.subheader("Consolidated feeder files")
            st.markdown(
                """
                - `forecast_long.csv`: future predictions in long format for every model.
                - `backtest_long.csv`: rolling-origin predictions in long format with actuals, errors, interval bounds, and coverage flags.
                - `series_summary.csv`: one row per series for analyst review and model feeds.
                - `model_audit.csv`: leaderboard enriched with weights and selected flags.
                - `model_win_rates.csv`: cross-series win rates against SeasonalNaive/Naive benchmarks.
                - `model_window_metrics.csv`, `residual_diagnostics.csv`, `residual_tests.csv`, `interval_diagnostics.csv`: practitioner diagnostics for CV, residual health, and interval calibration. Residual tests are heuristic screening, not formal adequacy certification.
                - `trust_summary.csv`: High/Medium/Low per-series readiness with caveats and next actions.
                - `scenario_assumptions.csv`, `scenario_forecast.csv`: event/scenario overlay assumptions and baseline-versus-scenario forecast rows.
                - `known_future_regressors.csv`, `driver_availability_audit.csv`, `driver_experiment_summary.csv`: declared known-future driver contracts, leakage/future-availability audit, and next-step summary. External regressors are audited, not auto-trained, in this release.
                - `audit/target_transform_audit.csv`: raw, adjusted, transformed, and modeled target trail for finance normalization/log transforms.
                - `audit/seasonality_diagnostics.csv`, `audit/seasonality_decomposition.csv`: cycle counts, credibility labels, and additive decomposition evidence.
                - `model_explainability.csv`: MLForecast/sklearn/LightGBM lag and date feature importance or coefficient magnitude when ML models run.
                - `llm_context.json`: fat LLM handoff packet with the executive headline, trust/horizon/interval/residual/seasonality/hierarchy/driver context, artifact index, guardrails, and recommended questions.
                - `manifest.json`: includes `model_policy_resolution`, showing which requested families were eligible, ran, or were skipped.
                - `hierarchy_coherence.csv`: parent forecast versus immediate-child sum when hierarchy metadata exists.
                - `hierarchy_contribution.csv`: parent/child contribution and gap attribution for hierarchy storytelling; allocation heuristic, not reconciliation output.
                - `hierarchy_reconciliation.csv`, `audit/hierarchy_backtest_comparison.csv`, `audit/hierarchy_unreconciled_forecast.csv`, `audit/hierarchy_coherence_pre.csv`, `audit/hierarchy_coherence_post.csv`: method summary, selected backtest accuracy comparison, and pre/post coherence audits when reconciliation is enabled. Reconciliation enforces coherence and may worsen node-level accuracy.
                """
            )
            st.subheader("forecast_long.csv")
            st.caption("Model feed columns keep `yhat`, `yhat_lo_80`, `yhat_hi_80`, `yhat_lo_95`, and `yhat_hi_95` adjacent so point forecasts and interval levels can be read together.")
            st.dataframe(forecast_long[ordered_model_feed_columns(forecast_long)].head(1000), width="stretch", hide_index=True)
            st.subheader("backtest_long.csv")
            st.dataframe(backtest_long[ordered_model_feed_columns(backtest_long)].head(1000), width="stretch", hide_index=True)
        '''
    ).strip() + "\n"


def _forecast_review_charts(payload: dict[str, Any]) -> str:
    forecast = payload.get("forecast", [])
    if not forecast:
        return '<p class="footnote">No forecast rows available.</p>'
    series_ids = sorted({str(row["unique_id"]) for row in forecast if row.get("unique_id") is not None})
    blocks: list[str] = []
    for uid in series_ids[:6]:
        selected = _selected_model(payload, uid)
        top = _top_model_names(payload, uid)
        all_model_rows = [row for row in payload.get("all_models", []) if str(row.get("unique_id")) == uid]
        model_cols = _model_columns_from_records(all_model_rows)
        other_count = len([model for model in model_cols if model != selected and model not in top])
        interval_levels = _selected_interval_levels(payload, uid)
        interval_model_count = len(_interval_model_names(payload, uid))
        subtitle = f"selected: {selected or 'not available'}"
        if top:
            subtitle += f" | top weighted: {', '.join(top)}"
        if other_count:
            subtitle += f" | {other_count} other candidates shown faint/unlabeled"
        if interval_model_count:
            subtitle += f" | interval bands: {interval_model_count} candidate models"
        interval_status = _payload_interval_status(payload, uid)
        if interval_levels:
            subtitle += f" | selected intervals: {', '.join(str(level) + '%' for level in interval_levels)} ({_interval_status_label(interval_status)})"
        else:
            subtitle += " | selected intervals unavailable"
        subtitle += f" | {_payload_horizon_summary(payload, uid)}"
        blocks.append(
            '<div class="chart-block">'
            f'<div class="chart-head"><strong>{_esc(uid)} - all candidate models</strong><span>{_esc(subtitle)}</span></div>'
            f'{_forecast_svg(payload, uid, show_all_models=True)}'
            "</div>"
        )
        blocks.append(
            '<div class="chart-block">'
            f'<div class="chart-head"><strong>{_esc(uid)} - selected plus top weighted alternatives</strong><span>clean decision view with interval bands where available</span></div>'
            f'{_forecast_svg(payload, uid, show_all_models=False)}'
            "</div>"
        )
    if len(series_ids) > 6:
        blocks.append(f'<p class="footnote">Showing 6 of {len(series_ids)} series. Use streamlit_app.py for the full interactive review.</p>')
    return "\n".join(blocks)


def _payload_interval_status(payload: dict[str, Any], uid: str) -> str:
    for row in payload.get("trust_summary", []):
        if str(row.get("unique_id")) == str(uid):
            return str(row.get("interval_status") or "unavailable")
    return "unavailable"


def _payload_horizon_summary(payload: dict[str, Any], uid: str) -> str:
    for row in payload.get("trust_summary", []):
        if str(row.get("unique_id")) == str(uid):
            requested = row.get("requested_horizon")
            validated = row.get("validated_through_horizon") or row.get("selection_horizon")
            full_claim = _truthy(row.get("full_horizon_claim_allowed"))
            cv_windows = row.get("cv_windows")
            state = str(row.get("horizon_trust_state") or "no_rolling_origin_evidence")
            if state == "full_horizon_validated":
                if not full_claim:
                    return f"requested horizon {requested} evaluated, but only {cv_windows} CV window(s); planning claim limited"
                return f"validated through requested horizon {validated}"
            if state == "partial_horizon_validated":
                return f"validated through {validated} of requested {requested}"
            return _horizon_trust_label(state)
    return "horizon validation unavailable"


def _forecast_svg(payload: dict[str, Any], series_id: str, *, show_all_models: bool) -> str:
    history = [row for row in payload.get("history", []) if str(row.get("unique_id")) == series_id and row.get("y") is not None]
    forecast = [row for row in payload.get("forecast", []) if str(row.get("unique_id")) == series_id and row.get("yhat") is not None]
    all_models = [row for row in payload.get("all_models", []) if str(row.get("unique_id")) == series_id]
    forecast_long = [row for row in payload.get("forecast_long", []) if str(row.get("unique_id")) == series_id]
    if len(history) + len(forecast) < 2:
        return f'<p class="footnote">{_esc(series_id)}: not enough rows to render a chart.</p>'

    selected = _selected_model(payload, series_id)
    top_models = [model for model in _top_model_names(payload, series_id) if model != selected][:3]
    model_cols = _dedupe_names([*_model_columns_from_records(all_models), *_forecast_long_model_names(payload, series_id)])

    n_hist = len(history)
    n_forecast = len(forecast)
    n_total = n_hist + n_forecast
    scale_values = _safe_floats([row.get("y") for row in history])
    scale_values += _safe_floats([row.get("yhat") for row in forecast])
    for row in forecast:
        for col in ("yhat_lo_95", "yhat_hi_95", "yhat_lo_80", "yhat_hi_80"):
            if col in row:
                scale_values += _safe_floats([row.get(col)])
    scale_models = model_cols if show_all_models else [selected, *top_models]
    for col in scale_models:
        if not col:
            continue
        long_rows = _forecast_long_rows(payload, series_id, col)
        if long_rows:
            for row in long_rows:
                scale_values += _safe_floats([row.get("yhat")])
                for level in (95, 80):
                    for bound in (f"yhat_lo_{level}", f"yhat_hi_{level}"):
                        if bound in row:
                            scale_values += _safe_floats([row.get(bound)])
        else:
            for row in all_models:
                scale_values += _safe_floats([row.get(col)])
                for level in (95, 80):
                    for bound in (f"{col}-lo-{level}", f"{col}-hi-{level}"):
                        if bound in row:
                            scale_values += _safe_floats([row.get(bound)])
    if not scale_values:
        return f'<p class="footnote">{_esc(series_id)}: no numeric values to render.</p>'
    y_min, y_max = min(scale_values), max(scale_values)
    y_span = y_max - y_min if y_max != y_min else 1.0
    y_min -= y_span * 0.08
    y_max += y_span * 0.08
    y_span = y_max - y_min

    width, height = 1180, 340
    left, top, right, bottom = 68, 24, 210, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    clip_id = f"clip-{abs(hash((series_id, show_all_models))) % 10_000_000}"
    line_labels: list[dict[str, Any]] = []

    def xy(idx: int, value: float) -> tuple[float, float]:
        return left + (idx / max(1, n_total - 1)) * plot_w, top + (1 - (value - y_min) / y_span) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;display:block">',
        f'<defs><clipPath id="{clip_id}"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
    ]
    for j in range(5):
        y = top + j * plot_h / 4
        label = _fmt(y_max - j * y_span / 4)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#edf0f4"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" fill="#6b7480" font-size="11">{_esc(label)}</text>')
    if n_forecast:
        x0 = xy(n_hist, y_min)[0]
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{width - right - x0:.1f}" height="{plot_h}" fill="#b65f32" opacity="0.035"/>')

    alt_colors = ["#516246", "#62518f", "#9a6b21"]

    def svg_model_color(model: str) -> str:
        if model == selected:
            return "#b65f32"
        if model in top_models:
            return alt_colors[top_models.index(model) % len(alt_colors)]
        return "#b9c0c9"

    def svg_model_opacity(model: str, level: int) -> str:
        if model == selected:
            return "0.16" if level == 80 else "0.08"
        if model in top_models:
            return "0.11" if level == 80 else "0.055"
        return "0.035" if level == 80 else "0.018"

    def append_model_band(model: str, level: int) -> None:
        upper: list[tuple[float, float]] = []
        lower: list[tuple[float, float]] = []
        long_rows = _forecast_long_rows(payload, series_id, model)
        if long_rows:
            lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
            for i, row in enumerate(long_rows):
                lo = row.get(lo_col)
                hi = row.get(hi_col)
                if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
                    continue
                try:
                    upper.append(xy(n_hist + i, float(hi)))
                    lower.append(xy(n_hist + i, float(lo)))
                except (TypeError, ValueError):
                    continue
        else:
            lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
            for i, row in enumerate(all_models):
                lo = row.get(lo_col)
                hi = row.get(hi_col)
                if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
                    continue
                try:
                    upper.append(xy(n_hist + i, float(hi)))
                    lower.append(xy(n_hist + i, float(lo)))
                except (TypeError, ValueError):
                    continue
        if upper and lower:
            parts.append(
                f'<path d="{_svg_closed_path(upper + list(reversed(lower)))}" '
                f'fill="{svg_model_color(model)}" opacity="{svg_model_opacity(model, level)}" clip-path="url(#{clip_id})"/>'
            )

    band_models = model_cols if show_all_models else [model for model in [selected, *top_models] if model]
    for model in band_models:
        if model == selected and _selected_interval_levels(payload, series_id):
            continue
        for level in (95, 80):
            append_model_band(model, level)

    for level, opacity in [(95, "0.08"), (80, "0.16")]:
        lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
        if forecast and all(row.get(lo_col) is not None and row.get(hi_col) is not None for row in forecast):
            upper = [xy(n_hist + i, float(row[hi_col])) for i, row in enumerate(forecast)]
            lower = [xy(n_hist + i, float(row[lo_col])) for i, row in enumerate(forecast)]
            if upper and lower:
                parts.append(f'<path d="{_svg_closed_path(upper + list(reversed(lower)))}" fill="#b65f32" opacity="{opacity}" clip-path="url(#{clip_id})"/>')

    if show_all_models:
        for col in model_cols:
            if col == selected or col in top_models:
                continue
            points = _future_model_points(payload, series_id, col, all_models, n_hist, xy)
            if points:
                parts.append(f'<path d="{_svg_path(points)}" fill="none" stroke="#b9c0c9" stroke-width="1" stroke-dasharray="2 4" opacity="0.38" clip-path="url(#{clip_id})"/>')

    for idx, model in enumerate(top_models):
        points = _future_model_points(payload, series_id, model, all_models, n_hist, xy)
        if points:
            parts.append(f'<path d="{_svg_path(points)}" fill="none" stroke="{alt_colors[idx % len(alt_colors)]}" stroke-width="2" stroke-dasharray="6 4" clip-path="url(#{clip_id})"/>')
            line_labels.append({"label": model, "y": points[-1][1], "color": alt_colors[idx % len(alt_colors)], "weight": 700})

    hist_points = [xy(i, float(row["y"])) for i, row in enumerate(history) if row.get("y") is not None]
    if hist_points:
        parts.append(f'<path d="{_svg_path(hist_points)}" fill="none" stroke="#17324d" stroke-width="2.6" stroke-linecap="round" clip-path="url(#{clip_id})"/>')

    selected_points: list[tuple[float, float]] = []
    if hist_points:
        selected_points.append(hist_points[-1])
    if selected:
        selected_points.extend(_future_model_points(payload, series_id, selected, all_models, n_hist, xy))
    if len(selected_points) <= 1:
        selected_points.extend([xy(n_hist + i, float(row["yhat"])) for i, row in enumerate(forecast) if row.get("yhat") is not None])
    if selected_points:
        parts.append(f'<path d="{_svg_path(selected_points)}" fill="none" stroke="#b65f32" stroke-width="3" stroke-linecap="round" clip-path="url(#{clip_id})"/>')
        line_labels.append({"label": f"{selected or 'Forecast'} (selected)", "y": selected_points[-1][1], "color": "#b65f32", "weight": 800})

    visible_band_models = [model for model in band_models if model in _interval_model_names(payload, series_id)]
    if visible_band_models:
        if show_all_models and len(visible_band_models) > 4:
            band_note = f"{len(visible_band_models)} model-colored interval ribbons; faint gray = unlabeled candidate bands"
        else:
            band_note = "interval ribbons: " + ", ".join(visible_band_models)
        parts.append(f'<text x="{left + 8}" y="{top + 14}" fill="#6b7480" font-size="11">{_esc(band_note)}</text>')

    last_hist_label = _date_label(history[-1].get("ds")) if history else ""
    first_fcst_label = _date_label(forecast[0].get("ds")) if forecast else ""
    last_fcst_label = _date_label(forecast[-1].get("ds")) if forecast else ""
    parts.append(_svg_line_labels(line_labels, x=width - right + 8, y_min=top + 10, y_max=top + plot_h - 10))
    parts.append(f'<text x="{left}" y="{height - 14}" fill="#6b7480" font-size="11">{_esc(_date_label(history[0].get("ds")) if history else "")}</text>')
    parts.append(f'<text x="{left + plot_w * 0.55:.1f}" y="{height - 14}" fill="#6b7480" font-size="11">forecast starts { _esc(first_fcst_label) }</text>')
    parts.append(f'<text x="{width - right}" y="{height - 14}" fill="#6b7480" font-size="11" text-anchor="end">{_esc(last_fcst_label or last_hist_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _backtest_review_charts(payload: dict[str, Any]) -> str:
    backtest = payload.get("backtest_predictions", [])
    if not backtest:
        return '<p class="footnote">No rolling-origin backtest predictions are available.</p>'
    df = pd.DataFrame(backtest)
    if df.empty or "cutoff" not in df.columns or "y" not in df.columns:
        return '<p class="footnote">Backtest data is incomplete.</p>'
    series_ids = sorted(df["unique_id"].astype(str).unique())[:4]
    blocks: list[str] = []
    for uid in series_ids:
        uid_df = df[df["unique_id"].astype(str) == uid].copy()
        uid_df["cutoff"] = pd.to_datetime(uid_df["cutoff"], errors="coerce")
        cutoffs = sorted(uid_df["cutoff"].dropna().unique())[-3:]
        for cutoff in cutoffs:
            cutoff_label = _date_label(cutoff)
            blocks.append(
                '<div class="chart-block" style="margin-top:12px">'
                f'<div class="chart-head"><strong>{_esc(uid)} - cutoff { _esc(cutoff_label) }</strong><span>fixed-axis train/holdout progression; other candidates shown faint/unlabeled</span></div>'
                f'{_backtest_svg(payload, uid, cutoff)}'
                "</div>"
            )
    if not blocks:
        return '<p class="footnote">Could not render backtest windows.</p>'
    return "\n".join(blocks)


def _backtest_svg(payload: dict[str, Any], series_id: str, cutoff: Any) -> str:
    history = pd.DataFrame(payload.get("history", []))
    backtest = pd.DataFrame(payload.get("backtest_predictions", []))
    if history.empty or backtest.empty:
        return '<p class="footnote">Backtest chart unavailable.</p>'
    history["ds"] = pd.to_datetime(history["ds"], errors="coerce")
    backtest["ds"] = pd.to_datetime(backtest["ds"], errors="coerce")
    backtest["cutoff"] = pd.to_datetime(backtest["cutoff"], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    series_history = history[history["unique_id"].astype(str) == series_id].sort_values("ds")
    series_backtest = backtest[backtest["unique_id"].astype(str) == series_id].sort_values(["cutoff", "ds"])
    train = series_history[series_history["ds"] <= cutoff_ts].sort_values("ds")
    window = series_backtest[series_backtest["cutoff"] == cutoff_ts].sort_values("ds")
    if window.empty:
        return '<p class="footnote">No rows for this cutoff.</p>'
    selected = _selected_model(payload, series_id)
    top_models = [model for model in _top_model_names(payload, series_id) if model != selected][:3]
    model_cols = [col for col in _model_columns_from_records(window.to_dict("records")) if col not in {"cutoff", "y"}]
    scale_values = _safe_floats(series_history["y"].tolist()) + _safe_floats(series_backtest["y"].tolist())
    for col in model_cols:
        if col and col in series_backtest.columns:
            scale_values += _safe_floats(series_backtest[col].tolist())
            for level in (95, 80):
                for bound in (f"{col}-lo-{level}", f"{col}-hi-{level}"):
                    if bound in series_backtest.columns:
                        scale_values += _safe_floats(series_backtest[bound].tolist())
    date_values = pd.concat([series_history["ds"], series_backtest["ds"]], ignore_index=True).dropna()
    if not scale_values or len(date_values) < 2:
        return '<p class="footnote">Not enough numeric backtest data.</p>'
    x_min, x_max = date_values.min(), date_values.max()
    x_span = max((x_max - x_min).total_seconds(), 1.0)
    y_min, y_max = min(scale_values), max(scale_values)
    y_span = y_max - y_min if y_max != y_min else 1.0
    y_min -= y_span * 0.08
    y_max += y_span * 0.08
    y_span = y_max - y_min
    width, height = 1180, 320
    left, top, right, bottom = 68, 24, 210, 38
    plot_w, plot_h = width - left - right, height - top - bottom
    clip_id = f"bt-{abs(hash((series_id, str(cutoff_ts)))) % 10_000_000}"
    line_labels: list[dict[str, Any]] = []

    def xy(date: Any, value: float) -> tuple[float, float]:
        offset = max((pd.Timestamp(date) - x_min).total_seconds(), 0.0)
        return left + (offset / x_span) * plot_w, top + (1 - (value - y_min) / y_span) * plot_h

    def series_points(frame: pd.DataFrame, value_col: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        if value_col not in frame.columns:
            return points
        for _, row in frame.iterrows():
            value = row.get(value_col)
            if pd.notna(row.get("ds")) and pd.notna(value):
                try:
                    points.append(xy(row["ds"], float(value)))
                except (TypeError, ValueError):
                    continue
        return points

    parts = [
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;display:block">',
        f'<defs><clipPath id="{clip_id}"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
    ]
    for j in range(4):
        y = top + j * plot_h / 3
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#edf0f4"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" fill="#6b7480" font-size="11">{_esc(_fmt(y_max - j * y_span / 3))}</text>')
    cutoff_x = xy(cutoff_ts, y_min)[0]
    if not train.empty:
        parts.append(f'<rect x="{left}" y="{top}" width="{max(0, cutoff_x - left):.1f}" height="{plot_h}" fill="#17324d" opacity="0.07"/>')
    x_test_start = xy(window["ds"].min(), y_min)[0]
    x_test_end = xy(window["ds"].max(), y_min)[0]
    parts.append(f'<rect x="{x_test_start:.1f}" y="{top}" width="{max(2, x_test_end - x_test_start):.1f}" height="{plot_h}" fill="#b65f32" opacity="0.10"/>')

    history_points = series_points(series_history, "y")
    if history_points:
        parts.append(f'<path d="{_svg_path(history_points)}" fill="none" stroke="#17324d" stroke-width="1.3" opacity="0.28" clip-path="url(#{clip_id})"/>')
    train_points = series_points(train, "y")
    if train_points:
        parts.append(f'<path d="{_svg_path(train_points)}" fill="none" stroke="#17324d" stroke-width="2.2" opacity="0.88" clip-path="url(#{clip_id})"/>')
    actual_points = series_points(window, "y")
    if actual_points:
        parts.append(f'<path d="{_svg_path(actual_points)}" fill="none" stroke="#17324d" stroke-width="2.6" clip-path="url(#{clip_id})"/>')

    if selected:
        for level, opacity in [(95, "0.08"), (80, "0.16")]:
            lo_col, hi_col = f"{selected}-lo-{level}", f"{selected}-hi-{level}"
            if lo_col in window.columns and hi_col in window.columns:
                upper = series_points(window.rename(columns={hi_col: "_hi"}), "_hi")
                lower = series_points(window.rename(columns={lo_col: "_lo"}), "_lo")
                if upper and lower:
                    parts.append(f'<path d="{_svg_closed_path(upper + list(reversed(lower)))}" fill="#b65f32" opacity="{opacity}" clip-path="url(#{clip_id})"/>')

    for col in model_cols:
        if col == selected or col in top_models or col not in window.columns:
            continue
        points = series_points(window, col)
        if points:
            parts.append(f'<path d="{_svg_path(points)}" fill="none" stroke="#b9c0c9" stroke-width="1" stroke-dasharray="2 4" opacity="0.35" clip-path="url(#{clip_id})"/>')
    alt_colors = ["#516246", "#62518f", "#9a6b21"]
    for idx, col in enumerate(top_models):
        if col in window.columns:
            points = series_points(window, col)
            if points:
                parts.append(f'<path d="{_svg_path(points)}" fill="none" stroke="{alt_colors[idx % len(alt_colors)]}" stroke-width="2" stroke-dasharray="6 4" clip-path="url(#{clip_id})"/>')
                line_labels.append({"label": col, "y": points[-1][1], "color": alt_colors[idx % len(alt_colors)], "weight": 700})
    if selected and selected in window.columns:
        points = series_points(window, selected)
        if points:
            parts.append(f'<path d="{_svg_path(points)}" fill="none" stroke="#b65f32" stroke-width="3" clip-path="url(#{clip_id})"/>')
            line_labels.append({"label": f"{selected} (selected)", "y": points[-1][1], "color": "#b65f32", "weight": 800})
    parts.append(f'<line x1="{cutoff_x:.1f}" x2="{cutoff_x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="#7d8791" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(_svg_line_labels(line_labels, x=width - right + 8, y_min=top + 10, y_max=top + plot_h - 10))
    parts.append(f'<text x="{left + 2}" y="{height - 12}" fill="#6b7480" font-size="11">fixed axis start {_esc(_date_label(x_min))}</text>')
    parts.append(f'<text x="{cutoff_x + 5:.1f}" y="{height - 12}" fill="#6b7480" font-size="11">cutoff {_esc(_date_label(cutoff_ts))}</text>')
    parts.append(f'<text x="{x_test_start + 6:.1f}" y="{top + 14}" fill="#8b4b2b" font-size="11">holdout</text>')
    parts.append(f'<text x="{width - right}" y="{height - 12}" fill="#6b7480" font-size="11" text-anchor="end">fixed axis end {_esc(_date_label(x_max))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _payload_from_run(run: ForecastRun) -> dict[str, Any]:
    from nixtla_scaffold.outputs import (
        build_forecast_long,
        build_hierarchy_backtest_comparison,
        build_hierarchy_contribution_frame,
        build_residual_test_summary,
        build_selected_forecast,
        build_trust_summary,
    )

    selected_forecast = build_selected_forecast(run)
    executive_headline = build_executive_headline(run).to_dict()
    return {
        "summary": {
            "title": _title_from_run(run),
            "narrative": _narrative_from_run(run),
            "engine": run.engine,
            "freq": run.profile.freq,
            "season_length": run.profile.season_length,
            "rows": run.profile.rows,
            "series_count": run.profile.series_count,
            "horizon": run.spec.horizon,
        },
        "history": _records(run.history),
        "forecast": _records(selected_forecast),
        "forecast_long": _records(build_forecast_long(run)),
        "all_models": _records(run.all_models),
        "model_selection": _records(run.model_selection),
        "backtest_metrics": _records(run.backtest_metrics),
        "model_win_rates": _model_win_rates_from_metrics(run.backtest_metrics),
        "residual_tests": _records(build_residual_test_summary(run)),
        "trust_summary": _records(build_trust_summary(run)),
        "target_transform_audit": _records(run.transformation_audit),
        "scenario_assumptions": _records(build_scenario_assumptions_frame(run.spec.events)),
        "scenario_forecast": _records(build_scenario_forecast_frame(selected_forecast)),
        "known_future_regressors": _records(build_known_future_regressors_frame(run.spec.regressors)),
        "driver_availability_audit": _records(run.driver_availability_audit),
        "driver_experiment_summary": _records(build_driver_experiment_summary_frame(run)),
        "backtest_predictions": _records(run.backtest_predictions),
        "backtest_windows": _records(backtest_windows_frame(run)),
        "model_weights": _records(run.model_weights),
        "seasonality_summary": _records(seasonality_summary_frame(run)),
        "seasonality_diagnostics": _records(seasonality_diagnostics_frame(run)),
        "seasonality_decomposition": _records(seasonality_decomposition_frame(run)),
        "hierarchy_reconciliation": _records(run.hierarchy_reconciliation),
        "hierarchy_contribution": _records(build_hierarchy_contribution_frame(run)),
        "hierarchy_backtest_comparison": _records(build_hierarchy_backtest_comparison(run)),
        "model_policy_resolution": run.model_policy_resolution,
        "best_practice_receipts": _records(best_practice_receipts_frame(run)),
        "executive_headline": executive_headline,
        "warnings": run.warnings,
    }


def _payload_from_directory(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    diagnostics = _read_json(run_dir / "diagnostics.json")
    profile = manifest.get("profile", {})
    spec = manifest.get("spec", {})
    forecast = _read_csv_records(run_dir / "forecast.csv")
    return {
        "summary": {
            "title": _title_from_records(forecast),
            "narrative": _narrative_from_manifest(manifest),
            "engine": manifest.get("engine", "unknown"),
            "freq": profile.get("freq", "unknown"),
            "season_length": profile.get("season_length", "unknown"),
            "rows": profile.get("rows", "unknown"),
            "series_count": profile.get("series_count", "unknown"),
            "horizon": spec.get("horizon", "unknown"),
        },
        "history": _read_csv_records(run_dir / "history.csv"),
        "forecast": forecast,
        "forecast_long": _read_artifact_records(run_dir, "forecast_long.csv"),
        "all_models": _read_artifact_records(run_dir, "all_models.csv"),
        "model_selection": _read_artifact_records(run_dir, "model_selection.csv"),
        "backtest_metrics": _read_artifact_records(run_dir, "backtest_metrics.csv"),
        "model_win_rates": _read_artifact_records(run_dir, "model_win_rates.csv"),
        "residual_tests": _read_artifact_records(run_dir, "residual_tests.csv"),
        "trust_summary": _read_artifact_records(run_dir, "trust_summary.csv"),
        "target_transform_audit": _read_artifact_records(run_dir, "target_transform_audit.csv"),
        "scenario_assumptions": _read_artifact_records(run_dir, "scenario_assumptions.csv"),
        "scenario_forecast": _read_artifact_records(run_dir, "scenario_forecast.csv"),
        "known_future_regressors": _read_artifact_records(run_dir, "known_future_regressors.csv"),
        "driver_availability_audit": _read_artifact_records(run_dir, "driver_availability_audit.csv"),
        "driver_experiment_summary": _read_artifact_records(run_dir, "driver_experiment_summary.csv"),
        "backtest_predictions": _read_artifact_records(run_dir, "backtest_predictions.csv"),
        "backtest_windows": _read_artifact_records(run_dir, "backtest_windows.csv"),
        "model_weights": _read_artifact_records(run_dir, "model_weights.csv"),
        "seasonality_summary": _read_artifact_records(run_dir, "seasonality_summary.csv"),
        "seasonality_diagnostics": _read_artifact_records(run_dir, "seasonality_diagnostics.csv"),
        "seasonality_decomposition": _read_artifact_records(run_dir, "seasonality_decomposition.csv"),
        "hierarchy_reconciliation": _read_artifact_records(run_dir, "hierarchy_reconciliation.csv"),
        "hierarchy_contribution": _read_artifact_records(run_dir, "hierarchy_contribution.csv"),
        "hierarchy_backtest_comparison": _read_artifact_records(run_dir, "hierarchy_backtest_comparison.csv"),
        "model_policy_resolution": manifest.get("model_policy_resolution", {}),
        "best_practice_receipts": _read_artifact_records(run_dir, "best_practice_receipts.csv"),
        "executive_headline": diagnostics.get("executive_headline", {}),
        "warnings": manifest.get("warnings", []),
    }


def _selected_model(payload: dict[str, Any], unique_id: str) -> str:
    for row in payload.get("model_selection", []):
        if str(row.get("unique_id")) == unique_id:
            return str(row.get("selected_model") or "")
    return ""


def _top_model_names(payload: dict[str, Any], unique_id: str) -> list[str]:
    rows = [row for row in payload.get("model_weights", []) if str(row.get("unique_id")) == unique_id]
    rows.sort(key=lambda row: float(row.get("weight") or 0), reverse=True)
    selected = _selected_model(payload, unique_id)
    return [str(row["model"]) for row in rows if row.get("model") and str(row["model"]) != selected][:3]


def _selected_interval_levels(payload: dict[str, Any], unique_id: str) -> list[int]:
    rows = [row for row in payload.get("forecast", []) if str(row.get("unique_id")) == unique_id]
    levels: list[int] = []
    for level in (80, 95):
        lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
        if any(row.get(lo_col) is not None and row.get(hi_col) is not None for row in rows):
            levels.append(level)
    return levels


def _interval_model_names(payload: dict[str, Any], unique_id: str) -> list[str]:
    rows = [row for row in payload.get("all_models", []) if str(row.get("unique_id")) == unique_id]
    model_cols = _dedupe_names([*_model_columns_from_records(rows), *_forecast_long_model_names(payload, unique_id)])
    models: list[str] = []
    selected = _selected_model(payload, unique_id)
    if selected and _selected_interval_levels(payload, unique_id):
        models.append(selected)
    for model in model_cols:
        if model in models:
            continue
        long_rows = _forecast_long_rows(payload, unique_id, model)
        if long_rows:
            for level in (80, 95):
                lo_col, hi_col = f"yhat_lo_{level}", f"yhat_hi_{level}"
                if any(row.get(lo_col) is not None and row.get(hi_col) is not None for row in long_rows):
                    models.append(model)
                    break
            if model in models:
                continue
        for level in (80, 95):
            lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
            if any(row.get(lo_col) is not None and row.get(hi_col) is not None for row in rows):
                models.append(model)
                break
    return models


def _top_weights(model_weights: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    if not model_weights:
        return []
    frame = pd.DataFrame(model_weights)
    if frame.empty or "weight" not in frame.columns:
        return model_weights[:limit]
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    if "family" not in frame.columns:
        frame["family"] = frame["model"].map(model_family) if "model" in frame.columns else ""
    top = frame.sort_values(["unique_id", "weight"], ascending=[True, False]).groupby("unique_id").head(4)
    return _records(top[["unique_id", "model", "family", "weight"]])[:limit]


def _trust_summary_display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        display_row = dict(row)
        if "interval_status" in display_row:
            display_row["interval_status"] = _interval_status_label(display_row.get("interval_status"))
        if "interval_method" in display_row:
            display_row["interval_method"] = _interval_method_label(display_row.get("interval_method"))
        if "cv_horizon_status" in display_row:
            display_row["cv_horizon_status"] = _cv_horizon_status_label(display_row.get("cv_horizon_status"))
        if "horizon_trust_state" in display_row:
            display_row["horizon_trust_state"] = _horizon_trust_label(display_row.get("horizon_trust_state"))
        if "horizon_gate_result" in display_row:
            display_row["horizon_gate_result"] = _horizon_gate_label(display_row.get("horizon_gate_result"))
        out.append(display_row)
    return out


def _interval_status_label(value: object) -> str:
    raw = str(value or "unavailable")
    return INTERVAL_STATUS_LABELS.get(raw, raw.replace("_", " "))


def _interval_method_label(value: object) -> str:
    raw = str(value or "not_applicable")
    return INTERVAL_METHOD_LABELS.get(raw, raw.replace("_", " "))


def _cv_horizon_status_label(value: object) -> str:
    raw = str(value or "unavailable")
    return CV_HORIZON_STATUS_LABELS.get(raw, raw.replace("_", " "))


def _horizon_trust_label(value: object) -> str:
    raw = str(value or "no_rolling_origin_evidence")
    return HORIZON_TRUST_LABELS.get(raw, raw.replace("_", " "))


def _horizon_gate_label(value: object) -> str:
    raw = str(value or "fail_no_rolling_origin_validation")
    return HORIZON_GATE_LABELS.get(raw, raw.replace("_", " "))


def _sort_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    sort_cols = [col for col in ["unique_id", "rmse", "mase", "mae", "wape"] if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    return _records(frame)


def _model_win_rates_from_metrics(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if metrics.empty:
        return []
    metric = "mase" if "mase" in metrics.columns and metrics["mase"].notna().any() else "rmse"
    benchmark_model = "SeasonalNaive" if (metrics["model"].astype(str) == "SeasonalNaive").any() else "Naive"
    benchmark = metrics[metrics["model"].astype(str) == benchmark_model][["unique_id", metric]].rename(columns={metric: "benchmark_metric"})
    if benchmark.empty:
        return []
    merged = metrics.merge(benchmark, on="unique_id", how="inner")
    merged = merged[merged[metric].notna() & merged["benchmark_metric"].notna() & (merged["benchmark_metric"] > 0)]
    rows: list[dict[str, Any]] = []
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
        return []
    frame = pd.DataFrame(rows).sort_values(["win_rate_vs_benchmark", "avg_skill_vs_benchmark", "model"], ascending=[False, False, True])
    return _records(frame)


def _model_columns_from_records(records: list[dict[str, Any]]) -> list[str]:
    excluded = {"unique_id", "ds", "cutoff", "y"}
    cols: list[str] = []
    for row in records:
        for col in row:
            if col not in excluded and "-lo-" not in col and "-hi-" not in col and col not in cols:
                cols.append(col)
    return cols


def _dedupe_names(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        name = str(value)
        if name and name != "nan" and name not in out:
            out.append(name)
    return out


def _forecast_long_model_names(payload: dict[str, Any], unique_id: str) -> list[str]:
    return _dedupe_names(
        [
            str(row.get("model"))
            for row in payload.get("forecast_long", [])
            if str(row.get("unique_id")) == str(unique_id) and row.get("model")
        ]
    )


def _forecast_long_rows(payload: dict[str, Any], unique_id: str, model: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in payload.get("forecast_long", [])
        if str(row.get("unique_id")) == str(unique_id) and str(row.get("model")) == str(model)
    ]
    return sorted(rows, key=lambda row: str(row.get("ds") or ""))


def _future_model_points(
    payload: dict[str, Any],
    unique_id: str,
    model: str,
    fallback_rows: list[dict[str, Any]],
    offset: int,
    xy: Any,
) -> list[tuple[float, float]]:
    long_rows = _forecast_long_rows(payload, unique_id, model)
    if long_rows:
        return _model_points(long_rows, "yhat", offset, xy)
    return _model_points(fallback_rows, model, offset, xy)


def _model_points(
    rows: list[dict[str, Any]],
    model: str,
    offset: int,
    xy: Any,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for idx, row in enumerate(rows):
        try:
            value = row.get(model)
            if value is not None and pd.notna(value):
                points.append(xy(offset + idx, float(value)))
        except (TypeError, ValueError):
            continue
    return points


def _limitations(summary: dict[str, Any], warnings: list[str]) -> str:
    default_items = [
        "This forecast is a statistical baseline, not the answer. It assumes future behavior resembles backtested historical behavior unless event or driver overlays were supplied.",
        "Champion selection uses backtested RMSE when available because it penalizes large misses; trust and benchmark review also show MAE, MASE, RMSSE, WAPE, and bias for scale-free and business-readable checks.",
        "Prediction intervals are empirically calibrated uncertainty bands only when interval_status is calibrated. Future-only, adjusted-not-recalibrated, and point-only-ensemble bands are planning aids, not validated coverage; even calibrated bands can be too narrow during structural breaks, launches, pricing changes, or market shocks.",
        "Review the cross-validation windows before using the forecast in a model or planning workflow.",
    ]
    warning_items = [warning for warning in warnings if _is_user_warning(warning)][:8]
    items = "".join(f"<li>{_esc(item)}</li>" for item in [*default_items, *warning_items])
    actions = (
        "<ol>"
        "<li>Validate the trajectory against plan, budget, prior year, and domain-owner expectations.</li>"
        "<li>Add known future events with --event when the history cannot know about launches, pricing changes, or contract movements.</li>"
        "<li>Search for regressors only when future values are known and leakage checks pass.</li>"
        "<li>Use forecast_long.csv and backtest_long.csv as model feeds; use granular files only for audit/debug.</li>"
        "</ol>"
    )
    return f'<ul class="warn">{items}</ul><h3 style="margin-top:14px">Action plan</h3>{actions}'


def _is_user_warning(warning: str) -> bool:
    lowered = warning.lower()
    return not (
        warning.startswith("StatsForecast ladder:")
        or "copy keyword is deprecated" in lowered
        or "backtest h=" in lowered
    )


def _metric_card(label: str, value: object) -> str:
    return f'<div class="card"><div class="label">{_esc(label)}</div><div class="value">{_esc(value)}</div></div>'


def _trust_summary_cards(rows: list[dict[str, Any]], *, limit: int) -> str:
    if not rows:
        return '<p class="footnote">No trust summary rows available.</p>'
    visible = rows[:limit]
    cards = "".join(_trust_summary_card(row) for row in visible)
    suffix = '<p class="footnote">Use <code>trust_summary.csv</code> for the raw wide table; this report renders one readable decision card per series.</p>'
    if len(rows) > limit:
        suffix += f'<p class="footnote">Showing {limit} of {len(rows)} series. Use the CSV artifacts for the full set.</p>'
    return f'<div class="decision-grid">{cards}</div>{suffix}'


def _trust_summary_card(row: dict[str, Any]) -> str:
    trust = str(row.get("trust_level") or "Unknown")
    score = row.get("trust_score_0_100")
    selected_model = row.get("selected_model") or "No selected model"
    selection_horizon = row.get("selection_horizon")
    requested_horizon = row.get("requested_horizon")
    horizon_pair = _horizon_pair(selection_horizon, requested_horizon)
    badge_class = _trust_badge_class(trust)
    chips = [
        _decision_chip("Trust score", score),
        _decision_chip("Validated horizon", horizon_pair),
        _decision_chip("Unvalidated steps", row.get("unvalidated_steps")),
        _decision_chip("Score cap", row.get("horizon_trust_score_cap")),
    ]
    fields = [
        _decision_field("Horizon status", row.get("horizon_trust_state")),
        _decision_field("Horizon gate", row.get("horizon_gate_result")),
        _decision_field("CV horizon", row.get("cv_horizon_status")),
        _decision_field("Planning-ready horizon", row.get("planning_ready_horizon")),
        _decision_field("Residual stability", row.get("residual_stability")),
        _decision_field("Intervals", _interval_summary(row)),
        _decision_field("Caveats", row.get("caveats"), wide=True),
        _decision_field("Next actions", row.get("next_actions"), wide=True),
    ]
    return f"""
        <article class="series-decision-card">
          <div class="decision-title">
            <div>
              <div class="series-name">{_display_value(row.get("unique_id"), fallback="Series")}</div>
              <div class="selected-model">Selected model: {_display_value(selected_model)}</div>
            </div>
            <span class="trust-badge {badge_class}">{_esc(trust)}</span>
          </div>
          <div class="decision-chips">{"".join(chips)}</div>
          <div class="decision-detail-grid">{"".join(fields)}</div>
        </article>
    """


def _trust_badge_class(trust_level: str) -> str:
    lowered = trust_level.lower()
    if "high" in lowered:
        return "trust-high"
    if "medium" in lowered:
        return "trust-medium"
    if "low" in lowered:
        return "trust-low"
    return "trust-unknown"


def _decision_chip(label: str, value: object) -> str:
    return f'<div class="decision-chip"><div class="chip-label">{_esc(label)}</div><div class="chip-value">{_display_value(value)}</div></div>'


def _decision_field(label: str, value: object, *, wide: bool = False) -> str:
    classes = "decision-field wide" if wide else "decision-field"
    value_class = "field-value long" if wide else "field-value"
    return f'<div class="{classes}"><div class="field-label">{_esc(label)}</div><div class="{value_class}">{_display_value(value)}</div></div>'


def _display_value(value: object, *, fallback: str = "Not available") -> str:
    if value is None:
        return f'<span class="muted">{_esc(fallback)}</span>'
    if isinstance(value, str) and not value.strip():
        return f'<span class="muted">{_esc(fallback)}</span>'
    return _format_cell(value)


def _horizon_pair(selection_horizon: object, requested_horizon: object) -> str:
    selection = _display_value(selection_horizon, fallback="?")
    requested = _display_value(requested_horizon, fallback="?")
    return f"{selection} / {requested}"


def _interval_summary(row: dict[str, Any]) -> str:
    status = row.get("interval_status") or "Not available"
    method = row.get("interval_method") or "Not available"
    return f"{status}; {method}"


def _output_item(name: str, description: str) -> str:
    return f'<div class="output-item"><strong><code>{_esc(name)}</code></strong><span class="footnote">{_esc(description)}</span></div>'


def _target_transform_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return f"""
    <section class="panel" style="margin-bottom:14px">
      <h2>Target transformation audit</h2>
      <p class="footnote">This table shows the raw target, any finance normalization factor, the adjusted target used for modeling, and the modeled transform scale. Forecast outputs are inverse-transformed when log/log1p is used; normalized runs remain in normalized units unless future factors are supplied externally.</p>
      {_table(rows, ["unique_id", "ds", "y_raw", "normalization_factor_col", "normalization_factor", "y_adjusted", "target_transform", "y_modeled", "output_scale", "notes"], limit=24)}
    </section>
    """


def _driver_assumptions_section(
    scenario_assumptions: list[dict[str, Any]],
    scenario_forecast: list[dict[str, Any]],
    known_future_regressors: list[dict[str, Any]],
    driver_availability_audit: list[dict[str, Any]],
    driver_experiment_summary: list[dict[str, Any]],
) -> str:
    if not (
        scenario_assumptions
        or scenario_forecast
        or known_future_regressors
        or driver_availability_audit
        or driver_experiment_summary
    ):
        return ""
    return f"""
    <section class="panel" style="margin-bottom:14px">
      <h2>Assumptions and drivers</h2>
      <p class="footnote">Scenario events are post-model overlays: baseline <code>yhat</code> remains separate from <code>yhat_scenario</code>. Known-future regressors are audited for leakage and future availability, but this release does not automatically train arbitrary external regressors; current MLForecast candidates use lag/date features.</p>
      <h3>Driver experiment summary</h3>
      {_table(driver_experiment_summary, ["driver_type", "name", "mode", "availability", "audit_status", "modeling_decision", "affected_or_required_rows", "next_step"], limit=16)}
      <h3 style="margin-top:14px">Scenario assumptions</h3>
      {_table(scenario_assumptions, ["assumption_type", "name", "start", "end", "effect", "magnitude", "confidence", "effective_magnitude", "affected_unique_ids", "notes"], limit=16)}
      <h3 style="margin-top:14px">Scenario forecast preview</h3>
      {_table(scenario_forecast, ["unique_id", "ds", "model", "horizon_step", "yhat", "yhat_scenario", "event_adjustment", "event_names"], limit=16)}
      <h3 style="margin-top:14px">Known-future regressor declarations</h3>
      {_table(known_future_regressors, ["name", "value_col", "availability", "mode", "future_file", "source_system", "owner", "contract_status"], limit=16)}
      <h3 style="margin-top:14px">Driver availability audit</h3>
      {_table(driver_availability_audit, ["name", "audit_status", "modeling_decision", "leakage_risk", "required_future_rows", "missing_future_rows", "known_as_of_violations", "audit_message"], limit=16)}
    </section>
    """


def _hierarchy_depth_section(
    hierarchy_contribution: list[dict[str, Any]],
    hierarchy_backtest_comparison: list[dict[str, Any]],
) -> str:
    if not hierarchy_contribution and not hierarchy_backtest_comparison:
        return ""
    return f"""
      <h2 style="margin-top:18px">Hierarchy depth review</h2>
      <p class="footnote">Use these before stakeholder planning: contribution rows show which children drive parent totals/gaps; backtest comparison shows whether reconciliation improves coherence at the cost of historical accuracy.</p>
      <h3>Hierarchy contribution preview</h3>
      {_table(hierarchy_contribution, ["value_col", "parent_unique_id", "child_unique_id", "ds", "child_value", "child_share_of_parent", "parent_child_gap", "gap_pct"], limit=16)}
      <h3 style="margin-top:14px">Reconciled vs unreconciled backtest preview</h3>
      {_table(hierarchy_backtest_comparison, ["unique_id", "model", "cutoff", "ds", "reconciliation_method", "applied_method", "comparison_status", "abs_error_unreconciled", "abs_error_reconciled", "abs_error_delta", "comparison_note"], limit=16)}
    """


def _model_policy_resolution_section(resolution: dict[str, Any]) -> str:
    rows = _policy_resolution_rows(resolution)
    if not rows:
        return ""
    policy = resolution.get("model_policy", "unknown") if isinstance(resolution, dict) else "unknown"
    return f"""
    <section class="panel" style="margin-bottom:14px">
      <h2>Model policy resolution</h2>
      <p class="footnote"><code>model_policy={_esc(policy)}</code>. <code>all</code> means all eligible open-source families; MLForecast requires enough history and installed optional dependencies. Empty or skipped families are explicit instead of silently hidden.</p>
      {_table(rows, ["family", "requested", "eligible", "ran", "reason_if_not_ran", "contributed_models"], limit=12)}
    </section>
    """


def _policy_resolution_rows(resolution: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(resolution, dict):
        return []
    rows: list[dict[str, Any]] = []
    for family in resolution.get("families", []) or []:
        if not isinstance(family, dict):
            continue
        rows.append(
            {
                "family": family.get("family", ""),
                "requested": family.get("requested", False),
                "eligible": family.get("eligible", False),
                "ran": family.get("ran", False),
                "reason_if_not_ran": family.get("reason_if_not_ran", ""),
                "contributed_models": ", ".join(str(model) for model in family.get("contributed_models", []) or []),
            }
        )
    return rows


def _table(rows: list[dict[str, Any]], columns: list[str], *, limit: int) -> str:
    if not rows:
        return '<p class="footnote">No rows available.</p>'
    visible = rows[:limit]
    header = "".join(f"<th>{_esc(col)}</th>" for col in columns)
    body = []
    for row in visible:
        body.append("<tr>" + "".join(f"<td>{_format_cell(row.get(col))}</td>" for col in columns) + "</tr>")
    suffix = ""
    if len(rows) > limit:
        suffix = f'<p class="footnote">Showing {limit} of {len(rows)} rows. Use the CSV artifacts for the full table.</p>'
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{suffix}'


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if str(value).lower() == "nan":
            return ""
        if abs(value) < 1 and value != 0:
            return _esc(f"{value:.2%}")
        return _esc(_fmt(value))
    return _esc(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: _safe_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _records(pd.read_csv(path))


def _read_artifact_records(run_dir: Path, name: str) -> list[dict[str, Any]]:
    for path in (run_dir / name, run_dir / "audit" / name):
        if path.exists():
            return _records(pd.read_csv(path))
    return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_floats(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            if value is not None and pd.notna(value):
                out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _truthy(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _svg_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first, *rest = points
    return " ".join([f"M {first[0]:.1f} {first[1]:.1f}", *(f"L {x:.1f} {y:.1f}" for x, y in rest)])


def _svg_line_labels(labels: list[dict[str, Any]], *, x: float, y_min: float, y_max: float) -> str:
    if not labels:
        return ""
    ordered = sorted(labels, key=lambda item: float(item["y"]))
    min_gap = 14.0
    placed: list[dict[str, Any]] = []
    for item in ordered:
        y = min(max(float(item["y"]), y_min), y_max)
        if placed and y - float(placed[-1]["placed_y"]) < min_gap:
            y = float(placed[-1]["placed_y"]) + min_gap
        placed.append(item | {"placed_y": y})
    overflow = placed[-1]["placed_y"] - y_max if placed else 0
    if overflow > 0:
        for item in placed:
            item["placed_y"] = max(y_min, float(item["placed_y"]) - overflow)
    output: list[str] = []
    for item in placed:
        y0 = float(item["y"])
        y1 = float(item["placed_y"])
        color = str(item.get("color", "#59636e"))
        weight = int(item.get("weight", 500))
        label = _esc(item.get("label", ""))
        output.append(f'<line x1="{x - 8:.1f}" x2="{x - 2:.1f}" y1="{y0:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="1" opacity="0.65"/>')
        output.append(
            f'<text x="{x:.1f}" y="{y1 + 4:.1f}" fill="{color}" font-size="11" font-weight="{weight}">{label}</text>'
        )
    return "".join(output)


def _svg_closed_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    return _svg_path(points) + " Z"


def _title_from_run(run: ForecastRun) -> str:
    ids = sorted(str(uid) for uid in run.forecast["unique_id"].dropna().unique())
    if len(ids) == 1:
        return f"{ids[0]} forecast"
    return f"{len(ids)} series forecast"


def _title_from_records(forecast: list[dict[str, Any]]) -> str:
    ids = sorted({str(row.get("unique_id")) for row in forecast if row.get("unique_id") is not None})
    if len(ids) == 1:
        return f"{ids[0]} forecast"
    return f"{len(ids)} series forecast"


def _narrative_from_run(run: ForecastRun) -> str:
    if run.model_selection.empty:
        return "Forecast generated without backtest-based model selection; treat it as a baseline and inspect limitations."
    selected = run.model_selection["selected_model"].value_counts().idxmax()
    metric = "RMSE" if "rmse" in run.model_selection.columns else "backtest error"
    return (
        f"Selected forecasts were produced with {run.engine}; the most common selected model is {selected}. "
        f"Champion selection is driven by backtested {metric}, with model disagreement, interval_status, and rolling-origin windows shown below."
    )


def _narrative_from_manifest(manifest: dict[str, Any]) -> str:
    engine = manifest.get("engine", "the configured engine")
    return (
        f"Selected forecasts were produced with {engine}. Review champion-vs-challenger disagreement, interval_status, "
        "rolling-origin windows, warnings, and consolidated feeder outputs before using this forecast operationally."
    )


def _date_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)[:10]


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)
