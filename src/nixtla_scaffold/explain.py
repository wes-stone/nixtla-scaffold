from __future__ import annotations

from nixtla_scaffold.headline import build_executive_headline
from nixtla_scaffold.schema import ForecastRun


def build_model_card(run: ForecastRun) -> str:
    profile = run.profile
    executive = build_executive_headline(run)
    lines = [
        "# Forecast model card",
        "",
        executive.paragraph,
        "",
        *_forecast_headlines(run),
        "",
        f"- Engine: `{run.engine}`",
        f"- Series: {profile.series_count}",
        f"- Rows: {profile.rows}",
        f"- Date range: {profile.start} to {profile.end}",
        f"- Inferred frequency: `{profile.freq}`",
        f"- Season length used: {profile.season_length}",
        f"- Horizon: {run.spec.horizon}",
        f"- Fill method: `{run.spec.fill_method}`",
        f"- Target transform: `{run.spec.transform.target}`",
        f"- Normalization factor column: `{run.spec.transform.normalization_factor_col or 'none'}`",
        f"- Hierarchy reconciliation: `{run.spec.hierarchy_reconciliation}`",
        f"- Weighted ensemble: {'enabled' if run.spec.weighted_ensemble else 'disabled'}",
        f"- Model-engine verbose logging: {'enabled' if run.spec.verbose else 'disabled'}",
        "- Diagnostics artifacts: always written",
        "",
        "## Selected models",
        "",
    ]
    if run.model_selection.empty:
        lines.append("No model selection table was produced.")
    else:
        for row in run.model_selection.to_dict("records"):
            metric = ""
            if "rmse" in row and row["rmse"] == row["rmse"]:
                metric = f" (RMSE {row['rmse']:.4g})"
            elif "wape" in row and row["wape"] == row["wape"]:
                metric = f" (WAPE {row['wape']:.1%})"
            cv_contract = ""
            selection_horizon = row.get("selection_horizon")
            requested_horizon = row.get("requested_horizon")
            if (
                selection_horizon is not None
                and requested_horizon is not None
                and selection_horizon == selection_horizon
                and requested_horizon == requested_horizon
            ):
                cv_contract = f", CV h={int(selection_horizon)}/{int(requested_horizon)}"
            lines.append(f"- `{row['unique_id']}`: `{row['selected_model']}`{metric}{cv_contract} - {row['selection_reason']}")

    from nixtla_scaffold.outputs import build_trust_summary

    trust_summary = build_trust_summary(run)
    lines.extend(["", "## Trust and action summary", ""])
    if trust_summary.empty:
        lines.append("No trust summary table was produced.")
    else:
        for row in trust_summary.to_dict("records"):
            caveats = row.get("caveats") or "No major caveats."
            actions = row.get("next_actions") or "No next actions recorded."
            lines.append(
                f"- `{row['unique_id']}`: **{row['trust_level']} trust** "
                f"({row['trust_score_0_100']}/100), model `{row['selected_model']}`. "
                f"Caveats: {caveats} Next: {actions}"
            )

    lines.extend(["", "## Data and modeling warnings", ""])
    warnings = _all_warnings(run)
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")

    lines.extend(["", "## Best-practice receipts", ""])
    for receipt in run.best_practice_receipts():
        lines.append(f"- `{receipt['id']}`: {receipt['status']} - {receipt['evidence']}")

    lines.extend(["", "## Diagnostics for future LLMs", ""])
    lines.append("- Quote `diagnostics.json.executive_headline.paragraph` verbatim when summarizing this run; do not rewrite it into a stronger claim.")
    lines.append("- Start with `diagnostics.json` for machine-readable run context, warnings, model weights, and next checks.")
    lines.append("- Use `failure_diagnostics.json` if a CLI run fails before normal artifacts are produced.")
    lines.append("- Use `forecast_long.csv`, `backtest_long.csv`, `series_summary.csv`, and `model_audit.csv` as the core feeder/review files.")
    lines.append("- Use `audit/backtest_windows.csv` and `audit/backtest_predictions.csv` to inspect forecast-vs-actual holdouts by cutoff.")
    if run.spec.transform.enabled:
        lines.append("- Use `audit/target_transform_audit.csv` to trace raw y, normalized y, modeled target values, and output scale.")
    if run.spec.hierarchy_reconciliation != "none":
        lines.append("- Use `hierarchy_reconciliation.csv` plus `audit/hierarchy_coherence_pre.csv` and `audit/hierarchy_coherence_post.csv` to inspect pre/post planning coherence.")
    lines.append("- Use `audit/seasonality_diagnostics.csv` and `audit/seasonality_decomposition.csv` to inspect cycle counts, credibility, trend/seasonal/remainder evidence, and seasonal warnings.")
    lines.append("- Open `interpretation.md` for a readable backtesting and seasonality walkthrough.")
    lines.append("- Open `report.html` for a portable visual report; `report_base64.txt` stores the same HTML for text-only handoff.")
    lines.append("- Run `streamlit run streamlit_app.py` in the output folder for an editable local dashboard scaffold.")
    if not run.model_weights.empty:
        lines.append("- `audit/model_weights.csv` shows inverse-error weights used by `WeightedEnsemble`.")
    else:
        lines.append("- `audit/model_weights.csv` is empty because weighted ensemble weights could not be learned from backtests.")

    lines.extend(["", "## Interpretation guardrails", ""])
    lines.append("- Treat the forecast as a statistical baseline unless driver/event assumptions are added.")
    lines.append("- Compare against prior-year, plan, and known business events before stakeholder use.")
    if run.spec.events:
        lines.append("- Scenario columns include user-provided driver/event assumptions; keep the statistical baseline for comparison.")
    else:
        lines.append("- No driver/event assumptions were applied; this is an all-else-equal statistical forecast.")
    if run.spec.transform.normalization_factor_col:
        lines.append("- Finance normalization was applied before modeling; forecasts remain in normalized units unless future normalization factors are supplied externally.")
    elif run.spec.transform.target != "none":
        lines.append("- Target transform outputs were inverse-transformed for reporting; interval interpretation after nonlinear inverse transforms is approximate.")
    if run.spec.hierarchy_reconciliation != "none":
        lines.append("- Hierarchy reconciliation produces coherent planning outputs, but independent model accuracy should still be judged from the unreconciled model tournament and rolling-origin CV.")
    lines.append("- If backtest metrics are missing, history was too short for a reliable champion test.")
    return "\n".join(lines) + "\n"


def _all_warnings(run: ForecastRun) -> list[str]:
    warnings = list(run.warnings)
    for series in run.profile.series:
        warnings.extend(f"{series.unique_id}: {warning}" for warning in series.warnings)
    return list(dict.fromkeys(warnings))


def _forecast_headlines(run: ForecastRun) -> list[str]:
    if run.forecast.empty:
        return []

    from nixtla_scaffold.outputs import build_selected_forecast

    forecast = build_selected_forecast(run)
    lines = ["## Forecast headline", "", "### Per-series detail", ""]
    for row in forecast.groupby("unique_id", sort=True).first().reset_index().head(5).to_dict("records"):
        interval = ""
        interval_status = str(row.get("interval_status") or "unavailable")
        if _should_show_headline_interval(row, interval_status):
            interval = (
                f" (80% interval: {_fmt_number(row['yhat_lo_80'])} to {_fmt_number(row['yhat_hi_80'])} "
                f"- {_interval_status_phrase(interval_status)})"
            )
        scenario = ""
        if "yhat_scenario" in row:
            scenario_interval = ""
            if _is_present(row.get("yhat_scenario_lo_80")) and _is_present(row.get("yhat_scenario_hi_80")):
                scenario_interval = (
                    f" (scenario 80% interval: {_fmt_number(row['yhat_scenario_lo_80'])} "
                    f"to {_fmt_number(row['yhat_scenario_hi_80'])} - scenario-adjusted, not recalibrated)"
                )
            scenario = f"; scenario: {_fmt_number(row['yhat_scenario'])}{scenario_interval}"
        lines.append(
            f"- `{row['unique_id']}` next period {row['ds'].date().isoformat()}: "
            f"{_fmt_number(row['yhat'])}{scenario}{interval}, model `{row['model']}`. "
            f"Horizon trust: {_horizon_phrase(row)}"
        )
    if run.profile.series_count > 5:
        lines.append(f"- Plus {run.profile.series_count - 5} additional series in forecast.csv.")
    return lines


def _should_show_headline_interval(row: dict[str, object], interval_status: str) -> bool:
    if interval_status in {"unavailable", "point_only_ensemble"}:
        return False
    return _is_present(row.get("yhat_lo_80")) and _is_present(row.get("yhat_hi_80"))


def _interval_status_phrase(status: str) -> str:
    return {
        "calibrated": "calibrated in rolling-origin CV",
        "calibration_warning": "calibration warning",
        "calibration_fail": "undercoverage risk",
        "insufficient_observations": "too few CV interval observations",
        "future_only": "future-only, not CV-calibrated",
        "adjusted_not_recalibrated": "adjusted after calibration, not recalibrated",
    }.get(status, status.replace("_", " "))


def _horizon_phrase(row: dict[str, object]) -> str:
    state = str(row.get("horizon_trust_state") or "no_rolling_origin_evidence")
    requested = row.get("requested_horizon")
    validated = row.get("validated_through_horizon")
    planning_eligible = _truthy(row.get("planning_eligible"))
    cv_windows = row.get("cv_windows")
    if state == "full_horizon_validated":
        if not planning_eligible:
            return (
                f"validated through requested horizon {validated or requested}, but only {_fmt_count(cv_windows)} "
                "rolling-origin window(s) support it; planning-ready champion claim is limited."
            )
        return f"validated through requested horizon {validated or requested}."
    if state == "partial_horizon_validated":
        return f"validated through {validated} of requested {requested}; later steps are directional."
    if state == "beyond_validated_horizon":
        return f"this row is beyond validated horizon {validated} of requested {requested}; directional only."
    return "no rolling-origin validation evidence; do not call this a validated champion forecast."


def _is_present(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(value) == float(value)
    except (TypeError, ValueError):
        return bool(value)


def _truthy(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _fmt_count(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _fmt_number(value: object) -> str:
    number = float(value)
    return f"{number:,.0f}" if abs(number) >= 100 else f"{number:,.2f}"

