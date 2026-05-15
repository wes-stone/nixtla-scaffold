from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import pandas as pd

from nixtla_scaffold.data import load_forecast_dataset
from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.outputs import build_model_audit, build_model_pareto_frontier, build_trust_summary
from nixtla_scaffold.schema import ForecastRun, ForecastSpec


EXPERIMENT_SCHEMA_VERSION = "nixtla_scaffold.experiment.v1"
COMPARE_MODELS_SCHEMA_VERSION = "nixtla_scaffold.compare_models.v1"
DEFAULT_EXPERIMENT_VARIANTS = ("baseline", "all_models", "events", "known_future_regressors", "hierarchy_methods")
EXPERIMENT_VARIANTS = (
    "baseline",
    "events",
    "known_future_regressors",
    "rolling_features",
    "all_models",
    "hierarchy_methods",
)
_HIERARCHY_COLUMNS = {"hierarchy_level", "hierarchy_depth", "parent_id", "bottom_level"}
_DRIVER_NAME_TERMS = {
    "adoption",
    "booking",
    "bookings",
    "budget",
    "capacity",
    "contract",
    "contracts",
    "event",
    "events",
    "lag",
    "lags",
    "lead",
    "leads",
    "marketing",
    "minute",
    "minutes",
    "pipeline",
    "plan",
    "price",
    "pricing",
    "rolling",
    "seat",
    "seats",
    "spend",
    "trailing",
    "usage",
    "user",
    "users",
    "xreg",
}
_CALENDAR_OR_METADATA_TERMS = {
    "calendar",
    "date",
    "day",
    "fiscal",
    "month",
    "quarter",
    "timestamp",
    "week",
    "year",
}
_AUDIT_OR_TARGET_LEAKAGE_TERMS = {
    "actual",
    "backtest",
    "error",
    "forecast",
    "fitted",
    "hi",
    "lo",
    "lower",
    "pred",
    "prediction",
    "residual",
    "target",
    "upper",
    "yhat",
}


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    description: str
    build_spec: Callable[[ForecastSpec], ForecastSpec]
    applicable: Callable[[ExperimentContext, ForecastSpec], tuple[bool, str]]


@dataclass(frozen=True)
class ExperimentContext:
    has_events: bool
    has_regressors: bool
    has_hierarchy: bool
    row_count: int
    series_count: int
    source_hash_sha256: str | None
    candidate_drivers: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    human_context_questions: tuple[str, ...] = field(default_factory=tuple)
    autoresearch_hypotheses: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_events": self.has_events,
            "has_regressors": self.has_regressors,
            "has_hierarchy": self.has_hierarchy,
            "row_count": self.row_count,
            "series_count": self.series_count,
            "source_hash_sha256": self.source_hash_sha256,
            "candidate_drivers": list(self.candidate_drivers),
            "human_context_questions": list(self.human_context_questions),
            "autoresearch_hypotheses": list(self.autoresearch_hypotheses),
        }


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    manifest: dict[str, Any]
    summary: pd.DataFrame
    recommendation_markdown: str
    llm_context: dict[str, Any]


def compare_models(
    data: str | Path | pd.DataFrame,
    spec: ForecastSpec | None = None,
    *,
    sheet: str | int | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the standard scaffold tournament and return an advisory model leaderboard."""

    run = run_forecast(data, spec or ForecastSpec(), sheet=sheet)
    leaderboard = build_model_leaderboard(run)
    if output_dir is not None:
        out = run.to_directory(output_dir)
        leaderboard.to_csv(out / "compare_models_leaderboard.csv", index=False)
        (out / "compare_models_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": COMPARE_MODELS_SCHEMA_VERSION,
                    "purpose": "Advisory leaderboard over the existing scaffold model tournament; does not override forecast.csv selection.",
                    "outputs": {
                        "leaderboard": "compare_models_leaderboard.csv",
                        "normal_run_manifest": "manifest.json",
                    },
                    "selected_models": _selected_model_counts(run),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    return leaderboard


def run_experiment(
    data: str | Path | pd.DataFrame,
    spec: ForecastSpec | None = None,
    *,
    sheet: str | int | None = None,
    output_dir: str | Path = "runs/experiment_latest",
    variants: Sequence[str] | None = None,
    max_variants: int = 4,
) -> ExperimentResult:
    """Run a bounded set of advisory forecast variants and write experiment artifacts."""

    base_spec = spec or ForecastSpec()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "variants").mkdir(parents=True, exist_ok=True)
    context = _experiment_context(data, base_spec, sheet=sheet)
    catalog = _variant_catalog()
    requested = _resolve_variants(variants, context)
    selected, capped = requested[:max_variants], requested[max_variants:]

    summary_rows: list[dict[str, Any]] = []
    child_runs: list[dict[str, Any]] = []
    for name in selected:
        variant = catalog.get(name)
        if variant is None:
            summary_rows.append(_skipped_row(name, "unknown_variant", f"unknown variant; choose from {list(EXPERIMENT_VARIANTS)}"))
            continue
        ok, reason = variant.applicable(context, base_spec)
        if not ok:
            summary_rows.append(_skipped_row(name, "not_applicable", reason, description=variant.description))
            continue
        variant_spec = variant.build_spec(base_spec)
        variant_dir = out / "variants" / name
        try:
            run = run_forecast(data, variant_spec, sheet=sheet)
            run.to_directory(variant_dir)
        except Exception as exc:
            summary_rows.append(_skipped_row(name, "failed", str(exc), description=variant.description, run_path=variant_dir))
            continue
        row = _successful_row(name, variant.description, variant_dir, run)
        summary_rows.append(row)
        child_runs.append(
            {
                "variant": name,
                "description": variant.description,
                "path": str(variant_dir),
                "spec": variant_spec.to_dict(),
                "selected_models": _selected_model_counts(run),
            }
        )

    for name in capped:
        summary_rows.append(
            _skipped_row(
                name,
                "max_variants_cap",
                f"not run because max_variants={max_variants}; rerun with --max-variants if this variant matters",
            )
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = _assign_advisory_ranks(summary)
        summary["_status_order"] = summary["status"].map({"success": 0}).fillna(1)
        summary = summary.sort_values(["_status_order", "advisory_rank", "variant"], na_position="last").drop(columns=["_status_order"]).reset_index(drop=True)
    recommendation = build_experiment_recommendation(summary, context)
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "purpose": "Bounded advisory forecast experiment; child runs are normal scaffold outputs and no recommendation mutates forecast.csv.",
        "context": context.to_dict(),
        "base_spec": base_spec.to_dict(),
        "requested_variants": list(requested),
        "max_variants": max_variants,
        "child_runs": child_runs,
        "outputs": {
            "summary": "experiment_summary.csv",
            "recommendation": "experiment_recommendation.md",
            "llm_context": "experiment_llm_context.json",
            "manifest": "experiment_manifest.json",
        },
    }
    llm_context = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "guardrails": [
            "Experiment ranking is advisory and does not override each child run's selected forecast.",
            "Compare variants by rolling-origin backtest and trust evidence before stakeholder use.",
            "Use the autoresearch_next_iteration block to test one next hypothesis at a time with a fixed budget.",
            "Automatically detected candidate drivers are advisory only; declare and audit one regressor before training it.",
        ],
        "summary": _records(summary),
        "recommendation": _recommendation_payload(summary, context),
        "human_context_questions": list(context.human_context_questions),
        "candidate_drivers": list(context.candidate_drivers),
        "autoresearch_hypotheses": list(context.autoresearch_hypotheses),
        "manifest": manifest,
    }
    summary.to_csv(out / "experiment_summary.csv", index=False)
    (out / "experiment_recommendation.md").write_text(recommendation, encoding="utf-8")
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "experiment_llm_context.json").write_text(json.dumps(llm_context, indent=2, default=str) + "\n", encoding="utf-8")
    return ExperimentResult(
        output_dir=out,
        manifest=manifest,
        summary=summary,
        recommendation_markdown=recommendation,
        llm_context=llm_context,
    )


def build_model_leaderboard(run: ForecastRun) -> pd.DataFrame:
    """Build a stable review leaderboard from existing model audit artifacts."""

    audit = build_model_audit(run)
    if audit.empty:
        selection = run.model_selection.copy()
        if selection.empty:
            return pd.DataFrame()
        selection = selection.rename(columns={"selected_model": "model"})
        selection["selected_model"] = selection["model"]
        selection["is_selected_model"] = True
        audit = selection
    pareto = build_model_pareto_frontier(run)
    if not pareto.empty and {"unique_id", "model", "is_pareto_optimal"}.issubset(pareto.columns):
        pareto_cols = ["unique_id", "model", "is_pareto_optimal", "pareto_status", "selection_alignment"]
        audit = audit.merge(pareto[[col for col in pareto_cols if col in pareto.columns]], on=["unique_id", "model"], how="left")
    trust = build_trust_summary(run)
    if not trust.empty and "unique_id" in trust.columns:
        trust_cols = [
            "unique_id",
            "trust_level",
            "trust_score_0_100",
            "horizon_trust_state",
            "full_horizon_claim_allowed",
            "validated_through_horizon",
        ]
        audit = audit.merge(trust[[col for col in trust_cols if col in trust.columns]], on="unique_id", how="left")
    sort_cols = [col for col in ["unique_id", "rmse", "mase", "mae", "wape", "abs_bias"] if col in audit.columns]
    if sort_cols:
        audit = audit.sort_values(sort_cols, na_position="last")
    audit["leaderboard_rank"] = audit.groupby("unique_id").cumcount() + 1
    if "is_pareto_optimal" in audit.columns:
        audit["is_pareto_optimal"] = audit["is_pareto_optimal"].fillna(False).astype(bool)
    columns = [
        "unique_id",
        "leaderboard_rank",
        "model",
        "family",
        "is_selected_model",
        "is_pareto_optimal",
        "pareto_status",
        "selection_alignment",
        "rmse",
        "mae",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "abs_bias",
        "observations",
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
        "trust_level",
        "trust_score_0_100",
        "horizon_trust_state",
        "full_horizon_claim_allowed",
        "validated_through_horizon",
        "selected_model",
        "selection_reason",
    ]
    return audit[[col for col in columns if col in audit.columns]].reset_index(drop=True)


def build_experiment_recommendation(summary: pd.DataFrame, context: ExperimentContext) -> str:
    payload = _recommendation_payload(summary, context)
    lines = [
        "# Experiment recommendation",
        "",
        "This recommendation is advisory. It helps an agent decide what to test next; it does not override any child run's official `forecast.csv` or champion selection.",
        "",
        "## Best current variant",
        "",
        f"- Variant: `{payload['best_variant']}`",
        f"- Rationale: {payload['rationale']}",
        "",
        "## Autoresearch next iteration",
        "",
        f"- Hypothesis: {payload['autoresearch_next_iteration']['hypothesis']}",
        f"- Metric: `{payload['autoresearch_next_iteration']['metric']}` ({payload['autoresearch_next_iteration']['direction']} is better)",
        f"- Budget: {payload['autoresearch_next_iteration']['budget']}",
        f"- Executor: `{payload['autoresearch_next_iteration']['executor']}`",
        f"- Keep rule: {payload['autoresearch_next_iteration']['keep_rule']}",
    ]
    command_hint = payload["autoresearch_next_iteration"].get("command_hint")
    if command_hint:
        lines.extend(["", "Suggested command seed:", "", f"```powershell\n{command_hint}\n```"])
    lines.extend(["", "## Context to collect from human", ""])
    lines.extend(f"- {question}" for question in payload["human_context_questions"])
    lines.extend(["", "## Automatically detected candidate drivers", ""])
    if payload["candidate_drivers"]:
        lines.append(_candidate_driver_markdown(payload["candidate_drivers"]))
    else:
        lines.append("No undeclared numeric driver-like columns were detected in the input.")
    lines.extend(
        [
            "",
            "Detected candidates are not trained automatically. Declare one candidate at a time as an audited regressor and keep it only if the fixed experiment metric improves.",
            "",
            "## Candidate next hypotheses",
            "",
        ]
    )
    if payload["autoresearch_hypotheses"]:
        for hypothesis_row in payload["autoresearch_hypotheses"]:
            lines.append(f"- `{hypothesis_row['id']}`: {hypothesis_row['hypothesis']}")
    else:
        lines.append("- No automatic driver hypothesis was generated; collect context first, then test one new event, hierarchy, or driver assumption.")
    lines.extend(["", "## Variant summary", ""])
    if summary.empty:
        lines.append("No variants were evaluated.")
    else:
        display_cols = [
            "variant",
            "status",
            "advisory_rank",
            "avg_rmse",
            "avg_mae",
            "pareto_selected_fraction",
            "avg_trust_score_0_100",
            "selected_models",
            "reason",
        ]
        lines.append(_frame_to_markdown(summary[[col for col in display_cols if col in summary.columns]]))
    lines.append("")
    return "\n".join(lines)


def _recommendation_payload(summary: pd.DataFrame, context: ExperimentContext) -> dict[str, Any]:
    success = summary[summary["status"].eq("success")].copy() if not summary.empty and "status" in summary.columns else pd.DataFrame()
    top_candidate = context.candidate_drivers[0] if context.candidate_drivers else None
    command_hint = top_candidate.get("experiment_command_seed") if top_candidate else None
    if success.empty:
        best_variant = "none"
        rationale = "No variant completed successfully; inspect failed/skipped reasons and fix the smallest blocker first."
        hypothesis = "Fix the first failed or skipped variant, then rerun the same bounded experiment."
    else:
        success = success.sort_values(["advisory_rank", "avg_rmse", "avg_mae"], na_position="last")
        best = success.iloc[0].to_dict()
        best_variant = str(best.get("variant", "unknown"))
        rationale = (
            f"`{best_variant}` had the strongest advisory rank with avg_rmse={_fmt(best.get('avg_rmse'))}, "
            f"avg_mae={_fmt(best.get('avg_mae'))}, pareto_selected_fraction={_fmt(best.get('pareto_selected_fraction'))}, "
            f"and avg_trust_score_0_100={_fmt(best.get('avg_trust_score_0_100'))}."
        )
        if context.has_events and best_variant != "events":
            hypothesis = "Test whether narrowing or resizing declared event assumptions improves rolling-origin RMSE versus the current best variant."
        elif context.has_regressors and best_variant != "known_future_regressors":
            hypothesis = "Test one audited known-future regressor family at a time and keep it only if rolling-origin RMSE improves."
        elif context.has_hierarchy and best_variant != "hierarchy_methods":
            hypothesis = "Test whether reconciliation improves planning coherence without materially degrading node-level backtest RMSE."
        elif top_candidate and not context.has_regressors:
            hypothesis = (
                f"Declare detected column `{top_candidate['value_col']}` as a historical-only model_candidate regressor, "
                "then rerun baseline versus known_future_regressors with a fixed metric budget."
            )
        else:
            hypothesis = "Add one business-plausible known-future driver or event and rerun the bounded experiment against the current best variant."
    autoresearch_next_iteration: dict[str, Any] = {
        "hypothesis": hypothesis,
        "metric": "avg_rmse primary, with avg_mae and RMSE/MAE Pareto status as secondary review",
        "direction": "lower avg_rmse",
        "budget": "one scaffold experiment variant plus generated child run artifacts",
        "executor": "uv run nixtla-scaffold experiment",
        "keep_rule": "Keep only if RMSE improves, or RMSE ties while MAE/Pareto review is no worse and no new trust/horizon caveats appear. Treat WAPE as diagnostic context only.",
    }
    if command_hint:
        autoresearch_next_iteration["command_hint"] = command_hint
    return {
        "best_variant": best_variant,
        "rationale": rationale,
        "autoresearch_next_iteration": autoresearch_next_iteration,
        "human_context_questions": list(context.human_context_questions),
        "candidate_drivers": list(context.candidate_drivers),
        "autoresearch_hypotheses": list(context.autoresearch_hypotheses),
    }


def _successful_row(name: str, description: str, path: Path, run: ForecastRun) -> dict[str, Any]:
    selection = run.model_selection.copy()
    trust = build_trust_summary(run)
    leaderboard = build_model_leaderboard(run)
    selected = leaderboard[leaderboard["is_selected_model"].eq(True)] if not leaderboard.empty and "is_selected_model" in leaderboard.columns else selection
    avg_rmse = _mean(selection, "rmse")
    avg_mae = _mean(selection, "mae")
    avg_wape = _mean(selection, "wape")
    avg_trust = _mean(trust, "trust_score_0_100")
    full_horizon_count = (
        int(trust["full_horizon_claim_allowed"].fillna(False).astype(bool).sum())
        if not trust.empty and "full_horizon_claim_allowed" in trust.columns
        else 0
    )
    pareto_fraction = (
        float(selected["is_pareto_optimal"].fillna(False).astype(bool).mean())
        if not selected.empty and "is_pareto_optimal" in selected.columns
        else None
    )
    rank_score = _rank_score(avg_rmse, avg_mae, avg_trust)
    return {
        "variant": name,
        "description": description,
        "status": "success",
        "reason": "",
        "run_path": str(path),
        "selected_models": ", ".join(f"{model}:{count}" for model, count in _selected_model_counts(run).items()),
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_wape": avg_wape,
        "avg_mase": _mean(selection, "mase"),
        "avg_rmsse": _mean(selection, "rmsse"),
        "avg_bias": _mean(selection, "bias"),
        "avg_trust_score_0_100": avg_trust,
        "full_horizon_claim_allowed_series": full_horizon_count,
        "pareto_selected_fraction": pareto_fraction,
        "warning_count": len(run.warnings),
        "advisory_rank_score": rank_score,
        "advisory_rank": None,
    }


def _skipped_row(
    name: str,
    status: str,
    reason: str,
    *,
    description: str = "",
    run_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "variant": name,
        "description": description,
        "status": status,
        "reason": reason,
        "run_path": str(run_path) if run_path is not None else "",
        "selected_models": "",
        "avg_rmse": None,
        "avg_mae": None,
        "avg_wape": None,
        "avg_mase": None,
        "avg_rmsse": None,
        "avg_bias": None,
        "avg_trust_score_0_100": None,
        "full_horizon_claim_allowed_series": None,
        "pareto_selected_fraction": None,
        "warning_count": None,
        "advisory_rank_score": None,
        "advisory_rank": None,
    }


def _rank_score(avg_rmse: float | None, avg_mae: float | None, avg_trust: float | None) -> float | None:
    if avg_rmse is None:
        return None
    score = float(avg_rmse)
    if avg_mae is not None:
        score += float(avg_mae) * 0.000001
    if avg_trust is not None:
        score -= float(avg_trust) * 0.000000001
    return score


def _candidate_driver_markdown(candidates: Sequence[dict[str, Any]]) -> str:
    rows = [
        {
            "value_col": candidate.get("value_col"),
            "suggested_availability": candidate.get("suggested_availability"),
            "score": _fmt(candidate.get("score")),
            "coverage": _fmt(candidate.get("non_null_fraction")),
            "best_lag": candidate.get("best_lag"),
            "timing": candidate.get("relationship_timing"),
            "why": candidate.get("why"),
        }
        for candidate in candidates[:5]
    ]
    return _frame_to_markdown(pd.DataFrame(rows))


def _detect_candidate_drivers(frame: pd.DataFrame, spec: ForecastSpec, *, limit: int = 5) -> tuple[dict[str, Any], ...]:
    if frame.empty:
        return ()
    excluded = _excluded_driver_columns(spec)
    target = pd.to_numeric(frame["y"], errors="coerce") if "y" in frame.columns else pd.Series(dtype="float64")
    target_tokens = _column_tokens(spec.target_col)
    candidates: list[dict[str, Any]] = []
    for column in frame.columns:
        column_name = str(column)
        if column_name in excluded or column_name.lower() in excluded:
            continue
        name_tokens = _column_tokens(column_name)
        if name_tokens & _AUDIT_OR_TARGET_LEAKAGE_TERMS:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = int(numeric.notna().sum())
        if non_null < max(6, int(len(frame) * 0.25)):
            continue
        unique_count = int(numeric.dropna().nunique())
        if unique_count <= 1:
            continue
        name_score, matched_terms = _driver_name_score(column_name, target_tokens)
        if unique_count <= 3 and name_score < 3:
            continue
        corr_abs = _panel_abs_correlation(frame, column_name)
        lag_evidence = _lagged_correlation_evidence(frame, column_name, same_period_correlation_abs=corr_abs)
        coverage = non_null / max(len(frame), 1)
        score = name_score + min(2.0, 2.0 * (corr_abs or 0.0)) + min(1.0, coverage)
        if lag_evidence.get("relationship_timing") in {"driver_leads_target", "target_leads_driver"}:
            score += 0.5
        if name_tokens & _CALENDAR_OR_METADATA_TERMS and not (name_tokens & target_tokens):
            score -= 1.0
        if score < 2.5:
            continue
        regressor_payload = {
            "name": column_name,
            "value_col": column_name,
            "availability": "historical_only",
            "mode": "model_candidate",
            "known_as_of_col": "known_as_of",
        }
        why = _candidate_driver_why(column_name, matched_terms, target_tokens, corr_abs, lag_evidence=lag_evidence)
        candidates.append(
            {
                "name": column_name,
                "value_col": column_name,
                "role": "detected_historical_input_column",
                "suggested_availability": "historical_only",
                "suggested_mode": "model_candidate",
                "score": round(float(score), 4),
                "non_null_fraction": round(float(coverage), 4),
                "unique_values": unique_count,
                "target_correlation_abs": round(float(corr_abs), 4) if corr_abs is not None else None,
                "same_period_correlation_abs": round(float(corr_abs), 4) if corr_abs is not None else None,
                "best_lag": lag_evidence.get("best_lag"),
                "best_lag_abs_correlation": lag_evidence.get("best_lag_abs_correlation"),
                "best_lag_paired_observations": lag_evidence.get("best_lag_paired_observations"),
                "lag_search_window": lag_evidence.get("lag_search_window"),
                "relationship_timing": lag_evidence.get("relationship_timing"),
                "lag_interpretation": lag_evidence.get("lag_interpretation"),
                "matched_name_terms": matched_terms,
                "why": why,
                "regressor_json": regressor_payload,
                "experiment_command_seed": _experiment_command_seed(regressor_payload),
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["value_col"])))
    return tuple(candidates[:limit])


def _human_context_questions(
    candidates: Sequence[dict[str, Any]],
    *,
    has_events: bool,
    has_hierarchy: bool,
    spec: ForecastSpec,
) -> tuple[str, ...]:
    questions = [
        "What decision, audience, and tolerance threshold will use this forecast?",
        f"What exactly does `{spec.target_col}` measure, including unit, grain, inclusion/exclusion rules, and known accounting or telemetry changes?",
        "Which business events, launches, pricing changes, contract changes, outages, or policy shifts could make the future differ from the past?",
        "What refresh cadence matters, and should refresh rerun model selection or intentionally pin a prior champion for governance?",
    ]
    if candidates:
        cols = ", ".join(f"`{candidate['value_col']}`" for candidate in candidates[:3])
        questions.extend(
            [
                f"Were detected candidate driver columns ({cols}) known at the forecast origin, or do any contain hindsight/leakage?",
                f"Can future values or scenario paths be supplied for any candidate driver, or should they be tested only as historical lag features?",
                "What lag direction and refresh latency should be assumed between the metric and each candidate driver?",
            ]
        )
    else:
        questions.append("Are there upstream usage, seat, spend, pipeline, plan, or pricing tables the agent should search for candidate regressors?")
    if not has_events:
        questions.append("Are there known one-time or recurring events that should be represented as explicit scenario overlays?")
    if not has_hierarchy:
        questions.append("Are there product, segment, region, customer, or parent/child rollups that must reconcile for planning?")
    return tuple(dict.fromkeys(questions))


def _autoresearch_hypotheses(candidates: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    hypotheses: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:3], start=1):
        value_col = str(candidate["value_col"])
        hypotheses.append(
            {
                "id": f"candidate-driver-{index}-{_slug(value_col)}",
                "category": "candidate_driver",
                "hypothesis": (
                    f"`{value_col}` contains useful leading or trailing signal; declaring it as a historical-only "
                    "MLForecast model_candidate will improve rolling-origin RMSE without adding trust caveats."
                ),
                "test_plan": "Run one bounded experiment comparing baseline against known_future_regressors with only this candidate declared.",
                "metric": "avg_rmse primary; review avg_mae, RMSE/MAE Pareto status, trust_score, and driver_model_cv_delta as guardrails",
                "keep_rule": "Keep only if RMSE improves, or RMSE ties while MAE/Pareto review is no worse and no new leakage/future-coverage warnings appear. Do not optimize on WAPE.",
                "command_seed": candidate["experiment_command_seed"],
            }
        )
    return tuple(hypotheses)


def _excluded_driver_columns(spec: ForecastSpec) -> set[str]:
    columns = {
        "unique_id",
        "ds",
        "y",
        spec.id_col,
        spec.time_col,
        spec.target_col,
        "known_as_of",
        *_HIERARCHY_COLUMNS,
    }
    for regressor in spec.regressors:
        if regressor.value_col:
            columns.add(regressor.value_col)
        columns.add(regressor.known_as_of_col)
    lowered = {column.lower() for column in columns}
    return columns | lowered


def _column_tokens(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)}
    return tokens | {token[:-1] for token in tokens if len(token) > 3 and token.endswith("s")}


def _driver_name_score(column: str, target_tokens: set[str]) -> tuple[float, list[str]]:
    lower = str(column).lower()
    tokens = _column_tokens(column)
    matched = sorted({term for term in _DRIVER_NAME_TERMS if term in lower or term in tokens})
    score = 0.0
    for term in matched:
        score += 2.0 if term in {"rolling", "trailing", "lag", "lags"} else 1.0
    overlap = sorted(tokens & target_tokens)
    if overlap and overlap != ["y"]:
        score += 3.0
        matched.extend(f"target:{token}" for token in overlap)
    if tokens & _CALENDAR_OR_METADATA_TERMS and not overlap:
        score -= 1.0
    return score, list(dict.fromkeys(matched))


def _abs_correlation(values: pd.Series, target: pd.Series) -> float | None:
    if target.empty:
        return None
    paired = pd.DataFrame({"x": values, "y": target}).dropna()
    if len(paired) < 4 or paired["x"].nunique() <= 1 or paired["y"].nunique() <= 1:
        return None
    corr = paired["x"].corr(paired["y"])
    return None if pd.isna(corr) else abs(float(corr))


def _panel_abs_correlation(frame: pd.DataFrame, value_col: str, *, target_col: str = "y", min_pairs: int = 4) -> float | None:
    if target_col not in frame.columns or value_col not in frame.columns:
        return None
    if "unique_id" not in frame.columns:
        return _abs_correlation(pd.to_numeric(frame[value_col], errors="coerce"), pd.to_numeric(frame[target_col], errors="coerce"))
    total_pairs = 0
    weighted_abs_corr = 0.0
    for _, group in frame.groupby("unique_id", sort=False):
        paired = pd.DataFrame(
            {
                "x": pd.to_numeric(group[value_col], errors="coerce"),
                "y": pd.to_numeric(group[target_col], errors="coerce"),
            }
        ).dropna()
        if len(paired) < min_pairs or paired["x"].nunique() <= 1 or paired["y"].nunique() <= 1:
            continue
        corr = paired["x"].corr(paired["y"])
        if pd.isna(corr):
            continue
        total_pairs += int(len(paired))
        weighted_abs_corr += abs(float(corr)) * float(len(paired))
    if total_pairs <= 0:
        return None
    return weighted_abs_corr / float(total_pairs)


def _lagged_correlation_evidence(
    frame: pd.DataFrame,
    value_col: str,
    *,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    max_lag: int = 3,
    min_pairs: int = 4,
    same_period_correlation_abs: float | None = None,
) -> dict[str, Any]:
    search_window = f"-{max_lag}..{max_lag}"
    empty = {
        "best_lag": None,
        "best_lag_abs_correlation": None,
        "best_lag_paired_observations": 0,
        "lag_search_window": search_window,
        "relationship_timing": "insufficient_lag_evidence",
        "lag_interpretation": (
            "Lag scan needs enough within-series paired observations; positive best_lag means the driver leads the target."
        ),
    }
    if frame.empty or value_col not in frame.columns or target_col not in frame.columns:
        return empty

    best_lag: int | None = None
    best_corr: float | None = None
    best_pairs = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        corr, pairs = _panel_lag_abs_correlation(
            frame,
            value_col,
            target_col=target_col,
            id_col=id_col,
            time_col=time_col,
            lag=lag,
            min_pairs=min_pairs,
        )
        if corr is None:
            continue
        if best_corr is None or corr > best_corr or (math.isclose(corr, best_corr) and abs(lag) < abs(best_lag or lag)):
            best_lag = lag
            best_corr = corr
            best_pairs = pairs
    if best_lag is None or best_corr is None:
        return empty

    improvement = best_corr - same_period_correlation_abs if same_period_correlation_abs is not None else None
    if improvement is None or improvement >= 0.05:
        timing = "driver_leads_target" if best_lag > 0 else "target_leads_driver"
    else:
        timing = "same_period_or_unclear"
    if best_lag > 0:
        interpretation = f"`{value_col}` leads `{target_col}` by {best_lag} period(s) in the strongest within-series lag screen."
    else:
        interpretation = f"`{target_col}` leads `{value_col}` by {abs(best_lag)} period(s) in the strongest within-series lag screen."
    if timing == "same_period_or_unclear":
        interpretation += " The lagged relationship does not clear the improvement margin over same-period correlation."
    return {
        "best_lag": int(best_lag),
        "best_lag_abs_correlation": round(float(best_corr), 4),
        "best_lag_paired_observations": int(best_pairs),
        "lag_search_window": search_window,
        "relationship_timing": timing,
        "lag_interpretation": interpretation,
    }


def _panel_lag_abs_correlation(
    frame: pd.DataFrame,
    value_col: str,
    *,
    target_col: str,
    id_col: str,
    time_col: str,
    lag: int,
    min_pairs: int,
) -> tuple[float | None, int]:
    group_keys = frame[id_col] if id_col in frame.columns else pd.Series(["__all__"] * len(frame), index=frame.index)
    total_pairs = 0
    weighted_abs_corr = 0.0
    work = frame.copy()
    work["_driver_group_key"] = group_keys.astype(str)
    for _, group in work.groupby("_driver_group_key", sort=False):
        if time_col in group.columns:
            group = group.sort_values(time_col)
        x = pd.to_numeric(group[value_col], errors="coerce")
        y = pd.to_numeric(group[target_col], errors="coerce").shift(-lag)
        paired = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(paired) < min_pairs or paired["x"].nunique() <= 1 or paired["y"].nunique() <= 1:
            continue
        corr = paired["x"].corr(paired["y"])
        if pd.isna(corr):
            continue
        total_pairs += int(len(paired))
        weighted_abs_corr += abs(float(corr)) * float(len(paired))
    if total_pairs <= 0:
        return None, 0
    return weighted_abs_corr / float(total_pairs), total_pairs


def _candidate_driver_why(
    column: str,
    matched_terms: Sequence[str],
    target_tokens: set[str],
    corr_abs: float | None,
    *,
    lag_evidence: dict[str, Any] | None = None,
) -> str:
    reasons: list[str] = []
    visible_terms = [term for term in matched_terms if not term.startswith("target:")]
    target_matches = [term.split(":", 1)[1] for term in matched_terms if term.startswith("target:")]
    if visible_terms:
        reasons.append(f"name suggests driver signal ({', '.join(visible_terms[:4])})")
    if target_matches:
        reasons.append(f"name overlaps target tokens ({', '.join(target_matches)})")
    if corr_abs is not None:
        reasons.append(f"absolute in-sample correlation to target is {corr_abs:.2f}")
    if lag_evidence and lag_evidence.get("relationship_timing") in {"driver_leads_target", "target_leads_driver"}:
        reasons.append(str(lag_evidence.get("lag_interpretation")))
    if not reasons:
        reasons.append(f"`{column}` is numeric and sufficiently populated")
    return "; ".join(reasons)


def _experiment_command_seed(regressor_payload: dict[str, Any]) -> str:
    regressor_json = json.dumps(regressor_payload, separators=(",", ":"))
    return (
        "uv run nixtla-scaffold experiment --input <data.csv> --preset standard --horizon <horizon> "
        "--model-policy mlforecast --train-known-future-regressors "
        f"--regressor '{regressor_json}' --variants baseline known_future_regressors --max-variants 2 --output runs\\experiment_driver_test"
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "driver"


def _variant_catalog() -> dict[str, ExperimentVariant]:
    always = lambda context, spec: (True, "")
    return {
        "baseline": ExperimentVariant(
            name="baseline",
            description="Current scaffold behavior with driver/event inputs removed for a clean statistical baseline.",
            build_spec=lambda spec: replace(spec, events=(), regressors=(), train_known_future_regressors=False, hierarchy_reconciliation="none"),
            applicable=always,
        ),
        "events": ExperimentVariant(
            name="events",
            description="Declared DriverEvent scenario overlays only.",
            build_spec=lambda spec: replace(spec, regressors=(), train_known_future_regressors=False),
            applicable=lambda context, spec: (
                context.has_events,
                "no events were declared; add --event or --event-file before testing event overlays",
            ),
        ),
        "known_future_regressors": ExperimentVariant(
            name="known_future_regressors",
            description="Opt-in MLForecast training with declared model_candidate regressors that pass leakage/future-coverage gates.",
            build_spec=lambda spec: replace(spec, events=(), train_known_future_regressors=True, model_policy="all" if spec.model_policy == "baseline" else spec.model_policy),
            applicable=lambda context, spec: (
                context.has_regressors,
                "no known-future regressors were declared; add --regressor or --regressor-file with future values/scenarios",
            ),
        ),
        "rolling_features": ExperimentVariant(
            name="rolling_features",
            description="Curated MLForecast rolling target feature policy.",
            build_spec=lambda spec: replace(spec, mlforecast_feature_policy="rolling", model_policy="all" if spec.model_policy == "baseline" else spec.model_policy),
            applicable=always,
        ),
        "all_models": ExperimentVariant(
            name="all_models",
            description="Broader explicit model policy; still uses the existing scaffold tournament and selection gates.",
            build_spec=lambda spec: replace(spec, model_policy="all"),
            applicable=always,
        ),
        "hierarchy_methods": ExperimentVariant(
            name="hierarchy_methods",
            description="Hierarchy reconciliation comparison via the existing both-mode bottom-up/top-down evidence path.",
            build_spec=lambda spec: replace(spec, hierarchy_reconciliation="both"),
            applicable=lambda context, spec: (
                context.has_hierarchy,
                "no hierarchy metadata was found; build hierarchy nodes first or include hierarchy columns",
            ),
        ),
    }


def _resolve_variants(variants: Sequence[str] | None, context: ExperimentContext) -> list[str]:
    if variants:
        names = []
        for value in variants:
            names.extend(part.strip() for part in str(value).replace(",", " ").split() if part.strip())
        if "all" in names:
            return list(EXPERIMENT_VARIANTS)
        return list(dict.fromkeys(names))
    names = ["baseline", "all_models"]
    if context.has_events:
        names.append("events")
    if context.has_regressors:
        names.append("known_future_regressors")
    if context.has_hierarchy:
        names.append("hierarchy_methods")
    return list(dict.fromkeys(names))


def _experiment_context(data: str | Path | pd.DataFrame, spec: ForecastSpec, *, sheet: str | int | None) -> ExperimentContext:
    frame = load_forecast_dataset(data, sheet=sheet, spec=spec)
    has_hierarchy = bool(_HIERARCHY_COLUMNS.intersection(frame.columns))
    candidate_drivers = _detect_candidate_drivers(frame, spec)
    return ExperimentContext(
        has_events=bool(spec.events),
        has_regressors=bool(spec.regressors),
        has_hierarchy=has_hierarchy,
        row_count=int(len(frame)),
        series_count=int(frame["unique_id"].nunique()) if "unique_id" in frame.columns else 0,
        source_hash_sha256=_source_hash(data, frame),
        candidate_drivers=candidate_drivers,
        human_context_questions=_human_context_questions(
            candidate_drivers,
            has_events=bool(spec.events),
            has_hierarchy=has_hierarchy,
            spec=spec,
        ),
        autoresearch_hypotheses=_autoresearch_hypotheses(candidate_drivers),
    )


def _source_hash(data: str | Path | pd.DataFrame, frame: pd.DataFrame) -> str | None:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists() and path.is_file():
            digest = sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        return sha256(hashed).hexdigest()
    return None


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def _selected_model_counts(run: ForecastRun) -> dict[str, int]:
    if run.model_selection.empty or "selected_model" not in run.model_selection.columns:
        return {}
    return {str(key): int(value) for key, value in run.model_selection["selected_model"].value_counts().sort_index().items()}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _assign_advisory_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "advisory_rank_score" not in summary.columns:
        return summary
    success_mask = summary["status"].eq("success")
    sort_cols = [col for col in ["avg_rmse", "avg_mae", "advisory_rank_score"] if col in summary.columns]
    ranked = summary.loc[success_mask].sort_values(sort_cols, na_position="last")
    summary.loc[ranked.index, "advisory_rank"] = range(1, len(ranked) + 1)
    return summary


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.where(pd.notna(frame), "")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)

