from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nixtla_scaffold.external import (
    ExternalForecastFormat,
    build_external_forecast_metadata,
    canonicalize_external_forecasts,
    load_external_forecasts,
)


FORECAST_COMPARISON_SCHEMA_VERSION = "nixtla_scaffold.forecast_comparison.v1"

FORECAST_COMPARISON_OUTPUT = "forecast_comparison.csv"
FORECAST_COMPARISON_SUMMARY_OUTPUT = "forecast_comparison_summary.csv"
FORECAST_COMPARISON_MANIFEST_OUTPUT = "comparison_manifest.json"
FORECAST_COMPARISON_WORKBOOK_OUTPUT = "forecast_comparison.xlsx"
FORECAST_COMPARISON_REPORT_OUTPUT = "comparison_report.html"
FORECAST_COMPARISON_LLM_CONTEXT_OUTPUT = "comparison_llm_context.json"

_DIRECTIONAL_WARNING = (
    "Directional triangulation only. External forecasts are not actuals or scaffold model candidates. "
    "Deltas are context, not prediction intervals, residuals, or accuracy/backtest metrics."
)


@dataclass(frozen=True)
class ForecastComparisonResult:
    """Directional comparison artifacts for a scaffold forecast and external forecasts."""

    comparison: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, Any]
    external_forecasts: pd.DataFrame

    def to_directory(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.comparison.to_csv(out / FORECAST_COMPARISON_OUTPUT, index=False)
        self.summary.to_csv(out / FORECAST_COMPARISON_SUMMARY_OUTPUT, index=False)
        _write_comparison_workbook(out / FORECAST_COMPARISON_WORKBOOK_OUTPUT, self)
        (out / FORECAST_COMPARISON_REPORT_OUTPUT).write_text(build_forecast_comparison_html(self), encoding="utf-8")
        (out / FORECAST_COMPARISON_LLM_CONTEXT_OUTPUT).write_text(
            json.dumps(build_forecast_comparison_llm_context(self), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (out / FORECAST_COMPARISON_MANIFEST_OUTPUT).write_text(json.dumps(self.manifest, indent=2, default=str) + "\n", encoding="utf-8")
        return out


def compare_forecasts(
    run_dir: str | Path,
    external_forecasts: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None = None,
    external_format: ExternalForecastFormat = "auto",
    external_model_name: str | None = None,
    external_source_id: str | None = None,
    external_id_col: str = "unique_id",
    external_time_col: str = "ds",
    external_value_col: str = "yhat",
    external_model_col: str = "model",
    scaffold_model: str | None = None,
) -> ForecastComparisonResult:
    """Compare selected scaffold forecast rows to external forecasts.

    This is a directional triangulation workflow only. It does not score
    external forecasts against actuals or modify the original forecast run.
    """

    run_path = Path(run_dir)
    scaffold = load_scaffold_forecast_for_comparison(run_path, scaffold_model=scaffold_model)
    external = _load_external_for_comparison(
        external_forecasts,
        sheet=sheet,
        external_format=external_format,
        external_model_name=external_model_name,
        external_source_id=external_source_id,
        external_id_col=external_id_col,
        external_time_col=external_time_col,
        external_value_col=external_value_col,
        external_model_col=external_model_col,
    )
    run_metadata = _load_run_metadata(run_path)
    comparison = build_forecast_comparison(scaffold, external, run_metadata=run_metadata)
    summary = build_forecast_comparison_summary(comparison)
    manifest = build_forecast_comparison_manifest(
        run_path,
        external_forecasts,
        comparison,
        summary,
        external,
        run_metadata=run_metadata,
        sheet=sheet,
        external_format=external_format,
        external_model_name=external_model_name,
        external_source_id=external_source_id,
        scaffold_model=scaffold_model,
    )
    return ForecastComparisonResult(comparison=comparison, summary=summary, manifest=manifest, external_forecasts=external)


def write_forecast_comparison(
    run_dir: str | Path,
    external_forecasts: str | Path | pd.DataFrame,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> ForecastComparisonResult:
    """Run a directional comparison and write comparison artifacts."""

    result = compare_forecasts(run_dir, external_forecasts, **kwargs)
    out = Path(output_dir) if output_dir is not None else Path(run_dir) / "comparison"
    manifest = dict(result.manifest)
    manifest["output_dir"] = str(out)
    result = ForecastComparisonResult(
        comparison=result.comparison,
        summary=result.summary,
        manifest=manifest,
        external_forecasts=result.external_forecasts,
    )
    result.to_directory(out)
    return result


def load_scaffold_forecast_for_comparison(run_dir: str | Path, *, scaffold_model: str | None = None) -> pd.DataFrame:
    """Load selected scaffold forecast rows from a run directory."""

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    if scaffold_model is None:
        path = run_path / "forecast.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
    else:
        path = run_path / "forecast_long.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            raise ValueError("forecast_long.csv must include a model column when scaffold_model is provided")
        frame = frame[frame["model"].astype(str) == str(scaffold_model)].copy()
        if frame.empty:
            raise ValueError(f"scaffold model '{scaffold_model}' was not found in forecast_long.csv")

    required = {"unique_id", "ds", "yhat", "model"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scaffold forecast is missing required comparison column(s): {missing}")
    out = frame.copy()
    out["unique_id"] = _clean_text(out["unique_id"], "unique_id", frame_name="scaffold forecast")
    out["scaffold_model"] = _clean_text(out["model"], "model", frame_name="scaffold forecast")
    out["ds"] = _coerce_datetime(out["ds"], "ds", frame_name="scaffold forecast")
    out["scaffold_yhat"] = _coerce_numeric(out["yhat"], "yhat", frame_name="scaffold forecast")
    duplicates = out.duplicated(["unique_id", "ds"], keep=False)
    if duplicates.any():
        sample = out.loc[duplicates, ["unique_id", "ds", "scaffold_model"]].head(5).to_dict("records")
        raise ValueError(f"scaffold forecast has duplicate unique_id/ds rows for comparison: {sample}")

    keep = [
        "unique_id",
        "ds",
        "scaffold_model",
        "scaffold_yhat",
        "horizon_step",
        "row_horizon_status",
        "horizon_trust_state",
        "planning_eligible",
        "planning_eligibility_scope",
        "interval_status",
    ]
    return out[[column for column in keep if column in out.columns]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_forecast_comparison(
    scaffold: pd.DataFrame,
    external: pd.DataFrame,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Outer-align scaffold and external forecasts and compute directional deltas."""

    scaffold_ready = _prepare_scaffold_frame(scaffold)
    external_ready = _prepare_external_frame(external)
    merged = scaffold_ready.merge(external_ready, on=["unique_id", "ds"], how="outer")
    if merged.empty:
        raise ValueError("forecast comparison has no rows after alignment")

    has_scaffold = merged["scaffold_yhat"].notna()
    has_external = merged["external_yhat"].notna()
    origin = _forecast_origin(run_metadata or {})
    unknown_cutoff_origin = has_scaffold & has_external & _unknown_cutoff_origin(merged, origin)
    cutoff_mismatch = has_scaffold & has_external & _cutoff_mismatch(merged, origin)
    merged["comparison_status"] = np.select(
        [
            has_scaffold & has_external & unknown_cutoff_origin,
            has_scaffold & has_external & cutoff_mismatch,
            has_scaffold & has_external,
            has_scaffold & ~has_external,
            ~has_scaffold & has_external,
        ],
        ["unknown_cutoff_origin", "mismatch_cutoff", "aligned", "scaffold_only", "external_only"],
        default="unusable",
    )

    comparable = merged["comparison_status"].eq("aligned")
    merged["comparison_delta_yhat"] = np.where(
        comparable,
        merged["external_yhat"] - merged["scaffold_yhat"],
        np.nan,
    )
    merged["comparison_delta_pct"] = _safe_pct_delta(merged["comparison_delta_yhat"], merged["scaffold_yhat"], comparable)
    merged["external_is_actual"] = False
    merged["is_comparison_scoreable"] = False
    merged["comparison_scope"] = "directional_unscored"
    merged["comparison_warning"] = [
        _comparison_warning(status, status_message, origin)
        for status, status_message in zip(merged["comparison_status"], merged.get("external_status_message", pd.Series([None] * len(merged))), strict=False)
    ]
    return _order_comparison_columns(merged).sort_values(_comparison_sort_columns(merged)).reset_index(drop=True)


def build_forecast_comparison_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize directional comparison coverage and deltas by external model/source/scenario/status."""

    if comparison.empty:
        return pd.DataFrame(columns=_summary_columns())
    frame = comparison.copy()
    for column in ["external_model", "external_source_id", "scenario_name", "comparison_evidence_status"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    group_cols = ["external_model", "external_source_id", "scenario_name", "comparison_evidence_status"]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        status_counts = group["comparison_status"].value_counts().to_dict()
        aligned = group[group["comparison_status"].eq("aligned")]
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=True))
        rows.append(
            {
                **key_map,
                "row_count": int(len(group)),
                "aligned_rows": int(len(aligned)),
                "rows_with_both_forecasts": int(group["scaffold_yhat"].notna().mul(group["external_yhat"].notna()).sum()),
                "scaffold_only_rows": int(status_counts.get("scaffold_only", 0)),
                "external_only_rows": int(status_counts.get("external_only", 0)),
                "mismatch_cutoff_rows": int(status_counts.get("mismatch_cutoff", 0)),
                "unknown_cutoff_origin_rows": int(status_counts.get("unknown_cutoff_origin", 0)),
                "series_count": int(group["unique_id"].astype(str).nunique()),
                "start": _date_or_none(group["ds"], "min"),
                "end": _date_or_none(group["ds"], "max"),
                "avg_delta_yhat_vs_scaffold": _mean_or_none(aligned["comparison_delta_yhat"]),
                "avg_abs_delta_yhat_vs_scaffold": _mean_or_none(aligned["comparison_delta_yhat"].abs()),
                "avg_pct_delta_vs_scaffold": _mean_or_none(aligned["comparison_delta_pct"]),
                "avg_abs_pct_delta_vs_scaffold": _mean_or_none(aligned["comparison_delta_pct"].abs()),
                "metric_definition": "Averages include only comparison_status='aligned' rows; cutoff warnings are count-only diagnostics.",
                "is_comparison_scoreable": False,
                "warning": _DIRECTIONAL_WARNING,
            }
        )
    return pd.DataFrame(rows, columns=_summary_columns())


def build_forecast_comparison_manifest(
    run_dir: str | Path,
    external_source: str | Path | pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    external: pd.DataFrame,
    *,
    run_metadata: dict[str, Any] | None = None,
    sheet: str | int | None = None,
    external_format: ExternalForecastFormat = "auto",
    external_model_name: str | None = None,
    external_source_id: str | None = None,
    scaffold_model: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable comparison manifest."""

    run_meta = run_metadata or {}
    forecast_origin = _forecast_origin(run_meta)
    status_counts = _value_counts(comparison, "comparison_status")
    aligned_rows = int(status_counts.get("aligned", 0))
    mismatch_cutoff_rows = int(status_counts.get("mismatch_cutoff", 0))
    unknown_cutoff_origin_rows = int(status_counts.get("unknown_cutoff_origin", 0))
    rows_with_both_forecasts = int(
        comparison["scaffold_yhat"].notna().mul(comparison["external_yhat"].notna()).sum()
    )
    scaffold_unique_rows = int(comparison.loc[comparison["scaffold_yhat"].notna(), ["unique_id", "ds"]].drop_duplicates().shape[0])
    external_key_cols = [
        column
        for column in ["unique_id", "ds", "external_model", "external_source_id", "scenario_name", "external_cutoff"]
        if column in comparison.columns
    ]
    external_unique_rows = int(
        comparison.loc[comparison["external_yhat"].notna(), external_key_cols]
        .drop_duplicates()
        .shape[0]
    )
    return {
        "schema_version": FORECAST_COMPARISON_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": "directional_unscored",
        "inputs": {
            "scaffold_run_dir": str(Path(run_dir)),
            "scaffold_forecast_file": "forecast.csv" if scaffold_model is None else "forecast_long.csv",
            "scaffold_model": scaffold_model or "selected",
            "scaffold_forecast_origin": forecast_origin.date().isoformat() if forecast_origin is not None else None,
            "scaffold_forecast_origin_status": _forecast_origin_status(comparison, forecast_origin),
            "external_source": str(external_source) if not isinstance(external_source, pd.DataFrame) else "dataframe",
            "external_sheet": sheet,
            "external_format": external_format,
            "external_model_name": external_model_name,
            "external_source_id": external_source_id,
        },
        "alignment": {
            "status": _alignment_status(aligned_rows, mismatch_cutoff_rows, unknown_cutoff_origin_rows),
            "rows_scaffold": scaffold_unique_rows,
            "rows_external": external_unique_rows,
            "rows_total": int(len(comparison)),
            "rows_aligned": aligned_rows,
            "rows_comparable": aligned_rows,
            "rows_with_both_forecasts": rows_with_both_forecasts,
            "scaffold_only_rows": int(status_counts.get("scaffold_only", 0)),
            "external_only_rows": int(status_counts.get("external_only", 0)),
            "mismatch_cutoff_rows": mismatch_cutoff_rows,
            "unknown_cutoff_origin_rows": unknown_cutoff_origin_rows,
            "alignment_coverage": float(aligned_rows / max(len(comparison), 1)),
        },
        "external": build_external_forecast_metadata(external),
        "comparison_status_distribution": status_counts,
        "comparison_evidence_status_distribution": _value_counts(comparison, "comparison_evidence_status"),
        "summary_rows": int(len(summary)),
        "guardrails": [
            "Comparison artifacts are directional triangulation only.",
            "External forecasts are imported yhat values, not actuals and not scaffold model candidates.",
            "Deltas are not residuals, prediction intervals, or accuracy/backtest metrics.",
            "Only comparison_status='aligned' rows contribute to summary delta averages.",
            "Cutoff-mismatched or unknown-origin rows are count-only diagnostics until externally scored.",
            "comparison_delta_pct is directional: (external_yhat - scaffold_yhat) / abs(scaffold_yhat).",
            "Historical cutoff-labeled external rows remain unscored until a later workflow joins them to actuals.",
            "is_comparison_scoreable is always false in this workstream.",
        ],
        "outputs": {
            "comparison": FORECAST_COMPARISON_OUTPUT,
            "summary": FORECAST_COMPARISON_SUMMARY_OUTPUT,
            "manifest": FORECAST_COMPARISON_MANIFEST_OUTPUT,
            "workbook": FORECAST_COMPARISON_WORKBOOK_OUTPUT,
            "html_report": FORECAST_COMPARISON_REPORT_OUTPUT,
            "llm_context": FORECAST_COMPARISON_LLM_CONTEXT_OUTPUT,
        },
    }


def build_forecast_comparison_llm_context(result: ForecastComparisonResult) -> dict[str, Any]:
    """Build a compact LLM handoff for a directional comparison run."""

    aligned = result.comparison[result.comparison["comparison_status"].eq("aligned")].copy()
    if "comparison_delta_yhat" in aligned.columns:
        aligned["_abs_delta_sort"] = pd.to_numeric(aligned["comparison_delta_yhat"], errors="coerce").abs()
        aligned = aligned.sort_values("_abs_delta_sort", ascending=False).drop(columns=["_abs_delta_sort"])
    guardrails = list(result.manifest.get("guardrails", []))
    return {
        "schema_version": "nixtla_scaffold.forecast_comparison.llm_context.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Paste or attach this JSON when asking an LLM to explain directional triangulation between a scaffold forecast and finance-owned external forecasts.",
        "prompt_starter": (
            "You are reviewing a directional forecast comparison. Start with alignment.status and guardrails. "
            "Do not call deltas residuals, accuracy, backtest metrics, prediction intervals, or model-selection evidence. "
            "External forecasts are not actuals and is_comparison_scoreable is false until score-external joins historical snapshots to actuals."
        ),
        "comparison_scope": result.manifest.get("comparison_scope", "directional_unscored"),
        "is_comparison_scoreable": False,
        "external_is_actual": False,
        "guardrails": guardrails,
        "inputs": result.manifest.get("inputs", {}),
        "alignment": result.manifest.get("alignment", {}),
        "comparison_status_distribution": result.manifest.get("comparison_status_distribution", {}),
        "comparison_evidence_status_distribution": result.manifest.get("comparison_evidence_status_distribution", {}),
        "summary": _records(result.summary),
        "largest_aligned_delta_rows": _records(aligned.head(50)),
        "sample_problem_rows": _records(
            result.comparison[~result.comparison["comparison_status"].eq("aligned")].head(50)
        ),
        "artifact_index": result.manifest.get("outputs", {}),
        "recommended_questions": [
            "How many rows aligned between the scaffold forecast and each external model/source/scenario?",
            "Which rows are scaffold-only, external-only, cutoff-mismatched, or unknown-origin diagnostics?",
            "Which external forecasts are directionally above or below the scaffold forecast on aligned rows?",
            "Does any external model have historical cutoff snapshots that should be scored with score-external before accuracy claims?",
            "What source, owner, model version, or scenario metadata should be added for monthly refreshability?",
        ],
    }


def build_forecast_comparison_html(result: ForecastComparisonResult) -> str:
    """Render a small standalone HTML report for directional forecast comparison."""

    manifest = result.manifest
    alignment = manifest.get("alignment", {})
    status_counts = manifest.get("comparison_status_distribution", {})
    aligned = result.comparison[result.comparison["comparison_status"].eq("aligned")].copy()
    if "comparison_delta_yhat" in aligned.columns:
        aligned["_abs_delta_sort"] = pd.to_numeric(aligned["comparison_delta_yhat"], errors="coerce").abs()
        aligned = aligned.sort_values("_abs_delta_sort", ascending=False).drop(columns=["_abs_delta_sort"])
    problem_rows = result.comparison[~result.comparison["comparison_status"].eq("aligned")].copy()
    cards = [
        ("Alignment status", alignment.get("status", "unknown")),
        ("Aligned rows", alignment.get("rows_aligned", 0)),
        ("Rows with both forecasts", alignment.get("rows_with_both_forecasts", 0)),
        ("Cutoff mismatch rows", alignment.get("mismatch_cutoff_rows", 0)),
        ("Unknown-origin rows", alignment.get("unknown_cutoff_origin_rows", 0)),
        ("Comparison scope", manifest.get("comparison_scope", "directional_unscored")),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Forecast comparison report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1, h2 {{ color: #111827; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 12px; }}
    .label {{ color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .warning {{ background: #fff7ed; border: 1px solid #fdba74; border-radius: 10px; padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2ff; }}
    code {{ background: #e5e7eb; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Forecast comparison report</h1>
  <p class="warning"><strong>Directional triangulation only.</strong> External forecasts are not actuals, not scaffold model candidates, and not scored here. Deltas are context, not residuals, prediction intervals, or accuracy metrics. Use <code>score-external</code> for historical cutoff-labeled forecast snapshots with realized actuals.</p>
  <section class="grid">
    {_html_cards(cards)}
  </section>
  <h2>Guardrails</h2>
  <ul>{''.join(f'<li>{_escape(item)}</li>' for item in manifest.get('guardrails', []))}</ul>
  <h2>Status distribution</h2>
  {_html_table(pd.DataFrame([status_counts]) if status_counts else pd.DataFrame())}
  <h2>Summary by external model/source/scenario</h2>
  <p>Summary delta averages include only rows where <code>comparison_status="aligned"</code>. Cutoff-mismatched or unknown-origin rows are count-only diagnostics.</p>
  {_html_table(result.summary)}
  <h2>Largest aligned directional deltas</h2>
  {_html_table(aligned.head(50))}
  <h2>Rows needing alignment review</h2>
  {_html_table(problem_rows.head(50))}
</body>
</html>
"""


def _write_comparison_workbook(path: Path, result: ForecastComparisonResult) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result.summary.to_excel(writer, sheet_name="Summary", index=False)
        result.comparison.to_excel(writer, sheet_name="Comparison Rows", index=False)
        pd.DataFrame([result.manifest.get("alignment", {})]).to_excel(writer, sheet_name="Alignment", index=False)
        pd.DataFrame({"guardrail": result.manifest.get("guardrails", [])}).to_excel(writer, sheet_name="Guardrails", index=False)
        pd.DataFrame(_records(result.external_forecasts.head(500))).to_excel(writer, sheet_name="External Forecasts", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                header = str(column_cells[0].value or "")
                width = min(max(len(header) + 2, 12), 48)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def _html_cards(cards: list[tuple[str, Any]]) -> str:
    return "".join(
        f'<div class="card"><div class="label">{_escape(label)}</div><div class="value">{_escape(value)}</div></div>'
        for label, value in cards
    )


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p><em>No rows.</em></p>"
    return frame.head(100).to_html(index=False, escape=True)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: _safe_json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _load_external_for_comparison(
    external_forecasts: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None,
    external_format: ExternalForecastFormat,
    external_model_name: str | None,
    external_source_id: str | None,
    external_id_col: str,
    external_time_col: str,
    external_value_col: str,
    external_model_col: str,
) -> pd.DataFrame:
    if isinstance(external_forecasts, pd.DataFrame):
        return canonicalize_external_forecasts(
            external_forecasts,
            format=external_format,
            model_name=external_model_name,
            source_id=external_source_id,
            id_col=external_id_col,
            time_col=external_time_col,
            value_col=external_value_col,
            model_col=external_model_col,
        )
    return load_external_forecasts(
        external_forecasts,
        sheet=sheet,
        format=external_format,
        model_name=external_model_name,
        source_id=external_source_id,
        id_col=external_id_col,
        time_col=external_time_col,
        value_col=external_value_col,
        model_col=external_model_col,
    )


def _prepare_scaffold_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "scaffold_yhat", "scaffold_model"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scaffold comparison frame is missing required column(s): {missing}")
    out = frame.copy()
    out["unique_id"] = _clean_text(out["unique_id"], "unique_id", frame_name="scaffold comparison frame")
    out["ds"] = _coerce_datetime(out["ds"], "ds", frame_name="scaffold comparison frame")
    out["scaffold_yhat"] = _coerce_numeric(out["scaffold_yhat"], "scaffold_yhat", frame_name="scaffold comparison frame")
    return out


def _prepare_external_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "model", "yhat", "source_id", "comparison_evidence_status", "status_message"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"external comparison frame is missing required column(s): {missing}")
    out = frame.copy()
    out["unique_id"] = _clean_text(out["unique_id"], "unique_id", frame_name="external comparison frame")
    out["ds"] = _coerce_datetime(out["ds"], "ds", frame_name="external comparison frame")
    out["external_model"] = _clean_text(out["model"], "model", frame_name="external comparison frame")
    out["external_source_id"] = _clean_text(out["source_id"], "source_id", frame_name="external comparison frame")
    out["external_yhat"] = _coerce_numeric(out["yhat"], "external_yhat", frame_name="external comparison frame")
    rename = {
        "cutoff": "external_cutoff",
        "horizon_step": "external_horizon_step",
        "source_file": "external_source_file",
        "sheet": "external_sheet",
        "status_message": "external_status_message",
        "is_backtested": "external_is_backtested",
        "backtest_status": "external_backtest_status",
    }
    out = out.rename(columns={old: new for old, new in rename.items() if old in out.columns})
    keep = [
        "unique_id",
        "ds",
        "external_model",
        "external_yhat",
        "external_source_id",
        "external_source_file",
        "external_sheet",
        "external_cutoff",
        "external_horizon_step",
        "scenario_name",
        "comparison_evidence_status",
        "external_is_backtested",
        "external_backtest_status",
        "external_status_message",
        "owner",
        "model_version",
        "notes",
    ]
    return out[[column for column in keep if column in out.columns]]


def _load_run_metadata(run_path: Path) -> dict[str, Any]:
    path = run_path / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_origin(run_metadata: dict[str, Any]) -> pd.Timestamp | None:
    origin = run_metadata.get("reproducibility", {}).get("forecast_origin")
    if origin is None:
        origin = run_metadata.get("profile", {}).get("end")
    if origin is None:
        return None
    parsed = pd.to_datetime(origin, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _cutoff_mismatch(frame: pd.DataFrame, forecast_origin: pd.Timestamp | None) -> pd.Series:
    if "external_cutoff" not in frame.columns or forecast_origin is None:
        return pd.Series(False, index=frame.index)
    cutoffs = pd.to_datetime(frame["external_cutoff"], errors="coerce").dt.normalize()
    return cutoffs.notna() & cutoffs.ne(forecast_origin)


def _unknown_cutoff_origin(frame: pd.DataFrame, forecast_origin: pd.Timestamp | None) -> pd.Series:
    if "external_cutoff" not in frame.columns or forecast_origin is not None:
        return pd.Series(False, index=frame.index)
    cutoffs = pd.to_datetime(frame["external_cutoff"], errors="coerce")
    return cutoffs.notna()


def _comparison_warning(status: str, external_status_message: Any, forecast_origin: pd.Timestamp | None) -> str:
    pieces: list[str] = []
    if isinstance(external_status_message, str) and external_status_message.strip():
        pieces.append(external_status_message.strip())
    if status == "mismatch_cutoff":
        origin_label = forecast_origin.date().isoformat() if forecast_origin is not None else "unknown"
        pieces.append(f"Forecast origins differ from the scaffold forecast origin ({origin_label}); this row is not scored.")
    elif status == "unknown_cutoff_origin":
        pieces.append("External forecast cutoff is present but scaffold forecast origin could not be verified; this row is not comparable or scored.")
    elif status == "scaffold_only":
        pieces.append("No external forecast row aligned to this scaffold forecast row.")
    elif status == "external_only":
        pieces.append("No selected scaffold forecast row aligned to this external forecast row.")
    pieces.append(_DIRECTIONAL_WARNING)
    return " ".join(pieces)


def _safe_pct_delta(delta: pd.Series, scaffold_yhat: pd.Series, mask: pd.Series) -> pd.Series:
    denom = scaffold_yhat.abs()
    valid = mask & denom.ne(0) & denom.notna()
    out = pd.Series(np.nan, index=delta.index, dtype="float64")
    out.loc[valid] = delta.loc[valid] / denom.loc[valid]
    return out


def _clean_text(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = values.astype("string").str.strip()
    missing = out.isna() | out.eq("")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} {frame_name} rows have blank {column} values")
    return out.astype(str)


def _coerce_datetime(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} {frame_name} rows have invalid {column} dates")
    return out


def _coerce_numeric(values: pd.Series, column: str, *, frame_name: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    invalid = out.isna() | ~np.isfinite(out.to_numpy(dtype="float64"))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} {frame_name} rows have missing or non-numeric {column} values")
    return out.astype(float)


def _order_comparison_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "unique_id",
        "ds",
        "comparison_status",
        "comparison_scope",
        "comparison_evidence_status",
        "scaffold_model",
        "scaffold_yhat",
        "external_model",
        "external_yhat",
        "comparison_delta_yhat",
        "comparison_delta_pct",
        "is_comparison_scoreable",
        "external_is_actual",
        "external_source_id",
        "external_source_file",
        "external_sheet",
        "external_cutoff",
        "scenario_name",
        "external_horizon_step",
        "external_is_backtested",
        "external_backtest_status",
        "horizon_step",
        "row_horizon_status",
        "horizon_trust_state",
        "planning_eligible",
        "planning_eligibility_scope",
        "interval_status",
        "comparison_warning",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extras]


def _comparison_sort_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["unique_id", "ds"]
    for column in ["external_model", "external_source_id", "scenario_name", "comparison_status"]:
        if column in frame.columns:
            columns.append(column)
    return columns


def _alignment_status(aligned_rows: int, mismatch_cutoff_rows: int, unknown_cutoff_origin_rows: int) -> str:
    if aligned_rows:
        if mismatch_cutoff_rows or unknown_cutoff_origin_rows:
            return "partially_aligned_with_cutoff_warnings"
        return "aligned"
    if unknown_cutoff_origin_rows:
        return "unknown_cutoff_origin_only"
    if mismatch_cutoff_rows:
        return "cutoff_mismatch_only"
    return "no_overlap"


def _forecast_origin_status(comparison: pd.DataFrame, forecast_origin: pd.Timestamp | None) -> str:
    has_external_cutoff = "external_cutoff" in comparison.columns and pd.to_datetime(comparison["external_cutoff"], errors="coerce").notna().any()
    if forecast_origin is not None:
        return "known"
    return "unknown_required_for_external_cutoffs" if has_external_cutoff else "not_required"


def _summary_columns() -> list[str]:
    return [
        "external_model",
        "external_source_id",
        "scenario_name",
        "comparison_evidence_status",
        "row_count",
        "aligned_rows",
        "rows_with_both_forecasts",
        "scaffold_only_rows",
        "external_only_rows",
        "mismatch_cutoff_rows",
        "unknown_cutoff_origin_rows",
        "series_count",
        "start",
        "end",
        "avg_delta_yhat_vs_scaffold",
        "avg_abs_delta_yhat_vs_scaffold",
        "avg_pct_delta_vs_scaffold",
        "avg_abs_pct_delta_vs_scaffold",
        "metric_definition",
        "is_comparison_scoreable",
        "warning",
    ]


def _mean_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float(clean.mean())


def _date_or_none(values: pd.Series, method: str) -> str | None:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    value = dates.min() if method == "min" else dates.max()
    return value.date().isoformat()


def _range_label(frame: pd.DataFrame, column: str) -> str:
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return "empty"
    return f"{dates.min().date().isoformat()}..{dates.max().date().isoformat()}"


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}
