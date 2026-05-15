from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nixtla_scaffold.comparisons import ForecastComparisonResult, write_forecast_comparison
from nixtla_scaffold.data import read_tabular_source
from nixtla_scaffold.external import ExternalForecastFormat, build_external_forecast_metadata, canonicalize_external_forecasts
from nixtla_scaffold.external_scoring import ExternalForecastScoreResult, write_external_forecast_scores
from nixtla_scaffold.hierarchy import aggregate_hierarchy_frame


BYO_MODEL_SCHEMA_VERSION = "nixtla_scaffold.byo_model.v1"
BYO_MODEL_FORECAST_OUTPUT = "byo_model_forecasts.csv"
BYO_MODEL_CONTRACT_OUTPUT = "byo_model_contract.csv"
BYO_MODEL_MANIFEST_OUTPUT = "byo_model_manifest.json"
BYO_MODEL_AUTOMATION_OUTPUT = "byo_model_automation.md"
BYO_MODEL_COMPARISON_SUMMARY_OUTPUT = "byo_model_comparison_summary.csv"
BYO_MODEL_SCORE_SUMMARY_OUTPUT = "byo_model_score_summary.csv"


@dataclass(frozen=True)
class BYOModelIngestResult:
    """Canonical imported Excel-owned forecast outputs."""

    forecasts: pd.DataFrame
    contract: pd.DataFrame
    manifest: dict[str, Any]

    def to_directory(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.forecasts.to_csv(out / BYO_MODEL_FORECAST_OUTPUT, index=False)
        self.contract.to_csv(out / BYO_MODEL_CONTRACT_OUTPUT, index=False)
        (out / BYO_MODEL_MANIFEST_OUTPUT).write_text(json.dumps(self.manifest, indent=2, default=str) + "\n", encoding="utf-8")
        _write_byo_model_automation(out, self.manifest)
        return out


@dataclass(frozen=True)
class BYOModelComparisonResult:
    """Directional comparison artifacts for BYO model forecasts."""

    ingest: BYOModelIngestResult
    comparison: ForecastComparisonResult
    byo_summary: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BYOModelScoreResult:
    """Actual-vs-forecast scoring artifacts for cutoff-labeled BYO forecasts."""

    ingest: BYOModelIngestResult
    scores: ExternalForecastScoreResult
    byo_summary: pd.DataFrame
    manifest: dict[str, Any]


def load_byo_model_forecasts(
    source: str | Path,
    *,
    sheets: Sequence[str | int] | None = None,
    format: ExternalForecastFormat = "auto",
    model_name: str | None = None,
    source_id: str | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    value_col: str = "yhat",
    model_col: str = "model",
    forecast_origin_col: str = "cutoff",
    scenario_col: str = "scenario_name",
    version_col: str = "version",
    group_cols: Sequence[str] | None = None,
    include_rollups: bool = True,
    total_label: str = "Total",
) -> pd.DataFrame:
    """Load one workbook/export with one or more finance-model versions.

    BYO models are imported forecast outputs. They are safe for comparison and
    external scoring, but they are not fed into scaffold champion selection.
    """

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    group_cols_tuple = _clean_column_sequence(group_cols, "group_cols")
    frames = _read_byo_source_frames(source_path, sheets=sheets)
    if not frames:
        raise ValueError(f"BYO model source has no readable sheets or tables: {source_path}")

    base_source_id = _validate_label(source_id or source_path.stem, "source_id")
    base_model_name = _validate_label(model_name or source_path.stem, "model_name")
    multi_sheet = len(frames) > 1
    outputs: list[pd.DataFrame] = []
    for sheet_name, raw in frames:
        sheet_label = _sheet_label(sheet_name)
        prepared = _prepare_sheet_frame(
            raw,
            sheet_label=sheet_label,
            model_name=base_model_name,
            id_col=id_col,
            model_col=model_col,
            scenario_col=scenario_col,
            version_col=version_col,
            group_cols=group_cols_tuple,
            include_rollups=include_rollups,
            total_label=total_label,
        )
        sheet_source_id = f"{base_source_id}::{sheet_label}" if multi_sheet else base_source_id
        canonical = canonicalize_external_forecasts(
            prepared,
            format=format,
            model_name=base_model_name,
            source_id=sheet_source_id,
            source_file=str(source_path),
            sheet=sheet_name,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            model_col=model_col,
            forecast_origin_col=forecast_origin_col,
        )
        if group_cols_tuple:
            canonical = add_byo_model_rollups(canonical, group_cols=group_cols_tuple, include_rollups=include_rollups, total_label=total_label)
        outputs.append(canonical)

    combined = pd.concat(outputs, ignore_index=True)
    if combined.empty:
        raise ValueError("BYO model import produced no forecast rows")
    return combined.sort_values(_byo_forecast_sort_columns(combined)).reset_index(drop=True)


def ingest_byo_model_forecasts(
    source: str | Path,
    *,
    sheets: Sequence[str | int] | None = None,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> BYOModelIngestResult:
    """Load BYO forecasts and build review metadata without writing unless asked."""

    forecasts = load_byo_model_forecasts(source, sheets=sheets, **kwargs)
    contract = build_byo_model_contract(forecasts)
    manifest = build_byo_model_manifest(source, forecasts, contract, sheets=sheets, output_dir=output_dir, operation="ingest", **_manifest_options(kwargs))
    return BYOModelIngestResult(forecasts=forecasts, contract=contract, manifest=manifest)


def write_byo_model_ingest(source: str | Path, output_dir: str | Path, *, sheets: Sequence[str | int] | None = None, **kwargs: Any) -> BYOModelIngestResult:
    """Write canonical BYO model forecasts and a small contract manifest."""

    result = ingest_byo_model_forecasts(source, sheets=sheets, output_dir=output_dir, **kwargs)
    result.to_directory(output_dir)
    return result


def write_byo_model_comparison(
    run_dir: str | Path,
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    sheets: Sequence[str | int] | None = None,
    scaffold_model: str | None = None,
    main_model_preference: str | None = None,
    **kwargs: Any,
) -> BYOModelComparisonResult:
    """Import BYO forecasts, compare to a scaffold run, and write BYO artifacts."""

    out = Path(output_dir) if output_dir is not None else Path(run_dir) / "byo_model"
    ingest = ingest_byo_model_forecasts(source, sheets=sheets, output_dir=out, **kwargs)
    ingest.to_directory(out)
    comparison = write_forecast_comparison(
        run_dir,
        ingest.forecasts,
        output_dir=out,
        external_format="long",
        scaffold_model=scaffold_model,
    )
    group_cols = tuple(kwargs.get("group_cols") or ())
    byo_summary = build_byo_model_comparison_summary(comparison.comparison, group_cols=group_cols)
    byo_summary.to_csv(out / BYO_MODEL_COMPARISON_SUMMARY_OUTPUT, index=False)
    manifest = {
        **ingest.manifest,
        "operation": "compare",
        "scaffold_run_dir": str(Path(run_dir)),
        "scaffold_model": scaffold_model or "selected",
        "main_model_preference": main_model_preference or "",
        "main_model_preference_scope": "display_only" if main_model_preference else "",
        "comparison": comparison.manifest,
        "automation_recommendations": build_byo_model_automation_recommendations(
            ingest.forecasts,
            operation="compare",
            options=_manifest_options(kwargs),
        ),
        "outputs": {
            **ingest.manifest.get("outputs", {}),
            "forecast_comparison": "forecast_comparison.csv",
            "forecast_comparison_summary": "forecast_comparison_summary.csv",
            "byo_model_comparison_summary": BYO_MODEL_COMPARISON_SUMMARY_OUTPUT,
        },
    }
    (out / BYO_MODEL_MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    _write_byo_model_automation(out, manifest)
    return BYOModelComparisonResult(ingest=ingest, comparison=comparison, byo_summary=byo_summary, manifest=manifest)


def write_byo_model_scores(
    source: str | Path,
    actuals: str | Path,
    output_dir: str | Path,
    *,
    sheets: Sequence[str | int] | None = None,
    actuals_sheet: str | int | None = None,
    actual_id_col: str = "unique_id",
    actual_time_col: str = "ds",
    actual_value_col: str = "y",
    season_length: int = 1,
    requested_horizon: int | None = None,
    **kwargs: Any,
) -> BYOModelScoreResult:
    """Score cutoff-labeled BYO forecasts against actuals, including optional rollups."""

    out = Path(output_dir)
    ingest = ingest_byo_model_forecasts(source, sheets=sheets, output_dir=out, **kwargs)
    ingest.to_directory(out)
    group_cols = tuple(kwargs.get("group_cols") or ())
    scoring_actuals = _load_byo_actuals_for_scoring(
        actuals,
        actuals_sheet=actuals_sheet,
        actual_id_col=actual_id_col,
        actual_time_col=actual_time_col,
        actual_value_col=actual_value_col,
        group_cols=group_cols,
    )
    scores = write_external_forecast_scores(
        ingest.forecasts,
        scoring_actuals,
        out,
        external_format="long",
        actual_id_col="unique_id",
        actual_time_col="ds",
        actual_value_col="y",
        season_length=season_length,
        requested_horizon=requested_horizon,
    )
    byo_summary = build_byo_model_score_summary(scores.backtest_long, group_cols=group_cols)
    byo_summary.to_csv(out / BYO_MODEL_SCORE_SUMMARY_OUTPUT, index=False)
    manifest = {
        **ingest.manifest,
        "operation": "score",
        "actuals_source": str(Path(actuals)) if isinstance(actuals, (str, Path)) else "dataframe",
        "scoring": scores.manifest,
        "automation_recommendations": build_byo_model_automation_recommendations(
            ingest.forecasts,
            operation="score",
            options=_manifest_options(kwargs),
        ),
        "outputs": {
            **ingest.manifest.get("outputs", {}),
            "external_backtest_long": "external_backtest_long.csv",
            "external_model_metrics": "external_model_metrics.csv",
            "byo_model_score_summary": BYO_MODEL_SCORE_SUMMARY_OUTPUT,
        },
    }
    (out / BYO_MODEL_MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    _write_byo_model_automation(out, manifest)
    return BYOModelScoreResult(ingest=ingest, scores=scores, byo_summary=byo_summary, manifest=manifest)


def build_byo_model_automation_markdown(manifest: dict[str, Any]) -> str:
    """Render the BYO automation recommendation artifact."""

    automation = manifest.get("automation_recommendations", {}) if isinstance(manifest, dict) else {}
    recommendations = automation.get("recommendations", []) if isinstance(automation, dict) else []
    detected = automation.get("detected_detail_columns", []) if isinstance(automation, dict) else []
    refresh_loop = automation.get("recommended_refresh_loop", []) if isinstance(automation, dict) else []
    lines = [
        "# BYO model automation guide",
        "",
        "Finance-owned driver models should remain the place where customer/SKU bridges, purchase type, renewal assumptions, PxQ, and scenario logic are authored.",
        "The scaffold imports their outputs for comparison, scoring, lineage, and refresh automation without overwriting `forecast.csv` or the scaffold champion.",
        "",
        "## Detected model contract",
        "",
        f"- Source: `{manifest.get('source', '')}`",
        f"- Operation: `{manifest.get('operation', '')}`",
        f"- Rows: `{manifest.get('rows', 0)}`",
        f"- Models/scenarios: `{', '.join(manifest.get('models', []) or [])}` / `{', '.join(manifest.get('scenarios', []) or [])}`",
        f"- Detail columns detected: `{', '.join(detected) if detected else 'none'}`",
        "",
        "## Recommended automation loop",
        "",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(refresh_loop, start=1))
    lines.extend(["", "## Automation recommendations", ""])
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        lines.append(f"### {recommendation.get('id', 'recommendation')} ({recommendation.get('status', 'recommended')})")
        lines.append("")
        lines.append(str(recommendation.get("recommendation", "")))
        lines.append("")
        lines.append(f"Why: {recommendation.get('why', '')}")
        lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "",
            "- Treat BYO outputs as external forecasts until cutoff-labeled snapshots are scored against actuals.",
            "- Keep ARR, billed revenue, net revenue, seats, and usage definitions explicit in the source model contract.",
            "- Use BYO model preference as display/storytelling context only; do not silently replace selected statistical `yhat`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_byo_model_automation(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / BYO_MODEL_AUTOMATION_OUTPUT).write_text(build_byo_model_automation_markdown(manifest), encoding="utf-8")


def add_byo_model_rollups(
    forecasts: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    include_rollups: bool = True,
    total_label: str = "Total",
) -> pd.DataFrame:
    """Attach hierarchy metadata and optional derived-sum rollup forecast rows."""

    group_cols_tuple = _clean_column_sequence(group_cols, "group_cols")
    if not group_cols_tuple:
        return forecasts.copy()
    missing = [column for column in group_cols_tuple if column not in forecasts.columns]
    if missing:
        raise ValueError(f"BYO model grouping columns are missing from imported forecasts: {missing}")
    _validate_group_values(forecasts, group_cols_tuple)
    _validate_stable_id_groups(forecasts, group_cols_tuple)

    leaf = forecasts.copy()
    leaf["hierarchy_level"] = "/".join(group_cols_tuple)
    leaf["hierarchy_depth"] = len(group_cols_tuple)
    leaf["external_rollup_source"] = "leaf"
    if not include_rollups:
        return leaf.sort_values(_byo_forecast_sort_columns(leaf)).reset_index(drop=True)

    nodes = [leaf]
    group_keys = _rollup_group_keys(leaf)
    for depth in range(0, len(group_cols_tuple)):
        prefix = group_cols_tuple[:depth]
        group_by = [*group_keys, *prefix]
        grouped = leaf.groupby(group_by, dropna=False, as_index=False)["yhat"].sum() if group_by else pd.DataFrame({"yhat": [leaf["yhat"].sum()]})
        grouped["unique_id"] = total_label if depth == 0 else grouped.apply(lambda row: _hierarchy_node_id(prefix, row), axis=1)
        grouped["hierarchy_level"] = "total" if depth == 0 else "/".join(prefix)
        grouped["hierarchy_depth"] = depth
        grouped["external_rollup_source"] = "derived_sum"
        for column in group_cols_tuple[depth:]:
            grouped[column] = None
        for column in _rollup_passthrough_columns(leaf):
            if column in grouped.columns:
                continue
            grouped[column] = _group_single_value(leaf, group_keys=group_by, grouped=grouped, column=column) if group_by else _single_or_none(leaf[column])
        nodes.append(grouped)

    out = pd.concat(nodes, ignore_index=True, sort=False)
    return out.sort_values(_byo_forecast_sort_columns(out)).reset_index(drop=True)


def build_byo_model_contract(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Build a compact sheet/version/source contract table."""

    if forecasts.empty:
        return pd.DataFrame(columns=_contract_columns())
    group_cols = [column for column in ["source_id", "source_file", "sheet", "model", "scenario_name", "model_version", "version"] if column in forecasts.columns]
    if not group_cols:
        group_cols = ["model"]
    rows: list[dict[str, Any]] = []
    for keys, group in forecasts.groupby(group_cols, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        row.update(
            {
                "rows": int(len(group)),
                "unique_ids": int(group["unique_id"].astype(str).nunique()),
                "start": _date_or_none(group["ds"], "min"),
                "end": _date_or_none(group["ds"], "max"),
                "has_cutoff": bool("cutoff" in group.columns and group["cutoff"].notna().any()),
                "hierarchy_levels": _json_counts(group, "hierarchy_level"),
                "rollup_sources": _json_counts(group, "external_rollup_source"),
                "evidence_status_distribution": _json_counts(group, "comparison_evidence_status"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=_contract_columns(rows))


def build_byo_model_manifest(
    source: str | Path,
    forecasts: pd.DataFrame,
    contract: pd.DataFrame,
    *,
    sheets: Sequence[str | int] | None = None,
    output_dir: str | Path | None = None,
    operation: str = "ingest",
    **options: Any,
) -> dict[str, Any]:
    """Build JSON-serializable BYO import metadata."""

    automation = build_byo_model_automation_recommendations(forecasts, operation=operation, options=options)
    return {
        "schema_version": BYO_MODEL_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "source": str(Path(source)),
        "sheets": [str(sheet) for sheet in sheets] if sheets else "all",
        "output_dir": str(Path(output_dir)) if output_dir is not None else None,
        "rows": int(len(forecasts)),
        "unique_ids": int(forecasts["unique_id"].astype(str).nunique()) if not forecasts.empty else 0,
        "models": sorted(forecasts["model"].dropna().astype(str).unique().tolist()) if "model" in forecasts.columns else [],
        "scenarios": sorted(forecasts["scenario_name"].dropna().astype(str).unique().tolist()) if "scenario_name" in forecasts.columns else [],
        "hierarchy_levels": _value_counts(forecasts, "hierarchy_level"),
        "rollup_sources": _value_counts(forecasts, "external_rollup_source"),
        "external_contract": build_external_forecast_metadata(forecasts),
        "contract_rows": int(len(contract)),
        "options": options,
        "guardrails": [
            "BYO Excel models are imported yhat outputs for comparison/scoring, not executable model candidates.",
            "Derived rollups are explicit rows tagged external_rollup_source='derived_sum'.",
            "Future-only BYO rows are directional comparison evidence only until cutoff-labeled snapshots join to actuals.",
            "Choosing a BYO model as a display preference never overwrites forecast.csv or scaffold champion selection.",
            "Customer/SKU, purchase-type, renewal, and PxQ bridges should live inside the finance-owned BYO model or its exported detail tables, not as a separate yhat-replacement layer.",
        ],
        "automation_recommendations": automation,
        "outputs": {
            "forecasts": BYO_MODEL_FORECAST_OUTPUT,
            "contract": BYO_MODEL_CONTRACT_OUTPUT,
            "manifest": BYO_MODEL_MANIFEST_OUTPUT,
            "automation": BYO_MODEL_AUTOMATION_OUTPUT,
        },
    }


def build_byo_model_automation_recommendations(
    forecasts: pd.DataFrame,
    *,
    operation: str = "ingest",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest how to automate and govern finance-owned BYO model outputs."""

    options = dict(options or {})
    columns = {str(column) for column in forecasts.columns}
    bridge_detail_columns = sorted(
        columns
        & {
            "account",
            "amount",
            "contract_id",
            "customer",
            "customer_id",
            "known_as_of",
            "next_sku",
            "prior_sku",
            "purchase_type",
            "quantity",
            "rate",
            "renewal_date",
            "seats",
            "sku",
            "subscription_id",
            "unit_price",
        }
    )
    group_cols = [str(column) for column in options.get("group_cols", ()) if str(column).strip()]
    has_cutoff = "cutoff" in forecasts.columns and forecasts["cutoff"].notna().any()
    has_known_as_of = "known_as_of" in forecasts.columns and forecasts["known_as_of"].notna().any()
    recommendations = [
        {
            "id": "make-source-executable",
            "status": "recommended",
            "recommendation": "Wrap the finance-owned workbook/model in a repeatable export step that writes the BYO forecast contract.",
            "why": "This keeps Excel/Python driver logic owned by finance while making refreshes reproducible.",
        },
        {
            "id": "snapshot-cutoffs",
            "status": "present" if has_cutoff else "missing",
            "recommendation": "Persist cutoff-labeled forecast snapshots before each official submission.",
            "why": "Cutoff snapshots are required before BYO forecasts can be scored against future actuals.",
        },
        {
            "id": "known-as-of-lineage",
            "status": "present" if has_known_as_of else "missing",
            "recommendation": "Add known_as_of lineage for manual assumptions, renewals, pricing, and customer-level overrides.",
            "why": "Known-as-of timestamps make renewal/add-on/churn assumptions auditable and reduce leakage risk.",
        },
        {
            "id": "component-detail-in-byo",
            "status": "present" if bridge_detail_columns else "optional",
            "recommendation": "Keep customer/SKU, purchase-type, renewal, and PxQ bridge detail in the BYO model export or a sibling detail table.",
            "why": "The bridge is part of the finance-owned bottoms-up model story; the scaffold should import, compare, score, and operationalize it rather than replace selected yhat.",
        },
        {
            "id": "metric-definition",
            "status": "recommended",
            "recommendation": "Define the metric upstream in the BYO model contract: ARR, billed revenue, net revenue, seats, usage, currency, FX, and normalization rules.",
            "why": "The scaffold should not infer conversions between ARR, billed revenue, net revenue, or quantity units.",
        },
    ]
    return {
        "operation": operation,
        "group_cols": group_cols,
        "detected_detail_columns": bridge_detail_columns,
        "has_cutoff_snapshots": bool(has_cutoff),
        "has_known_as_of_lineage": bool(has_known_as_of),
        "recommended_refresh_loop": [
            "Refresh finance workbook/model inputs.",
            "Export BYO forecasts with scenario_name, model_version, owner, cutoff, and known_as_of where applicable.",
            "Run `nixtla-scaffold byo-model compare` against the latest scaffold baseline.",
            "After actuals land, run `nixtla-scaffold byo-model score` on cutoff-labeled snapshots.",
            "Review deltas, score summaries, and assumption lineage before promoting any BYO scenario as the business narrative.",
        ],
        "recommendations": recommendations,
    }


def build_byo_model_comparison_summary(comparison: pd.DataFrame, *, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Summarize scaffold-vs-BYO alignment and deltas by version and rollup lens."""

    if comparison.empty:
        return pd.DataFrame(columns=_comparison_summary_columns())
    frame = comparison.copy()
    group_by = _existing_columns(
        frame,
        [
            "external_model",
            "scenario_name",
            "external_source_id",
            "external_sheet",
            "hierarchy_level",
            "hierarchy_depth",
            "external_rollup_source",
            *(group_cols or ()),
        ],
    )
    if not group_by:
        group_by = ["external_model"] if "external_model" in frame.columns else ["comparison_status"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_by, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        aligned = group[group["comparison_status"].eq("aligned")]
        scaffold_yhat = pd.to_numeric(aligned.get("scaffold_yhat"), errors="coerce")
        external_yhat = pd.to_numeric(aligned.get("external_yhat"), errors="coerce")
        delta = pd.to_numeric(aligned.get("comparison_delta_yhat"), errors="coerce")
        rows.append(
            {
                **dict(zip(group_by, key_values, strict=True)),
                "rows": int(len(group)),
                "aligned_rows": int(len(aligned)),
                "scaffold_yhat_sum": _float_or_none(scaffold_yhat.sum()) if scaffold_yhat.notna().any() else None,
                "external_yhat_sum": _float_or_none(external_yhat.sum()) if external_yhat.notna().any() else None,
                "delta_yhat_sum": _float_or_none(delta.sum()) if delta.notna().any() else None,
                "avg_delta_yhat": _float_or_none(delta.mean()) if delta.notna().any() else None,
                "avg_delta_pct": _float_or_none(pd.to_numeric(aligned.get("comparison_delta_pct"), errors="coerce").mean()) if not aligned.empty else None,
                "comparison_status_distribution": _json_counts(group, "comparison_status"),
            }
        )
    return pd.DataFrame(rows).sort_values([column for column in group_by if column in rows[0]]).reset_index(drop=True)


def build_byo_model_score_summary(backtest_long: pd.DataFrame, *, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Summarize cutoff-scored BYO rows by version and hierarchy lens."""

    if backtest_long.empty:
        return pd.DataFrame(columns=_score_summary_columns())
    scored = backtest_long[backtest_long["scoring_status"].eq("scored")].copy()
    if scored.empty:
        return pd.DataFrame(columns=_score_summary_columns())
    group_by = _existing_columns(
        scored,
        [
            "model",
            "scenario_name",
            "source_id",
            "sheet",
            "hierarchy_level",
            "hierarchy_depth",
            "external_rollup_source",
            *(group_cols or ()),
        ],
    )
    if not group_by:
        group_by = ["model"]
    rows: list[dict[str, Any]] = []
    for keys, group in scored.groupby(group_by, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        actual = pd.to_numeric(group["y_actual"], errors="coerce")
        abs_error = pd.to_numeric(group["abs_error"], errors="coerce")
        squared_error = pd.to_numeric(group["squared_error"], errors="coerce")
        forecast_error = pd.to_numeric(group["forecast_error"], errors="coerce")
        denom = float(actual.abs().sum())
        rows.append(
            {
                **dict(zip(group_by, key_values, strict=True)),
                "observations": int(len(group)),
                "cutoff_count": int(group["cutoff"].nunique()) if "cutoff" in group.columns else 0,
                "rmse": _float_or_none(np.sqrt(squared_error.mean())) if squared_error.notna().any() else None,
                "mae": _float_or_none(abs_error.mean()) if abs_error.notna().any() else None,
                "wape": _float_or_none(abs_error.sum() / denom) if denom else None,
                "bias": _float_or_none(forecast_error.sum() / denom) if denom else None,
                "start": _date_or_none(group["ds"], "min"),
                "end": _date_or_none(group["ds"], "max"),
                "scoring_status_distribution": _json_counts(group, "scoring_status"),
            }
        )
    return pd.DataFrame(rows).sort_values([column for column in group_by if column in rows[0]]).reset_index(drop=True)


def _read_byo_source_frames(source_path: Path, *, sheets: Sequence[str | int] | None) -> list[tuple[str | int | None, pd.DataFrame]]:
    if source_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        excel = pd.ExcelFile(source_path)
        selected = list(sheets) if sheets else list(excel.sheet_names)
        unknown = [sheet for sheet in selected if isinstance(sheet, str) and sheet not in excel.sheet_names]
        if unknown:
            raise ValueError(f"sheet(s) not found in {source_path}: {unknown}")
        return [(sheet, pd.read_excel(source_path, sheet_name=sheet)) for sheet in selected]
    if sheets:
        raise ValueError("BYO sheet selection is only supported for Excel workbooks")
    return [(None, read_tabular_source(source_path))]


def _prepare_sheet_frame(
    raw: pd.DataFrame,
    *,
    sheet_label: str,
    model_name: str,
    id_col: str,
    model_col: str,
    scenario_col: str,
    version_col: str,
    group_cols: tuple[str, ...],
    include_rollups: bool,
    total_label: str,
) -> pd.DataFrame:
    if raw.empty:
        raise ValueError(f"BYO model sheet '{sheet_label}' is empty")
    frame = raw.copy()
    if group_cols:
        _validate_group_values(frame, group_cols)
        _reject_reserved_group_rows(frame, group_cols=group_cols, total_label=total_label)
        if id_col in frame.columns:
            _reject_reserved_leaf_ids(
                frame,
                id_col=id_col,
                group_cols=group_cols,
                include_rollups=include_rollups,
                total_label=total_label,
            )
        else:
            frame[id_col] = frame.apply(lambda row: _hierarchy_node_id(group_cols, row), axis=1)
    if model_col not in frame.columns or _is_blank_series(frame[model_col]):
        frame[model_col] = model_name
    if scenario_col not in frame.columns or _is_blank_series(frame[scenario_col]):
        frame[scenario_col] = sheet_label
    if version_col in frame.columns and version_col != "model_version" and "model_version" not in frame.columns:
        frame["model_version"] = frame[version_col]
    return frame


def _load_byo_actuals_for_scoring(
    actuals: str | Path,
    *,
    actuals_sheet: str | int | None,
    actual_id_col: str,
    actual_time_col: str,
    actual_value_col: str,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    raw = read_tabular_source(actuals, sheet=actuals_sheet)
    if group_cols and set(group_cols).issubset(raw.columns):
        nodes = aggregate_hierarchy_frame(raw, hierarchy_cols=group_cols, time_col=actual_time_col, target_col=actual_value_col)
        return nodes[["unique_id", "ds", "y", "hierarchy_level", "hierarchy_depth", *group_cols]]
    frame = raw.copy()
    rename = {actual_id_col: "unique_id", actual_time_col: "ds", actual_value_col: "y"}
    return frame.rename(columns=rename)


def _rollup_group_keys(frame: pd.DataFrame) -> list[str]:
    return _existing_columns(
        frame,
        [
            "ds",
            "model",
            "source_id",
            "source_file",
            "sheet",
            "scenario_name",
            "cutoff",
            "horizon_step",
            "model_version",
            "version",
            "currency",
            "unit_label",
        ],
    )


def _rollup_passthrough_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["comparison_evidence_status", "is_backtested", "backtest_status", "status_message", "family", "is_external_forecast", "external_forecast_validation_required"]
    return [column for column in candidates if column in frame.columns]


def _group_single_value(leaf: pd.DataFrame, *, group_keys: list[str], grouped: pd.DataFrame, column: str) -> list[Any]:
    values: list[Any] = []
    if not group_keys:
        return [_single_or_none(leaf[column])] * len(grouped)
    indexed = leaf.groupby(group_keys, dropna=False)[column]
    lookup = {keys if isinstance(keys, tuple) else (keys,): _single_or_none(series) for keys, series in indexed}
    for row in grouped[group_keys].itertuples(index=False, name=None):
        values.append(lookup.get(tuple(row)))
    return values


def _validate_group_values(frame: pd.DataFrame, group_cols: tuple[str, ...]) -> None:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"BYO model group_cols are missing from the workbook: {missing}")
    invalid = frame[list(group_cols)].isna() | frame[list(group_cols)].astype(str).apply(lambda col: col.str.strip().eq(""))
    if invalid.any().any():
        sample = frame.loc[invalid.any(axis=1), list(group_cols)].head(5).to_dict("records")
        raise ValueError(f"BYO model group_cols cannot contain blank values: {sample}")


def _validate_stable_id_groups(frame: pd.DataFrame, group_cols: tuple[str, ...]) -> None:
    stability = frame[["unique_id", *group_cols]].drop_duplicates()
    conflicts = stability.groupby("unique_id", dropna=False).size()
    bad = conflicts[conflicts > 1]
    if not bad.empty:
        sample = stability[stability["unique_id"].isin(bad.index)].head(10).to_dict("records")
        raise ValueError(f"BYO model unique_id must map to one stable group path: {sample}")


def _reject_reserved_group_rows(frame: pd.DataFrame, *, group_cols: tuple[str, ...], total_label: str) -> None:
    reserved = {str(total_label).strip()}
    for _, row in frame.iterrows():
        for depth in range(1, len(group_cols)):
            reserved.add(_hierarchy_node_id(group_cols[:depth], row))
    for column in group_cols:
        values = frame[column].astype(str).str.strip()
        bad = values.isin(reserved) | values.str.lower().isin({"total", "subtotal", "all"})
        if bad.any():
            sample = frame.loc[bad, [*group_cols]].head(5).to_dict("records")
            raise ValueError(
                "BYO grouped import expects leaf-level rows only; "
                f"found subtotal/reserved labels in {column}: {sample}. "
                "Remove workbook subtotal rows and let --rollups create auditable derived totals."
            )


def _reject_reserved_leaf_ids(
    frame: pd.DataFrame,
    *,
    id_col: str,
    group_cols: tuple[str, ...],
    include_rollups: bool,
    total_label: str,
) -> None:
    ids = frame[id_col].astype(str).str.strip()
    reserved = {str(total_label).strip()}
    if include_rollups:
        for _, row in frame.iterrows():
            for depth in range(1, len(group_cols)):
                reserved.add(_hierarchy_node_id(group_cols[:depth], row))
    collisions = frame.loc[ids.isin(reserved), [id_col, *group_cols]].head(5).to_dict("records")
    if collisions:
        raise ValueError(
            "BYO model input appears to contain subtotal/total rows while group rollups are enabled; "
            f"remove subtotal rows or omit group_cols. Examples: {collisions}"
        )


def _hierarchy_node_id(cols: tuple[str, ...], row: pd.Series) -> str:
    return "|".join(f"{column}={row[column]}" for column in cols)


def _clean_column_sequence(columns: Sequence[str] | None, name: str) -> tuple[str, ...]:
    if not columns:
        return ()
    cleaned = tuple(str(column).strip() for column in columns if str(column).strip())
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} contains duplicate column names: {list(columns)}")
    return cleaned


def _validate_label(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} cannot be blank")
    return text


def _is_blank_series(series: pd.Series) -> bool:
    return series.isna().all() or series.dropna().astype(str).str.strip().eq("").all()


def _sheet_label(sheet: str | int | None) -> str:
    if sheet is None:
        return "table"
    return str(sheet).strip() or "sheet"


def _single_or_none(series: pd.Series) -> Any:
    values = series.dropna().drop_duplicates()
    if len(values) == 1:
        return values.iloc[0]
    return None


def _date_or_none(series: pd.Series, op: str) -> str | None:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return None
    selected = values.min() if op == "min" else values.max()
    return pd.Timestamp(selected).date().isoformat()


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("").astype(str).value_counts().sort_index().items()}


def _json_counts(frame: pd.DataFrame, column: str) -> str:
    return json.dumps(_value_counts(frame, column), sort_keys=True)


def _existing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _manifest_options(options: dict[str, Any]) -> dict[str, Any]:
    serializable = {}
    for key, value in options.items():
        if isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable


def _byo_forecast_sort_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["source_id", "sheet", "model", "scenario_name", "hierarchy_depth", "unique_id", "cutoff", "ds"]
    return [column for column in columns if column in frame.columns]


def _contract_columns(rows: list[dict[str, Any]] | None = None) -> list[str]:
    preferred = [
        "source_id",
        "source_file",
        "sheet",
        "model",
        "scenario_name",
        "model_version",
        "version",
        "rows",
        "unique_ids",
        "start",
        "end",
        "has_cutoff",
        "hierarchy_levels",
        "rollup_sources",
        "evidence_status_distribution",
    ]
    if not rows:
        return preferred
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return [column for column in preferred if any(column in row for row in rows)] + extras


def _comparison_summary_columns() -> list[str]:
    return [
        "external_model",
        "scenario_name",
        "external_source_id",
        "external_sheet",
        "hierarchy_level",
        "hierarchy_depth",
        "external_rollup_source",
        "rows",
        "aligned_rows",
        "scaffold_yhat_sum",
        "external_yhat_sum",
        "delta_yhat_sum",
        "avg_delta_yhat",
        "avg_delta_pct",
        "comparison_status_distribution",
    ]


def _score_summary_columns() -> list[str]:
    return [
        "model",
        "scenario_name",
        "source_id",
        "sheet",
        "hierarchy_level",
        "hierarchy_depth",
        "external_rollup_source",
        "observations",
        "cutoff_count",
        "rmse",
        "mae",
        "wape",
        "bias",
        "start",
        "end",
        "scoring_status_distribution",
    ]
