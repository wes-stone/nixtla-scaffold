from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nixtla_scaffold.data import canonicalize_forecast_frame, read_tabular_source

LEDGER_SCHEMA_VERSION = "1.0"
DEFAULT_LEDGER_PATH = Path("runs") / "forecast_ledger"

FORECAST_VERSION_COLUMNS = [
    "forecast_version_id",
    "forecast_key",
    "version_label",
    "run_dir",
    "forecast_origin",
    "horizon",
    "freq",
    "model_policy",
    "data_hash_sha256",
    "git_sha",
    "created_at",
    "created_by",
    "notes",
    "manifest_path",
    "diagnostics_path",
]

FORECAST_SNAPSHOT_COLUMNS = [
    "forecast_version_id",
    "forecast_key",
    "version_label",
    "forecast_origin",
    "unique_id",
    "ds",
    "model",
    "family",
    "yhat",
    "yhat_lo_80",
    "yhat_hi_80",
    "yhat_lo_95",
    "yhat_hi_95",
    "interval_status",
    "interval_method",
    "horizon_step",
    "row_horizon_status",
    "horizon_trust_state",
    "validated_through_horizon",
    "planning_eligible",
    "planning_eligibility_scope",
    "planning_eligibility_reason",
    "is_selected_model",
    "weight",
    "yhat_scenario",
    "event_adjustment",
    "event_names",
]

FORECAST_VERSION_METRIC_COLUMNS = [
    "forecast_version_id",
    "forecast_key",
    "unique_id",
    "selected_model",
    "rmse",
    "mae",
    "mase",
    "rmsse",
    "wape",
    "bias",
    "trust_level",
    "trust_score_0_100",
    "history_readiness",
    "interval_status",
    "horizon_trust_state",
    "planning_ready_horizon",
    "full_horizon_claim_allowed",
    "seasonality_status",
    "caveats",
    "next_actions",
    "executive_headline",
]

SOURCE_REFRESH_COLUMNS = [
    "source_refresh_id",
    "forecast_version_id",
    "forecast_key",
    "source_kind",
    "source_input",
    "source_output",
    "query_file",
    "metadata_file",
    "id_col",
    "time_col",
    "target_col",
    "id_value",
    "rows",
    "series_count",
    "source_start",
    "source_end",
    "created_at",
]

OFFICIAL_LOCK_COLUMNS = [
    "lock_id",
    "forecast_version_id",
    "forecast_key",
    "lock_label",
    "audience",
    "planning_cycle",
    "communication_date",
    "submitted_to",
    "lock_reason",
    "locked_at",
    "locked_by",
    "notes",
]

ACTUAL_REVISION_COLUMNS = [
    "actual_revision_id",
    "forecast_key",
    "revision_label",
    "unique_id",
    "ds",
    "y",
    "source_kind",
    "source_id",
    "source_path",
    "known_as_of",
    "created_at",
    "is_latest",
]

FORECAST_ACTUAL_COLUMNS = [
    "forecast_version_id",
    "forecast_key",
    "version_label",
    "unique_id",
    "ds",
    "model",
    "forecast_origin",
    "horizon_step",
    "yhat",
    "y_actual",
    "error",
    "abs_error",
    "pct_error",
    "interval_80_hit",
    "interval_95_hit",
    "actual_revision_id",
]

FORECAST_PERFORMANCE_COLUMNS = [
    "forecast_version_id",
    "forecast_key",
    "version_label",
    "unique_id",
    "model",
    "observed_periods",
    "mae",
    "rmse",
    "bias",
    "wape",
    "interval_80_coverage",
    "interval_95_coverage",
]

ADJUSTMENT_COLUMNS = [
    "adjustment_id",
    "forecast_key",
    "unique_id",
    "start_ds",
    "end_ds",
    "adjustment_type",
    "adjustment_value",
    "adjusted_y",
    "factor",
    "metric_mapping",
    "conversion_factor",
    "expected_elasticity",
    "confidence",
    "reason",
    "owner",
    "evidence_uri",
    "known_as_of",
    "approval_status",
    "approval_notes",
    "created_at",
]

CORRECTED_ACTUAL_COLUMNS = [
    "forecast_key",
    "unique_id",
    "ds",
    "raw_y",
    "corrected_y",
    "normalized_y",
    "is_excluded",
    "applied_adjustment_ids",
    "actual_revision_id",
]

REGIME_CHANGE_COLUMNS = [
    "regime_change_id",
    "forecast_key",
    "unique_id",
    "start_ds",
    "end_ds",
    "metric_mapping",
    "conversion_factor",
    "expected_elasticity",
    "confidence",
    "reason",
    "owner",
    "known_as_of",
    "approval_status",
    "created_at",
]

DELTA_COLUMNS = [
    "comparison_id",
    "forecast_key",
    "base_version_id",
    "comparison_version_id",
    "base_lock_label",
    "unique_id",
    "ds",
    "base_yhat",
    "comparison_yhat",
    "comparison_minus_base",
    "comparison_minus_base_pct",
    "actual_y",
    "actual_minus_base",
    "actual_minus_comparison",
    "status_label",
]

EXPORT_TABLES = {
    "forecast_versions": "forecast_versions.csv",
    "forecast_snapshot": "forecast_snapshot.csv",
    "forecast_version_metrics": "forecast_version_metrics.csv",
    "source_refreshes": "source_refreshes.csv",
    "official_forecast_locks": "official_forecast_locks.csv",
    "actual_revisions": "forecast_actual_revisions.csv",
    "forecast_actuals": "forecast_actuals.csv",
    "forecast_performance": "forecast_performance.csv",
    "forecast_adjustments": "forecast_adjustments.csv",
    "corrected_actuals": "corrected_actuals.csv",
    "regime_changes": "regime_changes.csv",
    "forecast_version_deltas": "forecast_version_deltas.csv",
}

ADJUSTMENT_TYPES = {
    "replacement",
    "additive_delta",
    "multiplicative_factor",
    "exclude",
    "fill",
    "interpolate",
    "business_model_normalization",
    "regime_change",
}
REQUIRED_ADJUSTMENT_METADATA = {"reason", "known_as_of", "approval_status"}


@dataclass(frozen=True)
class LedgerResult:
    ledger_path: str
    database_path: str
    exports_path: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_path": self.ledger_path,
            "database_path": self.database_path,
            "exports_path": self.exports_path,
            **self.payload,
        }


def init_ledger(ledger: str | Path = DEFAULT_LEDGER_PATH) -> LedgerResult:
    ledger_dir = _ledger_dir(ledger)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    exports = ledger_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
    return LedgerResult(str(ledger_dir), str(_database_path(ledger_dir)), str(exports), {"status": "initialized", "schema_version": LEDGER_SCHEMA_VERSION})


def register_run(
    ledger: str | Path,
    run_dir: str | Path,
    *,
    forecast_key: str,
    version_label: str = "",
    created_by: str = "",
    notes: str = "",
    source_metadata_path: str | Path | None = None,
    source_kind: str = "",
    export: bool = True,
) -> LedgerResult:
    ledger_dir = _ensure_ledger(ledger)
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json is required to register a forecast run: {manifest_path}")
    manifest = _read_json(manifest_path)
    diagnostics = _read_json(run_path / "diagnostics.json") if (run_path / "diagnostics.json").exists() else {}
    version = _version_record(
        run_path,
        manifest,
        diagnostics,
        forecast_key=forecast_key,
        version_label=version_label,
        created_by=created_by,
        notes=notes,
    )
    snapshot = _forecast_snapshot_frame(run_path, version)
    metrics = _forecast_version_metrics_frame(run_path, version, diagnostics)
    source_refreshes = _source_refresh_frame(
        run_path,
        version,
        source_metadata_path=source_metadata_path,
        source_kind=source_kind,
    )
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
        conn.execute("DELETE FROM forecast_snapshot WHERE forecast_version_id = ?", (version["forecast_version_id"],))
        conn.execute("DELETE FROM forecast_version_metrics WHERE forecast_version_id = ?", (version["forecast_version_id"],))
        conn.execute("DELETE FROM source_refreshes WHERE forecast_version_id = ?", (version["forecast_version_id"],))
        _upsert_frame(conn, "forecast_versions", pd.DataFrame([version]), FORECAST_VERSION_COLUMNS)
        _append_frame(conn, "forecast_snapshot", snapshot, FORECAST_SNAPSHOT_COLUMNS)
        _append_frame(conn, "forecast_version_metrics", metrics, FORECAST_VERSION_METRIC_COLUMNS)
        _append_frame(conn, "source_refreshes", source_refreshes, SOURCE_REFRESH_COLUMNS)
        _refresh_forecast_actuals(conn)
        _refresh_performance(conn)
    _write_run_ledger_context(run_path, ledger_dir, version)
    exported = export_ledger(ledger_dir).to_dict() if export else {}
    return LedgerResult(
        str(ledger_dir),
        str(_database_path(ledger_dir)),
        str(ledger_dir / "exports"),
        {"status": "registered", "forecast_version_id": version["forecast_version_id"], "run_dir": str(run_path), "export": exported.get("status")},
    )


def lock_version(
    ledger: str | Path,
    *,
    version_id: str,
    lock_label: str,
    audience: str = "leadership",
    planning_cycle: str = "",
    communication_date: str = "",
    submitted_to: str = "",
    reason: str = "",
    locked_by: str = "",
    notes: str = "",
    export: bool = True,
) -> LedgerResult:
    ledger_dir = _ensure_ledger(ledger)
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
        version_rows = pd.read_sql_query("SELECT * FROM forecast_versions WHERE forecast_version_id = ?", conn, params=(version_id,))
        if version_rows.empty:
            raise ValueError(f"forecast version not found in ledger: {version_id}")
        forecast_key = str(version_rows.iloc[0]["forecast_key"])
        now = _utc_now()
        lock = {
            "lock_id": _hash_key(version_id, forecast_key, lock_label, audience, planning_cycle, communication_date),
            "forecast_version_id": version_id,
            "forecast_key": forecast_key,
            "lock_label": lock_label,
            "audience": audience,
            "planning_cycle": planning_cycle,
            "communication_date": communication_date,
            "submitted_to": submitted_to,
            "lock_reason": reason,
            "locked_at": now,
            "locked_by": locked_by,
            "notes": notes,
        }
        _upsert_frame(conn, "official_forecast_locks", pd.DataFrame([lock]), OFFICIAL_LOCK_COLUMNS)
    exported = export_ledger(ledger_dir).to_dict() if export else {}
    return LedgerResult(
        str(ledger_dir),
        str(_database_path(ledger_dir)),
        str(ledger_dir / "exports"),
        {"status": "locked", "lock_id": lock["lock_id"], "forecast_version_id": version_id, "run_dir": str(version_rows.iloc[0].get("run_dir", "")), "export": exported.get("status")},
    )


def ingest_actuals(
    ledger: str | Path,
    source: str | Path,
    *,
    forecast_key: str,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    sheet: str | int | None = None,
    source_kind: str = "",
    source_id: str = "",
    revision_label: str = "",
    known_as_of: str = "",
    export: bool = True,
) -> LedgerResult:
    ledger_dir = _ensure_ledger(ledger)
    raw = read_tabular_source(source, sheet=sheet)
    actuals = canonicalize_forecast_frame(raw, id_col=id_col, time_col=time_col, target_col=target_col)
    created_at = _utc_now()
    revision_hash = _frame_hash(actuals, ["unique_id", "ds", "y"])
    actual_revision_id = _hash_key(forecast_key, source, revision_label, known_as_of, created_at, revision_hash)
    records = actuals[["unique_id", "ds", "y"]].copy()
    records["actual_revision_id"] = actual_revision_id
    records["forecast_key"] = forecast_key
    records["revision_label"] = revision_label
    records["source_kind"] = source_kind
    records["source_id"] = source_id
    records["source_path"] = str(source)
    records["known_as_of"] = known_as_of
    records["created_at"] = created_at
    records["is_latest"] = 1
    records = records[ACTUAL_REVISION_COLUMNS]
    records = _serialize_frame(records)
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
        _append_frame(conn, "actual_revisions", records, ACTUAL_REVISION_COLUMNS)
        _refresh_latest_actual_flags(conn)
        _refresh_forecast_actuals(conn)
        _refresh_performance(conn)
        _refresh_corrected_actuals(conn, forecast_key)
    exported = export_ledger(ledger_dir).to_dict() if export else {}
    return LedgerResult(
        str(ledger_dir),
        str(_database_path(ledger_dir)),
        str(ledger_dir / "exports"),
        {"status": "actuals_ingested", "actual_revision_id": actual_revision_id, "rows": int(len(records)), "export": exported.get("status")},
    )


def ingest_adjustments(
    ledger: str | Path,
    source: str | Path,
    *,
    forecast_key: str,
    sheet: str | int | None = None,
    export: bool = True,
) -> LedgerResult:
    ledger_dir = _ensure_ledger(ledger)
    raw = read_tabular_source(source, sheet=sheet)
    adjustments = _normalize_adjustments(raw, forecast_key)
    regime_changes = _regime_changes_from_adjustments(adjustments)
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
        _append_frame(conn, "forecast_adjustments", adjustments, ADJUSTMENT_COLUMNS)
        _append_frame(conn, "regime_changes", regime_changes, REGIME_CHANGE_COLUMNS)
        _refresh_corrected_actuals(conn, forecast_key)
    exported = export_ledger(ledger_dir).to_dict() if export else {}
    return LedgerResult(
        str(ledger_dir),
        str(_database_path(ledger_dir)),
        str(ledger_dir / "exports"),
        {"status": "adjustments_ingested", "rows": int(len(adjustments)), "regime_changes": int(len(regime_changes)), "export": exported.get("status")},
    )


def compare_versions(
    ledger: str | Path,
    *,
    forecast_key: str,
    against_lock: str | None = None,
    against_version_id: str | None = None,
    latest_version_id: str | None = None,
    watch_pct: float | None = None,
    call_up_pct: float | None = None,
    call_down_pct: float | None = None,
    export: bool = True,
) -> LedgerResult:
    ledger_dir = _ensure_ledger(ledger)
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
        base_version_id, base_lock_label = _resolve_base_version(conn, forecast_key, against_lock=against_lock, against_version_id=against_version_id)
        comparison_version_id = latest_version_id or _latest_version_id(conn, forecast_key, exclude_version_id=base_version_id)
        if not comparison_version_id:
            raise ValueError(f"no comparison version found for forecast_key={forecast_key!r}")
        deltas = _build_delta_frame(
            conn,
            forecast_key=forecast_key,
            base_version_id=base_version_id,
            comparison_version_id=comparison_version_id,
            base_lock_label=base_lock_label,
            watch_pct=watch_pct,
            call_up_pct=call_up_pct,
            call_down_pct=call_down_pct,
        )
        comparison_id = deltas["comparison_id"].iloc[0] if not deltas.empty else _hash_key(forecast_key, base_version_id, comparison_version_id)
        conn.execute("DELETE FROM forecast_version_deltas WHERE comparison_id = ?", (comparison_id,))
        _append_frame(conn, "forecast_version_deltas", deltas, DELTA_COLUMNS)
    exported = export_ledger(ledger_dir).to_dict() if export else {}
    return LedgerResult(
        str(ledger_dir),
        str(_database_path(ledger_dir)),
        str(ledger_dir / "exports"),
        {"status": "compared", "comparison_id": comparison_id, "rows": int(len(deltas)), "export": exported.get("status")},
    )


def export_ledger(ledger: str | Path, output: str | Path | None = None) -> LedgerResult:
    ledger_dir = _ensure_ledger(ledger)
    output_dir = Path(output) if output is not None else ledger_dir / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    with _connect(ledger_dir) as conn:
        _create_schema(conn)
        for table, filename in EXPORT_TABLES.items():
            frame = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            path = output_dir / filename
            frame.to_csv(path, index=False)
            written[table] = str(path)
    return LedgerResult(str(ledger_dir), str(_database_path(ledger_dir)), str(output_dir), {"status": "exported", "files": written})


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    _create_table(conn, "forecast_versions", FORECAST_VERSION_COLUMNS, primary_key="forecast_version_id")
    _create_table(conn, "forecast_snapshot", FORECAST_SNAPSHOT_COLUMNS)
    _create_table(conn, "forecast_version_metrics", FORECAST_VERSION_METRIC_COLUMNS)
    _create_table(conn, "source_refreshes", SOURCE_REFRESH_COLUMNS, primary_key="source_refresh_id")
    _create_table(conn, "official_forecast_locks", OFFICIAL_LOCK_COLUMNS, primary_key="lock_id")
    _create_table(conn, "actual_revisions", ACTUAL_REVISION_COLUMNS)
    _create_table(conn, "forecast_actuals", FORECAST_ACTUAL_COLUMNS)
    _create_table(conn, "forecast_performance", FORECAST_PERFORMANCE_COLUMNS)
    _create_table(conn, "forecast_adjustments", ADJUSTMENT_COLUMNS, primary_key="adjustment_id")
    _create_table(conn, "corrected_actuals", CORRECTED_ACTUAL_COLUMNS)
    _create_table(conn, "regime_changes", REGIME_CHANGE_COLUMNS, primary_key="regime_change_id")
    _create_table(conn, "forecast_version_deltas", DELTA_COLUMNS)
    conn.execute(
        "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES (?, ?)",
        ("schema_version", LEDGER_SCHEMA_VERSION),
    )


def _create_table(conn: sqlite3.Connection, name: str, columns: list[str], *, primary_key: str | None = None) -> None:
    definitions = []
    for column in columns:
        definition = f"{column} TEXT"
        if column == primary_key:
            definition += " PRIMARY KEY"
        definitions.append(definition)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(definitions)})")


def _version_record(
    run_path: Path,
    manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    forecast_key: str,
    version_label: str,
    created_by: str,
    notes: str,
) -> dict[str, Any]:
    repro = manifest.get("reproducibility", {})
    spec = manifest.get("spec", {})
    profile = manifest.get("profile", {})
    forecast_origin = str(repro.get("forecast_origin") or profile.get("end") or "")
    horizon = spec.get("horizon", "")
    freq = spec.get("freq") or repro.get("frequency") or profile.get("freq") or ""
    model_policy = spec.get("model_policy", "")
    data_hash = str(repro.get("data_hash_sha256") or "")
    version_id = _hash_key(forecast_key, version_label, forecast_origin, horizon, freq, model_policy, data_hash)
    return {
        "forecast_version_id": version_id,
        "forecast_key": forecast_key,
        "version_label": version_label,
        "run_dir": str(run_path),
        "forecast_origin": forecast_origin,
        "horizon": horizon,
        "freq": freq,
        "model_policy": model_policy,
        "data_hash_sha256": data_hash,
        "git_sha": repro.get("git_sha"),
        "created_at": _utc_now(),
        "created_by": created_by,
        "notes": notes or diagnostics.get("status", ""),
        "manifest_path": str(run_path / "manifest.json"),
        "diagnostics_path": str(run_path / "diagnostics.json") if (run_path / "diagnostics.json").exists() else "",
    }


def _forecast_snapshot_frame(run_path: Path, version: dict[str, Any]) -> pd.DataFrame:
    source_path = run_path / "forecast_long.csv"
    if not source_path.exists():
        source_path = run_path / "forecast.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"forecast.csv or forecast_long.csv is required to register a run: {run_path}")
    frame = pd.read_csv(source_path)
    out = pd.DataFrame(index=frame.index)
    out["forecast_version_id"] = version["forecast_version_id"]
    out["forecast_key"] = version["forecast_key"]
    out["version_label"] = version["version_label"]
    out["forecast_origin"] = version["forecast_origin"]
    for column in FORECAST_SNAPSHOT_COLUMNS:
        if column in out.columns:
            continue
        if column in frame.columns:
            out[column] = frame[column]
        elif column == "is_selected_model":
            out[column] = 1 if source_path.name == "forecast.csv" else 0
        elif column == "yhat_scenario" and "scenario_yhat" in frame.columns:
            out[column] = frame["scenario_yhat"]
        else:
            out[column] = None
    return _serialize_frame(out[FORECAST_SNAPSHOT_COLUMNS])


def _forecast_version_metrics_frame(run_path: Path, version: dict[str, Any], diagnostics: dict[str, Any]) -> pd.DataFrame:
    series = _read_csv_if_exists(run_path / "series_summary.csv")
    trust = _read_csv_if_exists(run_path / "trust_summary.csv")
    selection = _read_csv_if_exists(run_path / "audit" / "model_selection.csv")
    base_ids = _unique_ids([series, trust, selection])
    if not base_ids:
        snapshot = _read_csv_if_exists(run_path / "forecast.csv")
        base_ids = sorted(snapshot["unique_id"].astype(str).unique()) if not snapshot.empty and "unique_id" in snapshot.columns else []
    out = pd.DataFrame({"unique_id": base_ids})
    if not series.empty and "unique_id" in series.columns:
        out = out.merge(series, on="unique_id", how="left", suffixes=("", "_series"))
    if not trust.empty and "unique_id" in trust.columns:
        out = out.merge(trust, on="unique_id", how="left", suffixes=("", "_trust"))
    if not selection.empty and "unique_id" in selection.columns:
        out = out.merge(selection, on="unique_id", how="left", suffixes=("", "_selection"))
    headlines = _headline_by_series(diagnostics)
    metrics = pd.DataFrame(index=out.index)
    metrics["forecast_version_id"] = version["forecast_version_id"]
    metrics["forecast_key"] = version["forecast_key"]
    metrics["unique_id"] = out.get("unique_id", pd.Series(dtype=str)).astype(str)
    for column in FORECAST_VERSION_METRIC_COLUMNS:
        if column in {"forecast_version_id", "forecast_key", "unique_id"}:
            continue
        if column == "executive_headline":
            metrics[column] = metrics["unique_id"].map(headlines).fillna(str(diagnostics.get("executive_headline", {}).get("paragraph", "")))
        elif column == "selected_model":
            metrics[column] = _first_existing(out, ["selected_model", "model"])
        else:
            metrics[column] = out[column] if column in out.columns else None
    return _serialize_frame(metrics[FORECAST_VERSION_METRIC_COLUMNS])


def _source_refresh_frame(
    run_path: Path,
    version: dict[str, Any],
    *,
    source_metadata_path: str | Path | None,
    source_kind: str,
) -> pd.DataFrame:
    paths: list[Path] = []
    if source_metadata_path:
        paths.append(Path(source_metadata_path))
    paths.extend(sorted(run_path.glob("*.source.json")))
    rows = []
    for path in dict.fromkeys(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        metadata = _read_json(path)
        columns = metadata.get("columns", {})
        rows.append(
            {
                "source_refresh_id": _hash_key(version["forecast_version_id"], path, metadata.get("input"), metadata.get("output")),
                "forecast_version_id": version["forecast_version_id"],
                "forecast_key": version["forecast_key"],
                "source_kind": metadata.get("source_kind", source_kind),
                "source_input": metadata.get("input", ""),
                "source_output": metadata.get("output", ""),
                "query_file": metadata.get("query_file", ""),
                "metadata_file": str(path),
                "id_col": columns.get("id_col", ""),
                "time_col": columns.get("time_col", ""),
                "target_col": columns.get("target_col", ""),
                "id_value": columns.get("id_value", ""),
                "rows": metadata.get("rows", ""),
                "series_count": metadata.get("series_count", ""),
                "source_start": metadata.get("start", ""),
                "source_end": metadata.get("end", ""),
                "created_at": _utc_now(),
            }
        )
    if not rows and source_kind:
        rows.append(
            {
                "source_refresh_id": _hash_key(version["forecast_version_id"], source_kind),
                "forecast_version_id": version["forecast_version_id"],
                "forecast_key": version["forecast_key"],
                "source_kind": source_kind,
                "created_at": _utc_now(),
            }
        )
    return _serialize_frame(pd.DataFrame(rows, columns=SOURCE_REFRESH_COLUMNS))


def _normalize_adjustments(frame: pd.DataFrame, forecast_key: str) -> pd.DataFrame:
    missing = sorted(REQUIRED_ADJUSTMENT_METADATA - set(frame.columns))
    if missing:
        raise ValueError(f"adjustment file is missing required metadata columns: {missing}")
    out = frame.copy()
    if "adjustment_type" not in out.columns:
        raise ValueError("adjustment file is missing required column: adjustment_type")
    invalid = sorted(set(out["adjustment_type"].astype(str).str.strip()) - ADJUSTMENT_TYPES)
    if invalid:
        raise ValueError(f"unsupported adjustment_type values: {invalid}")
    for column in ADJUSTMENT_COLUMNS:
        if column not in out.columns:
            out[column] = None
    out["forecast_key"] = forecast_key
    out["created_at"] = _utc_now()
    out["adjustment_id"] = out.apply(
        lambda row: _hash_key(
            forecast_key,
            row.get("unique_id"),
            row.get("start_ds"),
            row.get("end_ds"),
            row.get("adjustment_type"),
            row.get("reason"),
            row.get("known_as_of"),
            row.get("adjustment_value"),
            row.get("adjusted_y"),
            row.get("factor"),
        ),
        axis=1,
    )
    return _serialize_frame(out[ADJUSTMENT_COLUMNS])


def _regime_changes_from_adjustments(adjustments: pd.DataFrame) -> pd.DataFrame:
    if adjustments.empty:
        return pd.DataFrame(columns=REGIME_CHANGE_COLUMNS)
    subset = adjustments[adjustments["adjustment_type"].astype(str).eq("regime_change")].copy()
    if subset.empty:
        return pd.DataFrame(columns=REGIME_CHANGE_COLUMNS)
    out = pd.DataFrame()
    out["regime_change_id"] = subset["adjustment_id"]
    for column in REGIME_CHANGE_COLUMNS:
        if column == "regime_change_id":
            continue
        out[column] = subset[column] if column in subset.columns else None
    return _serialize_frame(out[REGIME_CHANGE_COLUMNS])


def _refresh_latest_actual_flags(conn: sqlite3.Connection) -> None:
    actuals = pd.read_sql_query("SELECT * FROM actual_revisions", conn)
    if actuals.empty:
        return
    actuals["is_latest"] = 0
    actuals["_created_at_sort"] = pd.to_datetime(actuals["created_at"], errors="coerce")
    idx = actuals.sort_values("_created_at_sort").groupby(["forecast_key", "unique_id", "ds"], dropna=False).tail(1).index
    actuals.loc[idx, "is_latest"] = 1
    actuals = actuals.drop(columns=["_created_at_sort"])
    _replace_table(conn, "actual_revisions", actuals)


def _refresh_forecast_actuals(conn: sqlite3.Connection) -> None:
    snapshots = pd.read_sql_query("SELECT * FROM forecast_snapshot", conn)
    actuals = pd.read_sql_query("SELECT * FROM actual_revisions WHERE is_latest = 1", conn)
    if snapshots.empty or actuals.empty:
        _replace_table(conn, "forecast_actuals", pd.DataFrame(columns=FORECAST_ACTUAL_COLUMNS))
        return
    selected = snapshots[_bool_series(snapshots.get("is_selected_model"), default=True)].copy()
    if selected.empty:
        selected = snapshots.copy()
    merged = selected.merge(actuals[["forecast_key", "unique_id", "ds", "y", "actual_revision_id"]], on=["forecast_key", "unique_id", "ds"], how="inner")
    if merged.empty:
        _replace_table(conn, "forecast_actuals", pd.DataFrame(columns=FORECAST_ACTUAL_COLUMNS))
        return
    yhat = pd.to_numeric(merged["yhat"], errors="coerce")
    actual = pd.to_numeric(merged["y"], errors="coerce")
    out = pd.DataFrame(index=merged.index)
    for column in ["forecast_version_id", "forecast_key", "version_label", "unique_id", "ds", "model", "forecast_origin", "horizon_step"]:
        out[column] = merged.get(column)
    out["yhat"] = yhat
    out["y_actual"] = actual
    out["error"] = actual - yhat
    out["abs_error"] = (actual - yhat).abs()
    out["pct_error"] = (actual - yhat) / actual.abs().where(actual.abs() > 0)
    out["interval_80_hit"] = _interval_hit(merged, actual, 80)
    out["interval_95_hit"] = _interval_hit(merged, actual, 95)
    out["actual_revision_id"] = merged["actual_revision_id"]
    _replace_table(conn, "forecast_actuals", _serialize_frame(out[FORECAST_ACTUAL_COLUMNS]))


def _refresh_performance(conn: sqlite3.Connection) -> None:
    actuals = pd.read_sql_query("SELECT * FROM forecast_actuals", conn)
    if actuals.empty:
        _replace_table(conn, "forecast_performance", pd.DataFrame(columns=FORECAST_PERFORMANCE_COLUMNS))
        return
    actuals["error"] = pd.to_numeric(actuals["error"], errors="coerce")
    actuals["abs_error"] = pd.to_numeric(actuals["abs_error"], errors="coerce")
    actuals["y_actual"] = pd.to_numeric(actuals["y_actual"], errors="coerce")
    rows = []
    for keys, group in actuals.groupby(["forecast_version_id", "forecast_key", "version_label", "unique_id", "model"], dropna=False):
        error = group["error"].dropna()
        abs_error = group["abs_error"].dropna()
        denom = group["y_actual"].abs().sum()
        row = dict(zip(["forecast_version_id", "forecast_key", "version_label", "unique_id", "model"], keys))
        row["observed_periods"] = int(len(group))
        row["mae"] = float(abs_error.mean()) if not abs_error.empty else None
        row["rmse"] = float((error.pow(2).mean()) ** 0.5) if not error.empty else None
        row["bias"] = float(error.mean()) if not error.empty else None
        row["wape"] = float(abs_error.sum() / denom) if denom and pd.notna(denom) else None
        for level in (80, 95):
            hits = pd.to_numeric(group[f"interval_{level}_hit"], errors="coerce").dropna()
            row[f"interval_{level}_coverage"] = float(hits.mean()) if not hits.empty else None
        rows.append(row)
    _replace_table(conn, "forecast_performance", _serialize_frame(pd.DataFrame(rows, columns=FORECAST_PERFORMANCE_COLUMNS)))


def _refresh_corrected_actuals(conn: sqlite3.Connection, forecast_key: str) -> None:
    actuals = pd.read_sql_query("SELECT * FROM actual_revisions WHERE forecast_key = ? AND is_latest = 1", conn, params=(forecast_key,))
    adjustments = pd.read_sql_query("SELECT * FROM forecast_adjustments WHERE forecast_key = ?", conn, params=(forecast_key,))
    existing = pd.read_sql_query("SELECT * FROM corrected_actuals WHERE forecast_key != ?", conn, params=(forecast_key,))
    if actuals.empty:
        _replace_table(conn, "corrected_actuals", existing if not existing.empty else pd.DataFrame(columns=CORRECTED_ACTUAL_COLUMNS))
        return
    out = actuals[["forecast_key", "unique_id", "ds", "y", "actual_revision_id"]].copy()
    out = out.rename(columns={"y": "raw_y"})
    out["corrected_y"] = pd.to_numeric(out["raw_y"], errors="coerce")
    out["normalized_y"] = out["corrected_y"]
    out["is_excluded"] = 0
    out["applied_adjustment_ids"] = ""
    approved = adjustments[adjustments["approval_status"].astype(str).str.lower().isin({"approved", "accepted", "active"})].copy()
    if not approved.empty:
        out["_ds"] = pd.to_datetime(out["ds"], errors="coerce")
        for _, adj in approved.iterrows():
            mask = _adjustment_mask(out, adj)
            if not mask.any():
                continue
            adj_id = str(adj.get("adjustment_id", ""))
            adj_type = str(adj.get("adjustment_type", ""))
            value = _numeric_first(adj, ["adjustment_value", "adjusted_y", "factor", "conversion_factor"])
            if adj_type == "replacement":
                out.loc[mask, "corrected_y"] = _numeric_first(adj, ["adjusted_y", "adjustment_value"])
            elif adj_type == "additive_delta":
                out.loc[mask, "corrected_y"] = out.loc[mask, "corrected_y"] + (value or 0.0)
            elif adj_type == "multiplicative_factor" and value is not None:
                out.loc[mask, "corrected_y"] = out.loc[mask, "corrected_y"] * value
            elif adj_type == "exclude":
                out.loc[mask, "is_excluded"] = 1
            elif adj_type == "business_model_normalization" and value not in {None, 0}:
                out.loc[mask, "normalized_y"] = out.loc[mask, "corrected_y"] / value
            elif adj_type in {"fill", "interpolate"}:
                out.loc[mask, "corrected_y"] = pd.NA
            out.loc[mask, "applied_adjustment_ids"] = out.loc[mask, "applied_adjustment_ids"].apply(lambda text: _append_token(text, adj_id))
        out["corrected_y"] = out.groupby("unique_id")["corrected_y"].transform(lambda values: values.interpolate(limit_direction="both").ffill().bfill())
        out["normalized_y"] = out["normalized_y"].where(out["normalized_y"].notna(), out["corrected_y"])
        out = out.drop(columns=["_ds"])
    updated = pd.concat([existing, out[CORRECTED_ACTUAL_COLUMNS]], ignore_index=True)
    _replace_table(conn, "corrected_actuals", _serialize_frame(updated[CORRECTED_ACTUAL_COLUMNS]))


def _build_delta_frame(
    conn: sqlite3.Connection,
    *,
    forecast_key: str,
    base_version_id: str,
    comparison_version_id: str,
    base_lock_label: str,
    watch_pct: float | None,
    call_up_pct: float | None,
    call_down_pct: float | None,
) -> pd.DataFrame:
    snapshot = pd.read_sql_query("SELECT * FROM forecast_snapshot WHERE forecast_key = ?", conn, params=(forecast_key,))
    if snapshot.empty:
        return pd.DataFrame(columns=DELTA_COLUMNS)
    selected = snapshot[_bool_series(snapshot.get("is_selected_model"), default=True)].copy()
    if selected.empty:
        selected = snapshot.copy()
    base = selected[selected["forecast_version_id"] == base_version_id].copy()
    comp = selected[selected["forecast_version_id"] == comparison_version_id].copy()
    if base.empty:
        raise ValueError(f"base forecast version has no selected snapshot rows: {base_version_id}")
    if comp.empty:
        raise ValueError(f"comparison forecast version has no selected snapshot rows: {comparison_version_id}")
    merged = base[["unique_id", "ds", "yhat"]].rename(columns={"yhat": "base_yhat"}).merge(
        comp[["unique_id", "ds", "yhat"]].rename(columns={"yhat": "comparison_yhat"}),
        on=["unique_id", "ds"],
        how="inner",
    )
    actuals = pd.read_sql_query(
        "SELECT forecast_key, unique_id, ds, y FROM actual_revisions WHERE forecast_key = ? AND is_latest = 1",
        conn,
        params=(forecast_key,),
    )
    if not actuals.empty:
        merged = merged.merge(actuals[["unique_id", "ds", "y"]].rename(columns={"y": "actual_y"}), on=["unique_id", "ds"], how="left")
    else:
        merged["actual_y"] = None
    base_yhat = pd.to_numeric(merged["base_yhat"], errors="coerce")
    comp_yhat = pd.to_numeric(merged["comparison_yhat"], errors="coerce")
    actual_y = pd.to_numeric(merged["actual_y"], errors="coerce")
    comparison_id = _hash_key(forecast_key, base_version_id, comparison_version_id, base_lock_label)
    out = pd.DataFrame(index=merged.index)
    out["comparison_id"] = comparison_id
    out["forecast_key"] = forecast_key
    out["base_version_id"] = base_version_id
    out["comparison_version_id"] = comparison_version_id
    out["base_lock_label"] = base_lock_label
    out["unique_id"] = merged["unique_id"]
    out["ds"] = merged["ds"]
    out["base_yhat"] = base_yhat
    out["comparison_yhat"] = comp_yhat
    out["comparison_minus_base"] = comp_yhat - base_yhat
    out["comparison_minus_base_pct"] = (comp_yhat - base_yhat) / base_yhat.abs().where(base_yhat.abs() > 0)
    out["actual_y"] = actual_y
    out["actual_minus_base"] = actual_y - base_yhat
    out["actual_minus_comparison"] = actual_y - comp_yhat
    out["status_label"] = _status_labels(out["comparison_minus_base_pct"], watch_pct=watch_pct, call_up_pct=call_up_pct, call_down_pct=call_down_pct)
    return _serialize_frame(out[DELTA_COLUMNS])


def _resolve_base_version(
    conn: sqlite3.Connection,
    forecast_key: str,
    *,
    against_lock: str | None,
    against_version_id: str | None,
) -> tuple[str, str]:
    if against_version_id:
        return against_version_id, ""
    locks = pd.read_sql_query("SELECT * FROM official_forecast_locks WHERE forecast_key = ?", conn, params=(forecast_key,))
    if locks.empty:
        raise ValueError(f"no official locks found for forecast_key={forecast_key!r}; pass --against-version-id or create a lock")
    if against_lock:
        locks = locks[locks["lock_label"].astype(str).eq(str(against_lock))]
        if locks.empty:
            raise ValueError(f"official lock not found for forecast_key={forecast_key!r}: {against_lock}")
    locks["_locked_at_sort"] = pd.to_datetime(locks["locked_at"], errors="coerce")
    row = locks.sort_values("_locked_at_sort").iloc[-1]
    return str(row["forecast_version_id"]), str(row.get("lock_label", ""))


def _latest_version_id(conn: sqlite3.Connection, forecast_key: str, *, exclude_version_id: str | None = None) -> str:
    versions = pd.read_sql_query("SELECT * FROM forecast_versions WHERE forecast_key = ?", conn, params=(forecast_key,))
    if exclude_version_id:
        versions = versions[versions["forecast_version_id"].astype(str).ne(str(exclude_version_id))]
    if versions.empty:
        return ""
    versions["_created_at_sort"] = pd.to_datetime(versions["created_at"], errors="coerce")
    return str(versions.sort_values("_created_at_sort").iloc[-1]["forecast_version_id"])


def _status_labels(
    pct: pd.Series,
    *,
    watch_pct: float | None,
    call_up_pct: float | None,
    call_down_pct: float | None,
) -> pd.Series:
    if watch_pct is None and call_up_pct is None and call_down_pct is None:
        return pd.Series(["threshold_not_configured"] * len(pct), index=pct.index)
    labels = pd.Series(["on_track"] * len(pct), index=pct.index)
    if watch_pct is not None:
        labels[pct.abs() >= abs(watch_pct)] = "watch"
    if call_up_pct is not None:
        labels[pct >= abs(call_up_pct)] = "call_up"
    if call_down_pct is not None:
        labels[pct <= -abs(call_down_pct)] = "call_down"
    return labels


def _connect(ledger_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_database_path(ledger_dir))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_ledger(ledger: str | Path) -> Path:
    result = init_ledger(ledger)
    return Path(result.ledger_path)


def _ledger_dir(ledger: str | Path) -> Path:
    path = Path(ledger)
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        return path.parent
    return path


def _database_path(ledger_dir: Path) -> Path:
    return ledger_dir / "ledger.sqlite"


def _append_frame(conn: sqlite3.Connection, table: str, frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        return
    frame = _serialize_frame(_ensure_columns(frame, columns))
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)
    rows = [_sqlite_row(row) for row in frame[columns].itertuples(index=False, name=None)]
    conn.executemany(f"INSERT OR REPLACE INTO {table} ({column_sql}) VALUES ({placeholders})", rows)


def _upsert_frame(conn: sqlite3.Connection, table: str, frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        return
    primary_key = columns[0]
    for value in frame[primary_key].astype(str):
        conn.execute(f"DELETE FROM {table} WHERE {primary_key} = ?", (value,))
    _append_frame(conn, table, frame, columns)


def _replace_table(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    conn.execute(f"DELETE FROM {table}")
    if not frame.empty:
        _append_frame(conn, table, frame, list(frame.columns))


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]


def _serialize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.where(pd.notna(out), None)


def _sqlite_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    values: list[Any] = []
    for value in row:
        if pd.isna(value):
            values.append(None)
        else:
            values.append(value)
    return tuple(values)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _unique_ids(frames: list[pd.DataFrame]) -> list[str]:
    values: set[str] = set()
    for frame in frames:
        if not frame.empty and "unique_id" in frame.columns:
            values.update(frame["unique_id"].astype(str).dropna().tolist())
    return sorted(values)


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([None] * len(frame), index=frame.index)


def _headline_by_series(diagnostics: dict[str, Any]) -> dict[str, str]:
    headline = diagnostics.get("executive_headline", {})
    series = headline.get("series", []) if isinstance(headline, dict) else []
    if not isinstance(series, list):
        return {}
    return {str(row.get("unique_id")): str(row.get("paragraph")) for row in series if row.get("unique_id") and row.get("paragraph")}


def _bool_series(values: pd.Series | None, *, default: bool) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    normalized = values.astype(str).str.lower()
    truthy = normalized.isin({"1", "true", "yes", "y"})
    falsy = normalized.isin({"0", "false", "no", "n"})
    return truthy | (~truthy & ~falsy & default)


def _interval_hit(frame: pd.DataFrame, actual: pd.Series, level: int) -> pd.Series:
    lo_col = f"yhat_lo_{level}"
    hi_col = f"yhat_hi_{level}"
    if lo_col not in frame.columns or hi_col not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index)
    lo = pd.to_numeric(frame[lo_col], errors="coerce")
    hi = pd.to_numeric(frame[hi_col], errors="coerce")
    hit = (actual >= lo) & (actual <= hi)
    return hit.where(lo.notna() & hi.notna(), None)


def _adjustment_mask(actuals: pd.DataFrame, adjustment: pd.Series) -> pd.Series:
    mask = pd.Series([True] * len(actuals), index=actuals.index)
    unique_id = str(adjustment.get("unique_id", "") or "").strip()
    if unique_id:
        mask &= actuals["unique_id"].astype(str).eq(unique_id)
    start = pd.to_datetime(adjustment.get("start_ds"), errors="coerce")
    end = pd.to_datetime(adjustment.get("end_ds"), errors="coerce")
    if pd.notna(start):
        mask &= actuals["_ds"] >= start
    if pd.notna(end):
        mask &= actuals["_ds"] <= end
    return mask


def _numeric_first(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return None


def _append_token(text: Any, token: str) -> str:
    tokens = [part for part in str(text or "").split(";") if part]
    if token and token not in tokens:
        tokens.append(token)
    return ";".join(tokens)


def _write_run_ledger_context(run_path: Path, ledger_dir: Path, version: dict[str, Any]) -> None:
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "ledger_path": str(ledger_dir),
        "database_path": str(_database_path(ledger_dir)),
        "exports_path": str(ledger_dir / "exports"),
        "forecast_version_id": version["forecast_version_id"],
        "forecast_key": version["forecast_key"],
        "version_label": version["version_label"],
    }
    (run_path / "ledger_context.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return ""
    canonical = frame[available].copy()
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = pd.to_datetime(canonical[column], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    payload = canonical.sort_values(available).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_key(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
