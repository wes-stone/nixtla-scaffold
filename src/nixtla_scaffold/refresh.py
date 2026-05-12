from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REFRESH_DELTA_COLUMNS = [
    "delta_type",
    "unique_id",
    "ds",
    "field",
    "previous_value",
    "current_value",
    "absolute_delta",
    "pct_delta",
    "status",
    "note",
]


def write_refresh_artifacts(previous_run: str | Path, current_run: str | Path) -> dict[str, Any]:
    """Write compact refresh delta artifacts beside a newly generated run."""

    previous = Path(previous_run)
    current = Path(current_run)
    if not (previous / "manifest.json").exists():
        raise FileNotFoundError(previous / "manifest.json")
    if not (current / "manifest.json").exists():
        raise FileNotFoundError(current / "manifest.json")

    appendix = current / "appendix"
    appendix.mkdir(parents=True, exist_ok=True)
    previous_manifest = _read_json(previous / "manifest.json")
    current_manifest = _read_json(current / "manifest.json")
    delta = build_refresh_delta(previous, current, previous_manifest=previous_manifest, current_manifest=current_manifest)
    delta_path = appendix / "refresh_delta.csv"
    delta.to_csv(delta_path, index=False)

    payload = {
        "schema_version": "nixtla_scaffold.refresh.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "previous_run": str(previous),
        "current_run": str(current),
        "delta_rows": int(len(delta)),
        "delta_types": sorted(delta["delta_type"].dropna().astype(str).unique().tolist()) if not delta.empty else [],
        "artifacts": {"refresh_delta": "appendix/refresh_delta.csv"},
    }
    manifest_path = current / "refresh_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    _add_refresh_outputs_to_manifest(current / "manifest.json")
    return payload | {"paths": {"refresh_manifest": str(manifest_path), "refresh_delta": str(delta_path)}}


def build_refresh_delta(
    previous_run: str | Path,
    current_run: str | Path,
    *,
    previous_manifest: dict[str, Any] | None = None,
    current_manifest: dict[str, Any] | None = None,
) -> pd.DataFrame:
    previous = Path(previous_run)
    current = Path(current_run)
    previous_manifest = previous_manifest or _read_json(previous / "manifest.json")
    current_manifest = current_manifest or _read_json(current / "manifest.json")
    rows: list[dict[str, Any]] = []
    rows.extend(_spec_delta_rows(previous_manifest.get("spec", {}), current_manifest.get("spec", {})))
    rows.extend(_forecast_delta_rows(_read_csv(previous / "forecast.csv"), _read_csv(current / "forecast.csv")))
    rows.extend(
        _field_delta_rows(
            _read_csv(previous / "audit" / "model_selection.csv"),
            _read_csv(current / "audit" / "model_selection.csv"),
            fields=("selected_model", "selection_horizon", "cv_windows", "cv_horizon_matches_requested"),
            delta_type="model_selection_change",
            note="Selected-model or validation-contract change versus previous run.",
        )
    )
    rows.extend(
        _field_delta_rows(
            _read_csv(previous / "appendix" / "trust_summary.csv"),
            _read_csv(current / "appendix" / "trust_summary.csv"),
            fields=("trust_level", "trust_score_0_100", "horizon_trust_state", "full_horizon_claim_allowed"),
            delta_type="trust_change",
            note="Trust/readiness change versus previous run.",
        )
    )
    if not rows:
        return pd.DataFrame(columns=REFRESH_DELTA_COLUMNS)
    return pd.DataFrame(rows, columns=REFRESH_DELTA_COLUMNS)


def _spec_delta_rows(previous_spec: dict[str, Any], current_spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_flat = _flatten_dict(previous_spec)
    current_flat = _flatten_dict(current_spec)
    for key in sorted(set(previous_flat) | set(current_flat)):
        prev = previous_flat.get(key)
        curr = current_flat.get(key)
        if str(prev) == str(curr):
            continue
        rows.append(
            _delta_row(
                delta_type="spec_change",
                field=key,
                previous_value=prev,
                current_value=curr,
                status="changed",
                note="Explicit refresh override or changed persisted setup.",
            )
        )
    return rows


def _forecast_delta_rows(previous: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    if previous.empty or current.empty or not {"unique_id", "ds", "yhat"}.issubset(previous.columns) or not {"unique_id", "ds", "yhat"}.issubset(current.columns):
        return []
    prev = previous.copy()
    curr = current.copy()
    prev["unique_id"] = prev["unique_id"].astype(str)
    curr["unique_id"] = curr["unique_id"].astype(str)
    prev["ds"] = pd.to_datetime(prev["ds"], errors="coerce")
    curr["ds"] = pd.to_datetime(curr["ds"], errors="coerce")
    merged = prev[["unique_id", "ds", "yhat"]].rename(columns={"yhat": "previous_value"}).merge(
        curr[["unique_id", "ds", "yhat"]].rename(columns={"yhat": "current_value"}),
        on=["unique_id", "ds"],
        how="outer",
        indicator=True,
    )
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        status = "matched"
        if row.get("_merge") == "left_only":
            status = "missing_in_current"
        elif row.get("_merge") == "right_only":
            status = "new_in_current"
        prev_value = _numeric_or_none(row.get("previous_value"))
        curr_value = _numeric_or_none(row.get("current_value"))
        absolute_delta = curr_value - prev_value if prev_value is not None and curr_value is not None else None
        pct_delta = absolute_delta / prev_value if absolute_delta is not None and prev_value not in (None, 0) else None
        rows.append(
            _delta_row(
                delta_type="forecast_yhat_change",
                unique_id=row.get("unique_id"),
                ds=row.get("ds"),
                field="yhat",
                previous_value=row.get("previous_value"),
                current_value=row.get("current_value"),
                absolute_delta=absolute_delta,
                pct_delta=pct_delta,
                status=status,
                note="Selected forecast point change versus previous run.",
            )
        )
    return rows


def _field_delta_rows(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    fields: tuple[str, ...],
    delta_type: str,
    note: str,
) -> list[dict[str, Any]]:
    if previous.empty or current.empty or "unique_id" not in previous.columns or "unique_id" not in current.columns:
        return []
    prev = previous.copy()
    curr = current.copy()
    prev["unique_id"] = prev["unique_id"].astype(str)
    curr["unique_id"] = curr["unique_id"].astype(str)
    keep = ["unique_id", *[field for field in fields if field in prev.columns]]
    prev = prev[keep].drop_duplicates("unique_id", keep="last")
    keep = ["unique_id", *[field for field in fields if field in curr.columns]]
    curr = curr[keep].drop_duplicates("unique_id", keep="last")
    merged = prev.merge(curr, on="unique_id", how="outer", suffixes=("_previous", "_current"), indicator=True)
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        uid = row.get("unique_id")
        if row.get("_merge") != "both":
            rows.append(
                _delta_row(
                    delta_type=delta_type,
                    unique_id=uid,
                    field="unique_id",
                    previous_value="present" if row.get("_merge") == "left_only" else "",
                    current_value="present" if row.get("_merge") == "right_only" else "",
                    status="missing_in_current" if row.get("_merge") == "left_only" else "new_in_current",
                    note=note,
                )
            )
            continue
        for field in fields:
            prev_value = row.get(f"{field}_previous")
            curr_value = row.get(f"{field}_current")
            if str(prev_value) == str(curr_value):
                continue
            prev_num = _numeric_or_none(prev_value)
            curr_num = _numeric_or_none(curr_value)
            absolute_delta = curr_num - prev_num if prev_num is not None and curr_num is not None else None
            pct_delta = absolute_delta / prev_num if absolute_delta is not None and prev_num not in (None, 0) else None
            rows.append(
                _delta_row(
                    delta_type=delta_type,
                    unique_id=uid,
                    field=field,
                    previous_value=prev_value,
                    current_value=curr_value,
                    absolute_delta=absolute_delta,
                    pct_delta=pct_delta,
                    status="changed",
                    note=note,
                )
            )
    return rows


def _delta_row(
    *,
    delta_type: str,
    unique_id: Any = "",
    ds: Any = "",
    field: str,
    previous_value: Any = "",
    current_value: Any = "",
    absolute_delta: float | None = None,
    pct_delta: float | None = None,
    status: str,
    note: str,
) -> dict[str, Any]:
    return {
        "delta_type": delta_type,
        "unique_id": "" if unique_id is None else str(unique_id),
        "ds": "" if ds is None or pd.isna(ds) else str(pd.Timestamp(ds).date()) if _is_datetime_like(ds) else str(ds),
        "field": field,
        "previous_value": "" if previous_value is None or pd.isna(previous_value) else previous_value,
        "current_value": "" if current_value is None or pd.isna(current_value) else current_value,
        "absolute_delta": absolute_delta,
        "pct_delta": pct_delta,
        "status": status,
        "note": note,
    }


def _flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_dict(nested, child))
        return out
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, default=str)}
    return {prefix: value}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_datetime_like(value: Any) -> bool:
    try:
        pd.Timestamp(value)
        return not isinstance(value, (int, float))
    except (TypeError, ValueError):
        return False


def _add_refresh_outputs_to_manifest(manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    outputs = manifest.setdefault("outputs", {})
    outputs["refresh_manifest"] = "refresh_manifest.json"
    outputs["refresh_delta"] = "appendix/refresh_delta.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
