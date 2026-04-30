from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from nixtla_scaffold.data import read_tabular_source
from nixtla_scaffold.schema import DriverEvent, ForecastRun, ForecastSpec, KnownFutureRegressor


SCENARIO_ASSUMPTION_COLUMNS = [
    "assumption_type",
    "name",
    "start",
    "end",
    "effect",
    "magnitude",
    "confidence",
    "effective_magnitude",
    "affected_unique_ids",
    "notes",
]

SCENARIO_FORECAST_COLUMNS = [
    "unique_id",
    "ds",
    "model",
    "horizon_step",
    "yhat",
    "yhat_scenario",
    "event_adjustment",
    "event_names",
    "row_horizon_status",
    "horizon_trust_state",
    "validated_through_horizon",
    "planning_eligible",
    "planning_eligibility_scope",
    "planning_eligibility_reason",
    "horizon_warning",
    "validation_evidence",
]

KNOWN_FUTURE_REGRESSOR_COLUMNS = [
    "name",
    "value_col",
    "availability",
    "mode",
    "future_file",
    "known_as_of_col",
    "source_system",
    "source_query_file",
    "owner",
    "refresh_latency_days",
    "notes",
    "contract_status",
]

DRIVER_AUDIT_COLUMNS = [
    "name",
    "value_col",
    "availability",
    "mode",
    "audit_status",
    "modeling_decision",
    "leakage_risk",
    "future_scope",
    "historical_rows",
    "historical_non_null_rows",
    "future_file",
    "future_rows",
    "required_future_rows",
    "missing_future_rows",
    "missing_future_dates",
    "known_as_of_col",
    "known_as_of_violations",
    "source_system",
    "source_query_file",
    "owner",
    "refresh_latency_days",
    "audit_message",
    "notes",
]

DRIVER_EXPERIMENT_SUMMARY_COLUMNS = [
    "driver_type",
    "name",
    "mode",
    "availability",
    "audit_status",
    "modeling_decision",
    "affected_or_required_rows",
    "evidence",
    "next_step",
]

_TARGET_LEAKAGE_NAMES = {
    "y",
    "target",
    "targets",
    "actual",
    "actuals",
    "observed",
    "observation",
    "realized",
    "reported_actual",
}


def parse_driver_events(
    values: Sequence[str] | None = None,
    files: Sequence[str | Path] | None = None,
) -> tuple[DriverEvent, ...]:
    """Parse inline JSON and JSON/YAML/CSV event files into ordered event assumptions."""

    events: list[DriverEvent] = []
    for value in values or ():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --event JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("--event must be a JSON object")
        events.append(_event_from_mapping(payload, source="--event"))

    for path_value in files or ():
        path = Path(path_value)
        for payload in _payload_records(path, key="events"):
            events.append(_event_from_mapping(payload, source=f"--event-file {path}"))
    return tuple(events)


def parse_known_future_regressors(
    values: Sequence[str] | None = None,
    files: Sequence[str | Path] | None = None,
) -> tuple[KnownFutureRegressor, ...]:
    """Parse inline JSON and JSON/YAML/CSV regressor declarations."""

    regressors: list[KnownFutureRegressor] = []
    for value in values or ():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --regressor JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("--regressor must be a JSON object")
        regressors.append(_regressor_from_mapping(payload, source="--regressor"))

    for path_value in files or ():
        path = Path(path_value)
        for payload in _payload_records(path, key="regressors"):
            regressors.append(_regressor_from_mapping(payload, source=f"--regressor-file {path}"))
    return tuple(regressors)


def build_scenario_assumptions_frame(events: Sequence[DriverEvent]) -> pd.DataFrame:
    rows = [
        {
            "assumption_type": "post_model_scenario_overlay",
            "name": event.name,
            "start": event.start,
            "end": event.end or event.start,
            "effect": event.effect,
            "magnitude": event.magnitude,
            "confidence": event.confidence,
            "effective_magnitude": event.magnitude * event.confidence,
            "affected_unique_ids": "; ".join(event.affected_unique_ids) if event.affected_unique_ids else "all",
            "notes": event.notes,
        }
        for event in events
    ]
    return pd.DataFrame(rows, columns=SCENARIO_ASSUMPTION_COLUMNS)


def build_scenario_forecast_frame(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty or "yhat_scenario" not in forecast.columns:
        return pd.DataFrame(columns=SCENARIO_FORECAST_COLUMNS)
    forecast = forecast.copy()
    if "horizon_step" not in forecast.columns and {"unique_id", "ds"}.issubset(forecast.columns):
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
        forecast["horizon_step"] = (
            forecast.sort_values(["unique_id", "ds"]).groupby("unique_id").cumcount() + 1
        )
    scenario_cols = [
        column
        for column in forecast.columns
        if column.startswith("yhat_scenario")
        or column.startswith("yhat_lo_")
        or column.startswith("yhat_hi_")
        or column in {"event_adjustment", "event_names"}
    ]
    ordered = [column for column in SCENARIO_FORECAST_COLUMNS if column in forecast.columns]
    extra = [column for column in scenario_cols if column not in ordered]
    return forecast[ordered + extra].copy()


def build_known_future_regressors_frame(regressors: Sequence[KnownFutureRegressor]) -> pd.DataFrame:
    rows = []
    for regressor in regressors:
        data = regressor.to_dict()
        data["value_col"] = _regressor_value_col(regressor)
        data["contract_status"] = (
            "declared_for_audit_only"
            if regressor.mode == "audit_only"
            else "declared_model_candidate_requires_future_availability_audit"
        )
        rows.append(data)
    return pd.DataFrame(rows, columns=KNOWN_FUTURE_REGRESSOR_COLUMNS)


def audit_known_future_regressors(
    dataset: pd.DataFrame,
    forecast: pd.DataFrame,
    spec: ForecastSpec,
    *,
    forecast_origin: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fail-closed availability/leakage audit for declared external regressors.

    This intentionally audits the contract only. Current model engines continue to
    use lag/date features unless a future slice explicitly wires exogenous columns.
    """

    if not spec.regressors:
        return pd.DataFrame(columns=DRIVER_AUDIT_COLUMNS), []

    origin = pd.Timestamp(forecast_origin) if forecast_origin is not None else _max_timestamp(dataset.get("ds"))
    requirements = _future_requirements(forecast)
    warnings: list[str] = [
        "known-future regressors were declared and audited, but arbitrary external regressors are not automatically trained yet; current models use lag/date features only"
    ]
    rows: list[dict[str, Any]] = []
    for regressor in spec.regressors:
        value_col = _regressor_value_col(regressor)
        future_frame, future_error = _load_future_frame(regressor.future_file)
        future_cols = set(future_frame.columns) if future_frame is not None else set()
        source_cols = set(dataset.columns) | future_cols
        historical_rows = int(len(dataset)) if value_col in dataset.columns else 0
        historical_non_null = int(pd.to_numeric(dataset[value_col], errors="coerce").notna().sum()) if value_col in dataset.columns else 0

        coverage = _future_coverage(
            future_frame,
            requirements,
            value_col=value_col,
            known_as_of_col=regressor.known_as_of_col,
            forecast_origin=origin,
        )
        failures: list[str] = []
        cautions: list[str] = []
        leakage_risk = _leakage_risk(regressor.name, value_col)

        if leakage_risk != "none":
            failures.append(f"value column/name looks like target leakage ({leakage_risk})")
        if regressor.availability == "historical_only" and regressor.mode == "model_candidate":
            failures.append("historical_only regressors cannot be model candidates because future values are not known")
        if regressor.availability != "calendar" and value_col not in source_cols:
            failures.append(f"value_col '{value_col}' is absent from history and future_file")
        if (
            regressor.mode == "model_candidate"
            and regressor.availability != "calendar"
            and future_frame is not None
            and not future_frame.empty
            and regressor.known_as_of_col not in future_cols
        ):
            failures.append(f"future_file missing known_as_of_col '{regressor.known_as_of_col}' for model_candidate timing audit")
        if future_error:
            if regressor.mode == "model_candidate":
                failures.append(future_error)
            else:
                cautions.append(future_error)
        if regressor.mode == "model_candidate" and regressor.availability != "calendar":
            if coverage["required_future_rows"] and coverage["missing_future_rows"]:
                failures.append(
                    f"future values missing for {coverage['missing_future_rows']} of {coverage['required_future_rows']} required forecast rows"
                )
            if not coverage["required_future_rows"]:
                failures.append("no forecast horizon rows were available for future-value validation")
        elif regressor.mode == "audit_only" and regressor.availability != "calendar" and coverage["required_future_rows"] and coverage["missing_future_rows"]:
            cautions.append("future values are incomplete; retained as audit-only context")
        if coverage["known_as_of_violations"]:
            failures.append(
                f"{coverage['known_as_of_violations']} future row(s) have {regressor.known_as_of_col} after the forecast origin"
            )

        if failures:
            audit_status = "failed"
            modeling_decision = "blocked_model_candidate" if regressor.mode == "model_candidate" else "audit_only_failed_contract"
            message = "; ".join(failures)
        elif cautions:
            audit_status = "warning"
            modeling_decision = "audit_only_context" if regressor.mode == "audit_only" else "candidate_audited_not_trained"
            message = "; ".join(cautions)
        else:
            audit_status = "passed"
            modeling_decision = "audit_only_context" if regressor.mode == "audit_only" else "candidate_audited_not_trained"
            message = "future availability/leakage contract passed"

        if regressor.mode == "model_candidate":
            if failures:
                warnings.append(f"known-future regressor '{regressor.name}' blocked for modeling: {message}")
            else:
                warnings.append(
                    f"known-future regressor '{regressor.name}' passed availability audit but is not automatically trained in this release"
                )

        rows.append(
            {
                "name": regressor.name,
                "value_col": value_col,
                "availability": regressor.availability,
                "mode": regressor.mode,
                "audit_status": audit_status,
                "modeling_decision": modeling_decision,
                "leakage_risk": leakage_risk,
                "future_scope": coverage["future_scope"],
                "historical_rows": historical_rows,
                "historical_non_null_rows": historical_non_null,
                "future_file": regressor.future_file or "",
                "future_rows": coverage["future_rows"],
                "required_future_rows": coverage["required_future_rows"],
                "missing_future_rows": coverage["missing_future_rows"],
                "missing_future_dates": coverage["missing_future_dates"],
                "known_as_of_col": regressor.known_as_of_col,
                "known_as_of_violations": coverage["known_as_of_violations"],
                "source_system": regressor.source_system,
                "source_query_file": regressor.source_query_file,
                "owner": regressor.owner,
                "refresh_latency_days": regressor.refresh_latency_days,
                "audit_message": message,
                "notes": regressor.notes,
            }
        )
    return pd.DataFrame(rows, columns=DRIVER_AUDIT_COLUMNS), list(dict.fromkeys(warnings))


def build_driver_experiment_summary_frame(run: ForecastRun) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in run.spec.events:
        affected_rows = _event_affected_rows(run.forecast, event.name)
        rows.append(
            {
                "driver_type": "scenario_assumption",
                "name": event.name,
                "mode": "post_model_overlay",
                "availability": "scenario",
                "audit_status": "passed" if affected_rows else "warning",
                "modeling_decision": "scenario_adjusted_yhat_only",
                "affected_or_required_rows": affected_rows,
                "evidence": f"event window {event.start} to {event.end or event.start}; effect={event.effect}; effective_magnitude={event.magnitude * event.confidence:.6g}",
                "next_step": "Compare yhat to yhat_scenario; keep baseline forecast separate from scenario-adjusted forecast.",
            }
        )
    if not run.driver_availability_audit.empty:
        for row in run.driver_availability_audit.to_dict("records"):
            rows.append(
                {
                    "driver_type": "known_future_regressor",
                    "name": row.get("name"),
                    "mode": row.get("mode"),
                    "availability": row.get("availability"),
                    "audit_status": row.get("audit_status"),
                    "modeling_decision": row.get("modeling_decision"),
                    "affected_or_required_rows": row.get("required_future_rows"),
                    "evidence": row.get("audit_message"),
                    "next_step": (
                        "Treat as audited context until exogenous modeling is explicitly enabled; "
                        "rolling-origin driver experiments must beat the lag/calendar baseline before production use."
                    ),
                }
            )
    return pd.DataFrame(rows, columns=DRIVER_EXPERIMENT_SUMMARY_COLUMNS)


def _payload_records(path: Path, *, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return _records_from_structured_payload(payload, key=key, source=str(path))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _records_from_structured_payload(payload, key=key, source=str(path))
    frame = read_tabular_source(path)
    return _frame_records(frame)


def _records_from_structured_payload(payload: Any, *, key: str, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get(key, payload.get("items", payload))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"{source} must contain a list or a mapping with '{key}'")
    records = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{source} contains a non-object {key} entry")
        records.append(dict(item))
    return records


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict("records"):
        records.append({key: value for key, value in row.items() if not _is_blank(value)})
    return records


def _event_from_mapping(payload: dict[str, Any], *, source: str) -> DriverEvent:
    missing = [key for key in ("name", "start") if key not in payload or _is_blank(payload.get(key))]
    if missing:
        raise ValueError(f"{source} missing required field(s): {missing}")
    affected = _affected_unique_ids(payload.get("affected_unique_ids", ()))
    return DriverEvent(
        name=str(payload["name"]),
        start=str(payload["start"]),
        end=str(payload["end"]) if not _is_blank(payload.get("end")) else None,
        effect=str(payload.get("effect", "multiplicative")),
        magnitude=float(payload.get("magnitude", 0.0)),
        affected_unique_ids=affected,
        confidence=float(payload.get("confidence", 1.0)),
        notes=str(payload.get("notes", "")),
    )


def _regressor_from_mapping(payload: dict[str, Any], *, source: str) -> KnownFutureRegressor:
    if "name" not in payload or _is_blank(payload.get("name")):
        raise ValueError(f"{source} missing required field(s): ['name']")
    value_col = payload.get("value_col", payload.get("value_column", payload.get("column")))
    known_as_of_col = payload.get("known_as_of_col", payload.get("known_as_of", "known_as_of"))
    return KnownFutureRegressor(
        name=str(payload["name"]),
        value_col=str(value_col) if not _is_blank(value_col) else None,
        availability=str(payload.get("availability", "historical_only")),
        mode=str(payload.get("mode", "audit_only")),
        future_file=str(payload["future_file"]) if not _is_blank(payload.get("future_file")) else None,
        known_as_of_col=str(known_as_of_col),
        source_system=str(payload.get("source_system", "")),
        source_query_file=str(payload.get("source_query_file", "")),
        owner=str(payload.get("owner", "")),
        refresh_latency_days=_int_or_none(payload.get("refresh_latency_days")),
        notes=str(payload.get("notes", "")),
    )


def _affected_unique_ids(value: Any) -> tuple[str, ...]:
    if _is_blank(value):
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if not _is_blank(item))
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed if not _is_blank(item))
    delimiter = ";" if ";" in text else ","
    return tuple(part.strip() for part in text.split(delimiter) if part.strip())


def _future_requirements(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty or not {"unique_id", "ds"}.issubset(forecast.columns):
        return pd.DataFrame(columns=["unique_id", "ds"])
    out = forecast[["unique_id", "ds"]].drop_duplicates().copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    return out.dropna(subset=["ds"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _future_coverage(
    future_frame: pd.DataFrame | None,
    requirements: pd.DataFrame,
    *,
    value_col: str,
    known_as_of_col: str,
    forecast_origin: pd.Timestamp | None,
) -> dict[str, Any]:
    if requirements.empty:
        return {
            "future_scope": "unavailable",
            "future_rows": 0,
            "required_future_rows": 0,
            "missing_future_rows": 0,
            "missing_future_dates": "",
            "known_as_of_violations": 0,
        }
    if future_frame is None or future_frame.empty:
        return {
            "future_scope": "missing_future_file",
            "future_rows": 0,
            "required_future_rows": int(len(requirements)),
            "missing_future_rows": int(len(requirements)),
            "missing_future_dates": _date_sample(requirements["ds"]),
            "known_as_of_violations": 0,
        }
    if "ds" not in future_frame.columns:
        return {
            "future_scope": "future_file_missing_ds",
            "future_rows": int(len(future_frame)),
            "required_future_rows": int(len(requirements)),
            "missing_future_rows": int(len(requirements)),
            "missing_future_dates": _date_sample(requirements["ds"]),
            "known_as_of_violations": 0,
        }

    future = future_frame.copy()
    future["ds"] = pd.to_datetime(future["ds"], errors="coerce")
    future = future.dropna(subset=["ds"])
    has_unique_id = "unique_id" in future.columns
    future_scope = "unique_id_ds" if has_unique_id else "global_ds"
    if has_unique_id:
        future["unique_id"] = future["unique_id"].astype(str)
        merge_cols = ["unique_id", "ds", value_col] if value_col in future.columns else ["unique_id", "ds"]
        coverage = requirements.merge(future[merge_cols], on=["unique_id", "ds"], how="left")
        required_rows = int(len(requirements))
        future_rows = int(future[["unique_id", "ds"]].drop_duplicates().shape[0])
    else:
        required_dates = pd.DataFrame({"ds": sorted(requirements["ds"].dropna().unique())})
        merge_cols = ["ds", value_col] if value_col in future.columns else ["ds"]
        coverage = required_dates.merge(future[merge_cols], on="ds", how="left")
        required_rows = int(len(required_dates))
        future_rows = int(future["ds"].drop_duplicates().shape[0])

    if value_col in coverage.columns:
        missing_mask = coverage[value_col].isna()
    else:
        missing_mask = pd.Series(True, index=coverage.index)
    missing_dates = _date_sample(coverage.loc[missing_mask, "ds"])
    known_as_of_violations = _known_as_of_violations(future, known_as_of_col=known_as_of_col, forecast_origin=forecast_origin)
    return {
        "future_scope": future_scope,
        "future_rows": future_rows,
        "required_future_rows": required_rows,
        "missing_future_rows": int(missing_mask.sum()),
        "missing_future_dates": missing_dates,
        "known_as_of_violations": known_as_of_violations,
    }


def _load_future_frame(path_value: str | None) -> tuple[pd.DataFrame | None, str]:
    if not path_value:
        return None, ""
    path = Path(path_value)
    if not path.exists():
        return None, f"future_file not found: {path}"
    try:
        if path.suffix.lower() in {".json", ".jsonl"}:
            if path.suffix.lower() == ".jsonl":
                return pd.read_json(path, lines=True), ""
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("data", payload.get("rows", payload.get("records", payload)))
            if isinstance(payload, dict):
                payload = [payload]
            return pd.DataFrame(payload), ""
        return read_tabular_source(path), ""
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"future_file could not be read: {path} ({exc})"


def _known_as_of_violations(
    future: pd.DataFrame,
    *,
    known_as_of_col: str,
    forecast_origin: pd.Timestamp | None,
) -> int:
    if forecast_origin is None or known_as_of_col not in future.columns:
        return 0
    known_as_of = pd.to_datetime(future[known_as_of_col], errors="coerce")
    return int((known_as_of.notna() & (known_as_of > forecast_origin)).sum())


def _leakage_risk(name: str, value_col: str) -> str:
    risky = []
    for label, value in (("name", name), ("value_col", value_col)):
        normalized = _normalize_name(value)
        tokens = {part for part in normalized.split("_") if part}
        if normalized in _TARGET_LEAKAGE_NAMES or tokens.intersection(_TARGET_LEAKAGE_NAMES):
            risky.append(f"{label}={value}")
    return "; ".join(risky) if risky else "none"


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _regressor_value_col(regressor: KnownFutureRegressor) -> str:
    return str(regressor.value_col or regressor.name)


def _event_affected_rows(forecast: pd.DataFrame, event_name: str) -> int:
    if forecast.empty or "event_names" not in forecast.columns:
        return 0
    names = forecast["event_names"].fillna("").astype(str)
    return int(names.apply(lambda value: event_name in [part.strip() for part in value.split(";")]).sum())


def _date_sample(values: Iterable[Any], *, limit: int = 8) -> str:
    dates = pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna()
    if dates.empty:
        return ""
    unique_dates = dates.drop_duplicates().sort_values()
    formatted = [pd.Timestamp(value).date().isoformat() for value in unique_dates.head(limit)]
    suffix = "..." if len(unique_dates) > limit else ""
    return "; ".join(formatted) + suffix


def _max_timestamp(values: Any) -> pd.Timestamp | None:
    if values is None:
        return None
    dates = pd.to_datetime(values, errors="coerce")
    if dates is None or pd.isna(dates).all():
        return None
    return pd.Timestamp(dates.max())


def _int_or_none(value: Any) -> int | None:
    if _is_blank(value):
        return None
    return int(value)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()
