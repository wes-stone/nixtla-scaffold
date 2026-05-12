from __future__ import annotations

from dataclasses import dataclass, field
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

DRIVER_MODEL_FEATURE_COLUMNS = [
    "name",
    "value_col",
    "feature_columns",
    "feature_role",
    "status",
    "modeling_decision",
    "reason",
    "safe_lags",
    "historical_rows",
    "historical_non_null_rows",
    "training_non_null_min_rows",
    "future_file",
    "future_rows",
    "required_future_rows",
    "missing_future_rows",
    "known_as_of_violations",
    "source_system",
    "owner",
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


@dataclass(frozen=True)
class DriverFeatureBundle:
    historical: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["unique_id", "ds"]))
    future: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["unique_id", "ds"]))
    audit: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DRIVER_MODEL_FEATURE_COLUMNS))
    feature_columns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


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


def prepare_mlforecast_regressor_features(
    history: pd.DataFrame,
    spec: ForecastSpec,
    *,
    freq: str,
    horizon: int,
    season_length: int,
    min_training_rows: int,
    forecast_origin: str | pd.Timestamp | None,
) -> DriverFeatureBundle:
    """Build MLForecast dynamic feature frames from audit-passing known-future regressors."""

    if not spec.regressors or not spec.train_known_future_regressors:
        return DriverFeatureBundle()

    requirements = _future_grid_from_history(history, freq=freq, horizon=horizon)
    origin = pd.Timestamp(forecast_origin) if forecast_origin is not None else _max_timestamp(history.get("ds"))
    historical = history[["unique_id", "ds"]].copy()
    future = requirements.copy()
    rows: list[dict[str, Any]] = []
    warnings_out: list[str] = []
    included_cols: list[str] = []
    used_value_cols: set[str] = set()

    for regressor in spec.regressors:
        value_col = _regressor_value_col(regressor)
        future_frame, future_error = _load_future_frame(regressor.future_file)
        future_cols = set(future_frame.columns) if future_frame is not None else set()
        source_cols = set(history.columns) | future_cols
        historical_values = pd.to_numeric(history[value_col], errors="coerce") if value_col in history.columns else pd.Series(dtype="float64")
        coverage = (
            _historical_only_coverage()
            if regressor.availability == "historical_only"
            else _future_coverage(
                future_frame,
                requirements,
                value_col=value_col,
                known_as_of_col=regressor.known_as_of_col,
                forecast_origin=origin,
            )
        )
        failures = _regressor_contract_failures(
            regressor,
            value_col=value_col,
            source_cols=source_cols,
            future_cols=future_cols,
            future_error=future_error,
            coverage=coverage,
        )
        feature_role = "historical_only_lag_regressor" if regressor.availability == "historical_only" else "dynamic_known_future_regressor"
        status = "excluded"
        modeling_decision = "excluded_from_mlforecast"
        reason = "; ".join(failures)
        feature_columns: list[str] = []
        safe_lags: list[int] = []
        training_non_null_min_rows = 0

        if regressor.mode != "model_candidate":
            reason = "regressor mode is audit_only; not a model feature"
            modeling_decision = "audit_only_not_trained"
        elif not failures and value_col not in history.columns:
            reason = f"value_col '{value_col}' is absent from repaired history"
        elif not failures and historical_values.isna().any():
            reason = f"value_col '{value_col}' has {int(historical_values.isna().sum())} missing or non-numeric historical row(s)"
        elif not failures and regressor.availability == "historical_only":
            lag_result = _historical_only_lag_features(
                history,
                requirements,
                value_col=value_col,
                horizon=horizon,
                season_length=season_length,
                min_training_rows=min_training_rows,
            )
            if lag_result["included"]:
                status = "included"
                modeling_decision = "included_mlforecast_historical_lag_candidate"
                reason = lag_result["reason"]
                feature_columns = list(lag_result["feature_columns"])
                safe_lags = list(lag_result["safe_lags"])
                training_non_null_min_rows = int(lag_result["training_non_null_min_rows"])
                for column in feature_columns:
                    historical[column] = lag_result["historical"][column].to_numpy(dtype="float64")
                    future[column] = lag_result["future"][column].to_numpy(dtype="float64")
                included_cols.extend(column for column in feature_columns if column not in included_cols)
            else:
                reason = lag_result["reason"]
                training_non_null_min_rows = int(lag_result["training_non_null_min_rows"])
        elif not failures:
            future_values = _future_values_for_requirements(
                future_frame,
                requirements,
                value_col=value_col,
            )
            missing_future = int(pd.to_numeric(future_values[value_col], errors="coerce").isna().sum())
            if missing_future:
                reason = f"value_col '{value_col}' has {missing_future} missing or non-numeric future row(s)"
            else:
                status = "included"
                modeling_decision = "included_mlforecast_model_candidate"
                reason = "passed leakage, known-as-of, historical, and future-value checks"
                if value_col not in used_value_cols:
                    historical[value_col] = historical_values.astype("float64").to_numpy()
                    future[value_col] = pd.to_numeric(future_values[value_col], errors="coerce").astype("float64").to_numpy()
                    used_value_cols.add(value_col)
                    included_cols.append(value_col)
                    feature_columns = [value_col]
                    training_non_null_min_rows = int(
                        historical.groupby("unique_id")[value_col].apply(lambda values: pd.to_numeric(values, errors="coerce").notna().sum()).min()
                    )

        if status != "included" and regressor.mode == "model_candidate":
            warnings_out.append(f"external regressor '{regressor.name}' excluded from MLForecast training: {reason}")

        rows.append(
            {
                "name": regressor.name,
                "value_col": value_col,
                "feature_columns": "; ".join(feature_columns),
                "feature_role": feature_role,
                "status": status,
                "modeling_decision": modeling_decision,
                "reason": reason,
                "safe_lags": "; ".join(str(lag) for lag in safe_lags),
                "historical_rows": int(len(history)) if value_col in history.columns else 0,
                "historical_non_null_rows": int(historical_values.notna().sum()) if value_col in history.columns else 0,
                "training_non_null_min_rows": training_non_null_min_rows,
                "future_file": regressor.future_file or "",
                "future_rows": coverage["future_rows"],
                "required_future_rows": coverage["required_future_rows"],
                "missing_future_rows": coverage["missing_future_rows"],
                "known_as_of_violations": coverage["known_as_of_violations"],
                "source_system": regressor.source_system,
                "owner": regressor.owner,
            }
        )

    if not included_cols:
        warnings_out.append("train_known_future_regressors=True, but no declared regressor passed the MLForecast feature gate")

    return DriverFeatureBundle(
        historical=historical[["unique_id", "ds", *included_cols]].copy() if included_cols else pd.DataFrame(columns=["unique_id", "ds"]),
        future=future[["unique_id", "ds", *included_cols]].copy() if included_cols else pd.DataFrame(columns=["unique_id", "ds"]),
        audit=pd.DataFrame(rows, columns=DRIVER_MODEL_FEATURE_COLUMNS),
        feature_columns=tuple(included_cols),
        warnings=tuple(dict.fromkeys(warnings_out)),
    )


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
    warnings: list[str] = []
    if spec.train_known_future_regressors:
        warnings.append(
            "known-future regressor training was requested; only MLForecast model_candidate rows that pass the feature gate can enter modeling"
        )
    else:
        warnings.append(
            "known-future regressors were declared and audited, but arbitrary external regressors are not automatically trained unless train_known_future_regressors=True; current models use lag/date features only"
        )
    rows: list[dict[str, Any]] = []
    for regressor in spec.regressors:
        value_col = _regressor_value_col(regressor)
        future_frame, future_error = _load_future_frame(regressor.future_file)
        future_cols = set(future_frame.columns) if future_frame is not None else set()
        source_cols = set(dataset.columns) | future_cols
        historical_rows = int(len(dataset)) if value_col in dataset.columns else 0
        historical_non_null = int(pd.to_numeric(dataset[value_col], errors="coerce").notna().sum()) if value_col in dataset.columns else 0

        coverage = (
            _historical_only_coverage()
            if regressor.availability == "historical_only"
            else _future_coverage(
                future_frame,
                requirements,
                value_col=value_col,
                known_as_of_col=regressor.known_as_of_col,
                forecast_origin=origin,
            )
        )
        failures: list[str] = []
        cautions: list[str] = []
        leakage_risk = _leakage_risk(regressor.name, value_col)

        failures.extend(
            _regressor_contract_failures(
                regressor,
                value_col=value_col,
                source_cols=source_cols,
                future_cols=future_cols,
                future_error=future_error,
                coverage=coverage,
            )
        )
        if future_error and regressor.mode == "audit_only":
            cautions.append(future_error)
        if regressor.mode == "audit_only" and regressor.availability != "calendar" and coverage["required_future_rows"] and coverage["missing_future_rows"]:
            cautions.append("future values are incomplete; retained as audit-only context")

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
            if regressor.mode == "audit_only":
                modeling_decision = "audit_only_context"
            elif spec.train_known_future_regressors and regressor.availability == "historical_only":
                modeling_decision = "candidate_passed_for_historical_lag_training"
            elif spec.train_known_future_regressors:
                modeling_decision = "candidate_passed_for_opt_in_training"
            else:
                modeling_decision = "candidate_audited_not_trained"
            message = "future availability/leakage contract passed"

        if regressor.mode == "model_candidate":
            if failures:
                warnings.append(f"known-future regressor '{regressor.name}' blocked for modeling: {message}")
            else:
                if spec.train_known_future_regressors and regressor.availability == "historical_only":
                    warnings.append(
                        f"historical-only regressor '{regressor.name}' passed audit and may enter MLForecast as safe lag features if the feature gate includes it"
                    )
                elif spec.train_known_future_regressors:
                    warnings.append(
                        f"known-future regressor '{regressor.name}' passed availability audit and may be used by MLForecast if the feature gate includes it"
                    )
                else:
                    warnings.append(
                        f"known-future regressor '{regressor.name}' passed availability audit but is not automatically trained unless train_known_future_regressors=True"
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
            next_step = (
                "Review appendix/driver_model_features.csv and appendix/model_explainability.csv to confirm which audited drivers entered MLForecast."
                if row.get("modeling_decision") == "candidate_passed_for_opt_in_training"
                else (
                    "Treat as audited context until exogenous modeling is explicitly enabled; "
                    "rolling-origin driver experiments must beat the lag/calendar baseline before production use."
                )
            )
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
                    "next_step": next_step,
                }
            )
    driver_model_features = getattr(run, "driver_model_features", pd.DataFrame())
    if not driver_model_features.empty:
        for row in driver_model_features.to_dict("records"):
            rows.append(
                {
                    "driver_type": "mlforecast_feature",
                    "name": row.get("name"),
                    "mode": "model_candidate",
                    "availability": row.get("feature_role"),
                    "audit_status": row.get("status"),
                    "modeling_decision": row.get("modeling_decision"),
                    "affected_or_required_rows": row.get("required_future_rows"),
                    "evidence": row.get("reason"),
                    "next_step": "Compare appendix/model_explainability.csv, appendix/driver_model_cv_delta.csv, and appendix/model_audit.csv before trusting driver-enhanced candidates.",
                }
            )
    return pd.DataFrame(rows, columns=DRIVER_EXPERIMENT_SUMMARY_COLUMNS)


def _future_grid_from_history(history: pd.DataFrame, *, freq: str, horizon: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for uid, grp in history.groupby("unique_id", sort=True):
        last_date = pd.to_datetime(grp["ds"], errors="coerce").max()
        if pd.isna(last_date):
            continue
        dates = pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 1, freq=freq)[1:]
        rows.append(pd.DataFrame({"unique_id": str(uid), "ds": dates}))
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds"])
    return pd.concat(rows, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _historical_only_coverage() -> dict[str, Any]:
    return {
        "future_scope": "not_required_historical_only",
        "future_rows": 0,
        "required_future_rows": 0,
        "missing_future_rows": 0,
        "missing_future_dates": "",
        "known_as_of_violations": 0,
    }


def _historical_only_lag_features(
    history: pd.DataFrame,
    requirements: pd.DataFrame,
    *,
    value_col: str,
    horizon: int,
    season_length: int,
    min_training_rows: int,
) -> dict[str, Any]:
    lags = _candidate_historical_regressor_lags(horizon=horizon, season_length=season_length)
    if not lags:
        return _historical_lag_result(False, reason="no safe historical-only lags are available for the requested horizon")

    base = history[["unique_id", "ds", value_col]].copy()
    base["unique_id"] = base["unique_id"].astype(str)
    base["ds"] = pd.to_datetime(base["ds"], errors="coerce")
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    base = base.dropna(subset=["ds"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    future = requirements[["unique_id", "ds"]].copy()
    future["unique_id"] = future["unique_id"].astype(str)
    future["ds"] = pd.to_datetime(future["ds"], errors="coerce")
    feature_base = _normalize_name(value_col)

    included_columns: list[str] = []
    included_lags: list[int] = []
    historical_features = history[["unique_id", "ds"]].copy()
    future_features = requirements[["unique_id", "ds"]].copy()
    rejected: list[str] = []
    best_training_rows = 0

    for lag in lags:
        column = f"{feature_base}_lag_{lag}"
        lagged_history = _lag_history_feature(base, value_col=value_col, column=column, lag=lag)
        lagged_future = _lag_future_feature(base, future, value_col=value_col, column=column, lag=lag)
        training_counts = lagged_history.groupby("unique_id", sort=True)[column].apply(lambda values: pd.to_numeric(values, errors="coerce").notna().sum())
        min_non_null = int(training_counts.min()) if not training_counts.empty else 0
        best_training_rows = max(best_training_rows, min_non_null)
        missing_future = int(pd.to_numeric(lagged_future[column], errors="coerce").isna().sum()) if column in lagged_future.columns else len(future)
        if missing_future:
            rejected.append(f"lag {lag} has {missing_future} missing future feature row(s)")
            continue
        if min_non_null < min_training_rows:
            rejected.append(f"lag {lag} leaves only {min_non_null} non-null training row(s) per series; need at least {min_training_rows}")
            continue
        historical_features = historical_features.merge(lagged_history[["unique_id", "ds", column]], on=["unique_id", "ds"], how="left")
        future_features = future_features.merge(lagged_future[["unique_id", "ds", column]], on=["unique_id", "ds"], how="left")
        included_columns.append(column)
        included_lags.append(lag)

    if not included_columns:
        reason = "; ".join(rejected) if rejected else "no historical-only lag candidates passed the safety gate"
        return _historical_lag_result(False, reason=reason, training_non_null_min_rows=best_training_rows)

    reason = (
        "included safe historical-only lag feature(s) "
        f"{', '.join(included_columns)}; every future row uses source values known at the forecast origin"
    )
    return _historical_lag_result(
        True,
        reason=reason,
        feature_columns=tuple(included_columns),
        safe_lags=tuple(included_lags),
        training_non_null_min_rows=min(
            int(
                historical_features.groupby("unique_id", sort=True)[column]
                .apply(lambda values: pd.to_numeric(values, errors="coerce").notna().sum())
                .min()
            )
            for column in included_columns
        ),
        historical=historical_features,
        future=future_features,
    )


def _historical_lag_result(
    included: bool,
    *,
    reason: str,
    feature_columns: tuple[str, ...] = (),
    safe_lags: tuple[int, ...] = (),
    training_non_null_min_rows: int = 0,
    historical: pd.DataFrame | None = None,
    future: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "included": included,
        "reason": reason,
        "feature_columns": feature_columns,
        "safe_lags": safe_lags,
        "training_non_null_min_rows": training_non_null_min_rows,
        "historical": historical if historical is not None else pd.DataFrame(columns=["unique_id", "ds"]),
        "future": future if future is not None else pd.DataFrame(columns=["unique_id", "ds"]),
    }


def _candidate_historical_regressor_lags(*, horizon: int, season_length: int) -> list[int]:
    candidates = [max(1, int(horizon))]
    if season_length > horizon:
        candidates.append(int(season_length))
    return sorted(set(lag for lag in candidates if lag >= horizon and lag >= 1))


def _lag_history_feature(base: pd.DataFrame, *, value_col: str, column: str, lag: int) -> pd.DataFrame:
    out = base[["unique_id", "ds", value_col]].copy()
    out[column] = out.groupby("unique_id", sort=True)[value_col].shift(lag)
    return out[["unique_id", "ds", column]]


def _lag_future_feature(
    base: pd.DataFrame,
    future: pd.DataFrame,
    *,
    value_col: str,
    column: str,
    lag: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for uid, hist_grp in base.groupby("unique_id", sort=True):
        future_grp = future[future["unique_id"].astype(str) == str(uid)][["unique_id", "ds"]].copy()
        if future_grp.empty:
            continue
        hist_values = hist_grp[["unique_id", "ds", value_col]].copy()
        future_values = future_grp.copy()
        future_values[value_col] = pd.NA
        combined = pd.concat([hist_values, future_values], ignore_index=True).sort_values(["ds"]).reset_index(drop=True)
        combined[column] = combined[value_col].shift(lag)
        rows.append(combined[combined["ds"].isin(future_grp["ds"])][["unique_id", "ds", column]])
    if not rows:
        out = future[["unique_id", "ds"]].copy()
        out[column] = pd.NA
        return out
    return pd.concat(rows, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _regressor_contract_failures(
    regressor: KnownFutureRegressor,
    *,
    value_col: str,
    source_cols: set[str],
    future_cols: set[str],
    future_error: str,
    coverage: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    leakage_risk = _leakage_risk(regressor.name, value_col)
    if leakage_risk != "none":
        failures.append(f"value column/name looks like target leakage ({leakage_risk})")
    if regressor.availability != "calendar" and value_col not in source_cols:
        failures.append(f"value_col '{value_col}' is absent from history and future_file")
    if regressor.availability == "historical_only":
        return failures
    if (
        regressor.mode == "model_candidate"
        and regressor.availability != "calendar"
        and future_cols
        and regressor.known_as_of_col not in future_cols
    ):
        failures.append(f"future_file missing known_as_of_col '{regressor.known_as_of_col}' for model_candidate timing audit")
    if future_error and regressor.mode == "model_candidate":
        failures.append(future_error)
    if regressor.mode == "model_candidate" and regressor.availability != "calendar":
        if coverage["required_future_rows"] and coverage["missing_future_rows"]:
            failures.append(
                f"future values missing for {coverage['missing_future_rows']} of {coverage['required_future_rows']} required forecast rows"
            )
        if not coverage["required_future_rows"]:
            failures.append("no forecast horizon rows were available for future-value validation")
    if coverage["known_as_of_violations"]:
        failures.append(
            f"{coverage['known_as_of_violations']} future row(s) have {regressor.known_as_of_col} after the forecast origin"
        )
    return failures


def _future_values_for_requirements(
    future_frame: pd.DataFrame | None,
    requirements: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    if requirements.empty:
        return pd.DataFrame(columns=["unique_id", "ds", value_col])
    out = requirements[["unique_id", "ds"]].copy()
    if future_frame is None or future_frame.empty or "ds" not in future_frame.columns or value_col not in future_frame.columns:
        out[value_col] = pd.NA
        return out
    future = future_frame.copy()
    future["ds"] = pd.to_datetime(future["ds"], errors="coerce")
    future = future.dropna(subset=["ds"])
    if "unique_id" in future.columns:
        future["unique_id"] = future["unique_id"].astype(str)
        values = future[["unique_id", "ds", value_col]].drop_duplicates(["unique_id", "ds"], keep="last")
        return out.merge(values, on=["unique_id", "ds"], how="left")
    values = future[["ds", value_col]].drop_duplicates(["ds"], keep="last")
    return out.merge(values, on="ds", how="left")


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
