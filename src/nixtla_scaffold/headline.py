from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from nixtla_scaffold.schema import ForecastRun


@dataclass(frozen=True)
class ExecutiveHeadline:
    """Conservative forecast headline for agents and finance readers.

    Public attribute access is treated as stable in the 0.1.x package line.
    The serialized ``to_dict`` shape may append new optional fields; consumers
    should ignore unknown keys rather than failing closed.
    """

    paragraph: str
    scope: str
    direction: str
    direction_pct_vs_recent: float | None
    direction_abs_delta_vs_recent: float | None
    yoy_pct_vs_prior_year: float | None
    yoy_abs_delta_vs_prior_year: float | None
    value_unit_label: str
    horizon_end: str | None
    trust_distribution: dict[str, int]
    full_horizon_claim_allowed_count: int
    direction_split: dict[str, int]
    top_caveat: str
    next_action: str
    series: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph": self.paragraph,
            "scope": self.scope,
            "direction": self.direction,
            "direction_pct_vs_recent": self.direction_pct_vs_recent,
            "direction_abs_delta_vs_recent": self.direction_abs_delta_vs_recent,
            "yoy_pct_vs_prior_year": self.yoy_pct_vs_prior_year,
            "yoy_abs_delta_vs_prior_year": self.yoy_abs_delta_vs_prior_year,
            "value_unit_label": self.value_unit_label,
            "horizon_end": self.horizon_end,
            "trust_distribution": self.trust_distribution,
            "full_horizon_claim_allowed_count": self.full_horizon_claim_allowed_count,
            "direction_split": self.direction_split,
            "top_caveat": self.top_caveat,
            "next_action": self.next_action,
            "series": self.series,
        }


def build_executive_headline(run: ForecastRun) -> ExecutiveHeadline:
    """Build a deterministic, conservative executive headline for a forecast run.

    The headline is deliberately evidence-bound: it reuses selected forecasts,
    trust summary, interval provenance, horizon validation, and target-transform
    metadata rather than inventing new confidence claims.
    """

    from nixtla_scaffold.outputs import build_selected_forecast, build_trust_summary

    trust = build_trust_summary(run)
    forecast = build_selected_forecast(run)
    if forecast.empty:
        paragraph = "Executive headline unavailable: no selected forecast rows were produced."
        return ExecutiveHeadline(
            paragraph=paragraph,
            scope="empty",
            direction="unavailable",
            direction_pct_vs_recent=None,
            direction_abs_delta_vs_recent=None,
            yoy_pct_vs_prior_year=None,
            yoy_abs_delta_vs_prior_year=None,
            value_unit_label=_value_unit_label(run),
            horizon_end=None,
            trust_distribution={},
            full_horizon_claim_allowed_count=0,
            direction_split={},
            top_caveat="No selected forecast rows were produced.",
            next_action="Inspect diagnostics.json and failure_diagnostics.json if present.",
            series=[],
        )

    history = run.history.copy()
    history["ds"] = pd.to_datetime(history["ds"], errors="coerce")
    forecast = forecast.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    trust_records = _trust_by_uid(trust)
    series_rows = [
        _series_headline(run, uid, history, forecast, trust_records.get(uid, {}))
        for uid in sorted(forecast["unique_id"].astype(str).dropna().unique())
    ]
    series_rows = [row for row in series_rows if row]
    trust_distribution = _trust_distribution(trust)
    full_horizon_count = sum(1 for row in trust.to_dict("records") if _truthy(row.get("full_horizon_claim_allowed")))

    if len(series_rows) == 1:
        row = series_rows[0]
        return ExecutiveHeadline(
            paragraph=row["paragraph"],
            scope="single",
            direction=row["direction"],
            direction_pct_vs_recent=row["direction_pct_vs_recent"],
            direction_abs_delta_vs_recent=row["direction_abs_delta_vs_recent"],
            yoy_pct_vs_prior_year=row["yoy_pct_vs_prior_year"],
            yoy_abs_delta_vs_prior_year=row["yoy_abs_delta_vs_prior_year"],
            value_unit_label=row["value_unit_label"],
            horizon_end=row["horizon_end"],
            trust_distribution=trust_distribution,
            full_horizon_claim_allowed_count=full_horizon_count,
            direction_split=_direction_split(series_rows),
            top_caveat=row["top_caveat"],
            next_action=row["next_action"],
            series=series_rows,
        )

    portfolio = _portfolio_headline(run, series_rows, trust, trust_distribution, full_horizon_count)
    return ExecutiveHeadline(
        paragraph=portfolio["paragraph"],
        scope="portfolio",
        direction=portfolio["direction"],
        direction_pct_vs_recent=portfolio["direction_pct_vs_recent"],
        direction_abs_delta_vs_recent=portfolio["direction_abs_delta_vs_recent"],
        yoy_pct_vs_prior_year=portfolio["yoy_pct_vs_prior_year"],
        yoy_abs_delta_vs_prior_year=portfolio["yoy_abs_delta_vs_prior_year"],
        value_unit_label=portfolio["value_unit_label"],
        horizon_end=portfolio["horizon_end"],
        trust_distribution=trust_distribution,
        full_horizon_claim_allowed_count=full_horizon_count,
        direction_split=portfolio["direction_split"],
        top_caveat=portfolio["top_caveat"],
        next_action=portfolio["next_action"],
        series=series_rows,
    )


def _series_headline(
    run: ForecastRun,
    uid: str,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    trust_row: dict[str, Any],
) -> dict[str, Any]:
    series_forecast = forecast[forecast["unique_id"].astype(str) == uid].sort_values("ds")
    if series_forecast.empty:
        return {}
    final = series_forecast.iloc[-1].to_dict()
    series_history = history[history["unique_id"].astype(str) == uid]
    recent_average = _recent_average(series_history)
    yhat = _float_or_none(final.get("yhat"))
    value_label = _value_unit_label(run)
    direction, pct, abs_delta, direction_sentence = _direction_sentence(yhat, recent_average, final.get("ds"), value_label)
    yoy_pct, yoy_abs_delta, yoy_sentence = _yoy_sentence(
        yhat,
        _prior_year_actual(series_history, final.get("ds")),
        final.get("ds"),
        value_label,
    )
    trust_level = str(trust_row.get("trust_level") or "Unknown")
    trust_score = _display_value(trust_row.get("trust_score_0_100"), "N/A")
    selected_model = str(trust_row.get("selected_model") or final.get("model") or "selected model")
    caveat = _first_item(trust_row.get("caveats"), "No major caveats recorded.")
    action = _first_item(trust_row.get("next_actions"), "Validate with a domain expert before stakeholder use.")
    interval_sentence = _interval_sentence(final, trust_row)
    horizon_sentence = _horizon_sentence(trust_row, final)
    transform_sentence = _transform_sentence(run, value_label)
    intro = (
        f"{uid}: {trust_level} trust ({trust_score}/100), model {selected_model}."
        if trust_level != "Low"
        else f"{uid}: exploratory statistical baseline only - Low trust ({trust_score}/100), model {selected_model}."
    )
    paragraph = " ".join(
        part.strip()
        for part in [
            intro,
            transform_sentence,
            direction_sentence,
            yoy_sentence,
            horizon_sentence,
            interval_sentence,
            f"Biggest caveat: {caveat}.",
            f"Next: {action}",
        ]
        if part and part.strip()
    )
    return {
        "unique_id": uid,
        "paragraph": _squash(paragraph),
        "direction": direction,
        "direction_pct_vs_recent": pct,
        "direction_abs_delta_vs_recent": abs_delta,
        "yoy_pct_vs_prior_year": yoy_pct,
        "yoy_abs_delta_vs_prior_year": yoy_abs_delta,
        "value_unit_label": value_label,
        "horizon_end": _date_text(final.get("ds")),
        "trust_level": trust_level,
        "trust_score_0_100": trust_row.get("trust_score_0_100"),
        "selected_model": selected_model,
        "top_caveat": caveat,
        "next_action": action,
    }


def _portfolio_headline(
    run: ForecastRun,
    series_rows: list[dict[str, Any]],
    trust: pd.DataFrame,
    trust_distribution: dict[str, int],
    full_horizon_count: int,
) -> dict[str, Any]:
    count = len(series_rows)
    avg_score = None
    if not trust.empty and "trust_score_0_100" in trust.columns:
        scores = pd.to_numeric(trust["trust_score_0_100"], errors="coerce")
        if scores.notna().any():
            avg_score = float(scores.mean())
    direction, pct = _portfolio_direction(series_rows)
    abs_delta = _average_optional(series_rows, "direction_abs_delta_vs_recent")
    yoy_pct = _average_optional(series_rows, "yoy_pct_vs_prior_year")
    yoy_abs_delta = _average_optional(series_rows, "yoy_abs_delta_vs_prior_year")
    split = _direction_split(series_rows)
    horizon_end = max((row.get("horizon_end") for row in series_rows if row.get("horizon_end")), default=None)
    worst = min(series_rows, key=lambda row: _sort_score(row.get("trust_score_0_100")))
    top_caveat = str(worst.get("top_caveat") or "No major caveats recorded.")
    next_action = _portfolio_next_action(series_rows)
    trust_text = ", ".join(f"{level} {trust_distribution.get(level, 0)}" for level in ["High", "Medium", "Low"])
    avg_text = f", average trust {avg_score:.0f}/100" if avg_score is not None else ""
    direction_text = _portfolio_direction_text(direction, pct)
    split_text = _direction_split_text(split)
    watch_text = _watch_series_text(series_rows)
    value_label = _portfolio_value_label(series_rows, run)
    paragraph = (
        f"{count} series forecast through {horizon_end or 'the requested horizon'} on {run.profile.freq} grain: "
        f"{trust_text}{avg_text}. {direction_text} "
        f"{split_text} "
        f"{full_horizon_count} of {count} series allow full-horizon champion claims; filter forecast.csv to planning_eligible=True and review directional or exploratory rows before sharing. "
        f"Watch {watch_text}. "
        f"Biggest caveat: {top_caveat}. Next: {next_action}"
    )
    return {
        "paragraph": _squash(paragraph),
        "direction": direction,
        "direction_pct_vs_recent": pct,
        "direction_abs_delta_vs_recent": abs_delta,
        "yoy_pct_vs_prior_year": yoy_pct,
        "yoy_abs_delta_vs_prior_year": yoy_abs_delta,
        "value_unit_label": value_label,
        "horizon_end": horizon_end,
        "direction_split": split,
        "top_caveat": top_caveat,
        "next_action": next_action,
    }


def _trust_by_uid(trust: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trust.empty or "unique_id" not in trust.columns:
        return {}
    return {str(row["unique_id"]): row for row in trust.to_dict("records")}


def _trust_distribution(trust: pd.DataFrame) -> dict[str, int]:
    if trust.empty or "trust_level" not in trust.columns:
        return {}
    counts = trust["trust_level"].value_counts().to_dict()
    return {str(level): int(counts.get(level, 0)) for level in ["High", "Medium", "Low"]}


def _recent_average(history: pd.DataFrame) -> float | None:
    if history.empty or "y" not in history.columns:
        return None
    values = pd.to_numeric(history.sort_values("ds")["y"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.tail(min(3, len(values))).mean())


def _direction_sentence(
    yhat: float | None,
    recent_average: float | None,
    ds: object,
    value_label: str,
) -> tuple[str, float | None, float | None, str]:
    if yhat is None or recent_average is None:
        return "unavailable", None, None, "Direction versus recent actuals is unavailable."
    date_text = _date_text(ds) or "the horizon end"
    delta = yhat - recent_average
    if abs(recent_average) < 1e-12:
        return (
            "unavailable",
            None,
            float(delta),
            (
                f"Statistical baseline ends at {date_text} with point forecast {_format_measure(yhat, value_label)}; "
                f"recent average is zero, so percent direction is not meaningful. "
                f"absolute delta versus recent average is {_format_delta(delta, value_label)}."
            ),
        )
    pct = (yhat - recent_average) / abs(recent_average)
    if abs(pct) < 0.02:
        return (
            "flat",
            float(pct),
            float(delta),
            (
                "Statistical baseline is approximately flat versus the recent average "
                f"({_format_measure(recent_average, value_label)} to {_format_measure(yhat, value_label)} by {date_text}; "
                f"absolute delta {_format_delta(delta, value_label)})."
            ),
        )
    direction = "up" if pct > 0 else "down"
    return (
        direction,
        float(pct),
        float(delta),
        (
            f"Statistical baseline trends {direction} about {abs(pct):.0%} versus the recent average "
            f"({_format_measure(recent_average, value_label)} to {_format_measure(yhat, value_label)} by {date_text}; "
            f"absolute delta {_format_delta(delta, value_label)})."
        ),
    )


def _prior_year_actual(history: pd.DataFrame, forecast_ds: object) -> float | None:
    if history.empty or "ds" not in history.columns or "y" not in history.columns:
        return None
    try:
        target = pd.Timestamp(forecast_ds) - pd.DateOffset(years=1)
    except (TypeError, ValueError):
        return None
    frame = history.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    exact = frame[frame["ds"] == target]
    if exact.empty:
        return None
    values = pd.to_numeric(exact.sort_values("ds")["y"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _yoy_sentence(
    yhat: float | None,
    prior_year_actual: float | None,
    ds: object,
    value_label: str,
) -> tuple[float | None, float | None, str]:
    if yhat is None or prior_year_actual is None or abs(prior_year_actual) < 1e-12:
        return None, None, ""
    date_text = _date_text(ds) or "the horizon end"
    delta = yhat - prior_year_actual
    pct = delta / abs(prior_year_actual)
    if abs(pct) < 0.02:
        comparison = "approximately flat"
    else:
        comparison = f"{'up' if pct > 0 else 'down'} about {abs(pct):.0%}"
    sentence = (
        f"YoY: final-step baseline is {comparison} versus prior-year same period "
        f"({_format_measure(prior_year_actual, value_label)} to {_format_measure(yhat, value_label)} by {date_text}; "
        f"absolute delta {_format_delta(delta, value_label)})."
    )
    return float(pct), float(delta), sentence


def _interval_sentence(final: dict[str, Any], trust_row: dict[str, Any]) -> str:
    status = str(trust_row.get("interval_status") or final.get("interval_status") or "unavailable")
    lo = _float_or_none(final.get("yhat_lo_80"))
    hi = _float_or_none(final.get("yhat_hi_80"))
    if status in {"unavailable", "point_only_ensemble"} or lo is None or hi is None:
        if status == "point_only_ensemble":
            return "Intervals: WeightedEnsemble is point-only; no calibrated range is shown. "
        return "Intervals: prediction intervals unavailable; point forecast only. "
    label = {
        "calibrated": "80% interval is calibrated in rolling-origin CV",
        "calibration_warning": "80% interval is available but has a calibration warning",
        "calibration_fail": "80% interval is flagged for undercoverage risk",
        "future_only": "80% interval is future-only and not CV-calibrated",
        "adjusted_not_recalibrated": "80% interval was adjusted after calibration and not recalibrated",
        "insufficient_observations": "80% interval has too few CV observations for confident calibration",
    }.get(status, status.replace("_", " "))
    return f"Intervals: {label} ({_fmt_number(lo)} to {_fmt_number(hi)}). "


def _horizon_sentence(trust_row: dict[str, Any], final: dict[str, Any]) -> str:
    state = str(trust_row.get("horizon_trust_state") or final.get("horizon_trust_state") or "no_rolling_origin_evidence")
    requested = _display_value(trust_row.get("requested_horizon") or final.get("requested_horizon"), "the requested")
    validated = _display_value(trust_row.get("validated_through_horizon") or final.get("validated_through_horizon"), "0")
    full_claim = _truthy(trust_row.get("full_horizon_claim_allowed"))
    cv_windows = _display_value(trust_row.get("cv_windows") or final.get("cv_windows"), "unknown")
    if state == "full_horizon_validated" and full_claim:
        return f"Horizon: validated through the full requested {requested}-step horizon. "
    if state == "full_horizon_validated":
        return f"Horizon: full requested {requested}-step horizon was evaluated on only {cv_windows} CV window(s); planning claim limited. "
    if state == "partial_horizon_validated":
        return f"Horizon: validated through step {validated} of {requested}; later steps are directional. "
    return "Horizon: no rolling-origin validation; exploratory only. "


def _transform_sentence(run: ForecastRun, value_label: str) -> str:
    if run.spec.transform.normalization_factor_col:
        label = f" ({run.spec.transform.normalization_label})" if run.spec.transform.normalization_label else ""
        return (
            f"Values are in normalized units{label}; do not compare directly to raw reported actuals "
            "unless future normalization factors are supplied."
        )
    if run.spec.transform.target != "none":
        return f"Values are inverse-transformed into {value_label} for reporting; interval interpretation after nonlinear transforms is approximate."
    if run.spec.unit_label:
        return f"Values are reported in {value_label}."
    if run.spec.target_col != "y":
        return f"Values are reported in {value_label}."
    return ""


def _portfolio_direction(series_rows: list[dict[str, Any]]) -> tuple[str, float | None]:
    pcts = [_float_or_none(row.get("direction_pct_vs_recent")) for row in series_rows]
    pcts = [pct for pct in pcts if pct is not None]
    if not pcts:
        return "unavailable", None
    avg = float(sum(pcts) / len(pcts))
    if abs(avg) < 0.02:
        return "flat", avg
    return ("up" if avg > 0 else "down"), avg


def _portfolio_direction_text(direction: str, pct: float | None) -> str:
    if pct is None or direction == "unavailable":
        return "Aggregate direction versus recent actuals is unavailable."
    if direction == "flat":
        return "Average series direction is approximately flat versus recent actuals."
    return f"Average series direction is {direction} about {abs(pct):.0%} versus recent actuals."


def _direction_split(series_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"up": 0, "down": 0, "flat": 0, "unavailable": 0}
    for row in series_rows:
        direction = str(row.get("direction") or "unavailable")
        if direction not in counts:
            direction = "unavailable"
        counts[direction] += 1
    return counts


def _direction_split_text(split: dict[str, int]) -> str:
    return (
        "Direction split versus recent actuals: "
        f"up {split.get('up', 0)}, down {split.get('down', 0)}, flat {split.get('flat', 0)}, "
        f"unavailable {split.get('unavailable', 0)}."
    )


def _watch_series_text(series_rows: list[dict[str, Any]]) -> str:
    ranked = sorted(series_rows, key=lambda row: _sort_score(row.get("trust_score_0_100")))[:2]
    parts = []
    for row in ranked:
        score = _display_value(row.get("trust_score_0_100"), "N/A")
        action = str(row.get("next_action") or "validate before use").rstrip(".")
        parts.append(f"{row['unique_id']} ({row.get('trust_level', 'Unknown')} trust, {score}/100; next: {action})")
    return "; ".join(parts) if parts else "no specific series flagged"


def _portfolio_next_action(series_rows: list[dict[str, Any]]) -> str:
    low = [row for row in series_rows if row.get("trust_level") == "Low"]
    if low:
        if len(low) == len(series_rows):
            return "Treat the portfolio as diagnostic until Low-trust caveats are resolved."
        medium = [row for row in series_rows if row.get("trust_level") == "Medium"]
        medium_part = f", then validate {len(medium)} Medium-trust row(s)" if medium else ""
        return f"Address {len(low)} Low-trust row(s) in trust_summary.csv first{medium_part} before stakeholder sharing."
    medium = [row for row in series_rows if row.get("trust_level") == "Medium"]
    if medium:
        return "Review Medium-trust caveats and use only planning_eligible=True rows for stakeholder summaries."
    return "Validate the portfolio against plan, prior year, and known future events before stakeholder use."


def _average_optional(series_rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in series_rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _portfolio_value_label(series_rows: list[dict[str, Any]], run: ForecastRun) -> str:
    labels = {str(row.get("value_unit_label")) for row in series_rows if row.get("value_unit_label")}
    if len(labels) == 1:
        return next(iter(labels))
    return _value_unit_label(run)


def _first_item(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    for sep in ["; ", " | ", "\n"]:
        if sep in text:
            text = text.split(sep)[0]
            break
    return text.strip(". ") or fallback


def _truthy(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        return bool(value) and float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_score(value: object) -> float:
    score = _float_or_none(value)
    return score if score is not None else 101.0


def _date_text(value: object) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value) if value else None


def _display_value(value: object, fallback: str) -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else fallback


def _value_unit_label(run: ForecastRun) -> str:
    if run.spec.transform.normalization_factor_col:
        return "normalized units"
    label = str(run.spec.unit_label).strip() if run.spec.unit_label is not None else ""
    if label:
        return label
    if run.spec.target_col and run.spec.target_col != "y":
        return f"{run.spec.target_col} units"
    return "reported target units"


def _format_measure(value: float, unit_label: str) -> str:
    label = str(unit_label or "").strip()
    if _is_currency_label(label):
        prefix = "-" if value < 0 else ""
        suffix = " USD" if label.lower() == "usd" else ""
        return f"{prefix}${_fmt_number(abs(value))}{suffix}"
    if label:
        return f"{_fmt_number(value)} {label}"
    return _fmt_number(value)


def _format_delta(value: float, unit_label: str) -> str:
    label = str(unit_label or "").strip()
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    if _is_currency_label(label):
        suffix = " USD" if label.lower() == "usd" else ""
        return f"{sign}${_fmt_number(magnitude)}{suffix}"
    if label:
        return f"{sign}{_fmt_number(magnitude)} {label}"
    return f"{sign}{_fmt_number(magnitude)}"


def _is_currency_label(label: str) -> bool:
    return label in {"$"} or label.lower() in {"usd", "us dollars", "dollars"}


def _fmt_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _squash(text: str) -> str:
    return " ".join(text.split())
