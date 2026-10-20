"""Chronological deployed-policy qualification and fail-closed local delivery."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from nixtla_scaffold.configuration import screen_candidates
from nixtla_scaffold.conformal import calibrate_path, require_backend
from nixtla_scaffold.evaluation import training_slice
from nixtla_scaffold.finance import finance_scorecard
from nixtla_scaffold.models import _metrics_from_cv, model_columns, select_champions
from nixtla_scaffold.schema import ForecastRun, ForecastSpec, IntervalCalibrationSpec, _frame_hash


class ProductionReleaseError(RuntimeError):
    """Execution completed, but the saved statistical release contract failed."""


def reserve_production_output(output: Path) -> None:
    """Reserve one writer, including when callers supplied an empty directory."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Production output must be a new or empty directory; preserving {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".production-write-claim").open("x", encoding="utf-8") as claim:
        claim.write("Reserved production output; do not reuse or overwrite this directory.\n")


def prepare_production_spec(spec: ForecastSpec) -> ForecastSpec:
    policy = spec.production
    if policy is None:
        return spec
    if (spec.events or spec.hierarchy or spec.hierarchy_reconciliation != "none"
            or spec.custom_models or spec.challengers or spec.regressors or spec.driver_snapshots
            or spec.transform.normalization_factor_col):
        raise ValueError("Production v1 qualifies native, unadjusted, unreconciled paths only; scenarios, "
                         "hierarchies, external models, normalization and business drivers need their own exact replay.")
    if spec.weighted_ensemble or spec.ensemble.advisory_policies:
        raise ValueError("Production v1 requires weighted_ensemble=False; use production.ensemble_members "
                         "for a frozen equal-weight blend with exact component replay.")
    if (spec.run_mode != "accuracy-first" or spec.evaluation.strategy != "rolling-origin"
            or spec.selection_policy.configuration_guard != "exclude"):
        raise ValueError("Production requires accuracy-first, rolling-origin evaluation and configuration_guard='exclude'.")
    if any(type(level) is not int for level in spec.levels) or tuple(sorted(spec.levels)) != (80, 95):
        raise ValueError("Production v1 requires exactly the declared 80% and 95% interval levels.")
    if spec.interval_calibration.aggregate_totals:
        raise ValueError("Production v1 provides marginal policy intervals, not certified aggregate-total intervals.")
    require_backend()
    minimum = policy.minimum_selection_windows + policy.calibration_windows + policy.minimum_assessment_windows
    allowlist = spec.model_allowlist
    if allowlist:
        allowlist = tuple(dict.fromkeys((*allowlist, "Naive", "SeasonalNaive", *policy.ensemble_members)))
    return replace(
        spec, model_allowlist=allowlist, require_backtest=True, strict_cv_horizon=True,
        evaluation=replace(spec.evaluation, max_windows=max(minimum, spec.evaluation.max_windows),
                           step_size=max(spec.horizon, spec.evaluation.step_size or spec.horizon)),
    )


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _complete_prior(cv: pd.DataFrame, cutoff, horizon: int) -> pd.DataFrame:
    availability = cv.groupby("cutoff").agg(
        end=("ds", "max"), available=("available_at", "max"), rows=("ds", "size"),
    )
    landed = availability.index[
        availability.end.le(cutoff) & availability.available.le(cutoff) & availability.rows.eq(horizon)
    ]
    return cv.loc[cv.cutoff.isin(landed)]


def _choose(history, prior, paths, spec, *, cutoff, freq, season):
    policy = spec.production
    if policy is None or prior.cutoff.nunique() < policy.minimum_selection_windows:
        raise ValueError("Insufficient completely landed prior selection windows")
    candidates = [
        name for name in model_columns(paths)
        if name in prior and np.isfinite(paths[name]).all()
        and np.isfinite(prior[[name, "y"]]).all().all()
    ]
    if not candidates:
        raise ValueError("No complete candidate on the entire prior comparison cohort")
    evidence = prior[["unique_id", "ds", "cutoff", "y", *candidates]]
    scores = _metrics_from_cv(evidence)
    prefix = training_slice(history, cutoff, None)
    if spec.selection_policy.objective == "monthly_landing":
        finance, _ = finance_scorecard(prefix, evidence, candidates, horizon=spec.horizon)
        scores = scores.merge(finance, on=["unique_id", "model"], validate="one_to_one")
        scores["selection_eligible"] = np.isfinite(
            scores[["landing_mape_pct", "next28_mape_pct", "daily_wmape_pct"]]
        ).all(axis=1)
    else:
        scores["selection_eligible"] = np.isfinite(scores.rmse)
    screened, scores, admission = screen_candidates(
        prefix, paths[["unique_id", "ds", *candidates]], scores, spec, freq=freq, season=season,
    )
    if not scores.selection_eligible.any():
        raise ValueError("No configuration-admissible candidate with complete prior objective evidence")
    choice = select_champions(screened, scores, policy=spec.selection_policy).iloc[0].to_dict()
    receipt = {
        "cutoff": pd.Timestamp(cutoff).isoformat(), "selected_component": choice["selected_model"],
        "selection_reason": choice["selection_reason"], "prior_windows": prior.cutoff.nunique(),
        "latest_prior_target": prior.ds.max().isoformat(),
        "latest_prior_available_at": prior.available_at.max().isoformat(),
        "scores": _records(scores), "admission": admission,
    }
    return choice, receipt


def _add_frozen_blend(frame, members):
    result = frame.copy()
    if members:
        if not set(members).issubset(result):
            result["ProductionBlend"] = np.nan
        else:
            values = result[list(members)].to_numpy(float)
            result["ProductionBlend"] = np.where(np.isfinite(values).all(axis=1), values.mean(axis=1), np.nan)
    return result


def _replay(history, cv, future, spec, *, freq, season):
    policy = spec.production
    if policy is None:
        raise ValueError("A production policy is required")
    cv = _add_frozen_blend(cv, policy.ensemble_members)
    future = _add_frozen_blend(future, policy.ensemble_members)
    if "available_at" not in cv:
        releases = history[["unique_id", "ds", "available_at"]] if "available_at" in history else None
        cv = cv.merge(releases, on=["unique_id", "ds"], validate="many_to_one") if releases is not None else cv.assign(available_at=cv.ds)
    cv["available_at"] = pd.to_datetime(cv.available_at, errors="raise")
    if cv.available_at.isna().any() or cv.available_at.lt(cv.ds).any():
        raise ValueError("Production requires complete actual-availability timestamps not preceding target dates")
    if cv.duplicated(["cutoff", "ds"]).any():
        raise ValueError("Production replay has duplicate cutoff/target keys")
    frames, decisions = [], []
    last_end = None
    for index, (cutoff, group) in enumerate(cv.groupby("cutoff", sort=True)):
        group = group.sort_values("ds")
        expected = pd.date_range(cutoff, periods=spec.horizon + 1, freq=freq)[1:]
        if not pd.DatetimeIndex(group.ds).equals(expected) or (last_end is not None and group.ds.min() <= last_end):
            raise ValueError("Production requires complete, disjoint, origin-aligned evaluation horizons")
        last_end = group.ds.max()
        if index < policy.minimum_selection_windows:
            continue
        prior = _complete_prior(cv, cutoff, spec.horizon)
        paths = group.drop(columns=["cutoff", "y", "available_at"])
        row = group[["unique_id", "ds", "cutoff", "y", "available_at"]].copy()
        try:
            choice, decision = _choose(history, prior, paths, spec, cutoff=cutoff, freq=freq, season=season)
        except ValueError as exc:
            decision = {"cutoff": cutoff.isoformat(), "status": "selection_unavailable", "reason": str(exc)}
            row["Policy"], row["selected_component"] = np.nan, ""
        else:
            row["Policy"] = group[choice["selected_model"]]
            row["selected_component"] = choice["selected_model"]
        for benchmark in ("Naive", "SeasonalNaive"):
            row[benchmark] = group[benchmark] if benchmark in group else np.nan
        frames.append(row)
        decisions.append(decision)
    if not frames:
        raise ValueError("Production needs more than the selection warm-up origins; no policy evaluation is possible")
    origin = history.ds.max()
    choice, decision = _choose(
        history, _complete_prior(cv, origin, spec.horizon), future, spec,
        cutoff=origin, freq=freq, season=season,
    )
    decisions.append({**decision, "stage": "future"})
    forecast = future[["unique_id", "ds"]].copy()
    forecast["Policy"] = future[choice["selected_model"]]
    forecast["selected_component"] = choice["selected_model"]
    for benchmark in ("Naive", "SeasonalNaive"):
        forecast[benchmark] = future[benchmark] if benchmark in future else np.nan
    return pd.concat(frames, ignore_index=True), forecast, decisions


def _metric_scores(history, frame, names, spec):
    scores = _metrics_from_cv(frame[["unique_id", "ds", "cutoff", "y", *names]])
    if spec.selection_policy.objective == "monthly_landing":
        finance, _ = finance_scorecard(history, frame, names, horizon=spec.horizon)
        scores = scores.merge(finance, on=["unique_id", "model"], validate="one_to_one")
        scores["error_pct"], scores["bias_pct"] = scores.landing_mape_pct, scores.landing_bias_pct
        scores["benchmark_loss"] = scores.landing_mape_pct
    else:
        scores["error_pct"], scores["bias_pct"] = scores.wape * 100, scores.bias * 100
        scores["benchmark_loss"] = scores.rmse
    return scores


def _interval_score(frame, name, level):
    lo, hi, y = frame[f"{name}-lo-{level}"], frame[f"{name}-hi-{level}"], frame.y
    alpha = (100 - level) / 100
    score = hi - lo + 2 / alpha * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))
    return float(score.mean())


def _not_worse(value, reference):
    return (np.isfinite(value) and np.isfinite(reference)
            and (value <= reference or np.isclose(value, reference, rtol=1e-12, atol=0)))


def qualify_production_run(run: ForecastRun) -> ForecastRun:
    """Qualify every series, retaining failed cohorts instead of selecting surviving rows."""
    policy, spec = run.spec.production, run.spec
    if policy is None:
        return run
    history = run.observed_history if not run.observed_history.empty else run.history
    checks, series, replay_rows, choices, final_frames, interval_states = [], [], [], [], [], []

    def check(uid, name, passed, observed, required):
        checks.append({"unique_id": str(uid), "check": name, "passed": bool(passed),
                       "observed": observed, "required": required})

    for uid, actuals in history.groupby("unique_id", sort=True):
        actuals = actuals.sort_values("ds")
        cv = run.backtest_predictions.loc[run.backtest_predictions.unique_id.eq(uid)].copy()
        origins = run.model_policy_resolution.get("evaluation", {}).get("origins", [])
        expected = {
            (pd.Timestamp(item["cutoff"]), pd.Timestamp(target))
            for item in origins if str(item["unique_id"]) == str(uid) for target in item["targets"]
        }
        observed = set(zip(pd.to_datetime(cv.cutoff), pd.to_datetime(cv.ds)))
        if not expected or observed != expected or len(cv) != len(expected):
            raise ValueError(f"{uid}: production backtest differs from the frozen origin plan; no missing origin may be dropped")
        truths = cv[["ds", "y"]].merge(actuals[["ds", "y"]], on="ds", how="left",
                                     suffixes=("", "_source"), validate="many_to_one")
        if not (truths.y.eq(truths.y_source) | (truths.y.isna() & truths.y_source.isna())).all():
            raise ValueError(f"{uid}: production backtest actuals differ from the source snapshot")
        fc = run.all_models.loc[run.all_models.unique_id.eq(uid)].copy().sort_values("ds")
        replay, future, decisions = _replay(actuals, cv, fc, spec, freq=run.profile.freq, season=run.profile.season_length)
        choices.extend({"unique_id": str(uid), **decision} for decision in decisions)
        names = ["Policy", "Naive"] + (["SeasonalNaive"] if run.profile.season_length > 1 else [])
        full = np.isfinite(replay[["y", *names]]).all().all()
        check(uid, "complete_policy_and_benchmarks", full, int(replay[["y", *names]].isna().sum().sum()), 0)
        check(uid, "observed_history", np.isfinite(actuals.y).all(), int((~np.isfinite(actuals.y)).sum()), 0)
        if "available_at" in actuals:
            releases = pd.to_datetime(actuals.available_at, errors="raise")
            check(uid, "future_origin_availability", releases.notna().all() and releases.le(actuals.ds.max()).all(),
                  int((releases.isna() | releases.gt(actuals.ds.max())).sum()), 0)
        calibrated, future_calibrated = replay.copy(), future.copy()
        for name in names:
            calibrated, future_calibrated, state = calibrate_path(
                calibrated, future_calibrated,
                IntervalCalibrationSpec(method="split_conformal", max_windows=policy.calibration_windows),
                model=name, horizon=spec.horizon, freq=run.profile.freq, origin=actuals.ds.max(),
                levels=spec.levels, path_id=f"production-v1:{uid}:{name}",
            )
            interval_states.append({"unique_id": str(uid), "model": name, **state})
        assessment_cutoffs = sorted(replay.cutoff.unique())[policy.calibration_windows:]
        assessed = calibrated.loc[calibrated.cutoff.isin(assessment_cutoffs)]
        check(uid, "assessment_windows", len(assessment_cutoffs) >= policy.minimum_assessment_windows,
              len(assessment_cutoffs), f">={policy.minimum_assessment_windows}")
        scores = _metric_scores(actuals, assessed, names, spec)
        numeric = scores.set_index("model")
        policy_row = numeric.loc["Policy"] if "Policy" in numeric.index else pd.Series(dtype=float)
        error, bias = policy_row.get("error_pct", np.nan), policy_row.get("bias_pct", np.nan)
        loss = policy_row.get("benchmark_loss", np.nan)
        check(uid, "objective_error_pct", np.isfinite(error) and error <= policy.max_error_pct,
              float(error) if np.isfinite(error) else None, f"<={policy.max_error_pct}")
        check(uid, "absolute_bias_pct", np.isfinite(bias) and abs(bias) <= policy.max_abs_bias_pct,
              float(bias) if np.isfinite(bias) else None, f"absolute <= {policy.max_abs_bias_pct}")
        for name in names[1:]:
            reference = numeric.loc[name, "benchmark_loss"] if name in numeric.index else np.nan
            check(uid, f"benchmark_{name}", _not_worse(loss, reference),
                  float(loss) if np.isfinite(loss) else None, float(reference) if np.isfinite(reference) else None)
        coverage = []
        for level in spec.levels:
            columns = [f"{name}-{side}-{level}" for name in names for side in ("lo", "hi")]
            available = not assessed.empty and np.isfinite(assessed[columns]).all().all()
            check(uid, f"assessment_intervals_{level}", available, bool(available), True)
            future_available = np.isfinite(future_calibrated[[f"Policy-lo-{level}", f"Policy-hi-{level}"]]).all().all()
            check(uid, f"future_intervals_{level}", future_available, bool(future_available), True)
            observed = float(assessed.y.between(assessed[f"Policy-lo-{level}"], assessed[f"Policy-hi-{level}"]).mean()) if available else None
            check(uid, f"coverage_{level}", observed is not None and observed >= level / 100 - policy.coverage_tolerance,
                  observed, f">={level / 100 - policy.coverage_tolerance:.3f}")
            score = _interval_score(assessed, "Policy", level) if available else np.nan
            for name in names[1:]:
                baseline_score = _interval_score(assessed, name, level) if available else np.nan
                check(uid, f"interval_score_{level}_{name}",
                      _not_worse(score, baseline_score),
                      float(score) if np.isfinite(score) else None,
                      float(baseline_score) if np.isfinite(baseline_score) else None)
            coverage.append({"level": level, "coverage": observed, "observations": len(assessed),
                             "mean_interval_score": float(score) if np.isfinite(score) else None})
        calibrated["assessment"] = calibrated.cutoff.isin(assessment_cutoffs)
        replay_rows.extend(_records(calibrated))
        series.append({"unique_id": str(uid), "selected_component": future.selected_component.iloc[0],
                       "policy_windows": replay.cutoff.nunique(), "assessment_windows": len(assessment_cutoffs),
                       "scores": _records(scores), "intervals": coverage})
        selected = future_calibrated[["unique_id", "ds", "selected_component", "Policy",
                                     *[f"Policy-{side}-{level}" for level in spec.levels for side in ("lo", "hi")]]]
        selected = selected.rename(columns={
            "Policy": "yhat", **{f"Policy-{side}-{level}": f"yhat_{side}_{level}"
                               for level in spec.levels for side in ("lo", "hi")},
        }).assign(model=future.selected_component.iloc[0])
        final_frames.append(selected)
    passed = bool(checks) and all(item["passed"] for item in checks)
    report = {
        "schema_version": 1, "status": "qualified" if passed else "rejected",
        "release_allowed": passed, "policy": policy.to_dict(), "objective": spec.selection_policy.contract(),
        "threshold_authority": "declared_runtime_contract_not_business_SLA",
        "claim": "historical_rolling_policy_qualification_not_untouched_confirmation_or_business_approval",
        "interval_scope": "Marginal daily/period horizons only; never sum these bounds into monthly or simultaneous intervals.",
        "assumptions": [
            "Input snapshot is not a certified historical revision archive.",
            "Target timestamps stand in for publication times when available_at is absent.",
            "Disjoint residual windows do not prove independence or exchangeability under future regime shifts.",
        ],
        "selection_rule": "Full landed prior cohort; same objective, naive guard and configuration admission at every origin.",
        "calibration_backend": "conformalforecast 0.1.0 split conformal; existing adapter receipts retained",
        "series": series, "checks": checks,
        "failed_checks": [item for item in checks if not item["passed"]],
    }
    forecast = pd.concat(final_frames, ignore_index=True)
    report["binding"] = {
        "forecast_sha256": _frame_hash(forecast, columns=list(forecast)),
        "history_sha256": _frame_hash(history, columns=list(history)),
        "spec_sha256": sha256(_dump(spec.to_dict()).encode()).hexdigest(),
        "candidate_backtest_sha256": _frame_hash(run.backtest_predictions, columns=list(run.backtest_predictions)),
    }
    selection = run.model_selection.copy()
    for item in series:
        mask = selection.unique_id.astype(str).eq(item["unique_id"])
        selection.loc[mask, "selected_model"] = item["selected_component"]
        selection.loc[mask, "selection_reason"] = "Production prior-only policy; qualification in production_readiness.json"
    all_models = _add_frozen_blend(run.all_models, policy.ensemble_members)
    qualified = replace(run, forecast=forecast, all_models=all_models, model_selection=selection,
                   model_policy_resolution={**run.model_policy_resolution, "production": {
                       "readiness": report, "backtest": replay_rows, "decisions": choices,
                       "calibration": interval_states,
                   }})
    from nixtla_scaffold.configuration import build_configuration_report
    return replace(qualified, model_policy_resolution={
        **qualified.model_policy_resolution, "configuration": build_configuration_report(qualified),
    })


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n"


def _release_summary(run, report):
    lines = [
        "# Statistical forecast release", "",
        f"Origin: {run.profile.end}. Horizon: {run.spec.horizon} periods ({run.profile.freq}).",
        "Every series passed the declared historical qualification contract.", "",
        "| Series | Selected component | Assessment windows | Error % | Signed bias % | 80% coverage | 95% coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["series"]:
        scores = next(row for row in item["scores"] if row["model"] == "Policy")
        coverage = {row["level"]: row["coverage"] * 100 for row in item["intervals"]}
        uid = item["unique_id"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {uid} | {item['selected_component']} | {item['assessment_windows']} | "
            f"{scores['error_pct']:.2f} | {scores['bias_pct']:.2f} | {coverage[80]:.1f}% | {coverage[95]:.1f}% |"
        )
    lines.extend([
        "", "Error is monthly Landing MAPE for the monthly objective, otherwise period WAPE. "
        "Model selection and interval calibration use only previously landed outcomes; "
        "the table measures the later assessment cohort, not the final component's full-history score.",
        "", report["interval_scope"], "",
        "This local statistical release is not business approval, official adoption, untouched confirmation "
        "or a guarantee about future outcomes. The full thresholds, checks and assumptions are in qualification.json.",
        "",
    ])
    return "\n".join(lines)


def write_production_delivery(run: ForecastRun, output: Path, *, release: bool = True) -> None:
    """Persist diagnostics first; publish a new directory only after every check passes."""
    evidence = run.model_policy_resolution.get("production")
    if run.spec.production is None:
        return
    if not evidence or "readiness" not in evidence:
        raise ProductionReleaseError("Production evidence is missing; no release was created")
    report = evidence["readiness"]
    history = run.observed_history if not run.observed_history.empty else run.history
    binding = {
        "forecast_sha256": _frame_hash(run.forecast, columns=list(run.forecast)),
        "history_sha256": _frame_hash(history, columns=list(history)),
        "spec_sha256": sha256(_dump(run.spec.to_dict()).encode()).hexdigest(),
        "candidate_backtest_sha256": _frame_hash(run.backtest_predictions, columns=list(run.backtest_predictions)),
    }
    if report.get("binding") != binding:
        raise ProductionReleaseError("Forecast, history, backtest or spec changed after production qualification; rerun required")
    if report["release_allowed"] != (bool(report["checks"]) and all(item["passed"] is True for item in report["checks"])):
        raise ProductionReleaseError("Production status conflicts with its checks; no release created")
    (output / "production_readiness.json").write_text(_dump(report), encoding="utf-8")
    pd.DataFrame(evidence["backtest"]).to_csv(output / "audit" / "production_backtest.csv", index=False)
    (output / "audit" / "production_decisions.json").write_text(_dump(evidence["decisions"]), encoding="utf-8")
    (output / "audit" / "production_calibration.json").write_text(_dump(evidence["calibration"]), encoding="utf-8")
    if not release:
        return
    if not report["release_allowed"]:
        failures = ", ".join(f"{item['unique_id']}/{item['check']}" for item in report["failed_checks"])
        raise ProductionReleaseError(f"Production release rejected: {failures}. Diagnostics saved in {output}")
    destination = output / "release"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing production release: {destination}")
    with TemporaryDirectory(prefix=".release-", dir=output) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        run.forecast.to_csv(stage / "forecast.csv", index=False)
        (stage / "qualification.json").write_text(_dump(report), encoding="utf-8")
        (stage / "summary.md").write_text(_release_summary(run, report), encoding="utf-8")
        source_manifest = run.manifest()
        manifest = {
            "schema_version": 1, "status": "qualified_statistical_release",
            "business_approval": "not_granted", "publication": "local_only",
            "reproducibility": source_manifest.get("reproducibility", {}),
            "spec": run.spec.to_dict(),
            "sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(stage.iterdir())},
        }
        (stage / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
        stage.rename(destination)
