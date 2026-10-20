"""Driver admission and matched ablations for the existing chronological optimizer."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nixtla_scaffold.feature_sources import DriverSourceError, source_features
from nixtla_scaffold.schema import ForecastSpec, KnownFutureRegressor
from nixtla_scaffold.signals import screen_driver_relationships


def prepare_driver_research(tuning: pd.DataFrame, spec: ForecastSpec) -> tuple[ForecastSpec, dict]:
    """Screen only the supplied inner partition; all outcomes remain unadmitted."""
    from nixtla_scaffold.profile import infer_frequency, season_length_for_freq

    rows: list[dict[str, Any]] = []
    features = tuning[["unique_id", "ds", "y"]].copy()
    declarations: dict[str, KnownFutureRegressor] = {}
    frequency = spec.freq or infer_frequency(tuning)
    season = spec.season_length or season_length_for_freq(frequency, int(tuning.groupby("unique_id").size().min()))
    bounded = len(tuning) <= 100_000 and tuning.unique_id.nunique() <= 50
    for index, regressor in enumerate(spec.regressors):
        row = {"name": regressor.name, "status": "blocked", "training_admitted": False}
        rows.append(row)
        if index >= 8 or not bounded:
            row["reason"] = "Driver screening scope bound exceeded; no source read."
            continue
        if regressor.source is None or regressor.mode != "model_candidate":
            row["reason"] = "Predictive research requires a contracted model-candidate snapshot."
            continue
        if regressor.availability != "historical_only":
            row["reason"] = "Automatic lag research requires historical-only inputs; future/scenario declarations retain their explicit advisory experiment path."
            continue
        name = regressor.value_col or regressor.name
        if name in declarations or name in {"unique_id", "ds", "y"}:
            row["reason"] = "Duplicate or reserved feature key; no source read."
            continue
        try:
            joined, _, source = source_features(
                tuning, tuning[["unique_id", "ds"]].iloc[:0], spec, regressor,
                freq=frequency, origin=pd.to_datetime(tuning.ds).max(),
            )
        except (DriverSourceError, OSError, pd.errors.ParserError) as exc:
            row["reason"] = str(exc)
            continue
        features = features.merge(joined[["unique_id", "ds", name]],
                                  on=["unique_id", "ds"], validate="one_to_one")
        declarations[name] = regressor
        row.update(value_col=name, source=source, status="screened")
    screen = (screen_driver_relationships(features, features=tuple(declarations),
                                         horizon=spec.horizon, season_length=season)
              if declarations else {"cards": [], "tests": [], "tests_searched": 0})
    admitted = []
    for card in screen["cards"]:
        regressor = declarations[card["feature"]]
        row = next(row for row in rows if row.get("value_col") == card["feature"])
        row["screening"] = card
        row["status"] = card["disposition"]
        lag = card["best_safe_lag"]
        if lag is None:
            continue
        if regressor.historical_lags and lag not in regressor.historical_lags:
            row.update(status="blocked", reason="Screen-selected lag is outside the explicitly frozen lag recipe.")
            continue
        admitted.append(replace(regressor, historical_lags=(lag,)))
        row["frozen_lags"] = [lag]
    return replace(spec, regressors=tuple(admitted)), {
        "schema_version": "nixtla_scaffold.driver_research.v1",
        "partition": "inner_tuning_only", "sources": rows, "screening": screen,
        "claim": "Feature/lag selection is exploratory inside tuning; only later frozen confirmation can support a recommendation.",
        "automatic_adoption": False,
    }


def matched_driver_specs(spec: ForecastSpec, name: str, model: str) -> tuple[ForecastSpec, ForecastSpec]:
    regressors = tuple(item for item in spec.regressors if item.name == name)
    if len(regressors) != 1 or model not in {"Ridge", "RandomForest", "LightGBM"}:
        raise ValueError("A driver ablation requires one uniquely named feature and an explicit supported learner")
    if spec.model_allowlist and model not in spec.model_allowlist:
        raise ValueError("Driver learner is outside the explicitly requested model allowlist")
    if spec.model_policy in {"baseline", "statsforecast"}:
        raise ValueError("Driver research cannot silently upgrade an explicitly non-ML model policy")
    candidate = replace(spec, regressors=regressors, model_allowlist=(model,),
                        weighted_ensemble=False, events=(), challengers=(),
                        hierarchy_reconciliation="none", train_known_future_regressors=True)
    return replace(candidate, regressors=(), train_known_future_regressors=False), candidate


def driver_pair_contract(control_dir: Path, candidate_dir: Path, model: str) -> dict:
    """Compare actual executed recipes, allowing only the declared feature treatment."""
    from nixtla_scaffold.schema import forecast_spec_from_dict

    manifests = [json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                 for directory in (control_dir, candidate_dir)]
    specs = [forecast_spec_from_dict(manifest["spec"]) for manifest in manifests]
    control, candidate = specs
    expected = replace(candidate, regressors=(), train_known_future_regressors=False)
    same_spec = control == expected
    evidence = [json.loads((directory / "audit" / "candidate_evidence.json").read_text(encoding="utf-8"))
                for directory in (control_dir, candidate_dir)]
    normalized, fit_keys, feature_columns = [], [], []
    for index, item in enumerate(evidence):
        recipes = {}
        keys = set()
        columns = set()
        for fit in item.get("fits", []):
            recipe = item.get("recipes", {}).get(fit.get("recipe_id"), {})
            if recipe.get("model") != model or recipe.get("engine") != "mlforecast":
                continue
            if fit.get("status") != "succeeded":
                continue
            execution = dict(recipe["execution"])
            external = tuple(execution.pop("external_feature_columns", []))
            columns.add(external)
            sources = execution.pop("driver_source_receipts", [])
            if index and not sources:
                continue
            names = execution.get("actual_feature_names")
            if names is not None:
                if len(names) != len(set(names)) or not set(external).issubset(names):
                    continue
                execution["actual_feature_names"] = [name for name in names if name not in external]
            options = dict(execution.get("fit_options", {}))
            # No-feature frames contain only keys/target; [] is equivalent there.
            if options.get("static_features") == []:
                options.pop("static_features")
            execution["fit_options"] = options
            key = (str(fit["unique_id"]), str(fit["cutoff"]), str(fit.get("stage", "")))
            keys.add(key)
            recipes[key] = json.dumps({**recipe, "execution": execution}, sort_keys=True, default=str)
        normalized.append(recipes)
        fit_keys.append(keys)
        feature_columns.append(columns)
    frozen_columns = (len(feature_columns[1]) == 1 and bool(next(iter(feature_columns[1]), ()))
                      and feature_columns[0] == {()})
    matched = bool(same_spec and fit_keys[0] and fit_keys[0] == fit_keys[1]
                   and normalized[0] == normalized[1] and frozen_columns)
    return {
        "matched": matched, "model": model, "same_non_feature_spec": same_spec,
        "same_executed_recipes": bool(normalized[0] and normalized[0] == normalized[1]),
        "same_successful_origin_keys": bool(fit_keys[0] and fit_keys[0] == fit_keys[1]),
        "frozen_external_columns": frozen_columns,
        "candidate_feature_columns": [list(item) for item in sorted(feature_columns[1])],
        "reason": "" if matched else "Feature treatment changed unrelated configuration, lacked source receipts, or lost matching learner/origin support.",
    }


def assess_driver_ablation(control_dir: Path, candidate_dir: Path, model: str, *, primary: str,
                           secondary: str, policy) -> dict:
    from nixtla_scaffold.research import (
        _paired_tuning_evidence, _selected_backtest_rows, _cutoff_metrics,
        _METRIC_COLUMNS, _relative_improvement, _relative_regression,
    )

    contract = driver_pair_contract(control_dir, candidate_dir, model)
    paired = _paired_tuning_evidence(control_dir, candidate_dir, primary_metric=primary, model=model)
    paired.pop("frame")
    evidence = []
    full_support = True
    for directory in (control_dir, candidate_dir):
        frame = _selected_backtest_rows(directory, model=model)
        expected = pd.read_csv(directory / "appendix" / "cutoff_contract.csv", dtype={"unique_id": str})
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        horizon = manifest["spec"]["horizon"]
        keys = ["unique_id", "cutoff", "ds", "horizon_step"]
        for key in ("cutoff", "ds"):
            expected[key] = pd.to_datetime(expected[key])
        full_support &= (not expected.empty and len(frame) == len(expected)
                         and not frame.empty and frame[keys].merge(expected[keys], on=keys, how="outer",
                                                                   indicator=True)["_merge"].eq("both").all())
        full_support &= (expected["selection_horizon"].eq(horizon).all()
                         and expected.groupby(["unique_id", "cutoff"]).size().eq(horizon).all())
        if frame.empty:
            evidence.append({})
            continue
        metrics = _cutoff_metrics(frame.assign(variant="fixed_learner"))
        metric_columns = {key: value for key, value in _METRIC_COLUMNS.items() if value in metrics}
        # Equal weights within series, then across the declared panel.
        means = metrics.groupby("unique_id")[list(metric_columns.values())].mean().mean()
        valid = metrics[list(metric_columns.values())].apply(pd.to_numeric, errors="coerce")
        bias = (pd.to_numeric(frame.yhat) - pd.to_numeric(frame.y_actual)).groupby(frame.unique_id).mean()
        scale = frame.groupby("unique_id").rmsse_scale.first().apply(pd.to_numeric, errors="coerce")
        finance = {}
        if primary == "avg_landing_mape_pct":
            from nixtla_scaffold.finance import finance_scorecard
            history_path = manifest["outputs"].get("observed_history") or manifest["outputs"]["history"]
            history = pd.read_csv(directory / history_path, dtype={"unique_id": str}, keep_default_na=False)
            scores, _ = finance_scorecard(
                history, frame[["unique_id", "ds", "cutoff", "y_actual", "yhat"]].rename(
                    columns={"y_actual": "y", "yhat": model}), [model], horizon=horizon,
                clip_negative=manifest["spec"]["selection_policy"]["clip_negative"])
            finance = {f"avg_{name}": float(scores[name].mean()) if scores[name].notna().all() else None
                       for name in ("landing_mape_pct", "next28_mape_pct")}
        evidence.append({**{key: float(means[value]) if valid[value].notna().all() else None
                            for key, value in metric_columns.items()}, **finance,
                         "absolute_scaled_bias": float((bias.abs() / scale).mean())
                         if scale.gt(0).all() else None})
    improvement = _relative_improvement(evidence[0].get(primary), evidence[1].get(primary))
    secondary_regression = _relative_regression(evidence[0].get(secondary), evidence[1].get(secondary))
    biases = [item.get("absolute_scaled_bias") for item in evidence]
    bias_ok = all(value is not None for value in biases) and biases[1] <= biases[0] + policy.maximum_secondary_metric_regression
    passed = bool(contract["matched"] and full_support and paired.get("exact_match")
                  and paired.get("stable_window_fraction", 0) >= 0.6
                  and improvement is not None and improvement > 0
                  and improvement >= policy.minimum_primary_metric_improvement
                  and secondary_regression is not None
                  and secondary_regression <= policy.maximum_secondary_metric_regression and bias_ok)
    return {"passed": passed, "contract": contract, "paired": paired, "full_planned_support": bool(full_support),
            "control": evidence[0], "feature": evidence[1], "primary_metric": primary,
            "primary_improvement": improvement, "secondary_regression": secondary_regression,
            "absolute_scaled_bias_pass": bias_ok, "model": model,
            "claim": "Inner tuning ablation, not untouched confirmation or causality."}


def assess_driver_confirmation(runs, actuals, training, *, model, output_dir,
                               primary, secondary, policy, window) -> dict:
    from nixtla_scaffold.research import (
        _score_confirmation_run, _confirmation_metrics, _confirmation_window_comparison,
    )
    contract = driver_pair_contract(output_dir / "baseline", output_dir / "candidate", model)
    metrics, coverage = [], []
    biases = []
    for variant in ("baseline", "candidate"):
        scored, support = _score_confirmation_run(runs[variant], actuals, variant=variant, window=window, model=model)
        coverage.append(support["matched"] == len(actuals) and support["extra"] == 0)
        metrics.extend(_confirmation_metrics(scored, training, season_length=runs[variant].profile.season_length,
                                             variant=variant, window=window, spec=runs[variant].spec))
        per_series = []
        for uid, group in scored.groupby("unique_id"):
            values = training.loc[training.unique_id.eq(uid)].sort_values("ds").y
            scale = values.diff(runs[variant].profile.season_length).pow(2).mean() ** 0.5
            per_series.append(abs((group.yhat - group.y_actual).mean()) / scale if scale > 0 else float("nan"))
        biases.append(pd.Series(per_series, dtype=float))
    comparison = _confirmation_window_comparison(metrics, window=window, primary_metric=primary, secondary_metric=secondary)
    improvement = comparison["primary_improvement"]
    regression = comparison["secondary_regression"]
    bias_ok = bool(all(not item.empty and item.notna().all() for item in biases)
                   and biases[1].mean() <= biases[0].mean() + policy.maximum_secondary_metric_regression)
    passed = bool(contract["matched"] and all(coverage)
                  and improvement is not None and improvement > 0 and improvement >= policy.minimum_primary_metric_improvement
                  and regression is not None and regression <= policy.maximum_secondary_metric_regression and bias_ok)
    return {"window": window, "passed": passed, "contract": contract, "comparison": comparison,
            "complete_support": all(coverage), "absolute_scaled_bias_pass": bias_ok,
            "claim": "Frozen fixed-learner feature confirmation, not causality or adoption."}
