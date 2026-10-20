"""Frozen ensemble recipes and paired, as-of-safe evaluation.

Imported backtests are retrospective evidence, not a new untouched holdout.
The research optimizer owns independent confirmation and promotion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np
import pandas as pd

from nixtla_scaffold.ensemble import observable_backtest_rows
from nixtla_scaffold.model_families import model_family

POLICIES = ("equal", "fixed", "inverse_error", "top_k", "family_diverse", "greedy", "horizon_greedy")
LEARNED = {"greedy", "horizon_greedy"}
META = {"unique_id", "ds", "cutoff", "y", "available_at", "horizon_step", "horizon"}


@dataclass(frozen=True)
class BlendRecipe:
    policy: str = "equal"
    members: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    max_models: int = 3
    version: int = 2
    objective: str = "daily_rmse"
    max_steps: int = 32
    min_origins: int = 3
    horizon_breaks: tuple[int, ...] = ()
    shrinkage_origins: float = 8.

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"unknown blend policy: {self.policy}")
        if not self.members or len(self.members) > 12 or len(set(self.members)) != len(self.members):
            raise ValueError("choose 1 to 12 distinct component models")
        if any(not isinstance(member, str) or not member or member in META for member in self.members):
            raise ValueError("component model names must be nonempty model identifiers")
        if type(self.max_models) is not int or not 1 <= self.max_models <= 12:
            raise ValueError("max_models must be between 1 and 12")
        if self.version != 2 or self.objective not in {"daily_rmse", "monthly_landing"}:
            raise ValueError("Learned recipe version must be 2 and objective must be daily_rmse or monthly_landing")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 32:
            raise ValueError("max_steps must be between 1 and 32")
        if type(self.min_origins) is not int or self.min_origins < 3:
            raise ValueError("Learned ensembles require at least three prior origins")
        if len(self.horizon_breaks) > 2 or any(type(x) is not int or x < 1 for x in self.horizon_breaks) or tuple(sorted(set(self.horizon_breaks))) != self.horizon_breaks:
            raise ValueError("Declare at most two increasing positive horizon breaks")
        if not math.isfinite(self.shrinkage_origins) or self.shrinkage_origins < 1:
            raise ValueError("shrinkage_origins must be finite and at least 1")
        if self.policy == "horizon_greedy" and not self.horizon_breaks:
            raise ValueError("horizon_greedy requires predeclared horizon breaks")
        if self.policy != "horizon_greedy" and self.horizon_breaks:
            raise ValueError("horizon breaks apply only to horizon_greedy")
        if self.policy not in LEARNED and (self.objective != "daily_rmse" or self.max_steps != 32 or self.min_origins != 3 or self.shrinkage_origins != 8.):
            raise ValueError("Learning settings apply only to learned recipes")
        if self.policy == "fixed":
            if len(self.weights) != len(self.members):
                raise ValueError("provide one weight for each component")
            if any(not math.isfinite(w) or w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError("weights must be finite, nonnegative and have a positive sum")
            scaled = tuple(w / max(self.weights) for w in self.weights)
            object.__setattr__(self, "weights", tuple(w / sum(scaled) for w in scaled))
        elif self.weights:
            raise ValueError("weights are accepted only for the fixed policy")

    @classmethod
    def from_dict(cls, value: dict) -> BlendRecipe:
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid ensemble recipe fields")
        if not isinstance(value.get("members", ()), (list, tuple)) or not isinstance(value.get("weights", ()), (list, tuple)):
            raise ValueError("recipe members and weights must be arrays")
        return cls(
            policy=value.get("policy", "equal"),
            members=tuple(value.get("members", ())),
            weights=tuple(float(w) for w in value.get("weights", ())),
            max_models=value.get("max_models", 3),
            version=value.get("version", 2), objective=value.get("objective", "daily_rmse"),
            max_steps=value.get("max_steps", 32), min_origins=value.get("min_origins", 3),
            horizon_breaks=tuple(value.get("horizon_breaks", ())),
            shrinkage_origins=value.get("shrinkage_origins", 8.),
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.policy not in LEARNED:
            return {key: value[key] for key in ("policy", "members", "weights", "max_models")}
        return value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def component_models(frame: pd.DataFrame) -> list[str]:
    return [
        str(c) for c in frame
        if c not in META and c != "WeightedEnsemble"
        and "-lo-" not in c and "-hi-" not in c
        and pd.api.types.is_numeric_dtype(frame[c])
    ]


def _weights(recipe: BlendRecipe, prior: pd.DataFrame, *, history=None, horizon=None, clip_negative=False) -> dict:
    if recipe.policy in LEARNED:
        from nixtla_scaffold.greedy import learn
        return learn(recipe, prior, history=history, horizon=horizon, clip_negative=clip_negative)
    if recipe.policy == "fixed":
        return {m: w for m, w in zip(recipe.members, recipe.weights) if w > 0}
    if recipe.policy == "equal":
        return {m: 1 / len(recipe.members) for m in recipe.members}
    if prior.empty:
        return {}
    eligible = [m for m in recipe.members if m in prior and np.isfinite(prior[m]).all()]
    if not eligible or not np.isfinite(prior["y"]).all():
        return {}
    if recipe.policy == "inverse_error" and len(eligible) != len(recipe.members):
        return {}
    scores = {
        m: float(np.sqrt(np.mean(np.square(prior[m] - prior["y"]))))
        for m in eligible
    }
    ordered = sorted(eligible, key=lambda m: (scores[m], m))
    if recipe.policy == "family_diverse":
        families: set[str] = set()
        selected = []
        for model in ordered:
            family = model_family(model)
            if family not in families:
                families.add(family)
                selected.append(model)
        ordered = selected
    if recipe.policy in {"top_k", "family_diverse"}:
        selected = ordered[:recipe.max_models]
        return {m: 1 / len(selected) for m in selected}
    perfect = [m for m in ordered if scores[m] == 0]
    if perfect:
        return {m: 1 / len(perfect) for m in perfect}
    inverse = {m: 1 / max(scores[m], 1e-12) for m in ordered}
    total = sum(inverse.values())
    return {m: w / total for m, w in inverse.items()}


def _predict(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if isinstance(weights.get("groups"), list):
        steps = frame["horizon_step"] if "horizon_step" in frame else pd.to_datetime(frame.ds).rank(method="dense").astype(int)
        groups = np.searchsorted(weights["breaks"], steps, side="left")
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        for i, selected in enumerate(weights["groups"]):
            mask = groups == i
            out.loc[mask] = _predict(frame.loc[mask], selected)
        return out
    if not weights or any(m not in frame for m in weights):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[list(weights)].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1)
    return values.mul(pd.Series(weights)).sum(axis=1).where(valid)


def _score(frame: pd.DataFrame, prediction: str) -> dict:
    error = frame[prediction] - frame["y"]
    denom = float(frame["y"].abs().sum())
    return {
        "model": prediction,
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(error.abs().mean()),
        "wape": float(error.abs().sum() / denom) if denom else None,
        "bias": float(error.sum() / denom) if denom else None,
        "observations": len(frame),
        "cutoffs": int(frame["cutoff"].nunique()),
    }


def evaluate_blend(
    backtest: pd.DataFrame, future: pd.DataFrame, recipe: BlendRecipe, *,
    history: pd.DataFrame | None = None, horizon: int | None = None, clip_negative: bool = False,
    _allow_horizon_lab: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if recipe.policy == "horizon_greedy" and not _allow_horizon_lab:
        raise ValueError("Horizon grouping did not pass development qualification; use the global greedy policy.")
    required = {"unique_id", "ds", "cutoff", "y", *recipe.members}
    if not required.issubset(backtest):
        raise ValueError(f"backtest is missing required columns: {sorted(required - set(backtest))}")
    if not {"unique_id", "ds", *recipe.members}.issubset(future):
        raise ValueError("future forecasts do not contain all recipe components")
    cv = backtest.copy()
    cv["cutoff"] = pd.to_datetime(cv["cutoff"], errors="raise")
    cv["ds"] = pd.to_datetime(cv["ds"], errors="raise")
    future = future.copy()
    if history is not None:
        history = history.copy()
        history["ds"] = pd.to_datetime(history["ds"])
    if clip_negative:
        cv[list(recipe.members)] = cv[list(recipe.members)].clip(lower=0)
        future[list(recipe.members)] = future[list(recipe.members)].clip(lower=0)
    keys = ["unique_id", "cutoff", "ds"]
    if cv[keys].isna().any().any() or cv.duplicated(keys).any():
        raise ValueError("null or duplicate series/origin/period in backtest evidence")
    if future[["unique_id", "ds"]].isna().any().any() or future.duplicated(["unique_id", "ds"]).any():
        raise ValueError("null or duplicate future series/period")
    if not cv["ds"].gt(cv["cutoff"]).all():
        raise ValueError("backtest targets must be strictly after their origin")
    scores, predictions, forecasts = [], [], []
    excluded, warmup, planned = 0, 0, 0
    references = {}
    def weights_at(group, cutoff, selected=recipe):
        prior = observable_backtest_rows(group, cutoff)
        visible = history.loc[history.ds.le(cutoff)].copy() if history is not None else None
        if visible is not None and "available_at" in visible:
            released = pd.to_datetime(visible.available_at, errors="coerce")
            visible.loc[released.isna() | released.gt(cutoff), "y"] = np.nan
        return _weights(selected, prior, history=visible, horizon=horizon, clip_negative=clip_negative)
    for uid, group in cv.groupby("unique_id", sort=True):
        evaluated = []
        started = False
        reference = next((name for name in ("Naive", "SeasonalNaive", "RandomWalkWithDrift")
                          if name in group), recipe.members[0])
        references[str(uid)] = reference
        comparison_models = list(dict.fromkeys([*recipe.members, reference]))
        if clip_negative:
            group = group.copy()
            group[reference] = group[reference].clip(lower=0)
        if recipe.policy in LEARNED:
            comparison_models += ["EqualBlend", "PriorBest", "GlobalBlend"]
            references[str(uid)] = "PriorBest"
        for cutoff, current in group.groupby("cutoff", sort=True):
            weights = weights_at(group, cutoff)
            rows = current.copy()
            rows["Blend"] = _predict(rows, weights)
            if recipe.policy in LEARNED:
                global_recipe = replace(recipe, policy="greedy", horizon_breaks=())
                rows["EqualBlend"] = rows[list(recipe.members)].mean(axis=1, skipna=False)
                rows["PriorBest"] = _predict(rows, weights_at(group, cutoff, replace(global_recipe, max_steps=1, max_models=1)))
                rows["GlobalBlend"] = _predict(rows, weights if recipe.policy == "greedy" else weights_at(group, cutoff, global_recipe))
            # All comparison lines are evaluated on exactly the same rows.
            valid = np.isfinite(rows[["y", "Blend", *comparison_models]]).all(axis=1)
            if not started and not weights:
                warmup += len(rows)
            else:
                started = True
                planned += len(rows)
            excluded += int((~valid).sum())
            rows = rows.loc[valid].copy()
            rows["weights"] = json.dumps(weights, sort_keys=True)
            rows["horizon_step"] = (
                current["ds"].rank(method="dense").astype(int).loc[rows.index]
            )
            evaluated.append(rows)
        paired = pd.concat(evaluated, ignore_index=True) if evaluated else pd.DataFrame()
        if not paired.empty:
            predictions.append(paired)
            for model in ["Blend", *comparison_models]:
                scores.append({"unique_id": str(uid), **_score(paired, model)})
        future_rows = future[future["unique_id"].astype(str) == str(uid)].copy()
        if future_rows.empty:
            continue
        # Deployment can use outcomes in the supplied historical backtest only.
        origin = pd.to_datetime(future_rows["ds"]).min()
        landed = group[group["ds"] < origin]
        if "available_at" in landed:
            landed = landed[pd.to_datetime(landed["available_at"], errors="coerce") < origin]
        weights = weights_at(group, origin - pd.Timedelta(nanoseconds=1)) if recipe.policy in LEARNED else _weights(recipe, landed)
        future_rows["yhat"] = _predict(future_rows, weights)
        future_rows["weights"] = json.dumps(weights, sort_keys=True)
        future_rows["model"] = "Blend"
        forecasts.append(future_rows[["unique_id", "ds", "yhat", "model", "weights"]])
    backtest_out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    future_out = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    receipt = {
        "schema_version": "nixtla.workbench.blend.v1",
        "recipe": recipe.to_dict(), "recipe_fingerprint": recipe.fingerprint,
        "scoring": "paired_as_of_retrospective",
        "excluded_rows": excluded, "paired_rows": len(backtest_out),
        "warmup_rows": warmup, "planned_later_rows": planned,
        "comparison_eligible": bool(planned and len(backtest_out) == planned
                                    and not future_out.empty and np.isfinite(future_out["yhat"]).all()
                                    and len(future_out) == len(future)),
        "reference_models": references,
        "reference_policy": ("best_single_on_prior_complete_observable_origins" if recipe.policy in LEARNED else
                             "fixed_naive_then_seasonal_naive_then_drift_then_first_declared_member"),
        "cohort_fingerprint": hashlib.sha256(
            backtest_out[keys].sort_values(keys).to_csv(index=False).encode()
        ).hexdigest() if not backtest_out.empty else None,
        "interval_status": "point_only_ensemble",
        "confirmation_status": "not_independently_confirmed",
        "promotion_recommended": False,
        "advisory_only": True,
        "objective": recipe.objective if recipe.policy in LEARNED else "legacy_rmse_weight_policy",
        "composition": "clip_components_then_blend" if clip_negative else "blend_unclipped_components",
        "limitations": [
            "Retrospective paired evidence is not an untouched confirmation set.",
            "No calibrated blend intervals are available.",
            "Unavailable components exclude the row; weights are never silently redistributed.",
            "Incomplete later support is diagnostic only and cannot earn a recommendation.",
        ],
    }
    return backtest_out, future_out, pd.DataFrame(scores), receipt
