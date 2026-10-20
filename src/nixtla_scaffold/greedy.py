"""Small deterministic convex ensembles scored on their composed prediction path."""
from __future__ import annotations

import numpy as np
import pandas as pd


def objective_function(prior, history, *, objective, horizon):
    if objective == "daily_rmse":
        truth = prior.y.to_numpy(float)
        return lambda path: float(np.sqrt(np.mean(np.square(path - truth))))
    if objective != "monthly_landing":
        raise ValueError(f"Unsupported ensemble objective: {objective}")
    if history is None or horizon is None:
        raise ValueError("Monthly ensemble learning requires source history and the declared horizon.")
    from nixtla_scaffold.finance import total_windows
    reference = prior.assign(ObjectiveReference=0.)
    windows = total_windows(history, reference, ["ObjectiveReference"], horizon=horizon, kinds=("month",))
    if windows.empty or not windows.status.eq("complete").all():
        return None
    matrix = np.zeros((len(windows), len(prior)))
    offset = np.zeros(len(windows))
    for i, window in enumerate(windows.itertuples()):
        mask = (prior.unique_id.astype(str).eq(window.unique_id) & prior.cutoff.eq(window.cutoff)
                & prior.ds.between(window.start, window.end))
        matrix[i, mask.to_numpy()] = 100 / abs(window.actual)
        offset[i] = (window.prediction - window.actual) * 100 / abs(window.actual)
    return lambda path: float(np.mean(np.abs(matrix @ path + offset)))


def _optimize(values, score, members, *, max_steps, max_models):
    counts = np.zeros(len(members))
    best_weights, best_score = None, float("inf")
    for step in range(max_steps):
        trials = []
        for i, model in enumerate(members):
            if not counts[i] and np.count_nonzero(counts) >= max_models:
                continue
            candidate = counts.copy()
            candidate[i] += 1
            weights = candidate / (step + 1)
            trials.append((score(values @ weights), model, i, weights))
        loss, _, index, weights = min(trials, key=lambda item: (item[0], item[1]))
        if not np.isfinite(loss):
            break
        counts[index] += 1
        if loss < best_score - 1e-12:
            best_weights, best_score = weights.copy(), loss
        if best_score <= 1e-12:
            break
    return best_weights


def learn(recipe, prior, *, history, horizon, clip_negative):
    if horizon is None:
        raise ValueError("Learned ensembles require the declared horizon.")
    prior = prior.copy()
    # Admit whole origins only. A partially landed horizon is not a training fold.
    counts = prior.groupby(["unique_id", "cutoff"]).ds.transform("size")
    prior = prior.loc[counts.eq(horizon)].sort_values(["unique_id", "cutoff", "ds"]).reset_index(drop=True)
    if prior.cutoff.nunique() < recipe.min_origins or prior.empty:
        return {}
    if any(model not in prior for model in recipe.members) or not np.isfinite(prior[["y", *recipe.members]]).all().all():
        return {}
    prior["horizon_step"] = prior.groupby(["unique_id", "cutoff"]).cumcount() + 1
    if clip_negative:
        prior[list(recipe.members)] = prior[list(recipe.members)].clip(lower=0)
    score = objective_function(prior, history, objective=recipe.objective, horizon=horizon)
    if score is None:
        return {}
    members = tuple(sorted(recipe.members))
    values = prior[list(members)].to_numpy(float)
    global_weights = _optimize(values, score, members, max_steps=recipe.max_steps, max_models=recipe.max_models)
    if global_weights is None:
        return {}
    encode = lambda weights: {model: float(w) for model, w in zip(members, weights) if w > 0}
    if recipe.policy == "greedy":
        return encode(global_weights)
    if not recipe.horizon_breaks or recipe.horizon_breaks[-1] >= horizon:
        raise ValueError("Horizon-group breaks must be nonempty and below the declared horizon.")
    groups = np.searchsorted(recipe.horizon_breaks, prior.horizon_step.to_numpy(), side="left")
    path = values @ global_weights
    learned = []
    alpha = prior.cutoff.nunique() / (prior.cutoff.nunique() + recipe.shrinkage_origins)
    for group in range(len(recipe.horizon_breaks) + 1):
        mask = groups == group
        fixed = path.copy()
        def group_score(local_path):
            proposed = fixed.copy()
            proposed[mask] = alpha * local_path[mask] + (1 - alpha) * (values @ global_weights)[mask]
            return score(proposed)
        local = _optimize(values, group_score, members, max_steps=recipe.max_steps, max_models=recipe.max_models)
        weights = global_weights if local is None else alpha * local + (1 - alpha) * global_weights
        proposed = path.copy()
        proposed[mask] = (values @ weights)[mask]
        if score(proposed) >= score(path) - 1e-12:
            weights = global_weights
        else:
            path = proposed
        learned.append(encode(weights))
    return {"groups": learned, "breaks": list(recipe.horizon_breaks)}
