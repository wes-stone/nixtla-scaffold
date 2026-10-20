from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.blend import BlendRecipe, _weights, _predict, evaluate_blend
from nixtla_scaffold.finance import finance_scorecard
from nixtla_scaffold.greedy import objective_function


def evidence(horizon=6, windows=8):
    dates = pd.date_range("2024-01-01", periods=horizon * (windows + 1), freq="D")
    history = pd.DataFrame({"unique_id": "001", "ds": dates, "y": 100.})
    frames = []
    for i in range(windows):
        frame = history.iloc[(i + 1) * horizon:(i + 2) * horizon].copy()
        frame["cutoff"] = dates[(i + 1) * horizon - 1]
        frame["Low"], frame["High"] = 90., 110.
        frames.append(frame)
    future = pd.DataFrame({"unique_id": "001", "ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon),
                           "Low": 90., "High": 110.})
    return pd.concat(frames, ignore_index=True), future, history


def test_legacy_fingerprint_is_unchanged():
    old = {"policy": "equal", "members": ("Low", "High"), "weights": (), "max_models": 3}
    recipe = BlendRecipe.from_dict(old)
    assert recipe.to_dict() == old
    assert recipe.fingerprint == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("policy", ["greedy", "horizon_greedy"])
def test_convex_prior_only_matched_and_deterministic(policy):
    cv, future, history = evidence()
    recipe = BlendRecipe(policy, ("Low", "High"), horizon_breaks=(2, 4) if policy == "horizon_greedy" else ())
    bt, predicted, scores, receipt = evaluate_blend(cv, future, recipe, horizon=6, history=history, _allow_horizon_lab=True)
    assert len(bt) == 30 and receipt["warmup_rows"] == 18
    assert receipt["comparison_eligible"]
    assert bt.Blend.eq(100).all() and predicted.yhat.eq(100).all()
    assert set(scores.observations) == {30}
    altered = cv.copy()
    altered.loc[altered.cutoff.eq(altered.cutoff.max()), "y"] = 10000
    changed, _, _, _ = evaluate_blend(altered, future, recipe, horizon=6, history=history, _allow_horizon_lab=True)
    pd.testing.assert_series_equal(bt.Blend, changed.Blend)
    assert recipe.fingerprint == BlendRecipe.from_dict(recipe.to_dict()).fingerprint


def test_missing_later_support_is_ineligible():
    cv, future, history = evidence()
    cv.loc[len(cv) - 1, "Low"] = np.nan
    _, _, _, receipt = evaluate_blend(cv, future, BlendRecipe("greedy", ("Low", "High")), horizon=6, history=history)
    assert not receipt["comparison_eligible"]


def test_delayed_publication_and_partial_origins_do_not_train():
    cv, _, _ = evidence()
    recipe = BlendRecipe("greedy", ("Low", "High"))
    assert _weights(recipe, cv.iloc[:17], horizon=6) == {}
    cv["available_at"] = cv.ds + pd.Timedelta(days=60)
    _, _, _, receipt = evaluate_blend(cv, evidence()[1], recipe, horizon=6)
    assert receipt["paired_rows"] == 0


def test_monthly_loss_matches_shared_scorecard_and_optimizes_totals():
    cv, future, history = evidence(horizon=63)
    # Daily errors cancel within each complete calendar month, unlike the bias.
    cv["Low"] = np.where(cv.ds.dt.day % 2, 70., 130.)
    cv["High"] = 103.
    recipe = BlendRecipe("greedy", ("Low", "High"), objective="monthly_landing")
    score = objective_function(cv, history, objective="monthly_landing", horizon=63)
    for model in recipe.members:
        actual, _ = finance_scorecard(history, cv, [model], horizon=63)
        assert score(cv[model].to_numpy()) == pytest.approx(actual.landing_mape_pct.iloc[0])
    daily = _weights(replace(recipe, objective="daily_rmse", max_steps=1), cv, horizon=63)
    landing = _weights(replace(recipe, max_steps=1), cv, history=history, horizon=63)
    assert daily == {"High": 1.} and landing == {"Low": 1.}
    bt, _, _, receipt = evaluate_blend(cv, future, recipe, horizon=63, history=history)
    assert len(bt) > 0 and receipt["objective"] == "monthly_landing"


def test_clipping_is_before_composition_and_never_changes_actuals():
    cv, future, history = evidence()
    cv["Low"], cv["High"] = -100., 300.
    future["Low"], future["High"] = -100., 300.
    bt, forecast, _, receipt = evaluate_blend(cv, future, BlendRecipe("fixed", ("Low", "High"), (1., 1.)),
                                             horizon=6, clip_negative=True)
    assert bt.y.eq(100).all() and bt.Blend.eq(150).all()
    assert forecast.yhat.eq(150).all() and receipt["composition"] == "clip_components_then_blend"


def test_horizon_groups_shrink_and_preserve_joint_path():
    cv, future, history = evidence()
    steps = np.tile(np.arange(6), 8)
    cv["Low"] = np.where(steps < 3, 100., 120.)
    cv["High"] = np.where(steps < 3, 120., 100.)
    recipe = BlendRecipe("horizon_greedy", ("Low", "High"), horizon_breaks=(3,))
    weights = _weights(recipe, cv, horizon=6)
    assert weights["groups"][0]["Low"] > weights["groups"][0]["High"]
    assert weights["groups"][1]["High"] > weights["groups"][1]["Low"]
    assert all(sum(w.values()) == pytest.approx(1.) and all(x >= 0 for x in w.values()) for w in weights["groups"])


@pytest.mark.parametrize("kwargs", [{"objective": "unknown"}, {"max_steps": 33}, {"min_origins": 2},
                                    {"horizon_breaks": (2, 2)}, {"shrinkage_origins": 0}])
def test_invalid_learning_settings(kwargs):
    with pytest.raises(ValueError):
        BlendRecipe("greedy", ("Low", "High"), **kwargs)


@pytest.mark.parametrize("policy", ["greedy", "horizon_greedy"])
def test_protected_prediction_path_matches_exact_replay(policy):
    from types import SimpleNamespace
    from nixtla_scaffold.schema import ForecastSpec, SelectionPolicy
    from nixtla_scaffold.workbench_confirmation import _predictions
    cv, future, history = evidence()
    recipe = BlendRecipe(policy, ("Low", "High"), horizon_breaks=(2, 4) if policy == "horizon_greedy" else ())
    run = SimpleNamespace(
        all_models=future, backtest_predictions=cv, history=history, observed_history=history,
        spec=ForecastSpec(horizon=6, freq="D", selection_policy=SelectionPolicy(clip_negative=True)),
    )
    if policy == "horizon_greedy":
        with pytest.raises(ValueError, match="not qualified"):
            _predictions(run, recipe)
        with pytest.raises(ValueError, match="did not pass"):
            evaluate_blend(cv, future, recipe, horizon=6, history=history)
        return
    protected = _predictions(run, recipe)
    _, replay, _, _ = evaluate_blend(cv, future, recipe, horizon=6, history=history, clip_negative=True)
    np.testing.assert_array_equal(protected.Blend, replay.yhat)


def test_unavailable_month_totals_cannot_learn():
    cv, _, history = evidence(horizon=63)
    history.loc[history.ds.eq("2024-03-02"), "y"] = np.nan
    assert cv.y.notna().all()
    assert _weights(BlendRecipe("greedy", ("Low", "High"), objective="monthly_landing"),
                    cv, history=history, horizon=63) == {}


def test_replacement_search_can_discover_a_small_complementary_weight():
    cv, _, _ = evidence()
    cv["Low"], cv["High"] = 99., 109.
    weights = _weights(BlendRecipe("greedy", ("Low", "High")), cv, horizon=6)
    assert weights == pytest.approx({"Low": .9, "High": .1})
    np.testing.assert_allclose(_predict(cv, weights), 100., rtol=0, atol=3e-14)
