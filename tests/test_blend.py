import pandas as pd
import pytest

from nixtla_scaffold.blend import BlendRecipe, evaluate_blend


def evidence():
    cv = pd.DataFrame({
        "unique_id": ["A"] * 4,
        "cutoff": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]),
        "ds": pd.to_datetime(["2025-02-28", "2025-03-31", "2025-03-31", "2025-04-30"]),
        "y": [10., 100., 100., 20.],
        "Naive": [10., 0., 10., 10.],
        "Drift": [20., 100., 100., 20.],
    })
    future = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-05-31"], "Naive": [10.], "Drift": [20.]})
    return cv, future


def test_top_k_cannot_see_overlapping_future_outcome():
    cv, future = evidence()
    bt, _, metrics, receipt = evaluate_blend(cv, future, BlendRecipe("top_k", ("Naive", "Drift"), max_models=1))
    assert bt["Blend"].tolist() == [10., 10.]
    assert set(metrics["observations"]) == {2}
    assert receipt["excluded_rows"] == 2
    assert receipt["promotion_recommended"] is False


def test_fixed_weights_are_normalized_and_fingerprinted():
    recipe = BlendRecipe("fixed", ("Naive", "Drift"), (3., 1.))
    assert recipe.weights == (.75, .25)
    assert recipe.fingerprint == BlendRecipe.from_dict(recipe.to_dict()).fingerprint
    _, future, _, _ = evaluate_blend(*evidence(), recipe)
    assert future["yhat"].iloc[0] == 12.5


def test_missing_component_does_not_silently_change_blend():
    cv, future = evidence()
    cv.loc[2, "Naive"] = float("nan")
    future["Naive"] = float("nan")
    bt, forecast, metrics, _ = evaluate_blend(cv, future, BlendRecipe("equal", ("Naive", "Drift")))
    assert len(bt) == 3
    assert set(metrics["observations"]) == {3}
    assert forecast["yhat"].isna().all()


@pytest.mark.parametrize("weights", [(-1., 2.), (0., 0.), (float("nan"), 1.)])
def test_invalid_weights_fail(weights):
    with pytest.raises(ValueError):
        BlendRecipe("fixed", ("Naive", "Drift"), weights)


def test_large_finite_weights_do_not_overflow_normalization():
    assert BlendRecipe("fixed", ("Naive", "Drift"), (1e308, 1e308)).weights == (.5, .5)
