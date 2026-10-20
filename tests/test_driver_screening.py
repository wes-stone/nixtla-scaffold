import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.signals import screen_driver_relationships


def frame(n=120):
    rng = np.random.default_rng(751)
    x = rng.normal(size=n)
    y = -3 * np.roll(x, 4) + rng.normal(0, 0.15, n)
    return pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2010-01-01", periods=n, freq="MS"),
                         "y": y, "signal": x})


def screen(data, **kwargs):
    return screen_driver_relationships(data, features=("signal",), horizon=3, **kwargs)


def test_negative_leading_signal_survives_signed_changes_and_stability_checks():
    result = screen(frame())
    card = result["cards"][0]
    assert card["disposition"] == "eligible_for_predictive_testing"
    assert card["best_safe_lag"] == 4 and card["signed_direction"] == -1
    assert not card["training_admitted"]
    assert result["tests_searched"] == len(result["tests"]) == 25
    evidence = next(row for row in result["tests"] if row["lag"] == 4)
    assert evidence["change_correlation"] < -0.9
    assert evidence["residual_correlation"] < -0.9
    assert evidence["bonferroni_p_approximate"] < 0.05


def test_shared_trend_is_not_predictive_evidence():
    data = frame()
    t = np.arange(len(data))
    rng = np.random.default_rng(913)
    data["signal"] = 10 * t + rng.normal(size=len(t))
    data["y"] = 20 * t + rng.normal(size=len(t))
    result = screen(data)
    assert abs(next(row for row in result["tests"] if row["lag"] == 0)["level_correlation"]) > 0.99
    assert result["cards"][0]["disposition"] != "eligible_for_predictive_testing"


def test_shared_seasonality_is_removed_before_endorsement():
    data = frame()
    t = np.arange(len(data))
    data["signal"] = np.sin(2 * np.pi * t / 12)
    data["y"] = 100 + 5 * np.sin(2 * np.pi * t / 12) + 0.1 * t
    assert screen(data, season_length=12)["cards"][0]["disposition"] != "eligible_for_predictive_testing"


@pytest.mark.parametrize("shift", [0, -2, 2])
def test_affine_target_and_future_target_copy_controls_are_rejected(shift):
    data = frame()
    data["signal"] = 2 * data.y.shift(shift) + 7
    result = screen(data)
    assert result["cards"][0]["disposition"] == "rejected"
    assert not result["cards"][0]["training_admitted"]


def test_reverse_direction_and_short_unsafe_lags_do_not_qualify():
    data = frame()
    data["signal"] = data.y.shift(3) + np.random.default_rng(14).normal(0, 0.1, len(data))
    result = screen(data)
    reverse = max((row for row in result["tests"] if row["residual_correlation"] is not None),
                  key=lambda row: abs(row["residual_correlation"]))
    assert reverse["lag"] < 0
    assert result["cards"][0]["disposition"] != "eligible_for_predictive_testing"
    data = frame()
    data["y"] = data.signal.shift(1) + np.random.default_rng(7).normal(0, 0.1, len(data))
    assert screen(data)["cards"][0]["disposition"] != "eligible_for_predictive_testing"


def test_direction_reversal_does_not_pass_temporal_stability():
    data = frame()
    data.loc[60:, "y"] *= -1
    assert screen(data)["cards"][0]["disposition"] != "eligible_for_predictive_testing"


def test_panel_size_effect_and_opposite_directions_are_not_pooled():
    a = frame()
    b = frame().assign(unique_id="B")
    b["y"] *= -1
    result = screen(pd.concat([a, b], ignore_index=True))
    assert len(result["tests"]) == 50
    assert result["cards"][0]["disposition"] != "eligible_for_predictive_testing"


def test_short_or_irregular_history_has_truthful_insufficient_evidence():
    assert screen(frame(15))["cards"][0]["disposition"] == "insufficient_evidence"
    data = frame().drop(index=[12, 25])
    result = screen(data)
    assert not any(row["regular_calendar"] for row in result["tests"])
    assert result["cards"][0]["disposition"] != "eligible_for_predictive_testing"


def test_every_search_and_omitted_feature_is_counted_and_collinearity_is_visible():
    data = frame()
    data["copy"] = data.signal * 3
    data["not_searched"] = 1
    result = screen_driver_relationships(data, features=("signal", "copy", "not_searched"),
                                         horizon=3, max_features=2)
    assert result["tests_searched"] == len(result["tests"]) == 50
    assert result["features_omitted_by_bound"] == ["not_searched"]
    assert result["collinearity"][0]["residual_correlation"] == pytest.approx(1)


def test_changing_protected_tail_cannot_change_tuning_screen():
    data = frame()
    changed = data.copy()
    changed.loc[90:, ["y", "signal"]] *= 1000000
    assert screen(data.iloc[:90]) == screen(changed.iloc[:90])


def test_numeric_high_cardinality_random_features_do_not_gain_a_training_claim():
    data = frame()
    rng = np.random.default_rng(131)
    features = tuple(f"random_{i}" for i in range(8))
    for feature in features:
        data[feature] = rng.normal(size=len(data))
    result = screen_driver_relationships(data, features=features, horizon=3)
    assert result["tests_searched"] == 200
    assert not any(card["disposition"] == "eligible_for_predictive_testing" for card in result["cards"])


def test_optional_screening_bounds_preserve_baseline_context_without_sampling():
    from nixtla_scaffold.experiments import _experiment_context
    from nixtla_scaffold.schema import ForecastSpec
    data = pd.concat([frame(30).assign(unique_id=str(i)) for i in range(51)], ignore_index=True)
    context = _experiment_context(data, ForecastSpec(freq="MS"), sheet=None)
    assert context.row_count == len(data) and context.series_count == 51
    assert context.driver_screening["status"] == "scope_exceeds_bound"
    assert context.driver_screening["features_omitted_by_bound"] == ["signal"]
    assert not context.candidate_drivers and not context.autoresearch_hypotheses


def test_context_retains_all_rejected_and_unranked_search_rows():
    from nixtla_scaffold.experiments import _experiment_context
    from nixtla_scaffold.schema import ForecastSpec
    data = frame()
    data["future_target"] = data.y.shift(-1)
    context = _experiment_context(data, ForecastSpec(freq="MS", horizon=3, season_length=1), sheet=None)
    ledger = context.to_dict()["driver_screening"]
    assert ledger["tests_searched"] == len(ledger["tests"]) == 50
    assert next(card for card in ledger["cards"] if card["feature"] == "future_target")["disposition"] == "rejected"
