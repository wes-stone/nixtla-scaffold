import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import EvaluationPolicy, ForecastSpec, run_forecast
from nixtla_scaffold import outputs


def fixture():
    return run_forecast(
        pd.DataFrame({"unique_id": "A", "ds": pd.date_range("2020-01-01", periods=18, freq="MS"),
                      "y": np.arange(18) + 100.}),
        ForecastSpec(horizon=3, model_policy="baseline", evaluation=EvaluationPolicy(max_windows=2)),
    )


def test_output_batch_reuses_tables_without_exposing_mutable_cached_values(monkeypatch):
    run = fixture()
    expected = outputs.build_trust_summary(run)
    original = outputs.build_backtest_long
    calls = []

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(outputs, "build_backtest_long", counted)
    with outputs.cached_output_tables(run):
        first = outputs.build_trust_summary(run)
        count = len(calls)
        first.loc[:, "trust_score_0_100"] = -123
        second = outputs.build_trust_summary(run)
        assert len(calls) == count
        pd.testing.assert_frame_equal(second, expected)


def test_output_batch_never_reuses_evidence_after_scope_or_failure():
    run = fixture()
    with pytest.raises(RuntimeError, match="interrupted"):
        with outputs.cached_output_tables(run):
            assert not outputs.build_backtest_long(run).empty
            raise RuntimeError("interrupted")
    run.backtest_predictions = run.backtest_predictions.iloc[:0]
    with outputs.cached_output_tables(run):
        assert outputs.build_backtest_long(run).empty


def test_output_batches_are_run_scoped():
    first, second = fixture(), fixture()
    second.backtest_predictions = second.backtest_predictions.iloc[:0]
    with outputs.cached_output_tables(first):
        expected = outputs.build_backtest_long(first)
        with outputs.cached_output_tables(second):
            assert outputs.build_backtest_long(second).empty
        pd.testing.assert_frame_equal(outputs.build_backtest_long(first), expected)
