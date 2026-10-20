from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sqlite3

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import run_forecast
from nixtla_scaffold.models import ModelResult
from nixtla_scaffold.prediction_cache import PredictionCache, CacheError
from nixtla_scaffold.schema import EvaluationPolicy, ForecastSpec


def sample():
    return pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2020-01-01", periods=40, freq="MS"),
                         "y": 100 + np.arange(40) * .7})


def test_repeat_has_no_native_fits_and_preserves_points(tmp_path):
    spec = ForecastSpec(freq="MS", horizon=3, model_policy="baseline", levels=(), verbose=False,
                        evaluation=EvaluationPolicy(max_windows=3, cache_directory=str(tmp_path / "cache"),
                                                    evidence_directory=str(tmp_path / "exposure")))
    first = run_forecast(sample(), spec)
    second = run_forecast(sample(), spec)
    for name in ("all_models", "backtest_predictions", "backtest_metrics", "forecast"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
    assert first.model_policy_resolution["prediction_cache"]["misses"] == 4
    assert second.model_policy_resolution["prediction_cache"]["hits"] == 4
    assert second.model_policy_resolution["prediction_cache"]["native_fits"] == 0
    revised = sample()
    revised.loc[0, "y"] += 1
    changed = run_forecast(revised, spec)
    assert changed.model_policy_resolution["prediction_cache"]["misses"] == 4
    changed = run_forecast(sample(), replace(spec, horizon=2))
    assert changed.model_policy_resolution["prediction_cache"]["misses"] == 4


def test_cache_atomic_concurrent_and_corruption_visible(tmp_path):
    cache = PredictionCache(tmp_path, 1000000)
    calls = []
    frame = pd.DataFrame({"unique_id": ["001"], "ds": [pd.Timestamp("2025-01-01")],
                          "y": [np.nextafter(1., 2.)], "missing": [np.nan]})
    def compute():
        calls.append(1)
        return ModelResult(frame, pd.DataFrame(), pd.DataFrame(), "test", pd.DataFrame())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.get_or_fit("one", compute), range(4)))
    assert len(calls) == 1
    for result in results:
        pd.testing.assert_frame_equal(result.forecast, frame, check_exact=True)
    with sqlite3.connect(cache.path) as conn:
        conn.execute("UPDATE entries SET payload=? WHERE key='one'", (b"corrupt",))
    with pytest.raises(CacheError, match="Corrupt"):
        cache.get_or_fit("one", compute)
    assert len(calls) == 1


def test_payload_eviction_bound_and_runtime_invalidation(tmp_path):
    cache = PredictionCache(tmp_path, 2000)
    def compute():
        return ModelResult(pd.DataFrame({"y": range(30)}), pd.DataFrame(), pd.DataFrame(), "test", pd.DataFrame())
    for i in range(5):
        cache.get_or_fit(str(i), compute)
    with sqlite3.connect(cache.path) as conn:
        assert conn.execute("SELECT SUM(size) FROM entries").fetchone()[0] <= 2000
        assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] < 5
    from nixtla_scaffold.profile import profile_dataset
    from nixtla_scaffold.models import forecast_with_policy
    spec = ForecastSpec()
    profile = profile_dataset(sample(), spec)
    original = cache.key(sample(), profile, spec, forecast_with_policy)
    cache.runtime["python"]["version"] = "changed"
    assert original != cache.key(sample(), profile, spec, forecast_with_policy)


def test_failed_native_units_are_never_reused(tmp_path):
    cache = PredictionCache(tmp_path, 1000000)
    calls = []
    def compute():
        calls.append(1)
        return ModelResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "test", pd.DataFrame(),
                           model_policy_resolution={"candidate_evidence": {"fits": [{"status": "failed"}]}})
    for _ in range(2):
        cache.get_or_fit("failed", compute)
    assert len(calls) == 2
    assert cache.receipt()["failed_fit_units_uncached"] == 2
    assert cache.receipt()["hits"] == 0
    with sqlite3.connect(cache.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0


def test_append_only_reuses_unchanged_prior_origins(tmp_path):
    spec = ForecastSpec(freq="MS", horizon=3, model_policy="baseline", levels=(), verbose=False,
                        evaluation=EvaluationPolicy(max_windows=3, step_size=3, cache_directory=str(tmp_path)))
    history = sample()
    run_forecast(history, spec)
    extension = pd.DataFrame({"unique_id": "001", "ds": pd.date_range("2023-05-01", periods=3, freq="MS"),
                              "y": 100 + np.arange(40, 43) * .7})
    appended = pd.concat([history, extension], ignore_index=True)
    run = run_forecast(appended, spec)
    assert run.model_policy_resolution["prediction_cache"]["hits"] == 3
    assert run.model_policy_resolution["prediction_cache"]["native_fits"] == 1
    uncached = run_forecast(appended, replace(spec, evaluation=replace(spec.evaluation, cache_directory=None)))
    pd.testing.assert_frame_equal(run.backtest_predictions, uncached.backtest_predictions, check_exact=True)
