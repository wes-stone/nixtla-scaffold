from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.schema import ForecastSpec


@pytest.mark.parametrize("candidate", ["SmoothADAM_ZXZ", "SmoothADAM_CCC"])
def test_real_smooth_optional_smoke(candidate) -> None:
    pytest.importorskip("smooth")
    ds = pd.date_range("2021-01-31", periods=42, freq="ME")
    seasonal = np.sin(np.arange(len(ds)) * 2 * np.pi / 12.0) * 5.0
    trend = np.linspace(100.0, 150.0, len(ds))
    frame = pd.DataFrame({"unique_id": "finance_metric", "ds": ds, "y": trend + seasonal})

    run = run_forecast(
        frame,
        ForecastSpec(
            horizon=6,
            freq="ME",
            model_policy="all",
            model_allowlist=(candidate,),
            weighted_ensemble=False,
        ),
    )

    assert list(run.forecast["model"].unique()) == [candidate]
    assert run.forecast["yhat"].notna().all()
    assert (run.forecast["yhat"] > 0).all()
    assert any("smooth optional dependency active" in warning for warning in run.warnings)
    families = {row["family"]: row for row in run.model_policy_resolution["families"]}
    assert families["smooth"]["ran"]
    assert families["smooth"]["contributed_models"] == [candidate]


def test_nonfinite_smooth_predictions_are_rejected():
    from types import SimpleNamespace
    from nixtla_scaffold.models import _smooth_fit_predict

    class BrokenModel:
        def __init__(self, **kwargs):
            pass
        def fit(self, y):
            return self
        def predict(self, horizon):
            return pd.DataFrame({"mean": [float("nan")] * horizon})

    with pytest.raises(ValueError, match="nonfinite"):
        _smooth_fit_predict(SimpleNamespace(ADAM=BrokenModel), "SmoothADAM_ZXZ",
                            np.arange(30.), horizon=3, season_length=1)


def test_all_failed_smooth_models_preserve_the_reason(monkeypatch):
    from types import SimpleNamespace
    from nixtla_scaffold import models
    monkeypatch.setattr(models, "_load_smooth_module", lambda: object())
    monkeypatch.setattr(models, "_smooth_prediction_frame", lambda *args, **kwargs: (
        pd.DataFrame(columns=["unique_id", "ds"]), ["CustomPool: empty matrix at cutoff"]))
    with pytest.raises(RuntimeError, match="CustomPool: empty matrix"):
        models.forecast_with_smooth(pd.DataFrame(), SimpleNamespace(freq="D", season_length=1),
                                   ForecastSpec(model_allowlist=("SmoothADAM_CustomPool",)))


def test_smooth_110_custom_pool_is_an_explicit_failure():
    import importlib.metadata
    smooth = pytest.importorskip("smooth")
    if importlib.metadata.version("smooth") != "1.1.0":
        pytest.skip("Reproduction fixture for the pinned 1.1.0 upstream defect")
    from nixtla_scaffold.models import _smooth_fit_predict
    y = 100 + np.arange(42) + 5 * np.sin(np.arange(42) * 2 * np.pi / 12)
    with pytest.raises(RuntimeError, match="quarantined"):
        _smooth_fit_predict(smooth, "SmoothADAM_CustomPool", y, horizon=6, season_length=12)


@pytest.mark.parametrize("candidate", ["SmoothADAM_ZXZ", "SmoothADAM_CCC"])
@pytest.mark.parametrize("value", [0., 100.])
def test_constant_input_is_rejected_before_optional_fit(candidate, value):
    from nixtla_scaffold.models import _smooth_fit_predict
    with pytest.raises(ValueError, match="constant/all-zero"):
        _smooth_fit_predict(object(), candidate, np.full(42, value), horizon=6, season_length=12)
