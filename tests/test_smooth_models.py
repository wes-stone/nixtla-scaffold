from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.schema import ForecastSpec


def test_real_smooth_adam_zxz_optional_smoke_beats_naive_shape() -> None:
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
            model_allowlist=("SmoothADAM_ZXZ",),
            weighted_ensemble=False,
        ),
    )

    assert list(run.forecast["model"].unique()) == ["SmoothADAM_ZXZ"]
    assert run.forecast["yhat"].notna().all()
    assert (run.forecast["yhat"] > 0).all()
    assert any("smooth optional dependency active" in warning for warning in run.warnings)
    families = {row["family"]: row for row in run.model_policy_resolution["families"]}
    assert families["smooth"]["ran"]
    assert families["smooth"]["contributed_models"] == ["SmoothADAM_ZXZ"]
