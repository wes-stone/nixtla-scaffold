from __future__ import annotations

from pathlib import Path

import pandas as pd

from nixtla_scaffold import forecast_spec_preset, run_forecast


def build_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 18,
            "ds": pd.date_range("2024-01-31", periods=18, freq="ME"),
            "y": [100 + 5 * idx for idx in range(18)],
        }
    )


def run_example(output_dir: str | Path) -> Path:
    spec = forecast_spec_preset("quick", horizon=3, freq="ME")
    run = run_forecast(build_frame(), spec)
    return run.to_directory(output_dir)
