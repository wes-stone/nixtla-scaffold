from __future__ import annotations

from pathlib import Path

from nixtla_scaffold import DriverEvent, TransformSpec, forecast_spec_preset, run_forecast


ROOT = Path(__file__).resolve().parent


def run_example(output_dir: str | Path | None = None) -> Path:
    output = Path(output_dir) if output_dir is not None else ROOT / "runs" / "finance"
    spec = forecast_spec_preset(
        "finance",
        horizon=6,
        freq="ME",
        model_policy="baseline",
        transform=TransformSpec(normalization_factor_col="price_factor", normalization_label="FY24 pricing"),
        events=(
            DriverEvent(
                name="Sales capacity ramp",
                start="2025-03-31",
                end="2025-06-30",
                effect="multiplicative",
                magnitude=0.08,
                confidence=0.75,
            ),
        ),
    )
    run = run_forecast(ROOT / "input.csv", spec)
    return run.to_directory(output)


if __name__ == "__main__":
    print(run_example())
