from __future__ import annotations

from pathlib import Path

from nixtla_scaffold import CustomModelSpec, forecast_spec_preset, run_forecast


ROOT = Path(__file__).resolve().parent


def run_example(output_dir: str | Path | None = None) -> Path:
    output = Path(output_dir) if output_dir is not None else ROOT / "runs" / "custom_finance"
    script = ROOT / "finance_seasonality_model.py"
    spec = forecast_spec_preset(
        "finance",
        horizon=6,
        freq="ME",
        model_policy="baseline",
        weighted_ensemble=False,
        custom_models=(
            CustomModelSpec(
                name="MoM Growth FY Seasonal",
                script_path=str(script),
                timeout_seconds=30,
                extra_args=("--growth-window", "6", "--seasonality-years", "2"),
                source_id="examples.custom_finance_model",
                notes="Recent median MoM growth converted to an annual run-rate and allocated by historical month-of-year shares.",
            ),
        ),
    )
    run = run_forecast(ROOT / "input.csv", spec)
    return run.to_directory(output)


if __name__ == "__main__":
    print(run_example())
