from __future__ import annotations

from pathlib import Path

from nixtla_scaffold import forecast_spec_preset, run_forecast


ROOT = Path(__file__).resolve().parent


def run_example(output_dir: str | Path | None = None) -> Path:
    output = Path(output_dir) if output_dir is not None else ROOT / "runs" / "quickstart"
    spec = forecast_spec_preset("quick", horizon=3, freq="ME")
    run = run_forecast(ROOT / "input.csv", spec)
    return run.to_directory(output)


if __name__ == "__main__":
    print(run_example())
