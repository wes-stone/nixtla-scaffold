from __future__ import annotations

from pathlib import Path

from nixtla_scaffold import aggregate_hierarchy_frame, forecast_spec_preset, run_forecast


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "hierarchy_generic" / "input.csv"


def run_example(output_dir: str | Path | None = None) -> Path:
    output = Path(output_dir) if output_dir is not None else ROOT / "runs" / "hierarchy"
    nodes = aggregate_hierarchy_frame(
        SOURCE,
        hierarchy_cols=("region", "product"),
        time_col="ds",
        target_col="y",
    )
    spec = forecast_spec_preset("hierarchy", horizon=2, freq="ME", model_policy="baseline")
    run = run_forecast(nodes, spec)
    return run.to_directory(output)


if __name__ == "__main__":
    print(run_example())
