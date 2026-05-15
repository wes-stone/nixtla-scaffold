from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from nixtla_scaffold import DriverEvent, ForecastSpec, KnownFutureRegressor, TransformSpec, run_forecast


ROOT = Path(__file__).resolve().parent
TOURISM_EXAMPLE = ROOT.parent / "datasetsforecast_tourism_small" / "forecast_tourism_small.py"
DEMO_SCHEMA_VERSION = "air_tourism_demo.v1"
AIR_HORIZON = 12
AIR_SEASON_LENGTH = 12


def load_air_passengers_frame() -> pd.DataFrame:
    """Return canonical AirPassengers history with a descriptive series id."""

    try:
        from statsforecast.utils import AirPassengersDF
    except ModuleNotFoundError as exc:  # pragma: no cover - statsforecast is a core dependency
        raise RuntimeError("AirPassengers demo requires statsforecast, which is part of the core package.") from exc

    frame = AirPassengersDF.copy()
    required = {"unique_id", "ds", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"AirPassengersDF is missing required columns: {sorted(missing)}")
    frame = frame[["unique_id", "ds", "y"]].copy()
    frame["unique_id"] = "AirPassengers"
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype("float64")
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def infer_air_frequency(history: pd.DataFrame) -> str:
    """Infer the package's AirPassengers timestamp convention instead of assuming one."""

    dates = pd.Series(pd.to_datetime(history["ds"], errors="raise")).sort_values().drop_duplicates()
    freq = pd.infer_freq(dates)
    if not freq:
        raise ValueError("could not infer AirPassengers frequency from the local statsforecast fixture")
    return {"M": "ME", "Q": "QE"}.get(freq, freq)


def add_synthetic_air_capacity(history: pd.DataFrame) -> pd.DataFrame:
    """Add an explicitly illustrative capacity index for driver-audit demos."""

    enriched = history.copy()
    values: list[float] = []
    for index, ds in enumerate(pd.to_datetime(enriched["ds"])):
        seasonal = 2.5 * math.sin(2 * math.pi * (int(ds.month) - 1) / 12)
        values.append(round(100.0 + 0.32 * index + seasonal, 3))
    enriched["passenger_capacity_index"] = values
    return enriched


def write_synthetic_air_future_capacity(
    history: pd.DataFrame,
    output_dir: str | Path,
    *,
    horizon: int,
    freq: str,
) -> Path:
    """Write exact-horizon future capacity values for the driver availability audit."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    last_date = pd.to_datetime(history["ds"], errors="raise").max()
    future_dates = pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 1, freq=freq)[1:]
    rows: list[dict[str, Any]] = []
    start_index = len(history)
    for step, ds in enumerate(future_dates, start=1):
        seasonal = 2.5 * math.sin(2 * math.pi * (int(ds.month) - 1) / 12)
        ramp = 3.0 if step >= 4 else 0.0
        rows.append(
            {
                "unique_id": "AirPassengers",
                "ds": ds,
                "passenger_capacity_index": round(100.0 + 0.32 * (start_index + step - 1) + seasonal + ramp, 3),
                "known_as_of": last_date,
                "source": "synthetic_demo_plan",
            }
        )
    path = output / "air_passengers_future_capacity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_air_base(
    output_dir: str | Path,
    *,
    model_policy: str = "baseline",
    horizon: int = AIR_HORIZON,
) -> Path:
    history = load_air_passengers_frame()
    spec = ForecastSpec(
        horizon=horizon,
        freq=infer_air_frequency(history),
        season_length=AIR_SEASON_LENGTH,
        model_policy=model_policy,
        unit_label="passengers",
        verbose=False,
    )
    run = run_forecast(history, spec)
    return run.to_directory(output_dir)


def run_air_full(
    output_dir: str | Path,
    *,
    input_dir: str | Path,
    model_policy: str = "light",
    horizon: int = AIR_HORIZON,
) -> Path:
    history = add_synthetic_air_capacity(load_air_passengers_frame())
    freq = infer_air_frequency(history)
    future_file = write_synthetic_air_future_capacity(history, input_dir, horizon=horizon, freq=freq)
    future_dates = pd.read_csv(future_file, parse_dates=["ds"])["ds"]
    event_start = pd.Timestamp(future_dates.iloc[0]).strftime("%Y-%m-%d")
    event_end = pd.Timestamp(future_dates.iloc[min(2, len(future_dates) - 1)]).strftime("%Y-%m-%d")
    spec = ForecastSpec(
        horizon=horizon,
        freq=freq,
        season_length=AIR_SEASON_LENGTH,
        model_policy=model_policy,
        unit_label="passengers",
        transform=TransformSpec(target="log"),
        events=(
            DriverEvent(
                name="Synthetic 1961 capacity ramp",
                start=event_start,
                end=event_end,
                effect="multiplicative",
                magnitude=0.08,
                confidence=0.65,
                notes="Illustrative demo overlay only; AirPassengers does not include a real future event plan.",
            ),
        ),
        regressors=(
            KnownFutureRegressor(
                name="Synthetic passenger capacity plan",
                value_col="passenger_capacity_index",
                availability="plan",
                mode="model_candidate",
                future_file=str(future_file.resolve()),
                source_system="examples.air_tourism_demo",
                owner="demo",
                notes=(
                    "Synthetic known-future values used to demonstrate audit, leakage, and future-coverage "
                    "receipts. Training stays off unless train_known_future_regressors is explicitly enabled."
                ),
            ),
        ),
        train_known_future_regressors=False,
        verbose=False,
    )
    run = run_forecast(history, spec)
    return run.to_directory(output_dir)


def run_demo(
    output_dir: str | Path | None = None,
    *,
    include_tourism: bool = False,
    allow_download: bool = False,
    tourism_cache_dir: str | Path | None = None,
    air_base_model_policy: str = "baseline",
    air_full_model_policy: str = "light",
    tourism_model_policy: str = "baseline",
    air_horizon: int = AIR_HORIZON,
) -> Path:
    output = Path(output_dir) if output_dir is not None else ROOT / "runs" / "air_tourism_demo"
    inputs_dir = output / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    air_history = load_air_passengers_frame()
    air_freq = infer_air_frequency(air_history)
    air_history.to_csv(inputs_dir / "air_passengers_history.csv", index=False)
    enriched_air_history = add_synthetic_air_capacity(air_history)
    enriched_air_history.to_csv(inputs_dir / "air_passengers_enriched_history.csv", index=False)

    base_output = run_air_base(
        output / "air_passengers_base",
        model_policy=air_base_model_policy,
        horizon=air_horizon,
    )
    full_output = run_air_full(
        output / "air_passengers_full",
        input_dir=inputs_dir,
        model_policy=air_full_model_policy,
        horizon=air_horizon,
    )

    tourism_output: Path | None = None
    if include_tourism:
        tourism_module = _load_tourism_example()
        cache_dir = (
            Path(tourism_cache_dir)
            if tourism_cache_dir is not None
            else output / "datasetsforecast_cache"
        )
        tourism_output = tourism_module.run_example(
            output / "tourism_small_bottom_up",
            cache_dir=cache_dir,
            allow_download=allow_download,
            model_policy=tourism_model_policy,
        )

    manifest = _build_manifest(
        output=output,
        air_freq=air_freq,
        air_horizon=air_horizon,
        base_output=base_output,
        full_output=full_output,
        tourism_output=tourism_output,
        include_tourism=include_tourism,
        allow_download=allow_download,
        air_base_model_policy=air_base_model_policy,
        air_full_model_policy=air_full_model_policy,
        tourism_model_policy=tourism_model_policy,
    )
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "demo_summary.md").write_text(_render_summary(manifest), encoding="utf-8")
    return output


def _load_tourism_example():
    spec = importlib.util.spec_from_file_location("forecast_tourism_small_example", TOURISM_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load TourismSmall example from {TOURISM_EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_manifest(
    *,
    output: Path,
    air_freq: str,
    air_horizon: int,
    base_output: Path,
    full_output: Path,
    tourism_output: Path | None,
    include_tourism: bool,
    allow_download: bool,
    air_base_model_policy: str,
    air_full_model_policy: str,
    tourism_model_policy: str,
) -> dict[str, Any]:
    runs = [
        {
            "name": "air_passengers_base",
            "dataset": "statsforecast.utils.AirPassengersDF",
            "purpose": "fast statistical baseline and first control-pane view",
            "path": str(base_output),
            "workbench": str(base_output / "streamlit_app.py"),
            "launch": f"Set-Location {base_output}; .\\run_streamlit.ps1",
            "model_policy": air_base_model_policy,
        },
        {
            "name": "air_passengers_full",
            "dataset": "statsforecast.utils.AirPassengersDF + synthetic demo drivers",
            "purpose": "log transform, scenario overlay, known-future regressor audit, and richer workbench evidence",
            "path": str(full_output),
            "workbench": str(full_output / "streamlit_app.py"),
            "launch": f"Set-Location {full_output}; .\\run_streamlit.ps1",
            "model_policy": air_full_model_policy,
        },
    ]
    if tourism_output is not None:
        runs.append(
            {
                "name": "tourism_small_bottom_up",
                "dataset": "datasetsforecast.hierarchical TourismSmall",
                "purpose": "opt-in public real-data hierarchy with bottom-up reconciliation",
                "path": str(tourism_output),
                "workbench": str(tourism_output / "streamlit_app.py"),
                "launch": f"Set-Location {tourism_output}; .\\run_streamlit.ps1",
                "model_policy": tourism_model_policy,
            }
        )
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "output": str(output),
        "air_frequency": air_freq,
        "air_horizon": air_horizon,
        "tourism_included": include_tourism,
        "tourism_download_allowed": allow_download,
        "input_artifacts": {
            "air_history": str(output / "inputs" / "air_passengers_history.csv"),
            "air_enriched_history": str(output / "inputs" / "air_passengers_enriched_history.csv"),
            "air_future_capacity": str(output / "inputs" / "air_passengers_future_capacity.csv"),
        },
        "runs": runs,
        "limitations": [
            "AirPassengers future capacity and event assumptions are synthetic and illustrative.",
            "The base run is a statistical baseline, not a business plan.",
            "TourismSmall may download public data only when --allow-download is supplied.",
            "Open trust_summary.csv, model_card.md, and the Control Pane before presenting any run.",
        ],
    }


def _render_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# Air + Tourism forecast demo",
        "",
        "This demo is a thin orchestration layer over normal `nixtla-scaffold` runs. It keeps the generated run folders as the source of truth and uses the Streamlit Control Pane for the shared human/agent review surface.",
        "",
        "## Runs",
        "",
    ]
    for run in manifest["runs"]:
        lines.extend(
            [
                f"### {run['name']}",
                "",
                f"- Dataset: {run['dataset']}",
                f"- Purpose: {run['purpose']}",
                f"- Run folder: `{run['path']}`",
                f"- Launch: `{run['launch']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Suggested screenshot sequence",
            "",
            "1. Open the AirPassengers base Control Pane and capture the run summary plus generated rerun command.",
            "2. Open the AirPassengers base Forecast review to show the first-pass statistical baseline.",
            "3. Open the AirPassengers full Control Pane to show the log transform, scenario overlay, and driver audit levers.",
            "4. Open Assumptions & Drivers in the full run to show `scenario_assumptions.csv` and `driver_availability_audit.csv`.",
            "5. If TourismSmall was included, open the hierarchy/reconciliation sections to show coherent bottom-up public real-data validation.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in manifest["limitations"])
    lines.append("")
    return "\n".join(lines)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AirPassengers + optional TourismSmall demo and write a thin manifest/summary for screenshots."
        )
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "air_tourism_demo")
    parser.add_argument("--include-tourism", action="store_true", help="Also run the opt-in TourismSmall hierarchy example")
    parser.add_argument("--allow-download", action="store_true", help="Permit one-time public TourismSmall download")
    parser.add_argument("--tourism-cache-dir", type=Path, default=None)
    parser.add_argument("--air-base-model-policy", default="baseline", choices=["baseline", "statsforecast", "light", "auto", "all"])
    parser.add_argument("--air-full-model-policy", default="light", choices=["baseline", "statsforecast", "light", "auto", "all"])
    parser.add_argument("--tourism-model-policy", default="baseline", choices=["baseline", "statsforecast", "light", "auto", "all"])
    parser.add_argument("--air-horizon", type=int, default=AIR_HORIZON)
    args = parser.parse_args(argv)
    try:
        print(
            run_demo(
                args.output,
                include_tourism=args.include_tourism,
                allow_download=args.allow_download,
                tourism_cache_dir=args.tourism_cache_dir,
                air_base_model_policy=args.air_base_model_policy,
                air_full_model_policy=args.air_full_model_policy,
                tourism_model_policy=args.tourism_model_policy,
                air_horizon=args.air_horizon,
            )
        )
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
