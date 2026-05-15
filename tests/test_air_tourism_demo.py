from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_example():
    path = REPO_ROOT / "examples" / "air_tourism_demo" / "forecast_air_tourism_demo.py"
    spec = importlib.util.spec_from_file_location("forecast_air_tourism_demo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_air_passengers_loader_preserves_month_end_frequency() -> None:
    module = _load_example()

    history = module.load_air_passengers_frame()

    assert set(history["unique_id"]) == {"AirPassengers"}
    assert module.infer_air_frequency(history) == "ME"
    assert pd.to_datetime(history["ds"]).dt.is_month_end.all()


def test_air_tourism_demo_air_only_writes_base_full_and_audit_artifacts(tmp_path) -> None:
    module = _load_example()

    output = module.run_demo(
        tmp_path / "air_demo",
        include_tourism=False,
        air_base_model_policy="baseline",
        air_full_model_policy="baseline",
        air_horizon=3,
    )

    manifest = json.loads((output / "demo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == module.DEMO_SCHEMA_VERSION
    assert [run["name"] for run in manifest["runs"]] == ["air_passengers_base", "air_passengers_full"]
    assert (output / "demo_summary.md").exists()
    assert (output / "air_passengers_base" / "forecast.csv").exists()
    assert (output / "air_passengers_base" / "streamlit_app.py").exists()
    assert (output / "air_passengers_base" / "control_pane_state.json").exists()
    assert (output / "air_passengers_full" / "forecast.csv").exists()
    assert (output / "air_passengers_full" / "streamlit_app.py").exists()
    assert (output / "air_passengers_full" / "control_pane_state.json").exists()
    assert (output / "air_passengers_full" / "appendix" / "scenario_assumptions.csv").exists()
    assert (output / "air_passengers_full" / "appendix" / "known_future_regressors.csv").exists()
    assert (output / "air_passengers_full" / "appendix" / "driver_availability_audit.csv").exists()
    assert (output / "air_passengers_full" / "audit" / "target_transform_audit.csv").exists()

    future_capacity = pd.read_csv(output / "inputs" / "air_passengers_future_capacity.csv")
    assert len(future_capacity) == 3
    assert set(future_capacity["unique_id"]) == {"AirPassengers"}

    driver_audit = pd.read_csv(output / "air_passengers_full" / "appendix" / "driver_availability_audit.csv")
    assert driver_audit["audit_status"].tolist() == ["passed"]
    assert driver_audit["required_future_rows"].tolist() == [3]
    assert driver_audit["missing_future_rows"].tolist() == [0]
    assert driver_audit["modeling_decision"].tolist() == ["candidate_audited_not_trained"]
