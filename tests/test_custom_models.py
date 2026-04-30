from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nixtla_scaffold import CustomModelSpec, ForecastSpec, TransformSpec, run_forecast
from nixtla_scaffold.cli import main
from nixtla_scaffold.model_families import model_family


def _monthly_frame(rows: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * rows,
            "ds": pd.date_range("2024-01-31", periods=rows, freq="ME"),
            "y": [100 + i * 4 + (8 if i % 12 == 11 else 0) for i in range(rows)],
        }
    )


def _last_value_callable(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
    grid = pd.DataFrame(context["future_grid"])
    last = history.sort_values("ds").groupby("unique_id")["y"].last()
    grid["yhat"] = grid["unique_id"].astype(str).map(last)
    return grid[["unique_id", "ds", "yhat"]]


def _perfect_calendar_model(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
    grid = pd.DataFrame(context["future_grid"])
    start = pd.to_datetime(history["ds"]).min().to_period("M")
    periods = pd.to_datetime(grid["ds"]).dt.to_period("M")
    month_index = periods.apply(lambda value: value.ordinal - start.ordinal)
    grid["yhat"] = 100 + month_index.astype(float) ** 2
    return grid[["unique_id", "ds", "yhat"]]


def test_custom_callable_participates_in_tournament_and_artifacts(tmp_path) -> None:
    spec = ForecastSpec(
        horizon=2,
        freq="ME",
        model_policy="baseline",
        weighted_ensemble=False,
        custom_models=(CustomModelSpec(name="Finance Seasonality", callable=_last_value_callable),),
    )

    run = run_forecast(_monthly_frame(), spec)
    out_dir = run.to_directory(tmp_path / "custom_run")

    assert "Custom_Finance_Seasonality" in run.all_models.columns
    assert set(run.backtest_metrics["model"]) >= {"Custom_Finance_Seasonality"}
    assert model_family("Custom_Finance_Seasonality") == "custom"
    assert not run.custom_model_contracts.empty
    assert not run.custom_model_invocations.empty
    assert set(run.custom_model_invocations["status"]) == {"succeeded"}
    assert run.manifest()["outputs"]["custom_model_contracts"] == "custom_model_contracts.csv"
    assert (out_dir / "custom_model_contracts.csv").exists()
    assert (out_dir / "audit" / "custom_model_invocations.csv").exists()
    assert spec.to_dict()["custom_models"][0]["callable_object_ref"].endswith("_last_value_callable")


def test_custom_callable_gets_cutoff_limited_history() -> None:
    seen: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    def spy(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
        grid = pd.DataFrame(context["future_grid"])
        seen.append((pd.to_datetime(history["ds"]).max(), pd.Timestamp(cutoff), pd.to_datetime(grid["ds"]).min()))
        return _last_value_callable(history, horizon=horizon, freq=freq, cutoff=cutoff, levels=levels, context=context)

    run_forecast(
        _monthly_frame(),
        ForecastSpec(
            horizon=3,
            freq="ME",
            model_policy="baseline",
            weighted_ensemble=False,
            custom_models=(CustomModelSpec(name="Leakage Spy", callable=spy),),
        ),
    )

    assert seen
    assert all(history_max <= cutoff for history_max, cutoff, _ in seen)
    assert all(first_future > cutoff for _, cutoff, first_future in seen)


def test_custom_backtest_history_is_limited_to_future_grid_series() -> None:
    seen: list[tuple[set[str], set[str]]] = []

    def spy(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
        grid = pd.DataFrame(context["future_grid"])
        if context["invocation_kind"] == "backtest":
            seen.append((set(history["unique_id"].astype(str)), set(grid["unique_id"].astype(str))))
        return _last_value_callable(history, horizon=horizon, freq=freq, cutoff=cutoff, levels=levels, context=context)

    a = _monthly_frame(18)
    b = _monthly_frame(14).assign(unique_id="New Product")
    run_forecast(
        pd.concat([a, b], ignore_index=True),
        ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            weighted_ensemble=False,
            custom_models=(CustomModelSpec(name="Series Scoped", callable=spy),),
        ),
    )

    assert seen
    assert all(history_ids == grid_ids for history_ids, grid_ids in seen)


def test_custom_mid_run_failure_preserves_invocation_telemetry() -> None:
    calls = {"backtest": 0}

    def sometimes_fails(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
        if context["invocation_kind"] == "backtest":
            calls["backtest"] += 1
            if calls["backtest"] == 2:
                raise RuntimeError("simulated second-window failure")
        return _last_value_callable(history, horizon=horizon, freq=freq, cutoff=cutoff, levels=levels, context=context)

    run = run_forecast(
        _monthly_frame(),
        ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            weighted_ensemble=False,
            custom_models=(CustomModelSpec(name="Sometimes Fails", callable=sometimes_fails),),
        ),
    )

    assert "Custom_Sometimes_Fails" not in run.all_models.columns
    statuses = set(run.custom_model_invocations["status"])
    assert statuses == {"succeeded", "failed"}
    failed = run.custom_model_invocations[run.custom_model_invocations["status"] == "failed"].iloc[0]
    assert failed["invocation_kind"] == "backtest"
    assert str(failed["cutoff"])
    assert "simulated second-window failure" in failed["error"]


def test_custom_output_validation_failure_is_audited_and_excluded() -> None:
    def duplicate_rows(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
        out = _last_value_callable(history, horizon=horizon, freq=freq, cutoff=cutoff, levels=levels, context=context)
        return pd.concat([out, out.iloc[[0]]], ignore_index=True)

    run = run_forecast(
        _monthly_frame(),
        ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            weighted_ensemble=False,
            custom_models=(CustomModelSpec(name="Bad Duplicate", callable=duplicate_rows),),
        ),
    )

    assert "Custom_Bad_Duplicate" not in run.all_models.columns
    assert "custom model Custom_Bad_Duplicate failed and was excluded" in "\n".join(run.warnings)
    assert run.custom_model_contracts["status"].iloc[0] == "failed"
    assert run.custom_model_invocations["status"].iloc[0] == "failed"


def test_custom_output_requires_exact_grid_and_finite_values() -> None:
    def nonfinite(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
        out = _last_value_callable(history, horizon=horizon, freq=freq, cutoff=cutoff, levels=levels, context=context)
        out["yhat"] = out["yhat"].astype(float)
        out.loc[out.index[0], "yhat"] = np.inf
        return out

    run = run_forecast(
        _monthly_frame(),
        ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            weighted_ensemble=False,
            custom_models=(CustomModelSpec(name="Bad Infinite", callable=nonfinite),),
        ),
    )

    assert "Custom_Bad_Infinite" not in run.all_models.columns
    assert "finite numeric" in run.custom_model_contracts["error"].iloc[0]


def test_custom_model_is_rejected_with_target_normalization() -> None:
    frame = _monthly_frame().assign(price_factor=1.05)
    run = run_forecast(
        frame,
        ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            transform=TransformSpec(normalization_factor_col="price_factor"),
            custom_models=(CustomModelSpec(name="Normalized Custom", callable=_last_value_callable),),
        ),
    )

    assert "Custom_Normalized_Custom" not in run.all_models.columns
    assert "normalization_factor_col" in run.custom_model_contracts["error"].iloc[0]


def test_custom_model_can_win_selection_without_entering_weighted_ensemble() -> None:
    rows = 18
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * rows,
            "ds": pd.date_range("2024-01-31", periods=rows, freq="ME"),
            "y": [100 + i**2 for i in range(rows)],
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            weighted_ensemble=True,
            custom_models=(CustomModelSpec(name="Perfect Finance", callable=_perfect_calendar_model),),
        ),
    )

    assert run.model_selection["selected_model"].iloc[0] == "Custom_Perfect_Finance"
    assert "Custom_Perfect_Finance" in run.all_models.columns
    assert "WeightedEnsemble" in run.all_models.columns
    assert "Custom_Perfect_Finance" not in set(run.model_weights["model"])


def test_no_custom_run_does_not_emit_custom_artifacts(tmp_path) -> None:
    run = run_forecast(_monthly_frame(), ForecastSpec(horizon=2, freq="ME", model_policy="baseline"))
    out_dir = run.to_directory(tmp_path / "no_custom")

    assert run.custom_model_contracts.empty
    assert "custom_model_contracts" not in run.manifest()["outputs"]
    assert not (out_dir / "custom_model_contracts.csv").exists()
    assert not (out_dir / "audit" / "custom_model_invocations.csv").exists()


def test_custom_script_cli_uses_current_python_for_py_scripts(tmp_path) -> None:
    script_dir = tmp_path / "script dir"
    script_dir.mkdir()
    script = script_dir / "finance model.py"
    script.write_text(
        """
from __future__ import annotations

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--history", required=True)
parser.add_argument("--future-grid", required=True)
parser.add_argument("--context", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

history = pd.read_csv(args.history)
grid = pd.read_csv(args.future_grid)
last = history.sort_values("ds").groupby("unique_id")["y"].last()
grid["yhat"] = grid["unique_id"].astype(str).map(last)
grid[["unique_id", "ds", "yhat"]].to_csv(args.output, index=False)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.csv"
    _monthly_frame().to_csv(input_path, index=False)
    output_dir = tmp_path / "script_run"

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--freq",
            "ME",
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
            "--no-weighted-ensemble",
            "--custom-script",
            str(script),
            "--custom-model-name",
            "Script Model",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    invocations = pd.read_csv(output_dir / "audit" / "custom_model_invocations.csv")
    assert "Custom_Script_Model" in pd.read_csv(output_dir / "forecast_long.csv")["model"].unique()
    assert invocations["used_python_executable"].astype(bool).all()
    commands = [json.loads(command) for command in invocations["command_json"] if isinstance(command, str) and command]
    assert commands and all(command[0] == sys.executable for command in commands)


def test_custom_script_cli_resolves_relative_script_path(tmp_path, monkeypatch) -> None:
    script = tmp_path / "relative_finance_model.py"
    script.write_text(
        """
from __future__ import annotations

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--history", required=True)
parser.add_argument("--future-grid", required=True)
parser.add_argument("--context", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

history = pd.read_csv(args.history)
grid = pd.read_csv(args.future_grid)
last = history.sort_values("ds").groupby("unique_id")["y"].last()
grid["yhat"] = grid["unique_id"].astype(str).map(last)
grid[["unique_id", "ds", "yhat"]].to_csv(args.output, index=False)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _monthly_frame().to_csv(tmp_path / "input.csv", index=False)
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "forecast",
            "--input",
            "input.csv",
            "--freq",
            "ME",
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
            "--no-weighted-ensemble",
            "--custom-script",
            "relative_finance_model.py",
            "--custom-model-name",
            "Relative Script Model",
            "--output",
            "script_run",
        ]
    )

    assert exit_code == 0
    invocations = pd.read_csv(tmp_path / "script_run" / "audit" / "custom_model_invocations.csv")
    commands = [json.loads(command) for command in invocations["command_json"] if isinstance(command, str) and command]
    assert commands
    assert all(Path(command[1]).is_absolute() for command in commands)
