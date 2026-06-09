import json
from pathlib import Path

import pandas as pd
import pytest

from nixtla_scaffold.challengers import (
    ChallengerForecasts,
    ChallengerSkip,
    available_challenger_engines,
    build_challenger_leaderboard,
    build_finn_spec_runner,
    get_challenger_engine,
    register_challenger_engine,
    run_challengers,
)
from nixtla_scaffold.schema import ChallengerSpec


def _fake_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "appendix").mkdir(parents=True)
    (run_dir / "audit").mkdir(parents=True)
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")
    history = pd.DataFrame(
        {"unique_id": ["series_a"] * 24, "ds": dates, "y": [100.0 + index for index in range(24)]}
    )
    history.to_csv(run_dir / "appendix" / "history.csv", index=False)
    pd.DataFrame(
        [
            {
                "unique_id": "series_a",
                "model": "SeasonalNaive",
                "rmse": 5.0,
                "mae": 4.0,
                "wape": 0.04,
                "mase": 0.9,
                "rmsse": 0.8,
                "bias": 0.1,
                "abs_bias": 0.1,
                "observations": 12,
                "cv_windows": 4,
                "cv_horizon_matches_requested": True,
            }
        ]
    ).to_csv(run_dir / "audit" / "backtest_metrics.csv", index=False)
    spec = {"horizon": 3, "freq": "ME", "season_length": 12}
    (run_dir / "manifest.json").write_text(json.dumps({"spec": spec, "outputs": {}}), encoding="utf-8")
    (run_dir / "llm_context.json").write_text(json.dumps({"artifact_index": {}}), encoding="utf-8")
    return run_dir


class _StubEngine:
    name = "stub"

    def check_environment(self, spec: ChallengerSpec) -> dict:
        return {"available": True}

    def run(self, *, history, spec, run_spec, output_dir):
        cutoffs = pd.to_datetime(history["ds"]).iloc[-7:-4]
        rows = []
        for cutoff in cutoffs:
            for step in (1, 2, 3):
                ds = cutoff + pd.offsets.MonthEnd(step)
                rows.append(
                    {"unique_id": "series_a", "ds": ds, "model": "STUB_model", "yhat": 100.0, "cutoff": cutoff}
                )
        backtest = pd.DataFrame(rows)
        backtest = backtest[pd.to_datetime(backtest["ds"]) <= pd.to_datetime(history["ds"]).max()]
        return ChallengerForecasts(future=pd.DataFrame(), backtest=backtest)


class _UnavailableEngine:
    name = "offline"

    def check_environment(self, spec: ChallengerSpec) -> dict:
        return {"available": False, "reason": "engine offline", "install_hint": "install the engine"}

    def run(self, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("run should not be called when environment is unavailable")


register_challenger_engine("stub", _StubEngine)
register_challenger_engine("offline", _UnavailableEngine)


def test_engine_registry_round_trip() -> None:
    assert "finn" in available_challenger_engines()
    assert get_challenger_engine("stub").name == "stub"
    with pytest.raises(ChallengerSkip):
        get_challenger_engine("does_not_exist")


def test_run_challengers_stub_engine_scores_and_builds_leaderboard(tmp_path) -> None:
    run_dir = _fake_run_dir(tmp_path)
    spec = ChallengerSpec(engine="stub", source_id="stub", model_name="STUB")

    payload = run_challengers(run_dir, challengers=(spec,))

    assert payload["completed"] == 1
    status = json.loads((run_dir / "stub" / "challenger_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["advisory_only"] is True
    brief = json.loads((run_dir / "stub" / "agent_brief.json").read_text(encoding="utf-8"))
    assert brief["comparable_evidence"] == "stub/external_model_metrics.csv"
    assert (run_dir / "stub" / "external_model_metrics.csv").exists()

    leaderboard = pd.read_csv(run_dir / "appendix" / "challenger_leaderboard.csv")
    assert set(leaderboard["lane"]) == {"native", "challenger"}
    native = leaderboard[leaderboard["lane"] == "native"].iloc[0]
    assert native["source_id"] == "scaffold"
    assert bool(native["comparable"]) is True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["challenger_leaderboard"] == "appendix/challenger_leaderboard.csv"
    assert manifest["outputs"]["stub_challenger_status"] == "stub/challenger_status.json"


def test_run_challengers_soft_fails_when_environment_unavailable(tmp_path) -> None:
    run_dir = _fake_run_dir(tmp_path)
    spec = ChallengerSpec(engine="offline", source_id="offline")

    payload = run_challengers(run_dir, challengers=(spec,))

    assert payload["completed"] == 0
    assert payload["skipped"] == 1
    status = json.loads((run_dir / "offline" / "challenger_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped"
    assert "engine offline" in status["reason"]
    assert status["remediation"] == "install the engine"


def test_run_challengers_on_error_fail_raises(tmp_path) -> None:
    run_dir = _fake_run_dir(tmp_path)
    spec = ChallengerSpec(engine="offline", source_id="offline", on_error="fail")

    with pytest.raises(ChallengerSkip):
        run_challengers(run_dir, challengers=(spec,))


def test_disabled_challenger_records_disabled_status(tmp_path) -> None:
    run_dir = _fake_run_dir(tmp_path)
    spec = ChallengerSpec(engine="stub", source_id="stub", enabled=False)

    payload = run_challengers(run_dir, challengers=(spec,))

    assert payload["completed"] == 0
    status = json.loads((run_dir / "stub" / "challenger_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "disabled"


def test_finn_spec_runner_is_params_driven_and_auditable() -> None:
    runner = build_finn_spec_runner()
    assert 'get_arg("--params")' in runner
    assert "jsonlite::fromJSON(params_path)" in runner
    assert "finnts::forecast_time_series" in runner
    assert "back_test_data" in runner
    assert "set.seed" in runner


def test_leaderboard_empty_when_no_metrics(tmp_path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    assert build_challenger_leaderboard(run_dir).empty


def test_cli_forecast_finn_soft_fails_without_rscript(tmp_path) -> None:
    from nixtla_scaffold.cli import main

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "run"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 30,
            "ds": pd.date_range("2023-01-31", periods=30, freq="ME"),
            "y": [100.0 + index * 2 for index in range(30)],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--freq",
            "ME",
            "--horizon",
            "3",
            "--season-length",
            "12",
            "--model-policy",
            "baseline",
            "--no-verbose",
            "--finn",
            "--finn-rscript",
            r"C:\nonexistent\Rscript.exe",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    status = json.loads((output_path / "finn" / "challenger_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped"
    assert status.get("remediation")

    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    challenger_specs = manifest["spec"]["challengers"]
    assert challenger_specs[0]["engine"] == "finn"
    assert challenger_specs[0]["enabled"] is True
    from nixtla_scaffold.schema import _challenger_from_dict

    round_trip = _challenger_from_dict(challenger_specs[0])
    assert round_trip.rscript == r"C:\nonexistent\Rscript.exe"
    assert not (output_path / "appendix" / "challenger_leaderboard.csv").exists()
