from __future__ import annotations

import json

import pandas as pd
import pytest

from nixtla_scaffold.cli import main
from nixtla_scaffold.external_scoring import (
    EXTERNAL_SCORE_SCHEMA_VERSION,
    score_external_forecasts,
    write_external_forecast_scores,
)


def test_score_external_forecasts_scores_cutoff_snapshots_against_actuals() -> None:
    actuals = _actuals_frame()
    external = pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "cutoff": ["2026-03-31", "2026-03-31"],
            "ds": ["2026-04-30", "2026-05-31"],
            "yhat": [128.0, 145.0],
            "model": ["Finance plan", "Finance plan"],
            "source_id": ["fpna_workbook", "fpna_workbook"],
            "scenario_name": ["base", "base"],
        }
    )

    result = score_external_forecasts(external, actuals, season_length=1, requested_horizon=2)

    long = result.backtest_long
    assert long["scoring_status"].tolist() == ["scored", "scored"]
    assert long["error"].tolist() == [2.0, -5.0]
    assert long["forecast_error"].tolist() == [-2.0, 5.0]
    assert long["mase_scale"].tolist() == [10.0, 10.0]
    assert long["rmsse_scale"].tolist() == [10.0, 10.0]
    assert long["is_external_forecast"].tolist() == [True, True]
    assert long["external_is_actual"].tolist() == [False, False]
    assert long["backtest_status"].tolist() == ["scored_against_actuals", "scored_against_actuals"]

    metrics = result.model_metrics.iloc[0]
    assert metrics["family"] == "external"
    assert metrics["observations"] == 2
    assert metrics["cutoff_count"] == 1
    assert metrics["rmse"] == pytest.approx((14.5) ** 0.5)
    assert metrics["mae"] == pytest.approx(3.5)
    assert metrics["wape"] == pytest.approx(7.0 / 270.0)
    assert metrics["mase"] == pytest.approx(0.35)
    assert metrics["rmsse"] == pytest.approx((0.145) ** 0.5)
    assert metrics["bias"] == pytest.approx(3.0 / 270.0)
    assert metrics["requested_horizon"] == 2
    assert metrics["selection_horizon"] == 2
    assert bool(metrics["cv_horizon_matches_requested"])

    assert result.manifest["schema_version"] == EXTERNAL_SCORE_SCHEMA_VERSION
    assert result.manifest["rows"]["scored_rows"] == 2
    assert result.manifest["rows"]["missing_actual_rows"] == 0
    assert result.manifest["metric_status"] == "scored"
    assert "Rows must pass cutoff < ds" in "\n".join(result.manifest["guardrails"])


def test_score_external_forecasts_keeps_missing_actual_rows_unscored() -> None:
    actuals = _actuals_frame().iloc[:4]
    external = pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "cutoff": ["2026-03-31", "2026-03-31"],
            "ds": ["2026-04-30", "2026-05-31"],
            "yhat": [128.0, 145.0],
            "model": ["Finance plan", "Finance plan"],
            "source_id": ["fpna_workbook", "fpna_workbook"],
        }
    )

    result = score_external_forecasts(external, actuals)

    assert result.backtest_long["scoring_status"].tolist() == ["scored", "missing_actual"]
    assert result.backtest_long["backtest_status"].tolist() == ["scored_against_actuals", "missing_actual_unscored"]
    assert result.model_metrics.loc[0, "observations"] == 1
    assert result.manifest["rows"]["scored_rows"] == 1
    assert result.manifest["rows"]["missing_actual_rows"] == 1


def test_score_external_forecasts_requires_cutoff_labels() -> None:
    external = pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "ds": ["2026-04-30"],
            "yhat": [128.0],
            "model": ["Finance plan"],
            "source_id": ["fpna_workbook"],
        }
    )

    with pytest.raises(ValueError, match="requires cutoff"):
        score_external_forecasts(external, _actuals_frame())


def test_score_external_forecasts_rejects_invalid_scoring_parameters() -> None:
    external = _external_snapshot_frame()
    actuals = _actuals_frame()

    with pytest.raises(ValueError, match="season_length must be a positive integer"):
        score_external_forecasts(external, actuals, season_length=0)
    with pytest.raises(ValueError, match="requested_horizon must be a positive integer"):
        score_external_forecasts(external, actuals, requested_horizon=0)


def test_score_external_forecasts_no_match_error_explains_join_context() -> None:
    external = _external_snapshot_frame(ds="2026-07-31", cutoff="2026-06-30")

    with pytest.raises(ValueError) as exc_info:
        score_external_forecasts(external, _actuals_frame())

    message = str(exc_info.value)
    assert "Join key is unique_id + ds" in message
    assert "External target date range: 2026-07-31 to 2026-07-31" in message
    assert "actual date range: 2026-01-31 to 2026-05-31" in message
    assert "Unmatched external rows sample" in message
    assert "Revenue" in message


def test_score_external_forecasts_rejects_duplicate_actuals() -> None:
    actuals = pd.concat([_actuals_frame(), _actuals_frame().iloc[[0]]], ignore_index=True)
    external = pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "cutoff": ["2026-03-31"],
            "ds": ["2026-04-30"],
            "yhat": [128.0],
            "model": ["Finance plan"],
            "source_id": ["fpna_workbook"],
        }
    )

    with pytest.raises(ValueError, match="duplicate unique_id/ds"):
        score_external_forecasts(external, actuals)


def test_write_external_forecast_scores_outputs_artifacts(tmp_path) -> None:
    external_path = tmp_path / "finance_snapshots.csv"
    actuals_path = tmp_path / "actuals.csv"
    _actuals_frame().to_csv(actuals_path, index=False)
    pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "cutoff": ["2026-03-31"],
            "ds": ["2026-04-30"],
            "yhat": [128.0],
            "model": ["Finance plan"],
            "source_id": ["fpna_workbook"],
        }
    ).to_csv(external_path, index=False)

    result = write_external_forecast_scores(external_path, actuals_path, tmp_path / "scores")

    assert result.manifest["output_dir"] == str(tmp_path / "scores")
    assert (tmp_path / "scores" / "external_backtest_long.csv").exists()
    assert (tmp_path / "scores" / "external_model_metrics.csv").exists()
    manifest = json.loads((tmp_path / "scores" / "external_scoring_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["model_metrics"] == "external_model_metrics.csv"


def test_score_external_cli_writes_artifacts(tmp_path, capsys) -> None:
    external_path = tmp_path / "finance_snapshots.csv"
    actuals_path = tmp_path / "actuals.csv"
    _actuals_frame().to_csv(actuals_path, index=False)
    pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "cutoff": ["2026-03-31"],
            "ds": ["2026-04-30"],
            "yhat": [128.0],
            "model": ["Finance plan"],
            "source_id": ["fpna_workbook"],
        }
    ).to_csv(external_path, index=False)

    exit_code = main(
        [
            "score-external",
            "--external",
            str(external_path),
            "--actuals",
            str(actuals_path),
            "--output",
            str(tmp_path / "cli_scores"),
            "--season-length",
            "1",
            "--horizon",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == EXTERNAL_SCORE_SCHEMA_VERSION
    assert payload["outputs"]["backtest_long"] == "external_backtest_long.csv"
    assert (tmp_path / "cli_scores" / "external_backtest_long.csv").exists()


def test_score_external_cli_rejects_invalid_positive_integer_parameters(tmp_path, capsys) -> None:
    external_path = tmp_path / "finance_snapshots.csv"
    actuals_path = tmp_path / "actuals.csv"
    _external_snapshot_frame().to_csv(external_path, index=False)
    _actuals_frame().to_csv(actuals_path, index=False)

    exit_code = main(
        [
            "score-external",
            "--external",
            str(external_path),
            "--actuals",
            str(actuals_path),
            "--output",
            str(tmp_path / "bad_scores"),
            "--season-length",
            "0",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "season_length must be a positive integer" in captured.err
    payload = json.loads((tmp_path / "bad_scores" / "failure_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["command"] == "score-external"
    assert any("positive integers" in hint for hint in payload["likely_causes"])


def test_score_external_cli_rejects_invalid_horizon(tmp_path, capsys) -> None:
    external_path = tmp_path / "finance_snapshots.csv"
    actuals_path = tmp_path / "actuals.csv"
    _external_snapshot_frame().to_csv(external_path, index=False)
    _actuals_frame().to_csv(actuals_path, index=False)

    exit_code = main(
        [
            "score-external",
            "--external",
            str(external_path),
            "--actuals",
            str(actuals_path),
            "--output",
            str(tmp_path / "bad_horizon_scores"),
            "--horizon",
            "0",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "requested_horizon must be a positive integer" in captured.err
    payload = json.loads((tmp_path / "bad_horizon_scores" / "failure_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["command"] == "score-external"
    assert any("positive integers" in hint for hint in payload["likely_causes"])


def test_score_external_cli_no_match_writes_scoring_diagnostics(tmp_path, capsys) -> None:
    external_path = tmp_path / "future_snapshot.csv"
    actuals_path = tmp_path / "actuals.csv"
    _external_snapshot_frame(ds="2026-07-31", cutoff="2026-06-30").to_csv(external_path, index=False)
    _actuals_frame().to_csv(actuals_path, index=False)

    exit_code = main(
        [
            "score-external",
            "--external",
            str(external_path),
            "--actuals",
            str(actuals_path),
            "--output",
            str(tmp_path / "no_match_scores"),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Join key is unique_id + ds" in captured.err
    payload = json.loads((tmp_path / "no_match_scores" / "failure_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["command"] == "score-external"
    assert "Unmatched external rows sample" in payload["error"]
    assert any("target-date coverage" in hint for hint in payload["likely_causes"])
    assert not any("cutoff/forecast_origin" in hint for hint in payload["likely_causes"])
    assert any("realized values" in step for step in payload["next_diagnostic_steps"])
    assert (tmp_path / "no_match_scores" / "failure_diagnostics.md").exists()


def test_package_root_exports_external_scoring_helpers() -> None:
    import nixtla_scaffold as ns

    assert ns.EXTERNAL_SCORE_SCHEMA_VERSION == EXTERNAL_SCORE_SCHEMA_VERSION
    assert ns.score_external_forecasts is score_external_forecasts
    assert ns.write_external_forecast_scores is write_external_forecast_scores


def _actuals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 5,
            "ds": pd.date_range("2026-01-31", periods=5, freq="ME"),
            "y": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    )


def _external_snapshot_frame(*, ds: str = "2026-04-30", cutoff: str = "2026-03-31") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "cutoff": [cutoff],
            "ds": [ds],
            "yhat": [128.0],
            "model": ["Finance plan"],
            "source_id": ["fpna_workbook"],
        }
    )
