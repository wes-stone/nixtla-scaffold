from __future__ import annotations

import json

import pandas as pd

from nixtla_scaffold.cli import main
from nixtla_scaffold.scenario_lab import run_scenario_lab


def test_scenario_lab_scores_forecast_scenarios(tmp_path) -> None:
    payload = run_scenario_lab(count=8, output_dir=tmp_path / "lab", model_policy="baseline", seed=7)

    assert payload["summary"]["count"] == 8
    assert payload["summary"]["passed"] >= 6
    assert payload["summary"]["composite_score"] > 0
    scores = pd.read_csv(tmp_path / "lab" / "scenario_scores.csv")
    assert {"accuracy_score", "validity_score", "ease_score", "explainability_score", "composite_score", "error_metric"}.issubset(scores.columns)
    intermittent = scores[scores["archetype"] == "intermittent_demand"]
    if not intermittent.empty:
        assert intermittent["error_metric"].isin({"wape", "scale_normalized_absolute_error"}).all()
    assert (tmp_path / "lab" / "scenario_summary.json").exists()
    assert (tmp_path / "lab" / "scenario_recommendations.json").exists()


def test_scenario_lab_cli_writes_summary(tmp_path, capsys) -> None:
    exit_code = main(["scenario-lab", "--count", "5", "--model-policy", "baseline", "--output", str(tmp_path / "lab")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["count"] == 5
    assert (tmp_path / "lab" / "scenario_scores.csv").exists()
