from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import ForecastSpec, SelectionPolicy, resolve_forecast_spec
from nixtla_scaffold import cli
from nixtla_scaffold.accuracy import build_accuracy_gate
from nixtla_scaffold.models import select_champions
from nixtla_scaffold.pipelines import _forecast_spec_from_config, _refresh_spec
from nixtla_scaffold.schema import forecast_spec_from_dict
from nixtla_scaffold.workbench import _safe_spec


def _cli_spec(monkeypatch, command="forecast", *flags):
    captured = []
    monkeypatch.setattr(cli, "_run", lambda args: captured.append(cli._spec_from_args(args)) or 0)
    assert cli.main([command, "--input", "not-opened.csv", "--output", "not-created", *flags]) == 0
    return captured[0]


@pytest.mark.parametrize("command", ["forecast", "compare-models", "experiment", "optimize", "ingest"])
def test_new_requests_resolve_identically_across_surfaces(monkeypatch, command):
    expected = ForecastSpec().to_dict()
    assert resolve_forecast_spec().to_dict() == expected
    assert _forecast_spec_from_config({}).to_dict() == expected
    assert replace(_safe_spec({}), verbose=True).to_dict() == expected
    assert _cli_spec(monkeypatch, command).to_dict() == expected
    assert expected["context"]["decision"] == ""
    assert expected["context"]["research_budget"]["max_variants_per_iteration"] == 1


def test_explicit_flags_and_quick_intent_survive_resolution(monkeypatch):
    spec = _cli_spec(monkeypatch, "forecast", "--preset", "quick", "--horizon", "3",
                     "--model-policy", "statsforecast", "--no-strict-cv-horizon")
    assert spec.run_mode == "quick"
    assert spec.horizon == 3
    assert spec.model_policy == "statsforecast"
    assert spec.strict_cv_horizon is False
    assert spec.accuracy_first is False
    assert resolve_forecast_spec({"run_mode": "quick"}).model_policy == "baseline"


@pytest.mark.parametrize("command", ["forecast", "compare-models", "experiment", "optimize", "ingest"])
def test_cli_preserves_declared_exposure_scope(monkeypatch, command, tmp_path):
    spec = _cli_spec(monkeypatch, command, "--known-target-exposure", "--evidence-directory", str(tmp_path))
    assert spec.evaluation.known_target_exposure == "exposed"
    assert spec.evaluation.evidence_directory == str(tmp_path)


@pytest.mark.parametrize("mode", ["accuracy-first", "quick", "legacy"])
def test_versioned_specs_round_trip(mode):
    spec = resolve_forecast_spec({"run_mode": mode, "selection_policy": {"naive_guard_min_improvement": 0.1}})
    assert forecast_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_versionless_deserialization_and_refresh_preserve_old_defaults(tmp_path):
    original = {"horizon": 8, "freq": "MS", "model_policy": "auto"}
    before = json.dumps({"spec": original})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(before, encoding="utf-8")
    spec = _refresh_spec(tmp_path)
    assert spec.run_mode == "legacy" and spec.policy_version == 0
    assert spec.model_policy == "light" and spec.horizon == 8
    assert spec.paired_validation is False and spec.strict_cv_horizon is False
    assert spec.context is None
    assert forecast_spec_from_dict({}).horizon == 12
    assert resolve_forecast_spec(spec.to_dict()) == spec
    assert manifest.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("settings", [
    {"run_mode": "accuracy-first", "policy_version": 99},
    {"run_mode": "legacy", "policy_version": 1},
    {"run_mode": "unknown"},
    {"policy_version": True},
])
def test_unknown_or_inconsistent_policy_versions_fail(settings):
    with pytest.raises(ValueError):
        resolve_forecast_spec(settings)


def test_saved_new_intent_requires_a_version():
    with pytest.raises(ValueError, match="policy_version"):
        forecast_spec_from_dict({"run_mode": "accuracy-first"})


@pytest.mark.parametrize("margin", [np.nan, np.inf, -0.01, 1.0])
def test_invalid_selection_margins_fail(margin):
    with pytest.raises(ValueError, match="naive_guard"):
        SelectionPolicy(naive_guard_min_improvement=margin)


@pytest.mark.parametrize("margin,expected", [(0.05, "AutoETS"), (0.15, "Naive")])
def test_selection_uses_recorded_practical_margin(margin, expected):
    future = pd.DataFrame({"unique_id": ["A"], "ds": pd.to_datetime(["2026-01-01"]),
                           "AutoETS": [20.0], "Naive": [10.0]})
    metrics = pd.DataFrame({"unique_id": ["A", "A"], "model": ["AutoETS", "Naive"],
                            "rmse": [9.0, 10.0], "mae": [8.0, 9.0], "abs_bias": [0.1, 0.2]})
    result = select_champions(future, metrics, policy=SelectionPolicy(margin))
    assert result.iloc[0].selected_model == expected


def test_perfect_reference_is_retained_on_zero_error_tie():
    future = pd.DataFrame({"unique_id": ["A"], "ds": pd.to_datetime(["2026-01-01"]),
                           "AutoETS": [20.0], "Naive": [10.0]})
    metrics = pd.DataFrame({"unique_id": ["A", "A"], "model": ["AutoETS", "Naive"],
                            "rmse": [0.0, 0.0], "mae": [0.0, 0.0], "abs_bias": [0.0, 0.0]})
    assert select_champions(future, metrics).iloc[0].selected_model == "Naive"


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf])
def test_undefined_panel_metrics_cannot_drop_difficult_series(invalid):
    from nixtla_scaffold.evaluation import complete_metric_mean
    from nixtla_scaffold.research import _select_primary_metric
    frame = pd.DataFrame({"unique_id": ["easy", "difficult"], "rmsse": [0.5, invalid]})
    assert complete_metric_mean(frame, "rmsse") is None
    assert _select_primary_metric({"series_count": 2, "avg_rmsse": None, "avg_rmse": 1.0}) is None
    assert _select_primary_metric({"series_count": 1, "avg_rmsse": None, "avg_rmse": 1.0}) == "avg_rmse"


def test_panel_mean_weights_series_not_number_of_windows():
    from nixtla_scaffold.evaluation import complete_metric_mean
    frame = pd.DataFrame({"unique_id": ["A", "A", "B"], "rmsse": [1.0, 1.0, 3.0]})
    assert complete_metric_mean(frame, "rmsse") == 2.0


def _gate_run(mode="accuracy-first", paired=True):
    from nixtla_scaffold.schema import AccuracyPolicy, ForecastContext
    context = ForecastContext(accuracy_policy=AccuracyPolicy(require_context_discovery=False))
    spec = ForecastSpec(horizon=2, run_mode=mode, paired_validation=paired, context=context)
    run = SimpleNamespace(spec=spec, driver_availability_audit=pd.DataFrame(),
                          forecast=pd.DataFrame({"unique_id": ["A", "A"],
                                                 "ds": pd.date_range("2026-01-01", periods=2, freq="MS"),
                                                 "yhat": [10.0, 11.0]}))
    trust = pd.DataFrame([{"unique_id": "A", "trust_score_0_100": 100,
                           "full_horizon_claim_allowed": True, "horizon_gate_result": "passed",
                           "interval_status": "calibrated", "hierarchy_status": "no_hierarchy"}])
    return run, trust


@pytest.mark.parametrize("mode,paired,reason", [
    ("quick", True, "run_intent"), ("accuracy-first", False, "comparison_support"),
])
def test_exploratory_or_unpaired_run_cannot_claim_planning_ready(mode, paired, reason):
    run, trust = _gate_run(mode, paired)
    gate = build_accuracy_gate(run, trust)
    assert not gate["planning_ready_claim_allowed"]
    assert reason in gate["series"][0]["failed_gates"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "infinite", "nan"])
def test_claim_gate_rejects_incomplete_or_nonfinite_predictions(defect):
    run, trust = _gate_run()
    if defect == "missing":
        run.forecast = run.forecast.head(1)
    elif defect == "duplicate":
        run.forecast.loc[1, "ds"] = run.forecast.loc[0, "ds"]
    else:
        run.forecast.loc[1, "yhat"] = np.inf if defect == "infinite" else np.nan
    gate = build_accuracy_gate(run, trust)
    assert gate["status"] == "blocked"
    assert gate["series"][0]["failed_gates"] == ["forecast"]
