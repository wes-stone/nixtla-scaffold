from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from nixtla_scaffold import ForecastSpec, ProductionPolicy, forecast_spec_preset, run_forecast
from nixtla_scaffold.production import (
    ProductionReleaseError, _add_frozen_blend, _choose, _complete_prior, _replay,
    prepare_production_spec, qualify_production_run, write_production_delivery,
)
from nixtla_scaffold.schema import forecast_spec_from_dict
from nixtla_scaffold.cli import main


def _spec(**overrides):
    return forecast_spec_preset(
        "production", horizon=7, freq="D", season_length=7, model_policy="baseline",
        model_allowlist=("Naive", "SeasonalNaive"), verbose=False, **overrides,
    )


@pytest.fixture(scope="module")
def qualified_run():
    data = pd.DataFrame({"unique_id": "stable", "ds": pd.date_range("2020-01-01", periods=300), "y": 100.0})
    return run_forecast(data, _spec())


def test_policy_round_trip_and_legacy_preservation():
    original = _spec(production=ProductionPolicy(ensemble_members=("Naive", "SeasonalNaive")))
    assert forecast_spec_from_dict(original.to_dict()) == original
    assert forecast_spec_from_dict({}).production is None
    assert ForecastSpec().production is None
    prepared = prepare_production_spec(original)
    assert prepared.evaluation.step_size == prepared.horizon
    assert prepared.evaluation.max_windows == 32
    assert prepared.require_backtest and prepared.strict_cv_horizon


@pytest.mark.parametrize("kwargs", [
    {"max_error_pct": float("nan")}, {"max_abs_bias_pct": True}, {"coverage_tolerance": 0.2},
    {"calibration_windows": 19}, {"minimum_selection_windows": 2}, {"minimum_assessment_windows": 4},
    {"ensemble_members": ("Naive",)}, {"ensemble_members": ("Naive", "Naive")}, {"version": 2},
])
def test_policy_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        ProductionPolicy(**kwargs)


def test_unknown_policy_field_is_not_silently_ignored():
    payload = _spec().to_dict()
    payload["production"]["max_eror_pct"] = 0
    with pytest.raises(ValueError, match="declared production"):
        forecast_spec_from_dict(payload)


def test_benchmarks_are_mandatory_and_unsupported_paths_fail():
    spec = replace(_spec(), model_allowlist=("AutoETS",))
    assert prepare_production_spec(spec).model_allowlist == ("AutoETS", "Naive", "SeasonalNaive")
    with pytest.raises(ValueError, match="weighted_ensemble"):
        prepare_production_spec(replace(spec, weighted_ensemble=True))
    with pytest.raises(ValueError, match="80% and 95%"):
        prepare_production_spec(replace(spec, levels=()))
    with pytest.raises(ValueError, match="configuration_guard"):
        prepare_production_spec(replace(spec, selection_policy=replace(spec.selection_policy, configuration_guard="warn")))


def test_constant_native_run_qualifies_exact_policy(qualified_run):
    evidence = qualified_run.model_policy_resolution["production"]
    report = evidence["readiness"]
    assert report["release_allowed"], report["failed_checks"]
    assert report["series"][0]["assessment_windows"] == 9
    assert report["series"][0]["policy_windows"] == 29
    assert all(item["passed"] for item in report["checks"])
    assert np.allclose(qualified_run.forecast.yhat, 100)
    for decision in evidence["decisions"]:
        assert pd.Timestamp(decision["latest_prior_target"]) <= pd.Timestamp(decision["cutoff"])
        assert pd.Timestamp(decision["latest_prior_available_at"]) <= pd.Timestamp(decision["cutoff"])
    for state in evidence["calibration"]:
        for receipt in state["receipts"]:
            for window in receipt.get("calibration", {}).get("periods", []):
                assert pd.Timestamp(window["available_at"]) <= pd.Timestamp(receipt["cutoff"])


def test_atomic_release_hashes_and_no_overwrite(qualified_run, tmp_path):
    (tmp_path / "audit").mkdir()
    write_production_delivery(qualified_run, tmp_path)
    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text())
    assert manifest["business_approval"] == "not_granted"
    assert manifest["reproducibility"]["data_hash_sha256"]
    for name, digest in manifest["sha256"].items():
        assert sha256((tmp_path / "release" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="existing production release"):
        write_production_delivery(qualified_run, tmp_path)
    assert not list(tmp_path.glob(".release-*"))


def test_mutated_path_cannot_reuse_qualification(qualified_run, tmp_path):
    changed = qualified_run.forecast.copy()
    changed.loc[0, "yhat"] += 100
    with pytest.raises(ProductionReleaseError, match="changed after"):
        write_production_delivery(replace(qualified_run, forecast=changed), tmp_path)
    assert not (tmp_path / "release").exists()


def test_bad_forecast_is_rejected_with_diagnostics(qualified_run, tmp_path):
    cv, future = qualified_run.backtest_predictions.copy(), qualified_run.all_models.copy()
    for frame in (cv, future):
        frame["Naive"] = 200.0
        frame["SeasonalNaive"] = 200.0
    rejected = qualify_production_run(replace(qualified_run, backtest_predictions=cv, all_models=future))
    assert not rejected.model_policy_resolution["production"]["readiness"]["release_allowed"]
    (tmp_path / "audit").mkdir()
    with pytest.raises(ProductionReleaseError, match="objective_error_pct"):
        write_production_delivery(rejected, tmp_path)
    assert (tmp_path / "production_readiness.json").exists()
    assert not (tmp_path / "release").exists()


def test_qualifying_diagnostic_trial_never_creates_release(qualified_run, tmp_path):
    out = qualified_run.to_directory(tmp_path / "trial", release=False)
    manifest = json.loads((out / "manifest.json").read_text())
    readiness = json.loads((out / "production_readiness.json").read_text())
    assert readiness["release_allowed"] is True
    assert manifest["production_delivery"]["requested"] is False
    assert "production_release" not in manifest["outputs"]
    assert not (out / "release").exists()
    with pytest.raises(ValueError, match="boolean"):
        qualified_run.to_directory(tmp_path / "invalid", release="false")


def test_rejected_production_trials_still_compare_policy_without_release(qualified_run, monkeypatch, tmp_path):
    from nixtla_scaffold import experiments

    def forecast(data, spec, **kwargs):
        cv, future = qualified_run.backtest_predictions.copy(), qualified_run.all_models.copy()
        for frame in (cv, future):
            frame["Naive"] = frame["SeasonalNaive"] = 150 if spec.anomaly.mode == "adjust" else 200
        return qualify_production_run(replace(
            qualified_run, spec=spec, backtest_predictions=cv, all_models=future,
        ))

    monkeypatch.setattr(experiments, "run_forecast", forecast)
    result = experiments.run_experiment(
        qualified_run.history, qualified_run.spec, output_dir=tmp_path,
        variants=("anomaly_adjustment",), max_variants=1,
    )
    rows = result.summary.set_index("variant")
    assert rows.status.eq("success").all()
    assert rows.production_qualification.eq("rejected").all()
    assert rows.scoring_scope.eq("deployed_policy_later_assessment").all()
    assert rows.assessment_fingerprint.nunique() == 1
    assert rows.loc["baseline", "avg_rmse"] == pytest.approx(100)
    assert rows.loc["anomaly_adjustment", "avg_rmse"] == pytest.approx(50)
    for variant in ("baseline", "anomaly_adjustment"):
        assert (tmp_path / "variants" / variant / "production_readiness.json").exists()
        assert not (tmp_path / "variants" / variant / "release").exists()
    changed = result.summary.copy()
    changed.loc[changed.variant.eq("anomaly_adjustment"), "assessment_fingerprint"] = "different"
    comparison = experiments._apply_candidate_set_compatibility(changed).set_index("variant")
    assert not comparison.loc["anomaly_adjustment", "promotion_evidence_eligible"]
    assert comparison.loc["anomaly_adjustment", "candidate_set_comparison_status"] == "assessment_cohort_mismatch"
    context = experiments.ExperimentContext(**result.manifest["context"])
    recommendation = experiments._recommendation_payload(comparison.reset_index(), context)
    assert recommendation["best_variant"] == "baseline"
    tied = result.summary.copy()
    for column in ("avg_rmse", "avg_mae", "advisory_rank_score"):
        tied[column] = 0.0
    tied = experiments._assign_advisory_ranks(tied).set_index("variant")
    assert tied.loc["baseline", "advisory_rank"] == 1


def test_missing_policy_window_cannot_disappear_from_evidence(qualified_run):
    cv = qualified_run.backtest_predictions.copy()
    cutoff = sorted(cv.cutoff.unique())[27]
    cv.loc[cv.cutoff.eq(cutoff), ["Naive", "SeasonalNaive"]] = np.nan
    with pytest.raises(ValueError, match="complete candidate"):
        qualify_production_run(replace(qualified_run, backtest_predictions=cv))


def test_selection_ignores_current_and_future_actuals(qualified_run):
    run = qualified_run
    cv = run.backtest_predictions.copy().assign(available_at=lambda x: x.ds)
    cutoff = sorted(cv.cutoff.unique())[10]
    paths = cv.loc[cv.cutoff.eq(cutoff)].drop(columns=["cutoff", "y", "available_at"])
    prior = _complete_prior(cv, cutoff, 7)
    args = dict(cutoff=cutoff, freq="D", season=7)
    first = _choose(run.history, prior, paths, run.spec, **args)
    changed = run.history.copy()
    changed.loc[changed.ds.gt(cutoff), "y"] = 1e12
    cv.loc[cv.ds.gt(cutoff), "y"] = 1e12
    second = _choose(changed, _complete_prior(cv, cutoff, 7), paths, run.spec, **args)
    assert first[0]["selected_model"] == second[0]["selected_model"]
    assert first[1] == second[1]


def test_publication_delay_withholds_entire_window(qualified_run):
    cv = qualified_run.backtest_predictions.copy().assign(available_at=lambda x: x.ds)
    cutoff = sorted(cv.cutoff.unique())[8]
    prior_cutoff = sorted(cv.cutoff.unique())[7]
    cv.loc[cv.cutoff.eq(prior_cutoff), "available_at"] = pd.Timestamp(cutoff) + pd.Timedelta(days=2)
    prior = _complete_prior(cv, cutoff, 7)
    assert prior_cutoff not in prior.cutoff.unique()


def test_frozen_blend_never_reweights_missing_member():
    frame = pd.DataFrame({"Naive": [100.0, 100.0], "SeasonalNaive": [80.0, np.nan]})
    result = _add_frozen_blend(frame, ("Naive", "SeasonalNaive"))
    assert result.ProductionBlend.iloc[0] == 90
    assert pd.isna(result.ProductionBlend.iloc[1])


def test_future_model_failure_does_not_rewrite_historical_selection(qualified_run):
    run = qualified_run
    args = dict(freq="D", season=7)
    first, _, _ = _replay(run.history, run.backtest_predictions, run.all_models, run.spec, **args)
    future = run.all_models.copy()
    future["SeasonalNaive"] = np.nan
    second, _, _ = _replay(run.history, run.backtest_predictions, future, run.spec, **args)
    pd.testing.assert_frame_equal(first, second)


def test_cli_production_preset_is_not_lost_and_rejection_is_nonzero(monkeypatch, tmp_path, capsys):
    from nixtla_scaffold import cli

    class Rejected:
        def to_directory(self, output):
            raise ProductionReleaseError("qualification rejected; diagnostics retained")

    def run(data, spec, **kwargs):
        assert spec.production == ProductionPolicy()
        assert not spec.weighted_ensemble
        return Rejected()

    monkeypatch.setattr(cli, "run_forecast", run)
    assert main(["forecast", "--input", "unused.csv", "--preset", "production",
                 "--output", str(tmp_path / "new")]) == 2
    assert "qualification rejected" in capsys.readouterr().out


def test_pipeline_rejection_is_not_execution_success(qualified_run, monkeypatch, tmp_path):
    from nixtla_scaffold import pipelines

    config = tmp_path / "pipeline.yaml"
    config.write_text("extracts:\n  - name: source\nforecast:\n  preset: production\n")
    source = tmp_path / "source.csv"
    qualified_run.history.to_csv(source, index=False)
    monkeypatch.setattr(pipelines, "_run_extracts", lambda *a, **k: None)
    monkeypatch.setattr(pipelines, "_run_transform", lambda *a, **k: source)
    monkeypatch.setattr(pipelines, "_write_canonical_input", lambda *a, **k: source)

    class Rejected:
        def to_directory(self, output):
            raise ProductionReleaseError("objective_error_pct failed")

    monkeypatch.setattr(pipelines, "run_forecast", lambda *a, **k: Rejected())
    output = tmp_path / "pipeline"
    with pytest.raises(ProductionReleaseError):
        pipelines.run_pipeline(config, output)
    receipt = json.loads((output / "pipeline_manifest.json").read_text())
    assert receipt["status"] == "quality_rejected"
    assert receipt["forecast"]["release_status"] == "rejected"
    assert not (output / "forecast" / "release").exists()


def test_policy_output_provenance_and_immutable_full_export(qualified_run, tmp_path):
    from nixtla_scaffold.outputs import build_selected_forecast
    selected = build_selected_forecast(qualified_run)
    assert selected.interval_method.eq("chronological_selector_split_conformal").all()
    assert selected.interval_status.eq("policy_qualified").all()
    out = qualified_run.to_directory(tmp_path / "qualified")
    assert (out / "release" / "manifest.json").exists()
    assert (out / "audit" / "production_backtest.csv").exists()
    with pytest.raises(FileExistsError, match="new or empty"):
        qualified_run.to_directory(out)
    from nixtla_scaffold.ops import build_validation_receipt
    validation = build_validation_receipt(out)
    assert validation.loc[validation.check_id.eq("production_release"), "status"].item() == "pass"
    (out / "release" / "forecast.csv").write_text("damaged delivery")
    validation = build_validation_receipt(out)
    assert validation.loc[validation.check_id.eq("production_release"), "status"].item() == "fail"


def test_frozen_blend_can_be_selected_and_qualified(qualified_run):
    # Opposite errors cancel without fitting weights to the evaluated fold.
    cv, future = qualified_run.backtest_predictions.copy(), qualified_run.all_models.copy()
    for frame in (cv, future):
        frame["HistoricAverage"] = 90.0
        frame["WindowAverage"] = 110.0
        frame["Naive"] = 108.0
        frame["SeasonalNaive"] = 110.0
    spec = replace(qualified_run.spec, production=ProductionPolicy(
        ensemble_members=("HistoricAverage", "WindowAverage"),
    ))
    result = qualify_production_run(replace(qualified_run, spec=spec, backtest_predictions=cv, all_models=future))
    decisions = result.model_policy_resolution["production"]["decisions"]
    assert all("ProductionBlend" in {row["model"] for row in item["scores"]} for item in decisions)
    assert result.forecast.model.eq("ProductionBlend").all()
    assert result.model_policy_resolution["production"]["readiness"]["release_allowed"]


def test_coverage_failure_blocks_otherwise_accurate_points(qualified_run):
    cv = qualified_run.backtest_predictions.copy()
    cutoff = sorted(cv.cutoff.unique())[23]
    cv.loc[cv.cutoff.eq(cutoff), ["Naive", "SeasonalNaive"]] += 1.0
    result = qualify_production_run(replace(qualified_run, backtest_predictions=cv))
    report = result.model_policy_resolution["production"]["readiness"]
    failures = {row["check"] for row in report["failed_checks"]}
    assert "coverage_95" in failures
    assert "objective_error_pct" not in failures
    assert "absolute_bias_pct" not in failures
    assert not report["release_allowed"]


def test_missing_entire_planned_origin_is_rejected(qualified_run):
    cv = qualified_run.backtest_predictions
    missing = sorted(cv.cutoff.unique())[27]
    changed = replace(qualified_run, backtest_predictions=cv.loc[cv.cutoff.ne(missing)].copy())
    with pytest.raises(ValueError, match="frozen origin"):
        qualify_production_run(changed)


def test_pipeline_refuses_overwrite_before_running_extract(monkeypatch, tmp_path):
    from nixtla_scaffold import pipelines

    config = tmp_path / "pipeline.yaml"
    config.write_text("extracts:\n  - name: source\nforecast:\n  preset: production\n")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("previous release")
    monkeypatch.setattr(pipelines, "_run_extracts", lambda *a, **k: pytest.fail("must not execute extracts"))
    with pytest.raises(FileExistsError, match="new or empty"):
        pipelines.run_pipeline(config, output)
    assert sentinel.read_text() == "previous release"


def test_saved_production_refresh_keeps_the_release_contract(qualified_run, tmp_path):
    from nixtla_scaffold.pipelines import _refresh_spec

    (tmp_path / "manifest.json").write_text(json.dumps(qualified_run.manifest(), default=str))
    restored = _refresh_spec(tmp_path)
    assert restored.production == qualified_run.spec.production
    assert restored.evaluation == qualified_run.spec.evaluation
    assert restored.selection_policy == qualified_run.spec.selection_policy


def test_changed_scoring_actuals_cannot_qualify(qualified_run):
    cv = qualified_run.backtest_predictions.copy()
    cv.loc[0, "y"] += 1
    with pytest.raises(ValueError, match="actuals differ"):
        qualify_production_run(replace(qualified_run, backtest_predictions=cv))


def test_one_bad_series_blocks_the_panel(qualified_run):
    from nixtla_scaffold.profile import profile_dataset

    def panel(frame):
        return pd.concat([frame, frame.assign(unique_id="bad")], ignore_index=True)

    cv, future = panel(qualified_run.backtest_predictions), panel(qualified_run.all_models)
    for frame in (cv, future):
        frame.loc[frame.unique_id.eq("bad"), ["Naive", "SeasonalNaive"]] = 200.0
    history = panel(qualified_run.history)
    resolution = {**qualified_run.model_policy_resolution, "evaluation": {
        **qualified_run.model_policy_resolution["evaluation"],
        "origins": [
            *qualified_run.model_policy_resolution["evaluation"]["origins"],
            *[{**item, "unique_id": "bad"} for item in qualified_run.model_policy_resolution["evaluation"]["origins"]],
        ],
    }}
    result = qualify_production_run(replace(
        qualified_run, history=history, observed_history=history,
        profile=profile_dataset(history, qualified_run.spec), backtest_predictions=cv, all_models=future,
        model_selection=panel(qualified_run.model_selection), model_policy_resolution=resolution,
    ))
    report = result.model_policy_resolution["production"]["readiness"]
    assert not report["release_allowed"]
    assert len(report["series"]) == 2
    assert {row["unique_id"] for row in report["failed_checks"]} == {"bad"}


def test_concurrent_production_writers_cannot_share_a_directory(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from nixtla_scaffold.production import reserve_production_output

    output = tmp_path / "shared"

    def reserve():
        try:
            reserve_production_output(output)
            return "reserved"
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert sorted(results) == ["refused", "reserved"]


def test_benchmark_tolerance_does_not_depend_on_target_units():
    from nixtla_scaffold.production import _not_worse

    assert not _not_worse(1e-15, 1e-16)
    assert not _not_worse(1e-15, 0.0)
    assert not _not_worse(float("nan"), 1.0)
    assert _not_worse(0.0, 0.0)
    assert _not_worse(1.0 + 1e-13, 1.0)


def test_planning_gate_cannot_bypass_production_rejection(qualified_run):
    from nixtla_scaffold.accuracy import build_accuracy_gate
    from nixtla_scaffold.outputs import build_trust_summary

    summary = build_trust_summary(qualified_run)
    allowed = build_accuracy_gate(qualified_run, summary)
    assert allowed["series"][0]["gates"]["production_release"]["passed"]
    assert not allowed["planning_ready_claim_allowed"]  # Unknown business context remains unknown.
    production = qualified_run.model_policy_resolution["production"]
    changed = replace(qualified_run, model_policy_resolution={
        **qualified_run.model_policy_resolution, "production": {
            **production, "readiness": {**production["readiness"], "status": "rejected", "release_allowed": False},
        },
    })
    blocked = build_accuracy_gate(changed, summary)
    assert blocked["status"] == "blocked"
    assert "production_release" in blocked["series"][0]["failed_gates"]
    assert not blocked["planning_ready_claim_allowed"]
