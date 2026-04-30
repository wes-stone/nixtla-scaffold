from __future__ import annotations

import base64
import json

import pandas as pd
import pytest

from nixtla_scaffold import DataProfile, ForecastRun, ForecastSpec, TransformSpec, build_executive_headline, run_forecast
from nixtla_scaffold.citations import FPPY_CITATION
from nixtla_scaffold.cli import _coerce_sheet, main
from nixtla_scaffold.forecast import _apply_shrinkage_toward_last_actual
from nixtla_scaffold.models import (
    ModelResult,
    _add_weighted_ensemble_to_cv,
    _interval_windows,
    _metrics_from_cv,
    rebuild_result_metrics_on_output_scale,
)
from nixtla_scaffold.outputs import build_forecast_long, build_interval_diagnostics, build_residual_test_summary, build_selected_forecast, build_trust_summary
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.reports import write_report_artifacts_from_directory
import nixtla_scaffold.models as model_module


def _common_support_cv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": [pd.Timestamp("2025-04-30"), pd.Timestamp("2025-05-31"), pd.Timestamp("2025-07-31")],
            "cutoff": [pd.Timestamp("2025-03-31"), pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")],
            "y": [100.0, 100.0, 100.0],
            "Naive": [100.0, None, 1000.0],
            "AutoARIMA": [None, 0.0, 0.0],
            "LinearRegression": [None, 100.0, 100.0],
        }
    )


def _adjacent_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    present = [col for col in columns if col in frame.columns]
    if not present or present == ["yhat"]:
        return True
    if "yhat" not in present:
        return False
    positions = [frame.columns.get_loc(col) for col in present]
    return positions == list(range(positions[0], positions[0] + len(present)))


def test_baseline_forecast_smoke() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert len(run.forecast) == 3
    assert set(run.forecast.columns) >= {"unique_id", "ds", "yhat", "model"}
    assert run.model_selection["selected_model"].iloc[0] in {
        "HistoricAverage",
        "Naive",
        "RandomWalkWithDrift",
        "SeasonalNaive",
        "WindowAverage",
        "WeightedEnsemble",
    }
    assert run.profile.freq == "ME"
    receipts = {item["id"]: item["status"] for item in run.best_practice_receipts()}
    assert receipts["tidy_data_checked"] == "passed"
    assert receipts["benchmarks_included"] == "passed"
    assert receipts["weighted_ensemble_audited"] == "passed"
    assert all(FPPY_CITATION in item["source"] for item in run.best_practice_receipts())


def test_weighted_ensemble_is_audited_and_can_be_selected() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 18,
            "ds": pd.date_range("2024-01-31", periods=18, freq="ME"),
            "y": [100, 104, 109, 115, 122, 130, 139, 149, 160, 172, 185, 199, 214, 230, 247, 265, 284, 304],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))

    assert "WeightedEnsemble" in run.all_models.columns
    assert "WeightedEnsemble" in set(run.backtest_metrics["model"])
    assert not run.model_weights.empty
    assert set(run.model_weights.columns) == {"unique_id", "model", "family", "weight", "score_metric", "score_value"}
    assert set(run.model_weights["family"]) <= {"baseline", "statsforecast", "mlforecast", "ensemble", "unknown"}
    assert abs(run.model_weights["weight"].sum() - 1.0) < 1e-9


def test_long_statsforecast_ladder_includes_mstl_autoarima() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 36,
            "ds": pd.date_range("2023-01-31", periods=36, freq="ME"),
            "y": [100 + i * 4 + (18 if i % 12 == 11 else 0) - (8 if i % 12 == 6 else 0) for i in range(36)],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=3, freq="ME", model_policy="statsforecast"))

    assert "AutoARIMA" in run.all_models.columns
    assert "MSTL" in run.all_models.columns
    assert "MSTL_AutoARIMA" in run.all_models.columns


def test_statsforecast_cv_intervals_feed_interval_diagnostics() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 48,
            "ds": pd.date_range("2021-01-31", periods=48, freq="ME"),
            "y": [100 + i * 4 + (18 if i % 12 in {10, 11} else 0) for i in range(48)],
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=6,
            freq="ME",
            levels=(80,),
            model_policy="statsforecast",
            strict_cv_horizon=True,
            weighted_ensemble=False,
            verbose=False,
        ),
    )

    cv_interval_cols = [col for col in run.backtest_predictions.columns if "-lo-80" in col or "-hi-80" in col]
    diagnostics = build_interval_diagnostics(run)

    assert cv_interval_cols
    assert not diagnostics.empty
    assert {
        "interval_status",
        "interval_method",
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
    }.issubset(diagnostics.columns)
    assert set(diagnostics["interval_method"].dropna()) <= {"statsforecast_conformal_distribution"}
    assert set(diagnostics["interval_status"].dropna()) <= {
        "calibrated",
        "calibration_warning",
        "calibration_fail",
        "insufficient_observations",
    }
    assert any("StatsForecast CV conformal intervals enabled" in warning for warning in run.warnings)


def test_statsforecast_candidate_failure_does_not_drop_classical_family() -> None:
    values = [
        2061161.212,
        2113074.407,
        2110653.196,
        2228956.645,
        2222252.948,
        2332637.186,
        2357956.182,
        2191597.587,
        2471697.701,
        2452803.834,
        2583915.429,
        2567787.037,
        2711935.885,
        2778480.874,
        2740952.8,
        2872226.756,
        2824880.287,
        2949360.889,
        2967759.764,
        2841749.423,
        3072466.989,
        3002440.367,
        3131950.529,
        3078944.255,
        3224605.23,
        3260981.899,
        3196517.926,
        3340792.028,
        3279552.097,
        3414585.725,
        3425833.202,
        3117974.032,
        3398200.65,
        3307610.203,
        3427734.821,
        3350590.773,
        3488660.963,
        3529163.299,
        3440544.362,
        3566416.153,
        3482107.612,
        3607140.703,
        3604609.401,
        3307199.996,
        3684044.287,
    ]
    df = pd.DataFrame(
        {
            "unique_id": ["Demo Seats"] * len(values),
            "ds": pd.date_range("2022-07-01", periods=len(values), freq="MS"),
            "y": values,
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=16,
            freq="MS",
            levels=(80, 95),
            model_policy="statsforecast",
            strict_cv_horizon=True,
            verbose=False,
        ),
    )

    model_cols = {
        col
        for col in run.all_models.columns
        if col not in {"unique_id", "ds"} and "-lo-" not in col and "-hi-" not in col
    }
    assert run.engine == "statsforecast"
    assert {"AutoETS", "AutoARIMA", "MSTL", "MSTL_AutoARIMA", "MFLES", "AutoMFLES"}.issubset(model_cols)
    assert "WeightedEnsemble" in model_cols
    assert set(run.model_selection["selection_horizon"]) == {16}
    assert not any("StatsForecast failed; used baseline engine" in warning for warning in run.warnings)
    assert any("HoltWinters" in warning and "kept point forecast only" in warning for warning in run.warnings)


def test_mlforecast_ladder_uses_multiple_regressor_families() -> None:
    pytest.importorskip("mlforecast")
    pytest.importorskip("sklearn")
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 42,
            "ds": pd.date_range("2022-01-01", periods=42, freq="MS"),
            "y": [100 + i * 4 + (20 if i % 12 in {10, 11} else 0) for i in range(42)],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=6, freq="MS", model_policy="mlforecast", strict_cv_horizon=True))

    model_cols = {
        col
        for col in run.all_models.columns
        if col not in {"unique_id", "ds"} and "-lo-" not in col and "-hi-" not in col
    }
    assert {"LinearRegression", "Ridge", "BayesianRidge", "RandomForest", "ExtraTrees", "GradientBoosting"}.issubset(model_cols)
    assert len(model_cols) >= 10
    assert any("MLForecast ladder:" in warning for warning in run.warnings)
    assert any("MLForecast conformal intervals enabled" in warning for warning in run.warnings)
    assert {"yhat_lo_80", "yhat_hi_80", "yhat_lo_95", "yhat_hi_95"}.issubset(run.forecast.columns)
    assert run.effective_levels() == [80, 95]
    assert not run.model_explainability.empty


def test_auto_policy_uses_shared_cv_windows_for_statsforecast_and_mlforecast() -> None:
    pytest.importorskip("mlforecast")
    pytest.importorskip("sklearn")
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 42,
            "ds": pd.date_range("2022-01-01", periods=42, freq="MS"),
            "y": [100 + i * 4 + (20 if i % 12 in {10, 11} else 0) for i in range(42)],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=6, freq="MS", model_policy="auto", strict_cv_horizon=True, verbose=False))

    metrics = run.backtest_metrics.set_index("model")
    stats_contract = metrics.loc["AutoARIMA", ["selection_horizon", "cv_windows", "cv_step_size"]].to_dict()
    ml_contract = metrics.loc["LinearRegression", ["selection_horizon", "cv_windows", "cv_step_size"]].to_dict()
    assert ml_contract == stats_contract
    stats_cutoffs = set(pd.to_datetime(run.backtest_predictions.loc[run.backtest_predictions["AutoARIMA"].notna(), "cutoff"]))
    ml_cutoffs = set(pd.to_datetime(run.backtest_predictions.loc[run.backtest_predictions["LinearRegression"].notna(), "cutoff"]))
    assert ml_cutoffs == stats_cutoffs


def test_model_policy_all_discloses_mlforecast_history_gate() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "y": [100 + i * 3 for i in range(12)],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=2, freq="MS", model_policy="all", verbose=False))

    resolution = run.model_policy_resolution
    assert resolution["model_policy"] == "all"
    families = {row["family"]: row for row in resolution["families"]}
    assert families["mlforecast"]["requested"]
    assert not families["mlforecast"]["eligible"]
    assert not families["mlforecast"]["ran"]
    assert "min_history_below_threshold" in families["mlforecast"]["reason_if_not_ran"]
    assert any("MLForecast skipped for model_policy='all'" in warning for warning in run.warnings)


def test_model_policy_all_raises_when_eligible_mlforecast_is_unavailable(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 32,
            "ds": pd.date_range("2023-01-01", periods=32, freq="MS"),
            "y": [100 + i * 2 for i in range(32)],
        }
    )

    def fake_statsforecast(history, profile, spec):
        return model_module.forecast_with_baselines(history, profile, spec)

    def missing_mlforecast(history, profile, spec):
        raise ImportError("mlforecast missing")

    monkeypatch.setattr(model_module, "forecast_with_statsforecast", fake_statsforecast)
    monkeypatch.setattr(model_module, "forecast_with_mlforecast", missing_mlforecast)

    with pytest.raises(ImportError, match="MLForecast requested by model_policy='all'"):
        run_forecast(df, ForecastSpec(horizon=2, freq="MS", model_policy="all", verbose=False))


def test_cli_model_policy_all_mlforecast_failure_writes_diagnostics(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "run"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 32,
            "ds": pd.date_range("2023-01-01", periods=32, freq="MS"),
            "y": [100 + i * 2 for i in range(32)],
        }
    ).to_csv(input_path, index=False)

    def fake_statsforecast(history, profile, spec):
        return model_module.forecast_with_baselines(history, profile, spec)

    def broken_mlforecast(history, profile, spec):
        raise RuntimeError("mlforecast exploded")

    monkeypatch.setattr(model_module, "forecast_with_statsforecast", fake_statsforecast)
    monkeypatch.setattr(model_module, "forecast_with_mlforecast", broken_mlforecast)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "2",
            "--freq",
            "MS",
            "--model-policy",
            "all",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "MLForecast requested by model_policy='all'" in captured.err
    assert "Failure diagnostics written" in captured.err
    assert "Traceback" not in captured.err
    failure_diagnostics = json.loads((output_dir / "failure_diagnostics.json").read_text(encoding="utf-8"))
    assert failure_diagnostics["status"] == "failure"
    assert failure_diagnostics["command"] == "forecast"


def test_weighted_ensemble_backtest_does_not_score_on_weight_training_cutoff() -> None:
    cv = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.date_range("2026-01-31", periods=3, freq="ME"),
            "cutoff": [pd.Timestamp("2025-12-31")] * 3,
            "y": [0.0, 0.0, 1.0],
            "ModelA": [0.0, 0.0, 0.0],
            "ModelB": [0.0, 0.0, 2.0],
        }
    )

    metrics = _metrics_from_cv(cv, include_weighted=True)

    assert "WeightedEnsemble" not in set(metrics["model"])


def test_common_support_weighted_ensemble_cv_excludes_partial_support_models() -> None:
    cv = _common_support_cv_frame()

    out = _add_weighted_ensemble_to_cv(cv, common_support=True)

    current = out[out["cutoff"] == pd.Timestamp("2025-06-30")].iloc[0]
    assert current["WeightedEnsemble"] == 100.0


def test_transformed_combined_rebuild_preserves_common_support_weights() -> None:
    cv = _common_support_cv_frame()
    history = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": pd.date_range("2024-11-30", periods=8, freq="ME"),
            "y": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    spec = ForecastSpec(horizon=1, freq="ME")
    result = ModelResult(
        forecast=pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": [pd.Timestamp("2025-08-31")],
                "Naive": [1000.0],
                "AutoARIMA": [0.0],
                "LinearRegression": [100.0],
            }
        ),
        backtest_metrics=_metrics_from_cv(cv),
        backtest_predictions=cv,
        engine="statsforecast+mlforecast",
        model_weights=pd.DataFrame(),
    )

    rebuilt = rebuild_result_metrics_on_output_scale(result, history, profile_dataset(history, spec), spec)

    assert rebuilt.model_weights["model"].tolist() == ["LinearRegression"]
    assert rebuilt.forecast["WeightedEnsemble"].iloc[0] == 100.0
    current = rebuilt.backtest_predictions[rebuilt.backtest_predictions["cutoff"] == pd.Timestamp("2025-06-30")].iloc[0]
    assert current["WeightedEnsemble"] == 100.0


def test_weighted_ensemble_interval_trust_does_not_inherit_component_status() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 18,
            "ds": pd.date_range("2024-01-31", periods=18, freq="ME"),
            "y": [100 + i * 5 for i in range(18)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline"))
    run.model_selection["selected_model"] = "WeightedEnsemble"
    run.backtest_predictions["Naive-lo-80"] = pd.to_numeric(run.backtest_predictions["y"], errors="coerce") - 1.0
    run.backtest_predictions["Naive-hi-80"] = pd.to_numeric(run.backtest_predictions["y"], errors="coerce") + 1.0

    trust = build_trust_summary(run)

    row = trust[trust["unique_id"] == "Revenue"].iloc[0]
    assert row["interval_status"] == "point_only_ensemble"
    assert "point-only" in row["caveats"]
    headline = build_executive_headline(run)
    assert "WeightedEnsemble is point-only; no calibrated range is shown" in headline.paragraph
    assert "80% interval" not in headline.paragraph


def test_future_only_intervals_are_not_reported_as_calibrated() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 48,
            "ds": pd.date_range("2021-01-31", periods=48, freq="ME"),
            "y": [100 + i * 4 + (18 if i % 12 in {10, 11} else 0) for i in range(48)],
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=6,
            freq="ME",
            levels=(80,),
            model_policy="statsforecast",
            strict_cv_horizon=True,
            weighted_ensemble=False,
            verbose=False,
        ),
    )
    selected_model = run.model_selection["selected_model"].iloc[0]
    for row in run.forecast.to_dict("records"):
        mask = (run.all_models["unique_id"].astype(str) == str(row["unique_id"])) & (
            pd.to_datetime(run.all_models["ds"]) == pd.Timestamp(row["ds"])
        )
        run.all_models.loc[mask, selected_model] = row["yhat"]
        run.all_models.loc[mask, f"{selected_model}-lo-80"] = row["yhat_lo_80"]
        run.all_models.loc[mask, f"{selected_model}-hi-80"] = row["yhat_hi_80"]
    interval_cols = [col for col in run.backtest_predictions.columns if "-lo-" in col or "-hi-" in col]
    run.backtest_predictions = run.backtest_predictions.drop(columns=interval_cols)

    trust = build_trust_summary(run)
    forecast_long = build_forecast_long(run)
    selected_forecast = build_selected_forecast(run, forecast_long)

    assert trust.iloc[0]["interval_status"] == "future_only"
    assert "not empirically calibrated" in trust.iloc[0]["caveats"]
    assert selected_forecast.iloc[0]["interval_status"] == "future_only"
    assert selected_forecast.iloc[0]["interval_method"] == "statsforecast_conformal_distribution"
    assert "future-only, not CV-calibrated" in run.explanation()
    headline = build_executive_headline(run)
    assert "future-only and not CV-calibrated" in headline.paragraph
    assert "80% interval is calibrated in rolling-origin CV" not in headline.paragraph
    receipts = {item["id"]: item["status"] for item in run.best_practice_receipts()}
    assert receipts["intervals_supported_by_history"] == "warning"
    selected_rows = forecast_long[forecast_long["is_selected_model"]]
    assert "future_only" in set(selected_rows["interval_status"])


def test_trust_summary_does_not_borrow_non_selected_interval_evidence() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 18,
            "ds": pd.date_range("2024-01-31", periods=18, freq="ME"),
            "y": [100 + i * 5 for i in range(18)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline", weighted_ensemble=False))
    run.model_selection["selected_model"] = "Naive"
    run.forecast["model"] = "Naive"
    run.forecast["yhat_lo_80"] = pd.to_numeric(run.forecast["yhat"], errors="coerce") - 1.0
    run.forecast["yhat_hi_80"] = pd.to_numeric(run.forecast["yhat"], errors="coerce") + 1.0
    for row in run.forecast.to_dict("records"):
        mask = (run.all_models["unique_id"].astype(str) == str(row["unique_id"])) & (
            pd.to_datetime(run.all_models["ds"]) == pd.Timestamp(row["ds"])
        )
        run.all_models.loc[mask, "Naive"] = row["yhat"]
        run.all_models.loc[mask, "Naive-lo-80"] = row["yhat_lo_80"]
        run.all_models.loc[mask, "Naive-hi-80"] = row["yhat_hi_80"]
    run.backtest_predictions["HistoricAverage-lo-80"] = pd.to_numeric(run.backtest_predictions["y"], errors="coerce") - 1.0
    run.backtest_predictions["HistoricAverage-hi-80"] = pd.to_numeric(run.backtest_predictions["y"], errors="coerce") + 1.0

    trust = build_trust_summary(run)
    selected_forecast = build_selected_forecast(run)

    assert trust.iloc[0]["interval_status"] == "future_only"
    assert selected_forecast.iloc[0]["interval_status"] == "future_only"


def test_forecast_long_selected_rows_use_final_adjusted_forecast() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 48,
            "ds": pd.date_range("2021-01-31", periods=48, freq="ME"),
            "y": [100 + i * 4 + (18 if i % 12 in {10, 11} else 0) for i in range(48)],
        }
    )

    run = run_forecast(
        df,
        ForecastSpec(
            horizon=3,
            freq="ME",
            levels=(80,),
            model_policy="statsforecast",
            strict_cv_horizon=True,
            weighted_ensemble=False,
            verbose=False,
        ),
    )
    run.forecast["yhat"] = run.forecast["yhat"] + 10.0
    run.forecast["yhat_lo_80"] = run.forecast["yhat_lo_80"] + 10.0
    run.forecast["yhat_hi_80"] = run.forecast["yhat_hi_80"] + 10.0

    forecast_long = build_forecast_long(run)
    selected_rows = forecast_long[forecast_long["is_selected_model"]].sort_values("ds").reset_index(drop=True)

    assert selected_rows["yhat"].tolist() == run.forecast.sort_values("ds")["yhat"].tolist()
    assert set(selected_rows["interval_status"]) == {"adjusted_not_recalibrated"}


def test_weighted_ensemble_can_be_disabled() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline", weighted_ensemble=False))

    assert "WeightedEnsemble" not in run.all_models.columns
    assert run.model_weights.empty
    assert any("weighted ensemble disabled" in warning for warning in run.warnings)


def test_shrinkage_shifts_intervals_with_selected_yhat() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2026-01-31", periods=2, freq="ME"),
            "yhat": [200.0, 210.0],
            "model": ["ModelA", "ModelA"],
            "yhat_lo_80": [180.0, 190.0],
            "yhat_hi_80": [220.0, 230.0],
        }
    )
    history = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.date_range("2025-10-31", periods=3, freq="ME"),
            "y": [100.0, 101.0, 80.0],
        }
    )
    backtest = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.date_range("2025-10-31", periods=3, freq="ME"),
            "cutoff": [pd.Timestamp("2025-09-30")] * 3,
            "y": [100.0, 101.0, 102.0],
            "ModelA": [200.0, 50.0, 250.0],
        }
    )

    out, warnings = _apply_shrinkage_toward_last_actual(forecast, history, backtest)

    assert warnings
    assert out["yhat"].iloc[0] < forecast["yhat"].iloc[0]
    assert out["yhat_hi_80"].iloc[0] - out["yhat_lo_80"].iloc[0] == forecast["yhat_hi_80"].iloc[0] - forecast["yhat_lo_80"].iloc[0]
    assert (out["yhat_lo_80"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["yhat_hi_80"]).all()


def test_model_selection_records_cv_horizon_contract() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=6, model_policy="baseline"))

    assert {"requested_horizon", "selection_horizon", "cv_windows", "cv_step_size", "cv_horizon_matches_requested"}.issubset(
        run.model_selection.columns
    )
    row = run.model_selection.iloc[0]
    assert row["requested_horizon"] == 6
    assert row["selection_horizon"] < row["requested_horizon"]
    assert not bool(row["cv_horizon_matches_requested"])
    assert "validated through CV horizon" in row["selection_reason"]
    assert any("shorter than requested forecast horizon" in warning for warning in run.warnings)
    trust = build_trust_summary(run).iloc[0]
    assert trust["horizon_trust_state"] == "partial_horizon_validated"
    assert not bool(trust["full_horizon_claim_allowed"])
    assert trust["trust_level"] != "High"
    assert trust["trust_score_0_100"] <= 60
    headline = build_executive_headline(run)
    assert headline.scope == "single"
    assert "validated through step" in headline.paragraph
    assert "later steps are directional" in headline.paragraph
    forecast_long = build_forecast_long(run)
    selected_rows = forecast_long[forecast_long["is_selected_model"]]
    assert "beyond_validated_horizon" in set(selected_rows["horizon_trust_state"])
    assert "beyond_validated_horizon" in set(selected_rows["row_horizon_status"])
    assert selected_rows.loc[selected_rows["horizon_step"] > row["selection_horizon"], "planning_eligible"].eq(False).all()
    assert set(selected_rows["planning_eligibility_scope"]) == {"horizon_validation_only"}
    selected_forecast = build_selected_forecast(run, forecast_long)
    assert {
        "row_horizon_status",
        "horizon_trust_state",
        "validated_through_horizon",
        "planning_eligible",
        "planning_eligibility_scope",
        "planning_eligibility_reason",
        "horizon_warning",
    }.issubset(selected_forecast.columns)


def test_strict_cv_horizon_uses_requested_horizon_when_available() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 36,
            "ds": pd.date_range("2023-01-31", periods=36, freq="ME"),
            "y": [100 + i * 3 for i in range(36)],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=6, model_policy="baseline", strict_cv_horizon=True))

    assert set(run.model_selection["selection_horizon"]) == {6}
    assert set(run.model_selection["cv_horizon_matches_requested"]) == {True}
    trust = build_trust_summary(run)
    assert set(trust["horizon_trust_state"]) == {"full_horizon_validated"}
    forecast_long = build_forecast_long(run)
    assert set(forecast_long["horizon_trust_state"]) == {"full_horizon_validated"}


def test_horizon_planning_rows_expose_success_reason_and_review_context() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 36,
            "ds": pd.date_range("2023-01-31", periods=36, freq="ME"),
            "y": [100 + i * 3 for i in range(36)],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=6, model_policy="baseline", strict_cv_horizon=True))

    forecast_long = build_forecast_long(run)
    selected_rows = forecast_long[forecast_long["is_selected_model"]]
    assert selected_rows["planning_eligible"].eq(True).all()
    assert selected_rows["row_horizon_status"].equals(selected_rows["horizon_trust_state"])
    assert selected_rows["row_horizon_status"].equals(selected_rows["forecast_horizon_status"])
    assert selected_rows["validation_evidence"].equals(selected_rows["horizon_warning"])
    assert selected_rows["planning_eligibility_reason"].str.contains("Passes horizon-validation gate").all()

    selected_forecast = build_selected_forecast(run, forecast_long)
    assert selected_forecast["planning_eligible"].eq(True).all()
    assert selected_forecast["planning_eligibility_reason"].str.contains("Passes horizon-validation gate").all()

    trust_context = build_trust_summary(run)[
        ["unique_id", "trust_level", "interval_status", "caveats", "next_actions", "full_horizon_claim_allowed"]
    ].rename(columns={"interval_status": "series_interval_status"})
    stakeholder_review = selected_forecast[selected_forecast["planning_eligible"]].merge(trust_context, on="unique_id", how="left")
    assert not stakeholder_review.empty
    assert stakeholder_review["trust_level"].notna().all()
    assert stakeholder_review["series_interval_status"].notna().all()
    assert stakeholder_review["caveats"].notna().all()
    assert stakeholder_review["next_actions"].notna().all()


def test_single_window_full_horizon_discloses_limited_claim(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline", strict_cv_horizon=True))

    selected = run.model_selection.iloc[0]
    assert selected["selection_horizon"] == selected["requested_horizon"] == 3
    assert selected["cv_windows"] == 1
    trust = build_trust_summary(run).iloc[0]
    assert trust["horizon_trust_state"] == "full_horizon_validated"
    assert trust["horizon_gate_result"] == "warning_single_cv_window"
    assert not bool(trust["full_horizon_claim_allowed"])
    assert trust["trust_score_0_100"] <= 65
    forecast_long = build_forecast_long(run)
    selected_rows = forecast_long[forecast_long["is_selected_model"]]
    assert set(selected_rows["horizon_trust_state"]) == {"full_horizon_validated"}
    assert set(selected_rows["row_horizon_status"]) == {"full_horizon_validated"}
    assert selected_rows["planning_eligible"].eq(False).all()
    assert set(selected_rows["planning_eligibility_scope"]) == {"horizon_validation_only"}
    assert selected_rows["horizon_warning"].str.contains("only 1 rolling-origin window").all()
    selected_forecast = build_selected_forecast(run, forecast_long)
    assert selected_forecast["planning_eligible"].eq(False).all()
    assert set(selected_forecast["planning_eligibility_scope"]) == {"horizon_validation_only"}
    assert "only 1 rolling-origin window" in run.explanation()
    headline = build_executive_headline(run)
    assert "evaluated on only 1 CV window(s)" in headline.paragraph
    assert "planning claim limited" in headline.paragraph
    receipts = {item["id"]: item["status"] for item in run.best_practice_receipts()}
    assert receipts["horizon_claim_audited"] == "warning"

    output_dir = run.to_directory(tmp_path / "single_window")
    decoded_report = base64.b64decode((output_dir / "report_base64.txt").read_text(encoding="utf-8")).decode("utf-8")
    assert "only 1 CV window(s)" in decoded_report
    assert "planning claim limited" in decoded_report
    streamlit_app = (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert "planning-ready champion claim is limited" in streamlit_app


def test_no_backtest_fallback_is_low_trust_no_validated_champion() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 2,
            "ds": pd.date_range("2025-01-31", periods=2, freq="ME"),
            "y": [100, 105],
        }
    )

    run = run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline", strict_cv_horizon=True))

    trust = build_trust_summary(run).iloc[0]
    assert trust["horizon_trust_state"] == "no_rolling_origin_evidence"
    assert trust["trust_level"] == "Low"
    assert trust["trust_score_0_100"] <= 35
    assert not bool(trust["full_horizon_claim_allowed"])
    assert "no rolling-origin validation evidence" in trust["caveats"]
    forecast_long = build_forecast_long(run)
    assert set(forecast_long["horizon_trust_state"]) == {"no_rolling_origin_evidence"}
    assert set(forecast_long["row_horizon_status"]) == {"no_rolling_origin_evidence"}
    assert forecast_long["planning_eligible"].eq(False).all()
    assert set(forecast_long["planning_eligibility_scope"]) == {"horizon_validation_only"}
    selected_forecast = build_selected_forecast(run, forecast_long)
    assert selected_forecast["planning_eligible"].eq(False).all()
    headline = build_executive_headline(run)
    assert "exploratory statistical baseline only" in headline.paragraph
    assert "no rolling-origin validation" in headline.paragraph
    assert "planning-ready" not in headline.paragraph


def test_require_backtest_fails_when_unavailable() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 2,
            "ds": pd.date_range("2025-01-31", periods=2, freq="ME"),
            "y": [100, 105],
        }
    )

    try:
        run_forecast(df, ForecastSpec(horizon=3, model_policy="baseline", require_backtest=True))
    except ValueError as exc:
        assert "require_backtest=True" in str(exc)
    else:
        raise AssertionError("expected require_backtest to fail when metrics unavailable")


def test_mixed_history_series_forecasts_short_series_without_backtest_metrics() -> None:
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": ["Long"] * 8,
                    "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
                    "y": [100, 104, 106, 111, 116, 120, 123, 129],
                }
            ),
            pd.DataFrame(
                {
                    "unique_id": ["Short"] * 2,
                    "ds": pd.date_range("2025-07-31", periods=2, freq="ME"),
                    "y": [20, 22],
                }
            ),
        ],
        ignore_index=True,
    )

    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    assert set(run.model_selection["unique_id"]) == {"Long", "Short"}
    assert set(run.forecast["unique_id"]) == {"Long", "Short"}
    assert len(run.forecast) == 4
    short_selection = run.model_selection.set_index("unique_id").loc["Short"]
    assert "backtest not available" in short_selection["selection_reason"]
    trust = build_trust_summary(run).set_index("unique_id")
    assert trust.loc["Short", "horizon_trust_state"] == "no_rolling_origin_evidence"
    assert trust.loc["Short", "unvalidated_steps"] == 2
    assert trust.loc["Short", "horizon_trust_score_cap"] <= 35
    forecast_long = build_forecast_long(run)
    row_statuses = forecast_long[forecast_long["is_selected_model"]].groupby("unique_id")["row_horizon_status"].unique()
    assert "no_rolling_origin_evidence" in set(row_statuses.loc["Short"])
    assert set(forecast_long["planning_eligibility_scope"]) == {"horizon_validation_only"}


def test_auto_policy_backtests_eligible_series_in_mixed_history_panel() -> None:
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": ["Long"] * 8,
                    "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
                    "y": [100, 104, 106, 111, 116, 120, 123, 129],
                }
            ),
            pd.DataFrame(
                {
                    "unique_id": ["Short"] * 2,
                    "ds": pd.date_range("2025-07-31", periods=2, freq="ME"),
                    "y": [20, 22],
                }
            ),
        ],
        ignore_index=True,
    )

    run = run_forecast(df, ForecastSpec(horizon=2))

    assert "Long" in set(run.backtest_metrics["unique_id"])
    assert set(run.model_selection["unique_id"]) == {"Long", "Short"}
    assert "backtest not available" not in run.model_selection.set_index("unique_id").loc["Long", "selection_reason"]


def test_require_backtest_fails_when_any_series_lacks_metrics() -> None:
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": ["Long"] * 8,
                    "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
                    "y": [100, 104, 106, 111, 116, 120, 123, 129],
                }
            ),
            pd.DataFrame(
                {
                    "unique_id": ["Short"] * 2,
                    "ds": pd.date_range("2025-07-31", periods=2, freq="ME"),
                    "y": [20, 22],
                }
            ),
        ],
        ignore_index=True,
    )

    try:
        run_forecast(df, ForecastSpec(horizon=2, require_backtest=True))
    except ValueError as exc:
        assert "Short" in str(exc)
    else:
        raise AssertionError("expected strict backtest mode to fail for sparse series")


def test_demo_history_supports_conformal_interval_windows() -> None:
    assert _interval_windows(min_obs=24, horizon=6, season_length=12) == 2


def test_cli_require_backtest_flag_fails_without_traceback(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 2,
            "ds": pd.date_range("2025-01-31", periods=2, freq="ME"),
            "y": [100, 105],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "3",
            "--model-policy",
            "baseline",
            "--require-backtest",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "require_backtest=True" in captured.err
    assert "Failure diagnostics written" in captured.err
    assert "Traceback" not in captured.err
    failure_diagnostics = json.loads((tmp_path / "run" / "failure_diagnostics.json").read_text(encoding="utf-8"))
    assert failure_diagnostics["status"] == "failure"
    assert failure_diagnostics["command"] == "forecast"
    assert failure_diagnostics["likely_causes"]


def test_cli_forecast_accepts_explicit_interval_levels(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "run"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 24,
            "ds": pd.date_range("2024-01-31", periods=24, freq="ME"),
            "y": [100 + i * 4 for i in range(24)],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "3",
            "--model-policy",
            "baseline",
            "--levels",
            "70",
            "90",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec"]["levels"] == [70, 90]


def test_cli_operational_output_error_writes_failure_diagnostics(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.csv"
    blocked_output = tmp_path / "run"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130],
        }
    ).to_csv(input_path, index=False)
    blocked_output.write_text("not a directory", encoding="utf-8")

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "2",
            "--model-policy",
            "baseline",
            "--output",
            str(blocked_output),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Failure diagnostics written" in captured.err
    assert "Traceback" not in captured.err
    assert (tmp_path / "failure_diagnostics.json").exists()


def test_forecast_outputs_include_llm_diagnostics_and_model_weights(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130, 136, 141, 148, 155],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))

    output_dir = run.to_directory(tmp_path / "run")

    assert (output_dir / "diagnostics.json").exists()
    assert (output_dir / "diagnostics.md").exists()
    assert (output_dir / "llm_context.json").exists()
    assert (output_dir / "audit" / "model_weights.csv").exists()
    assert (output_dir / "audit" / "backtest_predictions.csv").exists()
    assert (output_dir / "audit" / "backtest_windows.csv").exists()
    assert (output_dir / "forecast_long.csv").exists()
    assert (output_dir / "backtest_long.csv").exists()
    assert (output_dir / "series_summary.csv").exists()
    assert (output_dir / "model_audit.csv").exists()
    assert (output_dir / "model_win_rates.csv").exists()
    assert (output_dir / "model_window_metrics.csv").exists()
    assert (output_dir / "residual_diagnostics.csv").exists()
    assert (output_dir / "interval_diagnostics.csv").exists()
    assert (output_dir / "trust_summary.csv").exists()
    assert (output_dir / "model_explainability.csv").exists()
    assert (output_dir / "audit" / "seasonality_profile.csv").exists()
    assert (output_dir / "audit" / "seasonality_summary.csv").exists()
    assert (output_dir / "audit" / "seasonality_diagnostics.csv").exists()
    assert (output_dir / "audit" / "seasonality_decomposition.csv").exists()
    assert (output_dir / "audit" / "interpretation.json").exists()
    assert (output_dir / "interpretation.md").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "report_base64.txt").exists()
    assert (output_dir / "streamlit_app.py").exists()
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["status"] == "success"
    executive = diagnostics["executive_headline"]
    executive_paragraph = executive["paragraph"]
    assert executive["scope"] == "single"
    assert "Revenue:" in executive_paragraph
    assert executive["series"][0]["paragraph"] == executive_paragraph
    assert "full_horizon_claim_allowed_count" in executive
    assert "full_horizon_validated_count" not in executive
    assert diagnostics["llm_triage_summary"]["weighted_ensemble_enabled"]
    assert "model_weights" in diagnostics
    assert "trust_summary" in diagnostics
    assert "reproducibility" in diagnostics
    assert "llm_context.json" in "\n".join(diagnostics["next_diagnostic_steps"])
    llm_context = json.loads((output_dir / "llm_context.json").read_text(encoding="utf-8"))
    assert llm_context["schema_version"] == "nixtla_scaffold.llm_context.v1"
    assert "prompt_starter" in llm_context
    assert "planning_eligible=True means horizon-validation only" in "\n".join(llm_context["guardrails"])
    assert llm_context["executive_headline"]["paragraph"] == executive_paragraph
    assert llm_context["series_reviews"][0]["forecast_rows"]
    assert "trust_summary" in llm_context["portfolio_tables"]
    assert "recommended_questions" in llm_context
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["reproducibility"]["data_hash_sha256"]) == 64
    assert manifest["reproducibility"]["forecast_origin"] == "2025-12-31"
    assert manifest["reproducibility"]["package_versions"]["pandas"] is not None
    assert manifest["outputs"]["model_win_rates"] == "model_win_rates.csv"
    assert manifest["outputs"]["trust_summary"] == "trust_summary.csv"
    assert manifest["outputs"]["llm_context"] == "llm_context.json"
    assert manifest["outputs"]["seasonality_diagnostics"] == "audit/seasonality_diagnostics.csv"
    assert manifest["outputs"]["seasonality_decomposition"] == "audit/seasonality_decomposition.csv"
    assert manifest["model_policy_resolution"]["model_policy"] == "baseline"
    policy_families = {row["family"]: row for row in manifest["model_policy_resolution"]["families"]}
    assert policy_families["baseline"]["requested"]
    assert policy_families["baseline"]["ran"]
    assert policy_families["statsforecast"]["reason_if_not_ran"] == "not_requested"
    assert "audit/backtest_windows.csv" in "\n".join(diagnostics["next_diagnostic_steps"])
    assert "model_win_rates.csv" in "\n".join(diagnostics["next_diagnostic_steps"])
    assert "trust_summary.csv" in "\n".join(diagnostics["next_diagnostic_steps"])
    assert "audit/seasonality_diagnostics.csv" in "\n".join(diagnostics["next_diagnostic_steps"])
    assert "executive_headline.paragraph" in "\n".join(diagnostics["next_diagnostic_steps"])
    diagnostics_md = (output_dir / "diagnostics.md").read_text(encoding="utf-8")
    assert "## Executive headline" in diagnostics_md
    assert executive_paragraph in diagnostics_md
    model_card = (output_dir / "model_card.md").read_text(encoding="utf-8")
    model_card_blocks = [block.strip() for block in model_card.split("\n\n") if block.strip()]
    assert model_card_blocks[1] == executive_paragraph
    interpretation = json.loads((output_dir / "audit" / "interpretation.json").read_text(encoding="utf-8"))
    assert interpretation["backtesting"]["window_count"] >= 1
    assert interpretation["seasonality"]["summary"]
    assert interpretation["seasonality"]["diagnostics"]
    decoded_report = base64.b64decode((output_dir / "report_base64.txt").read_text(encoding="utf-8")).decode("utf-8")
    assert "<!doctype html>" in decoded_report
    assert "Backtest" in decoded_report
    assert "Seasonality" in decoded_report
    assert "Seasonality credibility" in decoded_report
    assert "Forecast model review" in decoded_report
    assert "Executive forecast headline" in decoded_report
    assert executive_paragraph in decoded_report
    assert "Rolling-origin fixed-axis filmstrip" in decoded_report
    assert "Decision summary" in decoded_report
    assert "series-decision-card" in decoded_report
    assert "decision-detail-grid" in decoded_report
    assert "Use <code>trust_summary.csv</code> for the raw wide table" in decoded_report
    assert "Trust rubric" in decoded_report
    assert "High &gt;=75" in decoded_report
    assert "Medium 40-74" in decoded_report
    assert "Low &lt;40" in decoded_report
    assert "Champion selection uses backtested RMSE" in decoded_report
    assert "trust and benchmark review also show MAE, MASE, RMSSE, WAPE, and bias" in decoded_report
    assert "Prediction intervals are empirically calibrated uncertainty bands only when interval_status is calibrated" in decoded_report
    assert "shaded bands = future model bands; check interval_status before planning use" in decoded_report
    assert "validated through" in decoded_report
    assert "Model policy resolution" in decoded_report
    assert "Interval glossary" in decoded_report
    assert "llm_context.json" in decoded_report
    assert "component-model intervals" in decoded_report
    assert "future-only bands" in decoded_report
    assert "adjusted-not-recalibrated bands" in decoded_report
    assert "trust_summary.csv" in decoded_report
    assert "other candidates shown faint/unlabeled" in decoded_report
    assert "selected intervals" in decoded_report
    assert "(selected)" in decoded_report
    forecast = pd.read_csv(output_dir / "forecast.csv")
    assert _adjacent_columns(forecast, ["yhat", "yhat_lo_80", "yhat_hi_80", "yhat_lo_95", "yhat_hi_95"])
    assert {
        "interval_status",
        "interval_method",
        "interval_evidence",
        "horizon_step",
        "row_horizon_status",
        "horizon_trust_state",
        "validated_through_horizon",
        "planning_eligible",
        "planning_eligibility_scope",
        "planning_eligibility_reason",
        "horizon_warning",
    }.issubset(forecast.columns)
    forecast_long = pd.read_csv(output_dir / "forecast_long.csv")
    assert _adjacent_columns(forecast_long, ["yhat", "yhat_lo_80", "yhat_hi_80", "yhat_lo_95", "yhat_hi_95"])
    assert {"record_type", "unique_id", "ds", "model", "family", "yhat", "interval_status", "is_selected_model"}.issubset(
        forecast_long.columns
    )
    assert {"horizon_step", "interval_method", "interval_evidence", "requested_horizon", "selection_horizon"}.issubset(
        forecast_long.columns
    )
    assert {
        "row_horizon_status",
        "horizon_trust_state",
        "is_beyond_validated_horizon",
        "planning_eligible",
        "planning_eligibility_scope",
        "planning_eligibility_reason",
        "validation_evidence",
    }.issubset(forecast_long.columns)
    assert set(forecast_long["family"]) <= {"baseline", "statsforecast", "mlforecast", "ensemble", "unknown"}
    ensemble_status = set(forecast_long.loc[forecast_long["model"] == "WeightedEnsemble", "interval_status"])
    assert ensemble_status <= {"point_only_ensemble"}
    backtest_long = pd.read_csv(output_dir / "backtest_long.csv")
    assert _adjacent_columns(backtest_long, ["yhat", "yhat_lo_80", "yhat_hi_80", "yhat_lo_95", "yhat_hi_95"])
    assert {"h", "error", "abs_error", "squared_error", "pct_error"}.issubset(backtest_long.columns)
    model_audit = pd.read_csv(output_dir / "model_audit.csv")
    assert {"model", "family", "rmse", "mase", "rmsse", "is_selected_model"}.issubset(model_audit.columns)
    model_weights = pd.read_csv(output_dir / "audit" / "model_weights.csv")
    assert {"model", "family", "weight", "score_metric", "score_value"}.issubset(model_weights.columns)
    model_win_rates = pd.read_csv(output_dir / "model_win_rates.csv")
    assert {"benchmark_model", "metric", "model", "win_rate_vs_benchmark"}.issubset(model_win_rates.columns)
    window_metrics = pd.read_csv(output_dir / "model_window_metrics.csv")
    assert {"unique_id", "cutoff", "model", "rmse", "mase", "rmsse"}.issubset(window_metrics.columns)
    residual_diagnostics = pd.read_csv(output_dir / "residual_diagnostics.csv")
    assert {"unique_id", "model", "horizon_step", "rmse", "mase", "rmsse"}.issubset(residual_diagnostics.columns)
    residual_tests = pd.read_csv(output_dir / "residual_tests.csv")
    assert {
        "unique_id",
        "model",
        "residual_scope",
        "overall_status",
        "bias_status",
        "white_noise_residual_scope",
        "white_noise_status",
        "outlier_status",
        "structural_break_status",
        "interpretation",
    }.issubset(residual_tests.columns)
    interval_diagnostics = pd.read_csv(output_dir / "interval_diagnostics.csv")
    assert {
        "coverage_status",
        "interval_status",
        "interval_method",
        "requested_horizon",
        "selection_horizon",
        "cv_windows",
        "cv_step_size",
        "cv_horizon_matches_requested",
    }.issubset(interval_diagnostics.columns)
    trust_summary = pd.read_csv(output_dir / "trust_summary.csv")
    assert {
        "trust_level",
        "trust_score_0_100",
        "seasonality_status",
        "interval_method",
        "horizon_trust_state",
        "full_horizon_claim_allowed",
        "planning_ready_horizon",
        "unvalidated_steps",
        "horizon_trust_score_cap",
        "caveats",
        "next_actions",
    }.issubset(trust_summary.columns)
    seasonality_diagnostics = pd.read_csv(output_dir / "audit" / "seasonality_diagnostics.csv")
    assert {"cycle_count", "complete_cycles", "credibility_label", "warning", "interpretation"}.issubset(
        seasonality_diagnostics.columns
    )
    seasonality_decomposition = pd.read_csv(output_dir / "audit" / "seasonality_decomposition.csv")
    assert {"unique_id", "ds", "observed", "trend", "seasonal", "remainder"}.issubset(seasonality_decomposition.columns)
    workbook = pd.ExcelFile(output_dir / "forecast.xlsx")
    assert "Trust Summary" in workbook.sheet_names
    assert "Residual Tests" in workbook.sheet_names
    assert "Seasonality Diagnostics" in workbook.sheet_names
    assert "Seasonal Decomposition" in workbook.sheet_names
    streamlit_app = (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert "streamlit" in streamlit_app
    assert 'read_json("diagnostics.json")' in streamlit_app
    assert "executive_headline" in streamlit_app
    assert "Forecast headline" in streamlit_app
    assert "Quote diagnostics.json executive_headline.paragraph verbatim" in streamlit_app
    assert "Copy headline" in streamlit_app
    assert "copy_headline_" in streamlit_app
    assert "Decision summary" in streamlit_app
    assert "Trust level" in streamlit_app
    assert "Unvalidated steps" in streamlit_app
    assert "Horizon score cap" in streamlit_app
    assert 'data-baseweb="tab"' in streamlit_app
    assert "font-size: 1.04rem" in streamlit_app
    assert "llm_context.json" in streamlit_app
    assert "planning_eligible is a horizon-validation flag only" in streamlit_app
    assert "horizon_message_severity" in streamlit_app
    assert "st.error(horizon_message)" in streamlit_app
    assert "Trust rubric" in streamlit_app
    assert "High >=75, Medium 40-74, Low <40" in streamlit_app
    assert "All series trust summary" in streamlit_app
    assert 'read_json("manifest.json")' in streamlit_app
    assert "model_policy_resolution" in streamlit_app
    assert "Model policy resolution" in streamlit_app
    assert "did not run" in streamlit_app
    assert "No model was available for" in streamlit_app
    assert "Open the Model policy resolution expander above" in streamlit_app
    assert "WeightedEnsemble is point-only" in streamlit_app
    assert "Interval glossary" in streamlit_app
    assert "display_trust_summary" in streamlit_app
    assert "component-model intervals" in streamlit_app
    assert 'read_csv("trust_summary.csv")' in streamlit_app
    assert 'read_csv("residual_tests.csv")' in streamlit_app
    assert "Heuristic residual checks" in streamlit_app
    assert "structural-break checks" in streamlit_app
    assert "First-glance chart includes interval bands" in streamlit_app
    assert "Focused future forecast interval ownership" in streamlit_app
    assert "the legend names each interval band" in streamlit_app
    assert "Point forecasts and bands come from the same `forecast_long.csv` model feed" in streamlit_app
    assert "future_model_frame" in streamlit_app
    assert "ordered_model_feed_columns" in streamlit_app
    assert "Model feed columns keep `yhat`, `yhat_lo_80`, `yhat_hi_80`, `yhat_lo_95`, and `yhat_hi_95` adjacent" in streamlit_app
    assert '"Prediction intervals"' in streamlit_app
    assert "Prediction interval focus" in streamlit_app
    assert "All interval-bearing candidate models are selected by default" in streamlit_app
    assert "Models with interval bands" in streamlit_app
    assert "interval_bearing_models" in streamlit_app
    assert "interval_focus_models_frame" in streamlit_app
    assert "prediction_interval_focus_chart" in streamlit_app
    assert "prediction_interval_width_chart" in streamlit_app
    assert "Future interval width summary" in streamlit_app
    assert '"Seasonality"' in streamlit_app
    assert "Seasonality credibility" in streamlit_app
    assert "Seasonal year overlay" in streamlit_app
    assert "Beginning of year month" in streamlit_app
    assert "seasonality_year_profile_chart" in streamlit_app
    assert "seasonality_decomposition_chart" in streamlit_app
    assert 'read_csv("seasonality_diagnostics.csv")' in streamlit_app
    assert 'read_csv("seasonality_decomposition.csv")' in streamlit_app
    assert "CV window player" in streamlit_app
    assert "st.slider" in streamlit_app
    assert "st.session_state" in streamlit_app
    assert "slider_key = f\"{state_key}_slider\"" in streamlit_app
    assert "key=slider_key" in streamlit_app
    assert "on_change=sync_cutoff_slider" in streamlit_app
    assert "Auto-advance is active" in streamlit_app
    assert "Auto-advance loops until you switch it off" in streamlit_app
    assert "other candidates are shown as faint unlabeled context lines" in streamlit_app
    assert "No prediction interval bands were written for the models currently shown" in streamlit_app
    assert "The faint gray spread is model disagreement, not calibrated uncertainty" in streamlit_app
    assert "Check the Prediction intervals tab for calibration and row-level review" in streamlit_app
    assert "interval_method_label" in streamlit_app
    assert "active_model_horizon_summary" in streamlit_app
    assert "Validated through horizon" in streamlit_app
    assert "No empirical interval diagnostics available" in streamlit_app
    assert "Champion lens" in streamlit_app
    assert "Best StatsForecast/classical" in streamlit_app
    assert "Best MLForecast" in streamlit_app
    assert "Winner metric" in streamlit_app
    assert "When should I use each metric?" in streamlit_app
    assert "RMSE - penalize large misses" in streamlit_app
    assert "MAE - typical absolute miss" in streamlit_app
    assert "WAPE - business percentage error" in streamlit_app
    assert "Absolute bias - avoid systematic over/under" in streamlit_app
    assert '"Model investigation"' in streamlit_app
    assert "Models to investigate" in streamlit_app
    assert "Menu labels use `#rank | model | engine`" in streamlit_app
    assert "Model picker guide: rank, engine, and role" in streamlit_app
    assert "Rank comes from the current winner metric" in streamlit_app
    assert "native guide table italicizes the Engine column" in streamlit_app
    assert "model_menu_label" in streamlit_app
    assert "model_menu_table" in streamlit_app
    assert "model_picker_guide_style" in streamlit_app
    assert "model_picker_guide_html" not in streamlit_app
    assert "Interval model picker guide: rank and engine" in streamlit_app
    assert "Focused future forecast" in streamlit_app
    assert "Focused rolling-origin window" in streamlit_app
    assert "best_model_for_scope" in streamlit_app
    assert "model_family" in streamlit_app
    assert 'key=f"models_to_investigate_{uid}"' in streamlit_app
    assert 'key="forecast_context_chart"' in streamlit_app
    assert 'key="investigation_forecast_chart"' in streamlit_app
    assert 'key="investigation_backtest_chart"' in streamlit_app
    assert 'key="backtest_context_chart"' in streamlit_app
    assert 'key="residual_horizon_plot"' in streamlit_app
    assert 'key="interval_calibration_plot"' in streamlit_app
    focused_chart_idx = streamlit_app.rindex("show_all_models=False,")
    autoplay_idx = streamlit_app.index("if autoplay and len(cutoffs) > 1:")
    assert focused_chart_idx < autoplay_idx
    assert "Fixed date axis" in streamlit_app
    assert "add_period_shading" in streamlit_app
    assert "add_cutoff_marker" in streamlit_app
    assert "MLForecast interpretability" in streamlit_app
    assert "Win rate vs benchmark" in streamlit_app
    assert "benchmark_win_rate_chart" in streamlit_app
    assert "residual_horizon_chart" in streamlit_app
    assert "residual_time_chart" in streamlit_app
    assert "residual_distribution_chart" in streamlit_app
    assert "residual_acf_chart" in streamlit_app
    assert "Residual white-noise check" in streamlit_app
    assert "residual_outlier_table" in streamlit_app
    assert "interval_calibration_chart" in streamlit_app
    assert "Hierarchy roll-up / roll-down" in streamlit_app
    assert "add_vline" not in streamlit_app


def test_residual_test_summary_prefers_one_step_residuals_and_sample_std() -> None:
    errors = [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0]
    rows = []
    for idx, error in enumerate(errors, start=1):
        cutoff = pd.Timestamp("2025-01-31") + pd.offsets.MonthEnd(idx - 1)
        rows.append(
            {
                "unique_id": "Revenue",
                "cutoff": cutoff,
                "ds": cutoff + pd.offsets.MonthEnd(1),
                "y": 100.0 + error,
                "BiasedModel": 100.0,
            }
        )
        rows.append(
            {
                "unique_id": "Revenue",
                "cutoff": cutoff,
                "ds": cutoff + pd.offsets.MonthEnd(2),
                "y": 110.0 + error,
                "BiasedModel": 110.0,
            }
        )
    history = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 24,
            "ds": pd.date_range("2023-01-31", periods=24, freq="ME"),
            "y": range(100, 124),
        }
    )
    run = ForecastRun(
        history=history,
        forecast=pd.DataFrame(),
        all_models=pd.DataFrame(),
        model_selection=pd.DataFrame({"unique_id": ["Revenue"], "selected_model": ["BiasedModel"]}),
        backtest_metrics=pd.DataFrame(),
        profile=DataProfile(
            rows=len(history),
            series_count=1,
            freq="ME",
            season_length=12,
            start="2023-01-31",
            end="2024-12-31",
            min_obs_per_series=len(history),
            max_obs_per_series=len(history),
            duplicate_rows=0,
            missing_timestamps=0,
            null_y=0,
            zero_y=0,
            negative_y=0,
            data_freshness=None,
        ),
        spec=ForecastSpec(horizon=2, freq="ME", model_policy="baseline"),
        backtest_predictions=pd.DataFrame(rows),
    )

    residual_tests = build_residual_test_summary(run)

    row = residual_tests.loc[residual_tests["model"] == "BiasedModel"].iloc[0]
    assert row["residual_scope"] == "all_backtest_horizons"
    assert row["white_noise_residual_scope"] == "horizon_step_1"
    assert row["white_noise_observations"] == len(errors)
    significant_lags = [] if pd.isna(row["significant_acf_lags"]) else str(row["significant_acf_lags"]).split(",")
    assert row["significant_acf_lag_count"] == len(significant_lags)
    assert row["bias_status"] == "fail"
    assert row["residual_std"] == pytest.approx(pd.Series(errors * 2).std(ddof=1))


def test_report_regeneration_handles_missing_executive_headline_for_old_runs(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-31", periods=12, freq="ME"),
            "y": [100, 105, 108, 112, 118, 121, 127, 130, 136, 141, 148, 155],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=2, model_policy="baseline"))
    output_dir = run.to_directory(tmp_path / "old_run")
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.pop("executive_headline")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    write_report_artifacts_from_directory(output_dir)

    decoded_report = base64.b64decode((output_dir / "report_base64.txt").read_text(encoding="utf-8")).decode("utf-8")
    assert "Executive forecast headline" in decoded_report
    assert "Executive headline unavailable for this run." in decoded_report


def test_seasonality_diagnostics_warn_when_cycles_are_insufficient() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 18,
            "ds": pd.date_range("2025-01-01", periods=18, freq="MS"),
            "y": [100 + i * 2 + (30 if i % 12 == 11 else 0) for i in range(18)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, freq="MS", model_policy="baseline"))

    from nixtla_scaffold.interpretation import seasonality_decomposition_frame, seasonality_diagnostics_frame
    from nixtla_scaffold.outputs import build_trust_summary

    diagnostics = seasonality_diagnostics_frame(run)
    decomposition = seasonality_decomposition_frame(run)
    row = diagnostics.iloc[0]

    assert row["season_length"] == 12
    assert row["complete_cycles"] == 1
    assert row["credibility_label"] == "low"
    assert "need at least 2" in row["warning"]
    assert decomposition.empty
    trust = build_trust_summary(run).iloc[0]
    assert trust["seasonality_status"] == "low"
    assert "seasonality" in str(trust["caveats"]).lower()


def test_seasonality_decomposition_is_written_when_cycles_are_available() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 48,
            "ds": pd.date_range("2022-01-01", periods=48, freq="MS"),
            "y": [200 + i * 1.5 + (35 if i % 12 in {10, 11} else -12 if i % 12 in {5, 6} else 0) for i in range(48)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, freq="MS", model_policy="baseline"))

    from nixtla_scaffold.interpretation import seasonality_decomposition_frame, seasonality_diagnostics_frame

    diagnostics = seasonality_diagnostics_frame(run)
    decomposition = seasonality_decomposition_frame(run)
    row = diagnostics.iloc[0]

    assert row["complete_cycles"] == 4
    assert row["credibility_label"] in {"medium", "high"}
    assert row["warning"] is None or row["warning"] == ""
    assert not decomposition.empty
    assert {"observed", "trend", "seasonal", "remainder"}.issubset(decomposition.columns)
    assert decomposition["seasonal"].abs().max() > 0


def test_trust_summary_scores_short_history_below_richer_series() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Richer"] * 24 + ["Short"] * 6,
            "ds": pd.date_range("2023-01-01", periods=24, freq="MS").tolist()
            + pd.date_range("2024-07-01", periods=6, freq="MS").tolist(),
            "y": [100 + i * 3 for i in range(24)] + [50, 53, 55, 54, 56, 57],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, freq="MS", model_policy="baseline"))

    from nixtla_scaffold.outputs import build_trust_summary

    trust = build_trust_summary(run).set_index("unique_id")

    assert {"Richer", "Short"}.issubset(trust.index)
    assert trust.loc["Richer", "trust_score_0_100"] > trust.loc["Short", "trust_score_0_100"]
    assert trust.loc["Short", "trust_level"] in {"Low", "Medium"}
    assert "history" in str(trust.loc["Short", "caveats"]).lower()
    assert str(trust.loc["Short", "next_actions"])


def test_executive_headline_portfolio_summarizes_distribution_and_watch_series() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Richer"] * 24 + ["Short"] * 6,
            "ds": pd.date_range("2023-01-01", periods=24, freq="MS").tolist()
            + pd.date_range("2024-07-01", periods=6, freq="MS").tolist(),
            "y": [100 + i * 3 for i in range(24)] + [50, 53, 55, 54, 56, 57],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, freq="MS", model_policy="baseline"))

    headline = build_executive_headline(run)

    assert headline.scope == "portfolio"
    assert len(headline.series) == 2
    assert "2 series forecast" in headline.paragraph
    assert "High" in headline.paragraph and "Medium" in headline.paragraph and "Low" in headline.paragraph
    assert "series allow full-horizon champion claims" in headline.paragraph
    assert "filter forecast.csv to planning_eligible=True" in headline.paragraph
    assert "Direction split versus recent actuals" in headline.paragraph
    assert "Watch Short" in headline.paragraph
    assert headline.top_caveat
    assert headline.next_action


def test_executive_headline_includes_unit_absolute_delta_and_yoy() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 24,
            "ds": pd.date_range("2024-01-01", periods=24, freq="MS"),
            "y": [100 + i * 5 + (20 if i % 12 == 2 else 0) for i in range(24)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, freq="MS", model_policy="baseline", unit_label="$"))

    headline = build_executive_headline(run)

    assert headline.value_unit_label == "$"
    assert headline.direction_abs_delta_vs_recent is not None
    assert headline.yoy_pct_vs_prior_year is not None
    assert headline.yoy_abs_delta_vs_prior_year is not None
    assert "$" in headline.paragraph
    assert "absolute delta" in headline.paragraph
    assert "YoY:" in headline.paragraph


def test_executive_headline_omits_yoy_when_prior_year_is_unavailable() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 10,
            "ds": pd.date_range("2025-01-01", periods=10, freq="MS"),
            "y": [100 + i * 4 for i in range(10)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=2, freq="MS", model_policy="baseline", unit_label="ARR"))

    headline = build_executive_headline(run)

    assert headline.yoy_pct_vs_prior_year is None
    assert headline.yoy_abs_delta_vs_prior_year is None
    assert "YoY:" not in headline.paragraph


def test_executive_headline_normalized_unit_caveat_leads_numbers() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 12,
            "ds": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "y": [100 + i * 10 for i in range(12)],
            "price_factor": [1.0 + i * 0.02 for i in range(12)],
        }
    )
    run = run_forecast(
        df,
        ForecastSpec(
            horizon=2,
            freq="MS",
            model_policy="baseline",
            transform=TransformSpec(normalization_factor_col="price_factor", normalization_label="FY26 pricing"),
        ),
    )

    paragraph = build_executive_headline(run).paragraph

    assert "Values are in normalized units (FY26 pricing)" in paragraph
    assert paragraph.index("Values are in normalized units") < paragraph.index("Statistical baseline")
    assert paragraph.index("Values are in normalized units") < paragraph.index("Biggest caveat:")


def test_executive_headline_handles_empty_selected_forecast() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Revenue"] * 8,
            "ds": pd.date_range("2025-01-01", periods=8, freq="MS"),
            "y": [100 + i for i in range(8)],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=2, freq="MS", model_policy="baseline"))
    run.forecast = run.forecast.iloc[0:0].copy()

    headline = build_executive_headline(run)

    assert headline.scope == "empty"
    assert headline.direction == "unavailable"
    assert headline.direction_pct_vs_recent is None
    assert headline.direction_abs_delta_vs_recent is None
    assert "no selected forecast rows" in headline.paragraph.lower()
    assert headline.series == []


def test_executive_headline_zero_recent_average_keeps_absolute_delta() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Zero"] * 8,
            "ds": pd.date_range("2025-01-01", periods=8, freq="D"),
            "y": [0.0] * 8,
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=2, freq="D", model_policy="baseline"))

    headline = build_executive_headline(run)

    assert headline.direction == "unavailable"
    assert headline.direction_pct_vs_recent is None
    assert headline.direction_abs_delta_vs_recent is not None
    assert "recent average is zero" in headline.paragraph
    assert "absolute delta" in headline.paragraph


def test_executive_headline_portfolio_prioritizes_low_then_medium_actions() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Richer"] * 24 + ["Short"] * 6 + ["Tiny"] * 2,
            "ds": pd.date_range("2023-01-01", periods=24, freq="MS").tolist()
            + pd.date_range("2024-07-01", periods=6, freq="MS").tolist()
            + pd.date_range("2024-11-01", periods=2, freq="MS").tolist(),
            "y": [100 + i * 3 for i in range(24)] + [50, 53, 55, 54, 56, 57] + [25, 27],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=3, freq="MS", model_policy="baseline"))

    headline = build_executive_headline(run)

    assert headline.scope == "portfolio"
    assert "Address 1 Low-trust row(s) in trust_summary.csv first" in headline.next_action
    assert "Medium-trust" in headline.next_action
    assert headline.next_action in headline.paragraph


def test_executive_headline_marks_flat_direction_when_delta_is_small() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Stable"] * 12,
            "ds": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "y": [100.0] * 12,
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=2, freq="MS", model_policy="baseline"))

    headline = build_executive_headline(run)

    assert headline.direction == "flat"
    assert headline.direction_pct_vs_recent is not None
    assert abs(headline.direction_pct_vs_recent) < 0.02
    assert "approximately flat versus the recent average" in headline.paragraph


def test_trust_summary_handles_missing_backtest_without_crashing() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["Tiny"] * 4,
            "ds": pd.date_range("2026-01-01", periods=4, freq="D"),
            "y": [10, 11, 9, 12],
        }
    )
    run = run_forecast(df, ForecastSpec(horizon=4, freq="D", model_policy="baseline", require_backtest=False))

    from nixtla_scaffold.outputs import build_trust_summary

    trust = build_trust_summary(run)

    assert len(trust) == 1
    row = trust.iloc[0]
    assert row["trust_level"] in {"High", "Medium", "Low"}
    assert 0 <= row["trust_score_0_100"] <= 100
    assert "backtest" in str(row["caveats"]).lower() or "history" in str(row["caveats"]).lower()


def test_cli_coerces_numeric_sheet_index() -> None:
    assert _coerce_sheet("1") == 1
    assert _coerce_sheet("Data") == "Data"
    assert _coerce_sheet(None) is None


def test_xls_input_is_rejected_with_clear_error(tmp_path) -> None:
    input_path = tmp_path / "legacy.xls"
    input_path.write_text("not really excel", encoding="utf-8")

    exit_code = main(["profile", "--input", str(input_path)])

    assert exit_code == 2

