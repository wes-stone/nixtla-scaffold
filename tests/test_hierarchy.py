from __future__ import annotations

import pandas as pd
import pytest

import nixtla_scaffold.outputs as output_module
from nixtla_scaffold import (
    DataProfile,
    ForecastRun,
    ForecastSpec,
    aggregate_hierarchy_frame,
    hierarchy_coherence,
    hierarchy_summary,
    reconcile_hierarchy_forecast,
    run_forecast,
)
from nixtla_scaffold.cli import main
from nixtla_scaffold.outputs import build_hierarchy_backtest_comparison, build_hierarchy_contribution_frame


def _leaf_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["NA", "NA", "EMEA", "EMEA"] * 2,
            "product": ["ProductA", "ProductB", "ProductA", "ProductB"] * 2,
            "month": ["2025-01-31"] * 4 + ["2025-02-28"] * 4,
            "revenue": [100, 70, 80, 50, 108, 73, 84, 54],
        }
    )


def _hierarchy_profile(history: pd.DataFrame) -> DataProfile:
    return DataProfile(
        rows=len(history),
        series_count=history["unique_id"].nunique(),
        freq="ME",
        season_length=12,
        start=str(pd.to_datetime(history["ds"]).min().date()),
        end=str(pd.to_datetime(history["ds"]).max().date()),
        min_obs_per_series=int(history.groupby("unique_id").size().min()),
        max_obs_per_series=int(history.groupby("unique_id").size().max()),
        duplicate_rows=0,
        missing_timestamps=0,
        null_y=0,
        zero_y=0,
        negative_y=0,
        data_freshness=None,
    )


def test_aggregate_hierarchy_frame_creates_total_parent_and_leaf_nodes() -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )

    assert set(frame["hierarchy_level"]) == {"total", "region", "region/product"}
    assert frame["unique_id"].nunique() == 7
    jan = frame[frame["ds"] == pd.Timestamp("2025-01-31")]
    assert jan.loc[jan["unique_id"] == "Total", "y"].iloc[0] == 300
    assert jan.loc[jan["unique_id"] == "region=NA", "y"].iloc[0] == 170
    assert jan.loc[jan["unique_id"] == "region=NA|product=ProductA", "y"].iloc[0] == 100


def test_hierarchy_summary_reports_level_counts() -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )

    summary = hierarchy_summary(frame)

    assert summary["series_count"] == 7
    levels = {level["hierarchy_level"]: level["series_count"] for level in summary["levels"]}
    assert levels == {"total": 1, "region": 2, "region/product": 4}


def test_hierarchy_output_feeds_forecast_path() -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )

    run = run_forecast(frame, ForecastSpec(horizon=1, freq="ME", model_policy="baseline"))

    assert run.profile.series_count == 7
    assert len(run.forecast) == 7
    assert {"hierarchy_level", "hierarchy_depth"}.issubset(run.forecast.columns)
    assert any("not reconciled" in warning for warning in run.warnings)


def test_hierarchy_coherence_reports_parent_child_gaps(tmp_path) -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )
    run = run_forecast(frame, ForecastSpec(horizon=1, freq="ME", model_policy="baseline"))

    coherence = hierarchy_coherence(run.forecast)
    output_dir = run.to_directory(tmp_path / "run")

    assert not coherence.empty
    assert {"parent_unique_id", "parent_value", "child_sum", "gap_pct"}.issubset(coherence.columns)
    assert (output_dir / "hierarchy_coherence.csv").exists()
    assert (output_dir / "hierarchy_contribution.csv").exists()
    assert (output_dir / "best_practice_receipts.csv").exists()
    contribution = pd.read_csv(output_dir / "hierarchy_contribution.csv")
    assert {"parent_unique_id", "child_unique_id", "child_share_of_parent", "gap_contribution"}.issubset(contribution.columns)
    streamlit_app = (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Hierarchy roll-up / roll-down" in streamlit_app
    assert "hierarchy_rollup_chart" in streamlit_app
    assert 'read_csv("hierarchy_contribution.csv")' in streamlit_app
    assert "Child contribution to selected parent" in streamlit_app
    assert "hierarchy_coherence" in run.manifest()["outputs"]
    assert "hierarchy_contribution" in run.manifest()["outputs"]
    assert "best_practice_receipts" in run.manifest()["outputs"]


def test_hierarchy_contribution_allocates_parent_child_gap_by_child_share() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "ds": [pd.Timestamp("2025-03-31")] * 3,
            "yhat": [100.0, 30.0, 50.0],
            "model": ["Manual"] * 3,
            "hierarchy_level": ["total", "region", "region"],
            "hierarchy_depth": [0, 1, 1],
        }
    )
    history = forecast.rename(columns={"yhat": "y"}).copy()
    run = ForecastRun(
        history=history,
        forecast=forecast,
        all_models=pd.DataFrame(),
        model_selection=pd.DataFrame({"unique_id": ["Total"], "selected_model": ["Manual"]}),
        backtest_metrics=pd.DataFrame(),
        profile=_hierarchy_profile(history),
        spec=ForecastSpec(horizon=1, freq="ME", model_policy="baseline"),
    )

    contribution = build_hierarchy_contribution_frame(run)

    row = contribution[
        (contribution["parent_unique_id"] == "Total")
        & (contribution["child_unique_id"] == "region=NA")
        & (contribution["value_col"] == "yhat")
    ].iloc[0]
    assert abs(row["child_share_of_parent"] - 0.30) < 1e-9
    assert abs(row["child_share_of_children"] - 0.375) < 1e-9
    assert abs(row["gap_allocation_weight"] - 0.375) < 1e-9
    assert abs(row["parent_child_gap"] - 20.0) < 1e-9
    assert abs(row["gap_contribution"] - 7.5) < 1e-9
    assert row["gap_contribution_formula"] == "parent_child_gap * child_value / immediate_child_sum"


def test_bottom_up_reconciliation_sums_bottom_level_forecasts() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=NA|product=A", "region=NA|product=B"],
            "ds": [pd.Timestamp("2025-03-31")] * 4,
            "yhat": [999.0, 500.0, 40.0, 70.0],
            "model": ["Manual"] * 4,
            "hierarchy_level": ["total", "region", "region/product", "region/product"],
            "hierarchy_depth": [0, 1, 2, 2],
            "region": [None, "NA", "NA", "NA"],
            "product": [None, None, "ProductA", "ProductB"],
        }
    )

    reconciled, summary, warnings = reconcile_hierarchy_forecast(forecast, method="bottom_up")
    coherence = hierarchy_coherence(reconciled)

    assert reconciled.loc[reconciled["unique_id"] == "Total", "yhat"].iloc[0] == 110.0
    assert reconciled.loc[reconciled["unique_id"] == "region=NA", "yhat"].iloc[0] == 110.0
    assert float(coherence["gap"].abs().max()) == 0.0
    assert summary["applied_method"].unique().tolist() == ["bottom_up"]
    assert any("hierarchy reconciliation applied" in warning for warning in warnings)


def test_mint_ols_reconciliation_uses_hierarchicalforecast_when_available() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("hierarchicalforecast")
    forecast = pd.DataFrame(
        {
            "unique_id": ["Total", "Total", "region=NA", "region=NA", "region=NA|product=A", "region=NA|product=A", "region=NA|product=B", "region=NA|product=B"],
            "ds": [pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-30")] * 4,
            "yhat": [100.0, 110.0, 90.0, 95.0, 40.0, 42.0, 55.0, 58.0],
            "model": ["Manual"] * 8,
            "hierarchy_level": ["total", "total", "region", "region", "region/product", "region/product", "region/product", "region/product"],
            "hierarchy_depth": [0, 0, 1, 1, 2, 2, 2, 2],
            "region": [None, None, "NA", "NA", "NA", "NA", "NA", "NA"],
            "product": [None, None, None, None, "ProductA", "ProductA", "ProductB", "ProductB"],
        }
    )

    reconciled, summary, warnings = reconcile_hierarchy_forecast(forecast, method="mint_ols")
    coherence = hierarchy_coherence(reconciled)

    assert summary["applied_method"].unique().tolist() == ["mint_ols"]
    assert pd.to_numeric(coherence["gap"], errors="coerce").abs().max() < 1e-8
    assert not any("fallback" in warning for warning in warnings)


def test_forecast_hierarchy_reconciliation_writes_pre_post_artifacts(tmp_path) -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )
    run = run_forecast(frame, ForecastSpec(horizon=1, freq="ME", model_policy="baseline", hierarchy_reconciliation="bottom_up"))

    coherence = hierarchy_coherence(run.forecast)
    output_dir = run.to_directory(tmp_path / "run")

    assert not coherence.empty
    assert pd.to_numeric(coherence["gap"], errors="coerce").abs().max() < 1e-8
    assert not run.hierarchy_reconciliation.empty
    assert not run.unreconciled_forecast.empty
    assert any("reconciliation applied" in warning for warning in run.warnings)
    assert not any("not reconciled" in warning for warning in run.warnings)
    assert (output_dir / "hierarchy_reconciliation.csv").exists()
    assert (output_dir / "audit" / "hierarchy_unreconciled_forecast.csv").exists()
    assert (output_dir / "audit" / "hierarchy_backtest_comparison.csv").exists()
    assert (output_dir / "audit" / "hierarchy_coherence_pre.csv").exists()
    assert (output_dir / "audit" / "hierarchy_coherence_post.csv").exists()
    comparison = pd.read_csv(output_dir / "audit" / "hierarchy_backtest_comparison.csv")
    assert {"yhat_unreconciled", "yhat_reconciled", "abs_error_delta", "reconciliation_method", "comparison_status"}.issubset(comparison.columns)
    assert "hierarchy_reconciliation" in run.manifest()["outputs"]
    assert "hierarchy_backtest_comparison" in run.manifest()["outputs"]
    sheets = pd.ExcelFile(output_dir / "forecast.xlsx").sheet_names
    assert "Hierarchy Reconciliation" in sheets
    assert "Hierarchy Contribution" in sheets
    assert "Hierarchy Backtest" in sheets
    streamlit_app = (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Hierarchy reconciliation is enabled" in streamlit_app
    assert "Reconciled vs unreconciled backtest comparison" in streamlit_app


def test_hierarchy_backtest_comparison_is_empty_when_reconciliation_is_none() -> None:
    history = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "ds": [pd.Timestamp("2025-01-31")] * 3,
            "y": [80.0, 30.0, 50.0],
            "hierarchy_level": ["total", "region", "region"],
            "hierarchy_depth": [0, 1, 1],
        }
    )
    backtest_predictions = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "cutoff": [pd.Timestamp("2025-01-31")] * 3,
            "ds": [pd.Timestamp("2025-02-28")] * 3,
            "y": [80.0, 30.0, 50.0],
            "Manual": [100.0, 30.0, 50.0],
        }
    )
    run = ForecastRun(
        history=history,
        forecast=pd.DataFrame(),
        all_models=pd.DataFrame(),
        model_selection=pd.DataFrame({"unique_id": ["Total", "region=NA", "region=EMEA"], "selected_model": ["Manual"] * 3}),
        backtest_metrics=pd.DataFrame(),
        profile=_hierarchy_profile(history),
        spec=ForecastSpec(horizon=1, freq="ME", model_policy="baseline", hierarchy_reconciliation="none"),
        backtest_predictions=backtest_predictions,
    )

    comparison = build_hierarchy_backtest_comparison(run)

    assert comparison.empty
    assert {"yhat_unreconciled", "yhat_reconciled", "abs_error_delta", "reconciliation_method", "comparison_status", "comparison_note"}.issubset(
        comparison.columns
    )


def test_hierarchy_backtest_comparison_reconciles_selected_backtest_rows() -> None:
    history = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "ds": [pd.Timestamp("2025-01-31")] * 3,
            "y": [80.0, 30.0, 50.0],
            "hierarchy_level": ["total", "region", "region"],
            "hierarchy_depth": [0, 1, 1],
        }
    )
    backtest_predictions = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "cutoff": [pd.Timestamp("2025-01-31")] * 3,
            "ds": [pd.Timestamp("2025-02-28")] * 3,
            "y": [80.0, 30.0, 50.0],
            "Manual": [100.0, 30.0, 50.0],
        }
    )
    run = ForecastRun(
        history=history,
        forecast=pd.DataFrame(),
        all_models=pd.DataFrame(),
        model_selection=pd.DataFrame({"unique_id": ["Total", "region=NA", "region=EMEA"], "selected_model": ["Manual"] * 3}),
        backtest_metrics=pd.DataFrame(),
        profile=_hierarchy_profile(history),
        spec=ForecastSpec(horizon=1, freq="ME", model_policy="baseline", hierarchy_reconciliation="bottom_up"),
        backtest_predictions=backtest_predictions,
    )

    comparison = build_hierarchy_backtest_comparison(run)

    total = comparison.loc[comparison["unique_id"] == "Total"].iloc[0]
    assert total["reconciliation_method"] == "bottom_up"
    assert total["applied_method"] == "bottom_up"
    assert total["comparison_status"] == "compared"
    assert abs(total["yhat_unreconciled"] - 100.0) < 1e-9
    assert abs(total["yhat_reconciled"] - 80.0) < 1e-9
    assert abs(total["abs_error_delta"] + 20.0) < 1e-9


def test_hierarchy_backtest_comparison_surfaces_reconciliation_failures(monkeypatch) -> None:
    history = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "ds": [pd.Timestamp("2025-01-31")] * 3,
            "y": [80.0, 30.0, 50.0],
            "hierarchy_level": ["total", "region", "region"],
            "hierarchy_depth": [0, 1, 1],
        }
    )
    backtest_predictions = pd.DataFrame(
        {
            "unique_id": ["Total", "region=NA", "region=EMEA"],
            "cutoff": [pd.Timestamp("2025-01-31")] * 3,
            "ds": [pd.Timestamp("2025-02-28")] * 3,
            "y": [80.0, 30.0, 50.0],
            "Manual": [100.0, 30.0, 50.0],
        }
    )
    run = ForecastRun(
        history=history,
        forecast=pd.DataFrame(),
        all_models=pd.DataFrame(),
        model_selection=pd.DataFrame({"unique_id": ["Total", "region=NA", "region=EMEA"], "selected_model": ["Manual"] * 3}),
        backtest_metrics=pd.DataFrame(),
        profile=_hierarchy_profile(history),
        spec=ForecastSpec(horizon=1, freq="ME", model_policy="baseline", hierarchy_reconciliation="bottom_up"),
        backtest_predictions=backtest_predictions,
    )

    def fail_reconciliation(*args, **kwargs):
        raise ValueError("bad hierarchy structure")

    monkeypatch.setattr(output_module, "reconcile_hierarchy_forecast", fail_reconciliation)

    comparison = build_hierarchy_backtest_comparison(run)

    assert not comparison.empty
    assert set(comparison["comparison_status"]) == {"reconciliation_failed"}
    assert comparison["comparison_note"].str.contains("bad hierarchy structure").all()
    assert comparison["yhat_reconciled"].isna().all()


def test_hierarchy_contribution_rejects_malformed_deep_unique_ids() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["Total", "NA", "ProductA"],
            "ds": [pd.Timestamp("2025-03-31")] * 3,
            "yhat": [100.0, 80.0, 50.0],
            "hierarchy_level": ["total", "region", "region/product"],
            "hierarchy_depth": [0, 1, 2],
        }
    )
    history = forecast.rename(columns={"yhat": "y"}).copy()
    run = ForecastRun(
        history=history,
        forecast=forecast,
        all_models=pd.DataFrame(),
        model_selection=pd.DataFrame(),
        backtest_metrics=pd.DataFrame(),
        profile=_hierarchy_profile(history),
        spec=ForecastSpec(horizon=1, freq="ME", model_policy="baseline"),
    )

    with pytest.raises(ValueError, match="pipe-delimited unique_id ancestors"):
        build_hierarchy_contribution_frame(run)


def test_cli_forecast_hierarchy_reconciliation_flag_writes_artifacts(tmp_path) -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )
    input_path = tmp_path / "hierarchy_nodes.csv"
    output_dir = tmp_path / "cli_run"
    frame.to_csv(input_path, index=False)

    exit_code = main(
        [
            "forecast",
            "--input",
            str(input_path),
            "--horizon",
            "1",
            "--freq",
            "ME",
            "--model-policy",
            "baseline",
            "--hierarchy-reconciliation",
            "bottom_up",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    forecast = pd.read_csv(output_dir / "forecast.csv")
    coherence = hierarchy_coherence(forecast)
    assert pd.to_numeric(coherence["gap"], errors="coerce").abs().max() < 1e-8
    assert (output_dir / "hierarchy_reconciliation.csv").exists()
    assert (output_dir / "audit" / "hierarchy_unreconciled_forecast.csv").exists()
    assert (output_dir / "audit" / "hierarchy_backtest_comparison.csv").exists()
    assert (output_dir / "audit" / "hierarchy_coherence_pre.csv").exists()
    assert (output_dir / "audit" / "hierarchy_coherence_post.csv").exists()


def test_hierarchy_coherence_respects_custom_total_label() -> None:
    frame = aggregate_hierarchy_frame(
        _leaf_data(),
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
        total_label="All",
    )
    run = run_forecast(frame, ForecastSpec(horizon=1, freq="ME", model_policy="baseline"))

    coherence = hierarchy_coherence(run.forecast)

    assert "All" in set(coherence["parent_unique_id"])


def test_hierarchy_excel_preserves_na_region_code(tmp_path) -> None:
    input_path = tmp_path / "hierarchy.xlsx"
    _leaf_data().to_excel(input_path, index=False)

    frame = aggregate_hierarchy_frame(
        input_path,
        hierarchy_cols=("region", "product"),
        time_col="month",
        target_col="revenue",
    )

    assert "region=NA" in set(frame["unique_id"])


def test_hierarchy_rejects_reserved_delimiters_before_duplicate_ids() -> None:
    data = pd.DataFrame(
        {
            "region": ["x|product=y", "x"],
            "product": ["z", "y"],
            "month": ["2025-01-31", "2025-01-31"],
            "revenue": [10, 20],
        }
    )

    try:
        aggregate_hierarchy_frame(
            data,
            hierarchy_cols=("region", "product"),
            time_col="month",
            target_col="revenue",
        )
    except ValueError as exc:
        assert "reserved delimiter" in str(exc)
    else:
        raise AssertionError("expected reserved hierarchy delimiter values to fail")


def test_cli_hierarchy_writes_canonical_csv(tmp_path, capsys) -> None:
    input_path = tmp_path / "hierarchy.csv"
    output_path = tmp_path / "hierarchy_nodes.csv"
    _leaf_data().to_csv(input_path, index=False)

    exit_code = main(
        [
            "hierarchy",
            "--input",
            str(input_path),
            "--time-col",
            "month",
            "--target-col",
            "revenue",
            "--hierarchy-cols",
            "region",
            "product",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    output = pd.read_csv(output_path)
    assert {"unique_id", "ds", "y", "hierarchy_level", "hierarchy_depth"}.issubset(output.columns)
    assert "series_count" in capsys.readouterr().out
