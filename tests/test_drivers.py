from __future__ import annotations

import json

import pandas as pd
import pytest

from nixtla_scaffold import DriverEvent, ForecastSpec, KnownFutureRegressor, run_forecast
from nixtla_scaffold.cli import main
from nixtla_scaffold.drivers import parse_driver_events, parse_known_future_regressors


def _history_frame(periods: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2025-01-31", periods=periods, freq="ME")
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * periods,
            "ds": dates,
            "y": [100 + idx * 5 for idx in range(periods)],
            "seats_plan": [50 + idx for idx in range(periods)],
        }
    )


def _future_driver_file(path, *, periods: int = 2, known_as_of: str = "2025-12-15", value_col: str = "seats_plan") -> None:
    dates = pd.date_range("2026-01-31", periods=periods, freq="ME")
    pd.DataFrame(
        {
            "unique_id": ["Revenue"] * periods,
            "ds": dates,
            value_col: [70 + idx for idx in range(periods)],
            "known_as_of": [known_as_of] * periods,
        }
    ).to_csv(path, index=False)


def _ml_history_frame(periods: int = 36) -> pd.DataFrame:
    dates = pd.date_range("2023-01-31", periods=periods, freq="ME")
    seats = [50 + idx for idx in range(periods)]
    return pd.DataFrame(
        {
            "unique_id": ["Revenue"] * periods,
            "ds": dates,
            "y": [100 + idx * 3 + seats[idx] * 0.4 for idx in range(periods)],
            "seats_plan": seats,
        }
    )


def test_driver_file_parsers_support_events_and_regressors(tmp_path) -> None:
    event_file = tmp_path / "events.csv"
    event_file.write_text(
        "name,start,end,effect,magnitude,confidence,affected_unique_ids,notes\n"
        "Launch,2026-01-31,2026-02-28,multiplicative,0.1,0.8,Revenue,planned launch\n",
        encoding="utf-8",
    )
    regressor_file = tmp_path / "regressors.json"
    regressor_file.write_text(
        json.dumps(
            {
                "regressors": [
                    {
                        "name": "Seats plan",
                        "value_col": "seats_plan",
                        "availability": "plan",
                        "mode": "model_candidate",
                        "future_file": "future_seats.csv",
                        "source_system": "excel",
                        "owner": "fp&a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    events = parse_driver_events(files=[event_file])
    regressors = parse_known_future_regressors(files=[regressor_file])

    assert events == (
        DriverEvent(
            name="Launch",
            start="2026-01-31",
            end="2026-02-28",
            effect="multiplicative",
            magnitude=0.1,
            affected_unique_ids=("Revenue",),
            confidence=0.8,
            notes="planned launch",
        ),
    )
    assert regressors == (
        KnownFutureRegressor(
            name="Seats plan",
            value_col="seats_plan",
            availability="plan",
            mode="model_candidate",
            future_file="future_seats.csv",
            source_system="excel",
            owner="fp&a",
        ),
    )


def test_driver_contract_outputs_reports_and_workbook_surfaces(tmp_path) -> None:
    future_file = tmp_path / "future_seats.csv"
    _future_driver_file(future_file)
    event = DriverEvent(name="Launch", start="2026-02-28", effect="additive", magnitude=10)
    regressor = KnownFutureRegressor(
        name="Seats plan",
        value_col="seats_plan",
        availability="plan",
        mode="model_candidate",
        future_file=str(future_file),
        source_system="excel",
        owner="finance",
    )

    run = run_forecast(
        _history_frame(),
        ForecastSpec(horizon=2, freq="ME", model_policy="baseline", events=(event,), regressors=(regressor,)),
    )
    output_dir = run.to_directory(tmp_path / "run")
    appendix = output_dir / "appendix"

    assert (appendix / "scenario_assumptions.csv").exists()
    assert (appendix / "scenario_forecast.csv").exists()
    assert (appendix / "known_future_regressors.csv").exists()
    assert (appendix / "driver_availability_audit.csv").exists()
    assert (appendix / "driver_experiment_summary.csv").exists()
    audit = pd.read_csv(appendix / "driver_availability_audit.csv")
    assert audit["audit_status"].tolist() == ["passed"]
    assert audit["modeling_decision"].tolist() == ["candidate_audited_not_trained"]
    scenario_forecast = pd.read_csv(appendix / "scenario_forecast.csv")
    assert {
        "horizon_step",
        "row_horizon_status",
        "horizon_trust_state",
        "validated_through_horizon",
        "planning_eligible",
        "planning_eligibility_scope",
        "planning_eligibility_reason",
        "validation_evidence",
    }.issubset(scenario_forecast.columns)
    assert any("not automatically trained" in warning for warning in run.warnings)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["scenario_assumptions"] == "appendix/scenario_assumptions.csv"
    assert manifest["outputs"]["scenario_forecast"] == "appendix/scenario_forecast.csv"
    assert manifest["outputs"]["known_future_regressors"] == "appendix/known_future_regressors.csv"
    assert manifest["outputs"]["driver_availability_audit"] == "appendix/driver_availability_audit.csv"
    assert manifest["outputs"]["driver_experiment_summary"] == "appendix/driver_experiment_summary.csv"

    report_html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Assumptions and drivers" in report_html
    assert "Known-future regressors are audited for leakage and future availability" in report_html
    assert "scenario_assumptions.csv" in report_html
    assert "driver_availability_audit.csv" in report_html
    streamlit_app = (output_dir / "streamlit_app.py").read_text(encoding="utf-8")
    assert '"Assumptions & Drivers"' in streamlit_app
    assert 'read_csv("driver_availability_audit.csv")' in streamlit_app
    assert "Driver audit distribution" in streamlit_app
    workbook = pd.ExcelFile(output_dir / "forecast.xlsx")
    assert "Scenario Assumptions" in workbook.sheet_names
    assert "Scenario Forecast" in workbook.sheet_names
    scenario_sheet = workbook.parse("Scenario Forecast")
    assert {"horizon_step", "row_horizon_status", "planning_eligible", "planning_eligibility_scope"}.issubset(scenario_sheet.columns)
    assert "Known Future Regressors" in workbook.sheet_names
    assert "Driver Audit" in workbook.sheet_names
    assert "Driver Experiments" in workbook.sheet_names


def test_known_future_regressors_are_audit_only_by_default_even_with_mlforecast(tmp_path) -> None:
    pytest.importorskip("mlforecast")
    future_file = tmp_path / "future_seats.csv"
    _future_driver_file(future_file)
    regressor = KnownFutureRegressor(
        name="Seats plan",
        value_col="seats_plan",
        availability="plan",
        mode="model_candidate",
        future_file=str(future_file),
    )

    run = run_forecast(
        _ml_history_frame(),
        ForecastSpec(
            horizon=2,
            freq="ME",
            levels=(),
            model_policy="mlforecast",
            model_allowlist=("LinearRegression",),
            regressors=(regressor,),
            weighted_ensemble=False,
            verbose=False,
        ),
    )

    assert run.driver_model_features.empty
    audit = run.driver_availability_audit.set_index("name")
    assert audit.loc["Seats plan", "audit_status"] == "passed"
    assert audit.loc["Seats plan", "modeling_decision"] == "candidate_audited_not_trained"
    assert any("not automatically trained unless train_known_future_regressors=True" in warning for warning in run.warnings)


def test_opt_in_known_future_regressors_feed_mlforecast_artifacts(tmp_path) -> None:
    pytest.importorskip("mlforecast")
    future_file = tmp_path / "future_seats.csv"
    _future_driver_file(future_file)
    regressor = KnownFutureRegressor(
        name="Seats plan",
        value_col="seats_plan",
        availability="plan",
        mode="model_candidate",
        future_file=str(future_file),
    )

    run = run_forecast(
        _ml_history_frame(),
        ForecastSpec(
            horizon=2,
            freq="ME",
            levels=(),
            model_policy="mlforecast",
            model_allowlist=("LinearRegression",),
            regressors=(regressor,),
            train_known_future_regressors=True,
            weighted_ensemble=False,
            verbose=False,
        ),
    )
    output_dir = run.to_directory(tmp_path / "run")

    feature_gate = run.driver_model_features.set_index("name")
    assert feature_gate.loc["Seats plan", "status"] == "included"
    assert feature_gate.loc["Seats plan", "modeling_decision"] == "included_mlforecast_model_candidate"
    assert "seats_plan" in set(run.model_explainability["feature"])
    assert not run.driver_model_cv_delta.empty
    assert (output_dir / "appendix" / "driver_model_features.csv").exists()
    assert (output_dir / "appendix" / "driver_model_cv_delta.csv").exists()
    workbook = pd.ExcelFile(output_dir / "forecast.xlsx")
    assert "Driver Model Features" in workbook.sheet_names
    assert "Driver Model CV Delta" in workbook.sheet_names


def test_historical_only_regressor_lags_feed_mlforecast_without_future_file(tmp_path) -> None:
    pytest.importorskip("mlforecast")
    regressor = KnownFutureRegressor(
        name="Historical seats",
        value_col="seats_plan",
        availability="historical_only",
        mode="model_candidate",
    )

    run = run_forecast(
        _ml_history_frame(40),
        ForecastSpec(
            horizon=2,
            freq="ME",
            levels=(),
            model_policy="mlforecast",
            model_allowlist=("LinearRegression",),
            regressors=(regressor,),
            train_known_future_regressors=True,
            weighted_ensemble=False,
            verbose=False,
        ),
    )
    output_dir = run.to_directory(tmp_path / "historical_lag_run")

    feature_gate = run.driver_model_features.set_index("name")
    assert feature_gate.loc["Historical seats", "status"] == "included"
    assert feature_gate.loc["Historical seats", "modeling_decision"] == "included_mlforecast_historical_lag_candidate"
    assert "seats_plan_lag_2" in feature_gate.loc["Historical seats", "feature_columns"]
    assert feature_gate.loc["Historical seats", "future_rows"] == 0
    assert "seats_plan_lag_2" in set(run.model_explainability["feature"].astype(str))
    receipts = pd.read_csv(output_dir / "appendix" / "feature_selection_receipts.csv")
    receipt = receipts.set_index("feature").loc["seats_plan_lag_2"]
    assert receipt["feature_role"] == "historical_only_lag_regressor"
    assert receipt["final_decision"] == "included_and_seen_in_model"


def test_known_future_regressor_audit_fails_closed_for_leakage_and_future_contracts(tmp_path) -> None:
    missing_future = tmp_path / "missing_future.csv"
    _future_driver_file(missing_future, periods=1)
    leakage_future = tmp_path / "leakage_future.csv"
    _future_driver_file(leakage_future, value_col="actual")
    broad_leakage_future = tmp_path / "broad_leakage_future.csv"
    _future_driver_file(broad_leakage_future, value_col="actual_revenue")
    late_future = tmp_path / "late_future.csv"
    _future_driver_file(late_future, known_as_of="2026-01-15")
    no_asof_future = tmp_path / "no_asof_future.csv"
    pd.DataFrame(
        {
            "unique_id": ["Revenue", "Revenue"],
            "ds": ["2026-01-31", "2026-02-28"],
            "seats_plan": [70, 71],
        }
    ).to_csv(no_asof_future, index=False)

    regressors = (
        KnownFutureRegressor(
            name="Missing plan",
            value_col="seats_plan",
            availability="plan",
            mode="model_candidate",
            future_file=str(missing_future),
        ),
        KnownFutureRegressor(
            name="Actual",
            value_col="actual",
            availability="plan",
            mode="model_candidate",
            future_file=str(leakage_future),
        ),
        KnownFutureRegressor(
            name="Revenue proxy",
            value_col="actual_revenue",
            availability="plan",
            mode="model_candidate",
            future_file=str(broad_leakage_future),
        ),
        KnownFutureRegressor(
            name="Historical seats",
            value_col="seats_plan",
            availability="historical_only",
            mode="model_candidate",
        ),
        KnownFutureRegressor(
            name="Late plan",
            value_col="seats_plan",
            availability="plan",
            mode="model_candidate",
            future_file=str(late_future),
        ),
        KnownFutureRegressor(
            name="No timing provenance",
            value_col="seats_plan",
            availability="plan",
            mode="model_candidate",
            future_file=str(no_asof_future),
        ),
    )

    run = run_forecast(
        _history_frame(),
        ForecastSpec(horizon=2, freq="ME", model_policy="baseline", regressors=regressors),
    )

    audit = run.driver_availability_audit.set_index("name")
    assert set(audit["audit_status"]) == {"failed", "passed"}
    assert "future values missing" in audit.loc["Missing plan", "audit_message"]
    assert "target leakage" in audit.loc["Actual", "audit_message"]
    assert "target leakage" in audit.loc["Revenue proxy", "audit_message"]
    assert audit.loc["Historical seats", "modeling_decision"] == "candidate_audited_not_trained"
    assert audit.loc["Historical seats", "future_scope"] == "not_required_historical_only"
    assert "after the forecast origin" in audit.loc["Late plan", "audit_message"]
    assert "missing known_as_of_col" in audit.loc["No timing provenance", "audit_message"]


def test_cli_event_file_and_regressor_declaration_write_driver_artifacts(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.csv"
    _history_frame(periods=8).to_csv(input_path, index=False)
    event_file = tmp_path / "events.csv"
    event_file.write_text("name,start,effect,magnitude\nLaunch,2025-09-30,additive,5\n", encoding="utf-8")
    future_file = tmp_path / "future.csv"
    pd.DataFrame(
        {
            "unique_id": ["Revenue"],
            "ds": ["2025-09-30"],
            "seats_plan": [60],
            "known_as_of": ["2025-08-15"],
        }
    ).to_csv(future_file, index=False)
    output_dir = tmp_path / "cli_run"

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
            "--event-file",
            str(event_file),
            "--regressor",
            json.dumps(
                {
                    "name": "Seats plan",
                    "value_col": "seats_plan",
                    "availability": "plan",
                    "mode": "model_candidate",
                    "future_file": str(future_file),
                }
            ),
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    appendix = output_dir / "appendix"
    assert (appendix / "scenario_assumptions.csv").exists()
    assert (appendix / "scenario_forecast.csv").exists()
    assert (appendix / "known_future_regressors.csv").exists()
    assert (appendix / "driver_availability_audit.csv").exists()
    assert (appendix / "driver_experiment_summary.csv").exists()
    assert "Forecast written to" in capsys.readouterr().out
