from __future__ import annotations

import json
import py_compile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.hierarchy import aggregate_hierarchy_frame
from nixtla_scaffold.schema import CustomModelSpec, ForecastSpec, TransformSpec


GOLDEN_SCENARIOS = (
    "monthly_basic",
    "limited_history_new_product",
    "hierarchy_reconciled",
    "normalized_target_forecast",
    "custom_model_challenger",
)
SCENARIO_ALIASES = {
    "short_history": "limited_history_new_product",
    "transform_normalized": "normalized_target_forecast",
}
WORKBENCH_QA_SCENARIOS = GOLDEN_SCENARIOS + tuple(SCENARIO_ALIASES)
SCENARIO_DESCRIPTIONS = {
    "monthly_basic": "Flagship monthly single-series finance forecast.",
    "hierarchy_reconciled": "Flagship reconciled hierarchy forecast for parent/child finance rollups.",
    "limited_history_new_product": "Guardrail scenario for new products or metrics with limited history.",
    "normalized_target_forecast": "Guardrail scenario for pricing, FX, inflation, or definition-change normalization.",
    "custom_model_challenger": "Guardrail scenario for an opt-in custom finance model challenger.",
}
APP_TEST_TIMEOUT_SECONDS = 90


@dataclass(frozen=True)
class WorkbenchQAResult:
    scenario: str
    status: str
    usability_score: int
    run_dir: str
    rows: int
    series_count: int
    selected_models: str
    artifacts_status: str
    streamlit_compile_status: str
    app_test_status: str
    app_test_timeout_seconds: int
    required_text_missing: str
    notes: str


def run_workbench_qa(
    *,
    output_dir: str | Path = "runs/workbench_qa",
    scenarios: Iterable[str] | None = None,
    model_policy: str = "baseline",
    app_test: bool = True,
    app_test_timeout_seconds: int = APP_TEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generate and validate golden forecast workbench runs."""

    if app_test_timeout_seconds < 1:
        raise ValueError("app_test_timeout_seconds must be >= 1")
    requested = tuple(scenarios or GOLDEN_SCENARIOS)
    invalid = [name for name in requested if name not in WORKBENCH_QA_SCENARIOS]
    if invalid:
        raise ValueError(f"unknown workbench QA scenario(s): {invalid}; valid values are {list(WORKBENCH_QA_SCENARIOS)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [
        _run_one_workbench_scenario(
            name,
            output_dir=out / name,
            model_policy=model_policy,
            app_test=app_test,
            app_test_timeout_seconds=app_test_timeout_seconds,
        )
        for name in requested
    ]
    frame = pd.DataFrame(asdict(result) for result in results)
    summary = _summary_payload(
        frame,
        output_dir=out,
        model_policy=model_policy,
        app_test=app_test,
        app_test_timeout_seconds=app_test_timeout_seconds,
    )
    frame.to_csv(out / "workbench_qa_summary.csv", index=False)
    (out / "workbench_qa_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return {"summary": summary, "results": frame.to_dict("records"), "output_dir": str(out)}


def _run_one_workbench_scenario(
    scenario: str,
    *,
    output_dir: Path,
    model_policy: str,
    app_test: bool,
    app_test_timeout_seconds: int,
) -> WorkbenchQAResult:
    try:
        canonical_scenario = _canonical_scenario(scenario)
        data, spec = _scenario_payload(canonical_scenario, model_policy=model_policy)
        run = run_forecast(data, spec)
        run_dir = run.to_directory(output_dir)
        artifacts_status, missing_artifacts = _check_required_artifacts(run_dir, canonical_scenario)
        compile_status = _compile_streamlit(run_dir / "streamlit_app.py")
        app_test_status = (
            _app_test_streamlit(run_dir / "streamlit_app.py", timeout_seconds=app_test_timeout_seconds)
            if app_test
            else "skipped"
        )
        missing_text = _missing_required_streamlit_text(run_dir / "streamlit_app.py", canonical_scenario)
        score = _usability_score(
            artifacts_status=artifacts_status,
            compile_status=compile_status,
            app_test_status=app_test_status,
            missing_text=missing_text,
        )
        status = "passed" if score >= 90 else "failed"
        notes = _notes(missing_artifacts=missing_artifacts, missing_text=missing_text)
        selected_models = "; ".join(sorted(run.model_selection["selected_model"].dropna().astype(str).unique()))
        return WorkbenchQAResult(
            scenario=scenario,
            status=status,
            usability_score=score,
            run_dir=str(run_dir),
            rows=int(run.profile.rows),
            series_count=int(run.profile.series_count),
            selected_models=selected_models,
            artifacts_status=artifacts_status,
            streamlit_compile_status=compile_status,
            app_test_status=app_test_status,
            app_test_timeout_seconds=app_test_timeout_seconds,
            required_text_missing="; ".join(missing_text),
            notes=notes,
        )
    except Exception as exc:
        return WorkbenchQAResult(
            scenario=scenario,
            status="failed",
            usability_score=0,
            run_dir=str(output_dir),
            rows=0,
            series_count=0,
            selected_models="",
            artifacts_status="failed",
            streamlit_compile_status="not_run",
            app_test_status="not_run",
            app_test_timeout_seconds=app_test_timeout_seconds,
            required_text_missing="",
            notes=f"{type(exc).__name__}: {exc}",
        )


def _scenario_payload(scenario: str, *, model_policy: str) -> tuple[pd.DataFrame, ForecastSpec]:
    scenario = _canonical_scenario(scenario)
    if scenario == "monthly_basic":
        return _monthly_basic_frame(), ForecastSpec(horizon=3, freq="ME", model_policy=model_policy, verbose=False)
    if scenario == "limited_history_new_product":
        return _short_history_frame(), ForecastSpec(horizon=2, freq="ME", model_policy="baseline", verbose=False)
    if scenario == "hierarchy_reconciled":
        return _hierarchy_frame(), ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            hierarchy_reconciliation="bottom_up",
            verbose=False,
        )
    if scenario == "normalized_target_forecast":
        return _normalized_frame(), ForecastSpec(
            horizon=3,
            freq="ME",
            model_policy=model_policy,
            transform=TransformSpec(normalization_factor_col="price_factor", normalization_label="QA price index"),
            verbose=False,
        )
    if scenario == "custom_model_challenger":
        return _custom_model_frame(), ForecastSpec(
            horizon=2,
            freq="ME",
            model_policy="baseline",
            verbose=False,
            custom_models=(CustomModelSpec(name="QA finance challenger", callable=_qa_custom_last_value),),
        )
    raise ValueError(f"unknown workbench QA scenario: {scenario}")


def _monthly_basic_frame() -> pd.DataFrame:
    t = np.arange(36)
    y = 100 + 2.4 * t + 12 * np.sin(2 * np.pi * t / 12)
    return pd.DataFrame({"unique_id": "Revenue", "ds": pd.date_range("2023-01-31", periods=36, freq="ME"), "y": y})


def _short_history_frame() -> pd.DataFrame:
    t = np.arange(8)
    y = 50 + 3.0 * t
    return pd.DataFrame({"unique_id": "NewProduct", "ds": pd.date_range("2025-01-31", periods=8, freq="ME"), "y": y})


def _hierarchy_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-31", periods=18, freq="ME")
    rows: list[dict[str, Any]] = []
    for idx, ds in enumerate(dates):
        seasonal = 5 * np.sin(2 * np.pi * idx / 12)
        for region, offset in (("NA", 80), ("EMEA", 60)):
            for product, product_offset in (("ProductA", 12), ("ProductB", -8)):
                rows.append(
                    {
                        "region": region,
                        "product": product,
                        "month": ds,
                        "revenue": offset + product_offset + 1.5 * idx + seasonal,
                    }
                )
    return aggregate_hierarchy_frame(pd.DataFrame(rows), hierarchy_cols=("region", "product"), time_col="month", target_col="revenue")


def _normalized_frame() -> pd.DataFrame:
    t = np.arange(30)
    price_factor = np.where(t < 18, 1.0, 1.12)
    underlying = 100 + 1.8 * t + 6 * np.sin(2 * np.pi * t / 12)
    reported = underlying * price_factor
    return pd.DataFrame(
        {
            "unique_id": "Seats",
            "ds": pd.date_range("2023-01-31", periods=30, freq="ME"),
            "y": reported,
            "price_factor": price_factor,
        }
    )


def _custom_model_frame() -> pd.DataFrame:
    t = np.arange(18)
    y = 100 + 3.0 * t + np.where(t % 6 == 5, 10.0, 0.0)
    return pd.DataFrame({"unique_id": "CustomQA", "ds": pd.date_range("2024-01-31", periods=18, freq="ME"), "y": y})


def _qa_custom_last_value(
    history: pd.DataFrame,
    *,
    horizon: int,
    freq: str,
    cutoff: pd.Timestamp,
    levels: tuple[int, ...],
    context: dict[str, Any],
) -> pd.DataFrame:
    grid = pd.DataFrame(context["future_grid"])
    last = history.sort_values("ds").groupby("unique_id")["y"].last()
    grid["yhat"] = grid["unique_id"].astype(str).map(last)
    return grid[["unique_id", "ds", "yhat"]]


def _check_required_artifacts(run_dir: Path, scenario: str) -> tuple[str, list[str]]:
    scenario = _canonical_scenario(scenario)
    required = [
        "forecast.csv",
        "forecast_long.csv",
        "backtest_long.csv",
        "series_summary.csv",
        "model_audit.csv",
        "trust_summary.csv",
        "diagnostics.json",
        "llm_context.json",
        "model_card.md",
        "report.html",
        "report_base64.txt",
        "streamlit_app.py",
        "forecast.xlsx",
        "audit/model_selection.csv",
        "audit/backtest_metrics.csv",
        "audit/seasonality_diagnostics.csv",
    ]
    if scenario == "hierarchy_reconciled":
        required.extend(
            [
                "hierarchy_reconciliation.csv",
                "audit/hierarchy_unreconciled_forecast.csv",
                "audit/hierarchy_coherence_pre.csv",
                "audit/hierarchy_coherence_post.csv",
            ]
        )
    if scenario == "normalized_target_forecast":
        required.append("audit/target_transform_audit.csv")
    if scenario == "custom_model_challenger":
        required.extend(["custom_model_contracts.csv", "audit/custom_model_invocations.csv"])
    missing = [path for path in required if not (run_dir / path).exists()]
    return ("passed" if not missing else "failed", missing)


def _compile_streamlit(app_path: Path) -> str:
    try:
        py_compile.compile(str(app_path), doraise=True)
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"
    return "passed"


def _app_test_streamlit(app_path: Path, *, timeout_seconds: int = APP_TEST_TIMEOUT_SECONDS) -> str:
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        return "skipped_missing_streamlit"
    try:
        app = AppTest.from_file(str(app_path), default_timeout=timeout_seconds)
        app.run()
        if app.exception:
            return f"failed: {app.exception}"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"
    return "passed"


def _missing_required_streamlit_text(app_path: Path, scenario: str) -> list[str]:
    scenario = _canonical_scenario(scenario)
    text = app_path.read_text(encoding="utf-8") if app_path.exists() else ""
    required = [
        "Decision summary",
        "Trust level",
        "Unvalidated steps",
        "Horizon score cap",
        "planning_eligible is a horizon-validation flag only",
        "Model investigation",
        "Menu labels use `#rank | model | engine`",
        "Model picker guide: rank, engine, and role",
        "Interval model picker guide: rank and engine",
        "Focused future forecast interval ownership",
        "Point forecasts and bands come from the same `forecast_long.csv` model feed",
        "Model feed columns keep `yhat`",
        "First-glance chart includes interval bands",
        "CV window player",
        "Prediction intervals",
        "Models with interval bands",
        "Seasonality",
        "Seasonal year overlay",
        "Feeder outputs",
    ]
    if scenario == "hierarchy_reconciled":
        required.extend(["Hierarchy reconciliation is enabled", "Pre/post reconciliation gap audit"])
    if scenario == "normalized_target_forecast":
        required.append("Target transformation audit")
    return [needle for needle in required if needle not in text]


def _canonical_scenario(scenario: str) -> str:
    return SCENARIO_ALIASES.get(scenario, scenario)


def _usability_score(
    *,
    artifacts_status: str,
    compile_status: str,
    app_test_status: str,
    missing_text: list[str],
) -> int:
    score = 100
    if artifacts_status != "passed":
        score -= 35
    if compile_status != "passed":
        score -= 25
    if app_test_status not in {"passed", "skipped"}:
        score -= 25
    score -= min(15, 3 * len(missing_text))
    return max(0, score)


def _notes(*, missing_artifacts: list[str], missing_text: list[str]) -> str:
    notes: list[str] = []
    if missing_artifacts:
        notes.append(f"missing artifacts: {', '.join(missing_artifacts)}")
    if missing_text:
        notes.append(f"missing Streamlit text: {', '.join(missing_text)}")
    return "; ".join(notes)


def _summary_payload(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    model_policy: str,
    app_test: bool,
    app_test_timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "count": int(len(frame)),
        "passed": int((frame["status"] == "passed").sum()),
        "failed": int((frame["status"] == "failed").sum()),
        "min_usability_score": int(frame["usability_score"].min()) if not frame.empty else 0,
        "mean_usability_score": round(float(frame["usability_score"].mean()), 2) if not frame.empty else 0.0,
        "model_policy": model_policy,
        "app_test": bool(app_test),
        "app_test_timeout_seconds": int(app_test_timeout_seconds),
        "output_dir": str(output_dir),
        "scenarios": frame["scenario"].tolist(),
    }
