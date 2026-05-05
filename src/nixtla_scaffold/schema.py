from __future__ import annotations

import hashlib
from importlib import metadata
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from nixtla_scaffold.model_families import MODEL_ALLOWLIST_CANDIDATES, canonicalize_model_allowlist

FillMethod = Literal["ffill", "zero", "interpolate", "drop"]
ModelPolicy = Literal["auto", "baseline", "statsforecast", "mlforecast", "all"]
EventEffect = Literal["additive", "multiplicative"]
TargetTransform = Literal["none", "log", "log1p"]
HierarchyReconciliationMethod = Literal["none", "bottom_up", "mint_ols", "mint_wls_struct"]
RegressorAvailability = Literal["calendar", "contracted", "scheduled", "plan", "forecasted", "historical_only"]
RegressorMode = Literal["audit_only", "model_candidate"]


@dataclass(frozen=True)
class DriverEvent:
    """Auditable event or driver assumption applied after the statistical baseline."""

    name: str
    start: str
    end: str | None = None
    effect: EventEffect = "multiplicative"
    magnitude: float = 0.0
    affected_unique_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("driver event name is required")
        if self.effect not in {"additive", "multiplicative"}:
            raise ValueError("driver event effect must be 'additive' or 'multiplicative'")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("driver event confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["affected_unique_ids"] = list(self.affected_unique_ids)
        return data


@dataclass(frozen=True)
class KnownFutureRegressor:
    """Declared candidate model input with future-value and leakage audit metadata.

    Current scaffold releases audit this contract but do not automatically train
    StatsForecast or MLForecast candidates with arbitrary external regressors.
    """

    name: str
    value_col: str | None = None
    availability: RegressorAvailability = "historical_only"
    mode: RegressorMode = "audit_only"
    future_file: str | None = None
    known_as_of_col: str = "known_as_of"
    source_system: str = ""
    source_query_file: str = ""
    owner: str = ""
    refresh_latency_days: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("known-future regressor name is required")
        if self.value_col is not None and not str(self.value_col).strip():
            raise ValueError("known-future regressor value_col cannot be blank")
        if self.future_file is not None and not str(self.future_file).strip():
            raise ValueError("known-future regressor future_file cannot be blank")
        if not str(self.known_as_of_col).strip():
            raise ValueError("known-future regressor known_as_of_col cannot be blank")
        if self.availability not in {"calendar", "contracted", "scheduled", "plan", "forecasted", "historical_only"}:
            raise ValueError("known-future regressor availability must be one of: calendar, contracted, scheduled, plan, forecasted, historical_only")
        if self.mode not in {"audit_only", "model_candidate"}:
            raise ValueError("known-future regressor mode must be 'audit_only' or 'model_candidate'")
        if self.refresh_latency_days is not None and self.refresh_latency_days < 0:
            raise ValueError("known-future regressor refresh_latency_days must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformSpec:
    """Finance target preparation before modeling."""

    target: TargetTransform = "none"
    normalization_factor_col: str | None = None
    normalization_label: str = ""

    def __post_init__(self) -> None:
        if self.target not in {"none", "log", "log1p"}:
            raise ValueError("target transform must be one of: none, log, log1p")
        if self.normalization_factor_col is not None and not str(self.normalization_factor_col).strip():
            raise ValueError("normalization_factor_col cannot be blank")

    @property
    def enabled(self) -> bool:
        return self.target != "none" or self.normalization_factor_col is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustomModelSpec:
    """Opt-in executable model challenger with a strict forecast output contract."""

    name: str
    callable_path: str | None = None
    callable: Callable[..., pd.DataFrame] | None = field(default=None, repr=False, compare=False)
    script_path: str | None = None
    timeout_seconds: int = 120
    extra_args: tuple[str, ...] = ()
    source_id: str = "custom"
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("custom model name is required")
        invocation_count = sum(
            [
                self.callable_path is not None,
                self.callable is not None,
                self.script_path is not None,
            ]
        )
        if invocation_count != 1:
            raise ValueError("custom model requires exactly one of callable_path, callable, or script_path")
        if self.callable_path is not None and not str(self.callable_path).strip():
            raise ValueError("custom model callable_path cannot be blank")
        if self.script_path is not None and not str(self.script_path).strip():
            raise ValueError("custom model script_path cannot be blank")
        if self.timeout_seconds < 1:
            raise ValueError("custom model timeout_seconds must be >= 1")
        if not str(self.source_id).strip():
            raise ValueError("custom model source_id cannot be blank")
        if self.model_name == "WeightedEnsemble" or "-lo-" in self.model_name or "-hi-" in self.model_name:
            raise ValueError("custom model name conflicts with reserved forecast column naming")
        blank_args = [arg for arg in self.extra_args if not str(arg).strip()]
        if blank_args:
            raise ValueError("custom model extra_args cannot contain blank values")

    @property
    def invocation_type(self) -> str:
        if self.script_path is not None:
            return "script"
        if self.callable_path is not None:
            return "callable_path"
        return "callable_object"

    @property
    def model_name(self) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(self.name).strip()).strip("_")
        if not cleaned:
            cleaned = "model"
        return cleaned if cleaned.startswith("Custom_") else f"Custom_{cleaned}"

    def to_dict(self) -> dict[str, Any]:
        callable_ref = ""
        if self.callable is not None:
            callable_ref = f"{getattr(self.callable, '__module__', '')}.{getattr(self.callable, '__qualname__', getattr(self.callable, '__name__', 'callable'))}".strip(".")
        return {
            "name": self.name,
            "model_name": self.model_name,
            "invocation_type": self.invocation_type,
            "callable_path": self.callable_path,
            "callable_object_ref": callable_ref,
            "script_path": self.script_path,
            "timeout_seconds": self.timeout_seconds,
            "extra_args": list(self.extra_args),
            "source_id": self.source_id,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ForecastSpec:
    """User-facing forecast settings with safe defaults."""

    horizon: int = 12
    freq: str | None = None
    season_length: int | None = None
    levels: tuple[int, ...] = (80, 95)
    model_policy: ModelPolicy = "auto"
    fill_method: FillMethod = "ffill"
    id_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"
    unit_label: str | None = None
    model_allowlist: tuple[str, ...] = ()
    hierarchy: tuple[str, ...] = ()
    events: tuple[DriverEvent, ...] = field(default_factory=tuple)
    regressors: tuple[KnownFutureRegressor, ...] = field(default_factory=tuple)
    custom_models: tuple[CustomModelSpec, ...] = field(default_factory=tuple)
    transform: TransformSpec = field(default_factory=TransformSpec)
    hierarchy_reconciliation: HierarchyReconciliationMethod = "none"
    require_backtest: bool = False
    strict_cv_horizon: bool = False
    weighted_ensemble: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.season_length is not None and self.season_length < 1:
            raise ValueError("season_length must be >= 1")
        invalid_levels = [level for level in self.levels if level <= 0 or level >= 100]
        if invalid_levels:
            raise ValueError(f"interval levels must be between 0 and 100, got {invalid_levels}")
        if self.unit_label is not None and not str(self.unit_label).strip():
            raise ValueError("unit_label cannot be blank")
        canonical_models, unknown_models = canonicalize_model_allowlist(tuple(self.model_allowlist))
        if unknown_models:
            unknown = ", ".join(repr(model) for model in unknown_models)
            valid = ", ".join(sorted(MODEL_ALLOWLIST_CANDIDATES))
            raise ValueError(f"unknown model_allowlist entr{'y' if len(unknown_models) == 1 else 'ies'}: {unknown}. Known models: {valid}")
        object.__setattr__(self, "model_allowlist", canonical_models)
        if self.hierarchy_reconciliation not in {"none", "bottom_up", "mint_ols", "mint_wls_struct"}:
            raise ValueError("hierarchy_reconciliation must be one of: none, bottom_up, mint_ols, mint_wls_struct")
        if len(self.custom_models) > 1:
            raise ValueError("custom model v1 supports at most one custom model per run")
        custom_names = [custom.model_name for custom in self.custom_models]
        if len(custom_names) != len(set(custom_names)):
            raise ValueError("custom model names must be unique")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["levels"] = list(self.levels)
        data["model_allowlist"] = list(self.model_allowlist)
        data["hierarchy"] = list(self.hierarchy)
        data["events"] = [event.to_dict() for event in self.events]
        data["regressors"] = [regressor.to_dict() for regressor in self.regressors]
        if self.custom_models:
            data["custom_models"] = [custom.to_dict() for custom in self.custom_models]
        else:
            data.pop("custom_models", None)
        data["transform"] = self.transform.to_dict()
        return data


@dataclass(frozen=True)
class SeriesProfile:
    unique_id: str
    rows: int
    start: str | None
    end: str | None
    missing_timestamps: int
    null_y: int
    zero_y: int
    negative_y: int
    readiness: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataProfile:
    rows: int
    series_count: int
    freq: str
    season_length: int
    start: str | None
    end: str | None
    min_obs_per_series: int
    max_obs_per_series: int
    duplicate_rows: int
    missing_timestamps: int
    null_y: int
    zero_y: int
    negative_y: int
    data_freshness: str | None
    warnings: tuple[str, ...] = ()
    series: tuple[SeriesProfile, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["series"] = [profile.to_dict() for profile in self.series]
        return data


@dataclass
class ForecastRun:
    """Forecast result plus audit artifacts."""

    history: pd.DataFrame
    forecast: pd.DataFrame
    all_models: pd.DataFrame
    model_selection: pd.DataFrame
    backtest_metrics: pd.DataFrame
    profile: DataProfile
    spec: ForecastSpec
    backtest_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_explainability: pd.DataFrame = field(default_factory=pd.DataFrame)
    transformation_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_availability_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_invocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    unreconciled_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy_reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    engine: str = "baseline"
    model_policy_resolution: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        outputs = {
            "history": "history.csv",
            "forecast": "forecast.csv",
            "forecast_long": "forecast_long.csv",
            "backtest_long": "backtest_long.csv",
            "series_summary": "series_summary.csv",
            "model_audit": "model_audit.csv",
            "model_win_rates": "model_win_rates.csv",
            "model_window_metrics": "model_window_metrics.csv",
            "residual_diagnostics": "residual_diagnostics.csv",
            "residual_tests": "residual_tests.csv",
            "interval_diagnostics": "interval_diagnostics.csv",
            "trust_summary": "trust_summary.csv",
            "model_explainability": "model_explainability.csv",
            "all_models": "audit/all_models.csv",
            "model_selection": "audit/model_selection.csv",
            "backtest_metrics": "audit/backtest_metrics.csv",
            "backtest_predictions": "audit/backtest_predictions.csv",
            "backtest_windows": "audit/backtest_windows.csv",
            "model_weights": "audit/model_weights.csv",
            "target_transform_audit": "audit/target_transform_audit.csv",
            "seasonality_profile": "audit/seasonality_profile.csv",
            "seasonality_summary": "audit/seasonality_summary.csv",
            "seasonality_diagnostics": "audit/seasonality_diagnostics.csv",
            "seasonality_decomposition": "audit/seasonality_decomposition.csv",
            "interpretation": "audit/interpretation.json",
            "interpretation_markdown": "interpretation.md",
            "output_open_first": "OPEN_ME_FIRST.html",
            "output_index": "output/index.html",
            "output_workbook": "output/forecast_review.xlsx",
            "output_forecast": "output/forecast_for_review.csv",
            "output_decision_summary": "output/decision_summary.csv",
            "output_model_leaderboard": "output/appendix/model_leaderboard.csv",
            "output_forecast_brief": "output/appendix/forecast_brief.csv",
            "output_artifact_guide": "output/appendix/artifact_guide.csv",
            "html_report": "report.html",
            "html_report_base64": "report_base64.txt",
            "streamlit_app": "streamlit_app.py",
            "best_practice_receipts": "best_practice_receipts.csv",
            "diagnostics": "diagnostics.json",
            "diagnostics_markdown": "diagnostics.md",
            "llm_context": "llm_context.json",
            "model_card": "model_card.md",
            "workbook": "forecast.xlsx",
        }
        if "hierarchy_depth" in self.forecast.columns:
            outputs["hierarchy_coherence"] = "hierarchy_coherence.csv"
            outputs["hierarchy_contribution"] = "hierarchy_contribution.csv"
            outputs["hierarchy_backtest_comparison"] = "audit/hierarchy_backtest_comparison.csv"
            if not self.unreconciled_forecast.empty:
                outputs["hierarchy_unreconciled_forecast"] = "audit/hierarchy_unreconciled_forecast.csv"
                outputs["hierarchy_coherence_pre"] = "audit/hierarchy_coherence_pre.csv"
                outputs["hierarchy_coherence_post"] = "audit/hierarchy_coherence_post.csv"
            if not self.hierarchy_reconciliation.empty:
                outputs["hierarchy_reconciliation"] = "hierarchy_reconciliation.csv"
        if self.spec.events:
            outputs["scenario_assumptions"] = "scenario_assumptions.csv"
            outputs["scenario_forecast"] = "scenario_forecast.csv"
            outputs["driver_experiment_summary"] = "driver_experiment_summary.csv"
        if self.spec.regressors or not self.driver_availability_audit.empty:
            outputs["known_future_regressors"] = "known_future_regressors.csv"
            outputs["driver_availability_audit"] = "driver_availability_audit.csv"
            outputs["driver_experiment_summary"] = "driver_experiment_summary.csv"
        if self.spec.custom_models or not self.custom_model_contracts.empty or not self.custom_model_invocations.empty:
            outputs["custom_model_contracts"] = "custom_model_contracts.csv"
            outputs["custom_model_invocations"] = "audit/custom_model_invocations.csv"
        return {
            "engine": self.engine,
            "spec": self.spec.to_dict(),
            "effective_levels": self.effective_levels(),
            "profile": self.profile.to_dict(),
            "reproducibility": _reproducibility_metadata(self),
            "warnings": self.warnings,
            "model_policy_resolution": self.model_policy_resolution,
            "best_practice_receipts": self.best_practice_receipts(),
            "outputs": outputs,
        }

    def effective_levels(self) -> list[int]:
        levels = []
        for column in self.forecast.columns:
            if column.startswith("yhat_lo_"):
                levels.append(int(column.rsplit("_", 1)[-1]))
        return sorted(set(levels))

    def best_practice_receipts(self) -> list[dict[str, str]]:
        from nixtla_scaffold.best_practices import best_practice_receipts

        return best_practice_receipts(self)

    def diagnostics(self) -> dict[str, Any]:
        from nixtla_scaffold.diagnostics import build_run_diagnostics

        return build_run_diagnostics(self)

    def interpretation(self) -> dict[str, Any]:
        from nixtla_scaffold.interpretation import build_interpretation_payload

        return build_interpretation_payload(self)

    def explanation(self) -> str:
        from nixtla_scaffold.explain import build_model_card

        return build_model_card(self)

    def to_directory(self, output_dir: str | Path) -> Path:
        from nixtla_scaffold.outputs import write_run

        return write_run(self, output_dir)

    def to_excel(self, output_path: str | Path) -> Path:
        from nixtla_scaffold.outputs import write_workbook

        return write_workbook(self, output_path)


def _reproducibility_metadata(run: ForecastRun) -> dict[str, Any]:
    return {
        "data_hash_sha256": _frame_hash(run.history, columns=["unique_id", "ds", "y"]),
        "forecast_origin": run.profile.end,
        "frequency": run.profile.freq,
        "season_length": run.profile.season_length,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(
            [
                "nixtla-scaffold",
                "pandas",
                "numpy",
                "statsforecast",
                "utilsforecast",
                "mlforecast",
                "lightgbm",
                "hierarchicalforecast",
                "streamlit",
                "plotly",
            ]
        ),
        "git_sha": _git_sha(),
    }


def _frame_hash(frame: pd.DataFrame, *, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return ""
    canonical = frame[available].copy()
    if "ds" in canonical.columns:
        canonical["ds"] = pd.to_datetime(canonical["ds"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    canonical = canonical.sort_values(available).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip()
    return sha or None

