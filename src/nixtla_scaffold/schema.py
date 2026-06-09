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
from typing import Any, Literal, Sequence

import pandas as pd

from nixtla_scaffold.model_families import MODEL_ALLOWLIST_CANDIDATES, canonicalize_model_allowlist

FillMethod = Literal["ffill", "zero", "interpolate", "drop"]
ModelPolicy = Literal["standard", "light", "auto", "baseline", "statsforecast", "mlforecast", "all"]
EventEffect = Literal["additive", "multiplicative"]
TargetTransform = Literal["none", "log", "log1p"]
HierarchyReconciliationMethod = Literal["none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"]
RegressorAvailability = Literal["calendar", "contracted", "scheduled", "plan", "forecasted", "historical_only"]
RegressorMode = Literal["audit_only", "model_candidate"]
MLForecastFeaturePolicy = Literal["basic", "rolling"]
EnsemblePolicy = Literal["legacy_weighted", "top_k_average", "family_diverse_average"]
EnsembleScoringMode = Literal["prior_only"]
EnsembleDeploymentMode = Literal["full_backtest", "last_cutoff"]
ParallelProcessingMode = Literal["none", "local_machine", "spark"]
ChallengerOnError = Literal["skip", "fail"]


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

    Scaffold releases audit this contract by default. MLForecast can use passing
    model candidates only when opt-in regressor training is enabled.
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
class FeatureRecipeSpec:
    """FINN-inspired feature recipe metadata for audited native experiments."""

    fiscal_year_start: int = 1
    fourier_periods: tuple[int, ...] = ()
    lag_periods: tuple[int, ...] = ()
    rolling_window_periods: tuple[int, ...] = ()
    recipes_to_run: tuple[str, ...] = ()
    pca: bool | None = None
    feature_selection: bool = False
    weekly_to_daily: bool = True

    def __post_init__(self) -> None:
        if self.fiscal_year_start < 1 or self.fiscal_year_start > 12:
            raise ValueError("feature_recipe.fiscal_year_start must be between 1 and 12")
        object.__setattr__(self, "fourier_periods", _positive_int_tuple(self.fourier_periods, "feature_recipe.fourier_periods"))
        object.__setattr__(self, "lag_periods", _positive_int_tuple(self.lag_periods, "feature_recipe.lag_periods"))
        object.__setattr__(
            self,
            "rolling_window_periods",
            _positive_int_tuple(self.rolling_window_periods, "feature_recipe.rolling_window_periods"),
        )
        object.__setattr__(self, "recipes_to_run", tuple(str(item).strip() for item in self.recipes_to_run if str(item).strip()))

    @property
    def enabled(self) -> bool:
        return bool(self.fourier_periods or self.lag_periods or self.rolling_window_periods or self.recipes_to_run or self.pca or self.feature_selection)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fourier_periods"] = list(self.fourier_periods)
        data["lag_periods"] = list(self.lag_periods)
        data["rolling_window_periods"] = list(self.rolling_window_periods)
        data["recipes_to_run"] = list(self.recipes_to_run)
        return data


@dataclass(frozen=True)
class CleaningSpec:
    """FINN-inspired data cleaning and forecast-bound controls."""

    clean_missing_values: bool = True
    clean_outliers: bool = False
    negative_forecast: bool = True
    combo_cleanup_date: str | None = None

    def __post_init__(self) -> None:
        if self.combo_cleanup_date is not None and not str(self.combo_cleanup_date).strip():
            raise ValueError("cleaning.combo_cleanup_date cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleSpec:
    """Audited ensemble policies inspired by FINN's model averaging workflow."""

    policies: tuple[EnsemblePolicy, ...] = ("legacy_weighted",)
    max_models: int = 3
    scoring: EnsembleScoringMode = "prior_only"
    deployment: EnsembleDeploymentMode = "full_backtest"

    def __post_init__(self) -> None:
        allowed = {"legacy_weighted", "top_k_average", "family_diverse_average"}
        policies = tuple(dict.fromkeys(str(policy) for policy in self.policies))
        invalid = sorted(set(policies) - allowed)
        if invalid:
            raise ValueError(f"ensemble.policies contains unknown value(s): {invalid}; expected one of {sorted(allowed)}")
        if self.max_models < 1:
            raise ValueError("ensemble.max_models must be >= 1")
        if self.scoring != "prior_only":
            raise ValueError("ensemble.scoring must be 'prior_only'")
        if self.deployment not in {"full_backtest", "last_cutoff"}:
            raise ValueError("ensemble.deployment must be 'full_backtest' or 'last_cutoff'")
        object.__setattr__(self, "policies", policies or ("legacy_weighted",))

    @property
    def advisory_policies(self) -> tuple[EnsemblePolicy, ...]:
        return tuple(policy for policy in self.policies if policy != "legacy_weighted")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policies"] = list(self.policies)
        return data


@dataclass(frozen=True)
class ParallelSpec:
    """Local/remote execution intent metadata; no cloud orchestration is implied."""

    processing: ParallelProcessingMode = "none"
    inner_parallel: bool = False
    num_cores: int | None = None

    def __post_init__(self) -> None:
        if self.processing not in {"none", "local_machine", "spark"}:
            raise ValueError("parallel.processing must be one of: none, local_machine, spark")
        if self.num_cores is not None and self.num_cores < 1:
            raise ValueError("parallel.num_cores must be >= 1 when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChallengerSpec:
    """External challenger engine run beside the canonical pipeline in an advisory lane.

    Challengers never mutate the canonical ``forecast.csv``; they contribute
    compare/score artifacts under ``<run>/<source_id>`` for apples-to-apples review.
    """

    engine: str = "finn"
    enabled: bool = True
    on_error: ChallengerOnError = "skip"
    models: tuple[str, ...] = ()
    back_test_scenarios: int | None = None
    back_test_spacing: int | None = None
    forecast_approach: str = "bottoms_up"
    run_ensemble_models: bool = False
    feature_selection: bool = False
    rscript: str = "Rscript"
    timeout_seconds: int = 3600
    seed: int = 123
    source_id: str = ""
    model_name: str = ""
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        engine = str(self.engine).strip().lower()
        if not engine:
            raise ValueError("challenger engine is required")
        object.__setattr__(self, "engine", engine)
        if self.on_error not in {"skip", "fail"}:
            raise ValueError("challenger on_error must be 'skip' or 'fail'")
        models = tuple(dict.fromkeys(str(model).strip() for model in self.models if str(model).strip()))
        object.__setattr__(self, "models", models)
        if self.back_test_scenarios is not None and self.back_test_scenarios < 1:
            raise ValueError("challenger back_test_scenarios must be >= 1 when provided")
        if self.back_test_spacing is not None and self.back_test_spacing < 1:
            raise ValueError("challenger back_test_spacing must be >= 1 when provided")
        if self.forecast_approach not in {"bottoms_up", "standard_hierarchy", "grouped_hierarchy"}:
            raise ValueError("challenger forecast_approach must be one of: bottoms_up, standard_hierarchy, grouped_hierarchy")
        if self.timeout_seconds < 1:
            raise ValueError("challenger timeout_seconds must be >= 1")
        if not str(self.rscript).strip():
            raise ValueError("challenger rscript cannot be blank")
        source_id = str(self.source_id).strip() or engine
        object.__setattr__(self, "source_id", source_id)
        model_name = str(self.model_name).strip() or engine.upper()
        object.__setattr__(self, "model_name", model_name)
        extra: list[tuple[str, str]] = []
        for item in self.extra:
            key, value = item
            key = str(key).strip()
            if not key:
                raise ValueError("challenger extra keys cannot be blank")
            extra.append((key, str(value)))
        object.__setattr__(self, "extra", tuple(extra))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "enabled": self.enabled,
            "on_error": self.on_error,
            "models": list(self.models),
            "back_test_scenarios": self.back_test_scenarios,
            "back_test_spacing": self.back_test_spacing,
            "forecast_approach": self.forecast_approach,
            "run_ensemble_models": self.run_ensemble_models,
            "feature_selection": self.feature_selection,
            "rscript": self.rscript,
            "timeout_seconds": self.timeout_seconds,
            "seed": self.seed,
            "source_id": self.source_id,
            "model_name": self.model_name,
            "extra": {key: value for key, value in self.extra},
        }


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
    model_policy: ModelPolicy = "light"
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
    challengers: tuple[ChallengerSpec, ...] = field(default_factory=tuple)
    transform: TransformSpec = field(default_factory=TransformSpec)
    feature_recipe: FeatureRecipeSpec = field(default_factory=FeatureRecipeSpec)
    cleaning: CleaningSpec = field(default_factory=CleaningSpec)
    ensemble: EnsembleSpec = field(default_factory=EnsembleSpec)
    parallel: ParallelSpec = field(default_factory=ParallelSpec)
    hierarchy_reconciliation: HierarchyReconciliationMethod = "none"
    train_known_future_regressors: bool = False
    mlforecast_feature_policy: MLForecastFeaturePolicy = "basic"
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
        model_policy = "light" if self.model_policy == "auto" else self.model_policy
        if model_policy not in {"standard", "light", "baseline", "statsforecast", "mlforecast", "all"}:
            raise ValueError("model_policy must be one of: standard, light, baseline, statsforecast, mlforecast, all")
        object.__setattr__(self, "model_policy", model_policy)
        canonical_models, unknown_models = canonicalize_model_allowlist(tuple(self.model_allowlist))
        if unknown_models:
            unknown = ", ".join(repr(model) for model in unknown_models)
            valid = ", ".join(sorted(MODEL_ALLOWLIST_CANDIDATES))
            raise ValueError(f"unknown model_allowlist entr{'y' if len(unknown_models) == 1 else 'ies'}: {unknown}. Known models: {valid}")
        object.__setattr__(self, "model_allowlist", canonical_models)
        if self.hierarchy_reconciliation not in {"none", "bottom_up", "top_down", "both", "mint_ols", "mint_wls_struct"}:
            raise ValueError("hierarchy_reconciliation must be one of: none, bottom_up, top_down, both, mint_ols, mint_wls_struct")
        if self.mlforecast_feature_policy not in {"basic", "rolling"}:
            raise ValueError("mlforecast_feature_policy must be one of: basic, rolling")
        if len(self.custom_models) > 1:
            raise ValueError("custom model v1 supports at most one custom model per run")
        custom_names = [custom.model_name for custom in self.custom_models]
        if len(custom_names) != len(set(custom_names)):
            raise ValueError("custom model names must be unique")
        challenger_engines = [challenger.engine for challenger in self.challengers]
        if len(challenger_engines) != len(set(challenger_engines)):
            raise ValueError("challenger engines must be unique")
        challenger_source_ids = [challenger.source_id for challenger in self.challengers]
        if len(challenger_source_ids) != len(set(challenger_source_ids)):
            raise ValueError("challenger source_ids must be unique")

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
        if self.challengers:
            data["challengers"] = [challenger.to_dict() for challenger in self.challengers]
        else:
            data.pop("challengers", None)
        data["transform"] = self.transform.to_dict()
        data["feature_recipe"] = self.feature_recipe.to_dict()
        data["cleaning"] = self.cleaning.to_dict()
        data["ensemble"] = self.ensemble.to_dict()
        data["parallel"] = self.parallel.to_dict()
        return data


def _positive_int_tuple(values: Sequence[Any], field_name: str) -> tuple[int, ...]:
    out = tuple(int(value) for value in values)
    invalid = [value for value in out if value < 1]
    if invalid:
        raise ValueError(f"{field_name} values must be positive integers, got {invalid}")
    return out


def forecast_spec_from_dict(data: dict[str, Any]) -> ForecastSpec:
    """Rehydrate a persisted manifest spec into a ForecastSpec."""

    transform_data = data.get("transform") if isinstance(data.get("transform"), dict) else {}
    feature_recipe_data = data.get("feature_recipe") if isinstance(data.get("feature_recipe"), dict) else {}
    cleaning_data = data.get("cleaning") if isinstance(data.get("cleaning"), dict) else {}
    ensemble_data = data.get("ensemble") if isinstance(data.get("ensemble"), dict) else {}
    parallel_data = data.get("parallel") if isinstance(data.get("parallel"), dict) else {}
    events = tuple(_driver_event_from_dict(item) for item in data.get("events", []) if isinstance(item, dict))
    regressors = tuple(_known_future_regressor_from_dict(item) for item in data.get("regressors", []) if isinstance(item, dict))
    custom_models = tuple(_custom_model_from_dict(item) for item in data.get("custom_models", []) if isinstance(item, dict))
    challengers = tuple(_challenger_from_dict(item) for item in data.get("challengers", []) if isinstance(item, dict))
    return ForecastSpec(
        horizon=int(data.get("horizon", 12)),
        freq=data.get("freq"),
        season_length=_optional_int(data.get("season_length")),
        levels=tuple(int(level) for level in data.get("levels", (80, 95))),
        model_policy=data.get("model_policy", "light"),
        fill_method=data.get("fill_method", "ffill"),
        id_col=data.get("id_col", "unique_id"),
        time_col=data.get("time_col", "ds"),
        target_col=data.get("target_col", "y"),
        unit_label=data.get("unit_label"),
        model_allowlist=tuple(data.get("model_allowlist", ())),
        hierarchy=tuple(data.get("hierarchy", ())),
        events=events,
        regressors=regressors,
        custom_models=custom_models,
        challengers=challengers,
        transform=TransformSpec(
            target=transform_data.get("target", "none"),
            normalization_factor_col=transform_data.get("normalization_factor_col"),
            normalization_label=transform_data.get("normalization_label", ""),
        ),
        feature_recipe=FeatureRecipeSpec(
            fiscal_year_start=int(feature_recipe_data.get("fiscal_year_start", 1)),
            fourier_periods=tuple(feature_recipe_data.get("fourier_periods", ())),
            lag_periods=tuple(feature_recipe_data.get("lag_periods", ())),
            rolling_window_periods=tuple(feature_recipe_data.get("rolling_window_periods", ())),
            recipes_to_run=tuple(feature_recipe_data.get("recipes_to_run", ())),
            pca=feature_recipe_data.get("pca"),
            feature_selection=bool(feature_recipe_data.get("feature_selection", False)),
            weekly_to_daily=bool(feature_recipe_data.get("weekly_to_daily", True)),
        ),
        cleaning=CleaningSpec(
            clean_missing_values=bool(cleaning_data.get("clean_missing_values", True)),
            clean_outliers=bool(cleaning_data.get("clean_outliers", False)),
            negative_forecast=bool(cleaning_data.get("negative_forecast", True)),
            combo_cleanup_date=str(cleaning_data["combo_cleanup_date"]) if cleaning_data.get("combo_cleanup_date") not in (None, "") else None,
        ),
        ensemble=EnsembleSpec(
            policies=tuple(ensemble_data.get("policies", ("legacy_weighted",))),
            max_models=int(ensemble_data.get("max_models", 3)),
            scoring=ensemble_data.get("scoring", "prior_only"),
            deployment=ensemble_data.get("deployment", "full_backtest"),
        ),
        parallel=ParallelSpec(
            processing=parallel_data.get("processing", "none"),
            inner_parallel=bool(parallel_data.get("inner_parallel", False)),
            num_cores=_optional_int(parallel_data.get("num_cores")),
        ),
        hierarchy_reconciliation=data.get("hierarchy_reconciliation", "none"),
        train_known_future_regressors=bool(data.get("train_known_future_regressors", False)),
        mlforecast_feature_policy=data.get("mlforecast_feature_policy", "basic"),
        require_backtest=bool(data.get("require_backtest", False)),
        strict_cv_horizon=bool(data.get("strict_cv_horizon", False)),
        weighted_ensemble=bool(data.get("weighted_ensemble", True)),
        verbose=bool(data.get("verbose", True)),
    )


def _driver_event_from_dict(data: dict[str, Any]) -> DriverEvent:
    affected = data.get("affected_unique_ids", ())
    if isinstance(affected, str):
        affected_tuple = tuple(part.strip() for part in affected.replace(";", ",").split(",") if part.strip())
    else:
        affected_tuple = tuple(str(item) for item in affected or ())
    return DriverEvent(
        name=str(data.get("name", "")),
        start=str(data.get("start", "")),
        end=str(data["end"]) if data.get("end") not in (None, "") else None,
        effect=data.get("effect", "multiplicative"),
        magnitude=float(data.get("magnitude", 0.0)),
        affected_unique_ids=affected_tuple,
        confidence=float(data.get("confidence", 1.0)),
        notes=str(data.get("notes", "")),
    )


def _known_future_regressor_from_dict(data: dict[str, Any]) -> KnownFutureRegressor:
    return KnownFutureRegressor(
        name=str(data.get("name", "")),
        value_col=str(data["value_col"]) if data.get("value_col") not in (None, "") else None,
        availability=data.get("availability", "historical_only"),
        mode=data.get("mode", "audit_only"),
        future_file=str(data["future_file"]) if data.get("future_file") not in (None, "") else None,
        known_as_of_col=str(data.get("known_as_of_col", "known_as_of")),
        source_system=str(data.get("source_system", "")),
        source_query_file=str(data.get("source_query_file", "")),
        owner=str(data.get("owner", "")),
        refresh_latency_days=_optional_int(data.get("refresh_latency_days")),
        notes=str(data.get("notes", "")),
    )


def _challenger_from_dict(data: dict[str, Any]) -> ChallengerSpec:
    extra_data = data.get("extra")
    if isinstance(extra_data, dict):
        extra = tuple((str(key), str(value)) for key, value in extra_data.items())
    else:
        extra = tuple((str(key), str(value)) for key, value in (extra_data or ()))
    return ChallengerSpec(
        engine=str(data.get("engine", "finn")),
        enabled=bool(data.get("enabled", True)),
        on_error=data.get("on_error", "skip"),
        models=tuple(str(model) for model in data.get("models", ())),
        back_test_scenarios=_optional_int(data.get("back_test_scenarios")),
        back_test_spacing=_optional_int(data.get("back_test_spacing")),
        forecast_approach=str(data.get("forecast_approach", "bottoms_up")),
        run_ensemble_models=bool(data.get("run_ensemble_models", False)),
        feature_selection=bool(data.get("feature_selection", False)),
        rscript=str(data.get("rscript", "Rscript")),
        timeout_seconds=int(data.get("timeout_seconds", 3600)),
        seed=int(data.get("seed", 123)),
        source_id=str(data.get("source_id", "")),
        model_name=str(data.get("model_name", "")),
        extra=extra,
    )


def _custom_model_from_dict(data: dict[str, Any]) -> CustomModelSpec:
    callable_path = data.get("callable_path")
    script_path = data.get("script_path")
    if not callable_path and not script_path:
        raise ValueError("cannot refresh a manifest custom model without callable_path or script_path")
    return CustomModelSpec(
        name=str(data.get("name") or data.get("model_name") or "custom_model"),
        callable_path=str(callable_path) if callable_path else None,
        script_path=str(script_path) if script_path else None,
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        extra_args=tuple(str(arg) for arg in data.get("extra_args", ())),
        source_id=str(data.get("source_id", "custom")),
        notes=str(data.get("notes", "")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


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
    driver_model_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_model_cv_delta: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_model_invocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    unreconciled_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy_reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy_reconciliation_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    engine: str = "baseline"
    model_policy_resolution: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        outputs = {
            "history": "appendix/history.csv",
            "forecast": "forecast.csv",
            "forecast_long": "appendix/forecast_long.csv",
            "backtest_long": "appendix/backtest_long.csv",
            "series_summary": "appendix/series_summary.csv",
            "series_features": "appendix/series_features.csv",
            "borrowed_strength_advisor": "appendix/borrowed_strength_advisor.csv",
            "model_audit": "appendix/model_audit.csv",
            "model_win_rates": "appendix/model_win_rates.csv",
            "model_tradeoff_scores": "appendix/model_tradeoff_scores.csv",
            "model_pareto_frontier": "appendix/model_pareto_frontier.csv",
            "feature_selection_receipts": "appendix/feature_selection_receipts.csv",
            "ensemble_policy_receipts": "appendix/ensemble_policy_receipts.csv",
            "model_window_metrics": "appendix/model_window_metrics.csv",
            "residual_diagnostics": "appendix/residual_diagnostics.csv",
            "residual_tests": "appendix/residual_tests.csv",
            "interval_diagnostics": "appendix/interval_diagnostics.csv",
            "trust_summary": "appendix/trust_summary.csv",
            "model_explainability": "appendix/model_explainability.csv",
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
            "streamlit_requirements": "streamlit_requirements.txt",
            "streamlit_launcher_ps1": "run_streamlit.ps1",
            "streamlit_launcher_cmd": "run_streamlit.cmd",
            "control_pane_state": "control_pane_state.json",
            "training_progress": "audit/training_progress.jsonl",
            "best_practice_receipts": "appendix/best_practice_receipts.csv",
            "run_receipt": "appendix/run_receipt.json",
            "run_receipt_markdown": "appendix/run_receipt.md",
            "validation_receipt": "appendix/validation_receipt.csv",
            "validation_receipt_json": "appendix/validation_receipt.json",
            "diagnostics": "diagnostics.json",
            "diagnostics_markdown": "diagnostics.md",
            "llm_context": "llm_context.json",
            "model_card": "model_card.md",
            "workbook": "forecast.xlsx",
        }
        if "hierarchy_depth" in self.forecast.columns:
            outputs["hierarchy_rollup"] = "appendix/hierarchy_rollup.csv"
            outputs["hierarchy_coherence"] = "appendix/hierarchy_coherence.csv"
            outputs["hierarchy_contribution"] = "appendix/hierarchy_contribution.csv"
            outputs["hierarchy_backtest_comparison"] = "audit/hierarchy_backtest_comparison.csv"
            if not self.unreconciled_forecast.empty:
                outputs["hierarchy_unreconciled_forecast"] = "audit/hierarchy_unreconciled_forecast.csv"
                outputs["hierarchy_coherence_pre"] = "audit/hierarchy_coherence_pre.csv"
                outputs["hierarchy_coherence_post"] = "audit/hierarchy_coherence_post.csv"
            if not self.hierarchy_reconciliation.empty:
                outputs["hierarchy_reconciliation"] = "appendix/hierarchy_reconciliation.csv"
            if not self.hierarchy_reconciliation_comparison.empty:
                outputs["hierarchy_reconciliation_comparison"] = "appendix/hierarchy_reconciliation_comparison.csv"
        if self.spec.events:
            outputs["scenario_assumptions"] = "appendix/scenario_assumptions.csv"
            outputs["scenario_forecast"] = "appendix/scenario_forecast.csv"
            outputs["driver_experiment_summary"] = "appendix/driver_experiment_summary.csv"
        if self.spec.regressors or not self.driver_availability_audit.empty:
            outputs["known_future_regressors"] = "appendix/known_future_regressors.csv"
            outputs["driver_availability_audit"] = "appendix/driver_availability_audit.csv"
            outputs["driver_experiment_summary"] = "appendix/driver_experiment_summary.csv"
        if self.spec.ensemble.advisory_policies:
            outputs["ensemble_backtest"] = "appendix/ensemble_backtest.csv"
            outputs["ensemble_selection"] = "appendix/ensemble_selection.csv"
            outputs["ensemble_forecast"] = "appendix/ensemble_forecast.csv"
        if not self.driver_model_features.empty:
            outputs["driver_model_features"] = "appendix/driver_model_features.csv"
        if not self.driver_model_cv_delta.empty:
            outputs["driver_model_cv_delta"] = "appendix/driver_model_cv_delta.csv"
        if self.spec.custom_models or not self.custom_model_contracts.empty or not self.custom_model_invocations.empty:
            outputs["custom_model_contracts"] = "appendix/custom_model_contracts.csv"
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
