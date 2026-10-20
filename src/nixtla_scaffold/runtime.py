"""Read-only environment inspection, separate from model execution evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

from nixtla_scaffold.model_families import (
    BASELINE_MODELS, MLFORECAST_MODELS, SMOOTH_MODELS, STATSFORECAST_MODELS,
)
from nixtla_scaffold.models import _with_resolved_candidate_identity


_ENGINES = (
    ("statsforecast", ("statsforecast",), BASELINE_MODELS | STATSFORECAST_MODELS, "2.1.1", "built_in"),
    ("mlforecast", ("mlforecast", "sklearn", "lightgbm"), MLFORECAST_MODELS, "1.1.0", "built_in"),
    ("smooth", ("smooth",), SMOOTH_MODELS, "1.1.0", "optional"),
    ("conformalforecast", ("conformalforecast",), set(), "0.1.0", "experimental_intervals"),
    ("hierarchicalforecast", ("hierarchicalforecast",), set(), "1.5.3", "reconciliation_adapter"),
)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _probe(modules: tuple[str, ...], *, timeout: float = 30) -> dict[str, str]:
    # Isolate import crashes/noisy optional dependencies from the workbench worker.
    code = "import importlib,sys; [importlib.import_module(name) for name in sys.argv[1:]]"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, *modules],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Import probe exceeded {timeout:g} seconds; no fit was attempted."}
    except OSError as exc:
        return {"status": "probe_failed", "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        return {"status": "import_failed", "error": detail or f"Import process exited {result.returncode}."}
    return {"status": "importable", "error": ""}


def runtime_capabilities() -> dict[str, Any]:
    """Inspect only this selected interpreter; never discover or switch runtimes."""
    identity = _with_resolved_candidate_identity({}, resolved_candidates=[])
    started = perf_counter()
    versions = {package: _version(package) for package, *_ in _ENGINES}
    with ThreadPoolExecutor(max_workers=len(_ENGINES)) as pool:
        pending = {package: pool.submit(_probe, modules)
                   for package, modules, *_ in _ENGINES if versions[package]}
        probes = {package: future.result() for package, future in pending.items()}
    engines = []
    for package, modules, recipes, validated_version, integration in _ENGINES:
        version = versions[package]
        probe = probes.get(package, {"status": "not_installed", "error": ""})
        engines.append({
            "package": package, "version": version, "recipes": sorted(recipes),
            "integration": integration, **probe,
            "compatibility": (
                "not_installed" if version is None else
                "version_matched" if version == validated_version else "unverified_version"
            ),
            "tested_version": validated_version,
            "data_eligibility": "not_evaluated",
            "execution": "not_attempted",
            "limitations": [
                "CustomPool is registered for historical evidence but quarantined before fit: all five reference fixtures failed on 1.1.0 and 1.2.0. No recipe substitution.",
                "CCC and ZXZ reject constant/all-zero inputs before fit; positive seasonal and intermittent fixtures passed.",
                "1.2.0 adds a short-series CCC regression ('float' has no attribute 'rename'); the supported pin remains 1.1.0.",
            ] if package == "smooth" and version in {"1.1.0", "1.2.0"} else [
                "Experimental SplitConformal/NexCP only; insufficient chronological residual support stays point-only. Not a planning certificate.",
            ] if package == "conformalforecast" else [],
        })
    return {
        "schema_version": "nixtla_scaffold.runtime.v1",
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "probe_seconds": round(perf_counter() - started, 3),
        "cached": False,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "package_version": _version("nixtla-scaffold"),
        "fingerprint": identity["resolved_candidate_fingerprint"],
        "identity": identity["resolved_candidate_identity"],
        "registered_recipes": sum(len(engine["recipes"]) for engine in engines),
        "importable_recipes": sum(len(engine["recipes"]) for engine in engines if engine["status"] == "importable"),
        "engines": engines,
        "scope": "Environment imports only. Version matching is not fit certification. Dataset eligibility and saved-run outcomes are separate.",
    }


def inspect_runtime(python: str | None = None) -> dict[str, Any]:
    if python is None:
        return runtime_capabilities()
    try:
        result = subprocess.run(
            [python, "-B", "-m", "nixtla_scaffold.cli", "runtime"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=55,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Selected runtime inspection exceeded 55 seconds; no fit was attempted.") from exc
    if result.returncode:
        raise RuntimeError(f"Selected runtime inspection failed: {(result.stderr or result.stdout).strip()[-2000:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Selected runtime did not return a valid capability receipt.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "nixtla_scaffold.runtime.v1":
        raise RuntimeError("Selected runtime has no matching capability contract; do not mix package versions.")
    return payload
