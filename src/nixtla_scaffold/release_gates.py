from __future__ import annotations

from datetime import datetime, timezone
import json
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import importlib.util
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any
from urllib.request import urlopen
import zipfile

import pandas as pd

from nixtla_scaffold.forecast import run_forecast
from nixtla_scaffold.scenario_lab import run_scenario_lab
from nixtla_scaffold.schema import ForecastSpec
from nixtla_scaffold.workbench_qa import APP_TEST_TIMEOUT_SECONDS, run_workbench_qa


OPTIONAL_EXTRAS: dict[str, tuple[str, ...]] = {
    "ml": ("mlforecast", "lightgbm"),
    "hierarchy": ("hierarchicalforecast",),
    "neural": ("neuralforecast",),
    "datasets": ("datasetsforecast",),
}
REQUIRED_PACKAGE_MEMBERS = (
    "nixtla_scaffold/cli.py",
    "nixtla_scaffold/custom_models.py",
    "nixtla_scaffold/forecast.py",
    "nixtla_scaffold/headline.py",
    "nixtla_scaffold/reports.py",
    "nixtla_scaffold/workbench_qa.py",
    "nixtla_scaffold/scenario_lab.py",
    "nixtla_scaffold/release_gates.py",
)
REQUIRED_RUN_OUTPUTS = (
    "history.csv",
    "forecast.csv",
    "forecast_long.csv",
    "backtest_long.csv",
    "series_summary.csv",
    "model_audit.csv",
    "trust_summary.csv",
    "diagnostics.json",
    "diagnostics.md",
    "llm_context.json",
    "model_card.md",
    "report.html",
    "report_base64.txt",
    "streamlit_app.py",
    "forecast.xlsx",
    "best_practice_receipts.csv",
    "audit/model_selection.csv",
    "audit/backtest_metrics.csv",
    "audit/backtest_predictions.csv",
    "audit/backtest_windows.csv",
    "audit/seasonality_diagnostics.csv",
)


@dataclass(frozen=True)
class ReleaseGateResult:
    gate: str
    status: str
    details: dict[str, Any]
    artifact: str | None = None
    remediation: str = ""
    duration_seconds: float | None = None


def run_release_gates(
    *,
    output_dir: str | Path = "runs/release_gates",
    extended: bool = False,
    build: bool = True,
    install_smoke: bool = True,
    scenario_count: int = 8,
    scenario_model_policy: str = "baseline",
    workbench_qa: bool = True,
    workbench_model_policy: str = "baseline",
    workbench_app_test: bool = True,
    workbench_app_test_timeout_seconds: int = APP_TEST_TIMEOUT_SECONDS,
    live_streamlit: bool = True,
    live_streamlit_timeout: int = 45,
    require_optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run local release-readiness gates and write a machine-readable summary."""

    started_at = time.perf_counter()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if extended:
        scenario_count = max(scenario_count, 20)
        scenario_model_policy = "auto"
        if workbench_model_policy == "baseline":
            workbench_model_policy = "all"
        require_optional = _dedupe((*require_optional, "ml", "hierarchy"))
    results: list[ReleaseGateResult] = []
    options = {
        "extended": extended,
        "build": build,
        "install_smoke": install_smoke,
        "scenario_count": scenario_count,
        "scenario_model_policy": scenario_model_policy,
        "workbench_qa": workbench_qa,
        "workbench_model_policy": workbench_model_policy,
        "workbench_app_test": workbench_app_test,
        "workbench_app_test_timeout_seconds": workbench_app_test_timeout_seconds,
        "live_streamlit": live_streamlit,
        "live_streamlit_timeout": live_streamlit_timeout,
        "require_optional": list(require_optional),
    }

    gate_started = time.perf_counter()
    results.append(_with_duration(_package_metadata_gate(), gate_started))

    gate_started = time.perf_counter()
    build_result, wheel_path = _build_gate(out) if build else (_skipped("build", "build disabled"), None)
    results.append(_with_duration(build_result, gate_started))

    gate_started = time.perf_counter()
    install_result = _install_smoke_gate(out, wheel_path) if install_smoke else _skipped("install_smoke", "install smoke disabled")
    results.append(_with_duration(install_result, gate_started))

    gate_started = time.perf_counter()
    scenario_result = _scenario_lab_gate(out, count=scenario_count, model_policy=scenario_model_policy)
    results.append(_with_duration(scenario_result, gate_started))

    gate_started = time.perf_counter()
    quick_result, quick_run_dir = _quick_forecast_gate(out)
    results.append(_with_duration(quick_result, gate_started))

    gate_started = time.perf_counter()
    hygiene_result = (
        _artifact_hygiene_gate(quick_run_dir)
        if quick_run_dir is not None
        else _failed("artifact_hygiene", {"reason": "quick forecast did not produce a run directory"})
    )
    results.append(_with_duration(hygiene_result, gate_started))

    gate_started = time.perf_counter()
    results.append(_optional_extras_gate(require_optional))
    results[-1] = _with_duration(results[-1], gate_started)

    gate_started = time.perf_counter()
    if workbench_qa:
        workbench_result = _workbench_qa_gate(
            out,
            model_policy=workbench_model_policy,
            app_test=workbench_app_test,
            app_test_timeout_seconds=workbench_app_test_timeout_seconds,
        )
    else:
        workbench_result = _skipped("workbench_qa", "workbench QA disabled")
    results.append(_with_duration(workbench_result, gate_started))

    gate_started = time.perf_counter()
    if live_streamlit:
        live_result = (
            _live_streamlit_gate(quick_run_dir, timeout_seconds=live_streamlit_timeout)
            if quick_run_dir is not None
            else _failed("live_streamlit", {"reason": "quick forecast did not produce a run directory"})
        )
    else:
        live_result = _skipped("live_streamlit", "live Streamlit smoke disabled")
    results.append(_with_duration(live_result, gate_started))

    payload = _payload(
        results,
        output_dir=out,
        options=options,
        total_duration_seconds=round(time.perf_counter() - started_at, 3),
    )
    (out / "release_gate_summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(_results_table(payload)).to_csv(out / "release_gate_results.csv", index=False)
    (out / "release_gate_summary.md").write_text(format_release_gate_markdown(payload), encoding="utf-8")
    return payload


def _package_metadata_gate() -> ReleaseGateResult:
    pyproject = _repo_root() / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    required = [
        'name = "nixtla-scaffold"',
        'nixtla-scaffold = "nixtla_scaffold.cli:main"',
        "[project.optional-dependencies]",
        "hierarchy",
        "ml",
        "neural",
        "datasets",
    ]
    missing = [needle for needle in required if needle not in text]
    return ReleaseGateResult(
        gate="package_metadata",
        status="passed" if not missing else "failed",
        details={"pyproject": str(pyproject), "missing": missing},
        remediation=_remediation_for("package_metadata") if missing else "",
    )


def _build_gate(output_dir: Path) -> tuple[ReleaseGateResult, Path | None]:
    build_dir = output_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    command = ["uv", "build", "--out-dir", str(build_dir)]
    completed = _run_command(command, cwd=_repo_root(), timeout=240)
    if completed["returncode"] != 0:
        return _failed("build", completed, artifact=str(build_dir)), None
    wheels = sorted(build_dir.glob("*.whl"))
    sdists = sorted(build_dir.glob("*.tar.gz"))
    inspection = _inspect_package_artifacts(wheels, sdists)
    status = "passed" if wheels and sdists and not inspection["missing"] else "failed"
    details = {**completed, **inspection, "wheels": [str(path) for path in wheels], "sdists": [str(path) for path in sdists]}
    return (
        ReleaseGateResult(
            gate="build",
            status=status,
            details=details,
            artifact=str(build_dir),
            remediation=_remediation_for("build") if status == "failed" else "",
        ),
        wheels[-1] if wheels else None,
    )


def _install_smoke_gate(output_dir: Path, wheel_path: Path | None) -> ReleaseGateResult:
    if wheel_path is None:
        return _failed("install_smoke", {"reason": "no built wheel available"})
    venv_dir = output_dir / "install_smoke_venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    create = _run_command(["uv", "venv", str(venv_dir)], cwd=output_dir, timeout=180)
    if create["returncode"] != 0:
        return _failed("install_smoke", {"step": "uv venv", **create}, artifact=str(venv_dir))
    python_exe = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    install = _run_command(["uv", "pip", "install", "--python", str(python_exe), str(wheel_path)], cwd=output_dir, timeout=300)
    if install["returncode"] != 0:
        return _failed("install_smoke", {"step": "uv pip install", **install}, artifact=str(venv_dir))
    code = (
        "from nixtla_scaffold import ForecastSpec, build_executive_headline, forecast_spec_preset; "
        "from nixtla_scaffold.cli import main; "
        "assert ForecastSpec(horizon=1).horizon == 1; "
        "assert forecast_spec_preset('quick').model_policy == 'baseline'; "
        "assert callable(build_executive_headline); "
        "raise SystemExit(main(['guide', 'presets']))"
    )
    smoke = _run_command([str(python_exe), "-c", code], cwd=output_dir, timeout=120)
    console_script = venv_dir / ("Scripts/nixtla-scaffold.exe" if sys.platform == "win32" else "bin/nixtla-scaffold")
    console_smoke = _run_command([str(console_script), "guide", "presets"], cwd=output_dir, timeout=120)
    failures: list[str] = []
    if smoke["returncode"] != 0 or not _command_output_contains(smoke, '"quick"'):
        failures.append("public API smoke failed")
    if console_smoke["returncode"] != 0 or not _command_output_contains(console_smoke, '"quick"'):
        failures.append("installed console-script smoke failed")
    status = "passed" if not failures else "failed"
    return ReleaseGateResult(
        gate="install_smoke",
        status=status,
        details={
            "venv": str(venv_dir),
            "wheel": str(wheel_path),
            "python_executable": str(python_exe),
            "console_script": str(console_script),
            "public_api_smoke": smoke,
            "console_script_smoke": console_smoke,
            "failures": failures,
        },
        artifact=str(venv_dir),
        remediation=_remediation_for("install_smoke") if status == "failed" else "",
    )


def _scenario_lab_gate(output_dir: Path, *, count: int, model_policy: str) -> ReleaseGateResult:
    lab_dir = output_dir / "scenario_lab"
    _remove_owned_path(lab_dir)
    payload = run_scenario_lab(count=count, output_dir=lab_dir, model_policy=model_policy, seed=7)
    summary = payload["summary"]
    failures: list[str] = []
    if summary["crashed"] != 0:
        failures.append(f"{summary['crashed']} scenario(s) crashed")
    if summary["passed"] < max(1, count - 2):
        failures.append(f"only {summary['passed']} of {count} scenario(s) passed")
    if float(summary["composite_score"]) < 70:
        failures.append(f"composite_score below 70: {summary['composite_score']}")
    if float(summary["validity_score"]) < 85:
        failures.append(f"validity_score below 85: {summary['validity_score']}")
    if float(summary["explainability_score"]) < 80:
        failures.append(f"explainability_score below 80: {summary['explainability_score']}")
    return ReleaseGateResult(
        gate="scenario_lab_numeric",
        status="passed" if not failures else "failed",
        details={"summary": summary, "failures": failures},
        artifact=str(lab_dir),
        remediation=_remediation_for("scenario_lab_numeric") if failures else "",
    )


def _quick_forecast_gate(output_dir: Path) -> tuple[ReleaseGateResult, Path | None]:
    run_dir = output_dir / "quick_forecast"
    _remove_owned_path(run_dir)
    df = _release_smoke_frame()
    run = run_forecast(df, ForecastSpec(horizon=3, freq="ME", model_policy="baseline", verbose=False))
    run.to_directory(run_dir)
    forecast = pd.read_csv(run_dir / "forecast.csv")
    trust = pd.read_csv(run_dir / "trust_summary.csv")
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    yhat = pd.to_numeric(forecast["yhat"], errors="coerce")
    interval_checks = _interval_sanity_failures(forecast)
    failures: list[str] = []
    if len(forecast) != 3:
        failures.append(f"expected 3 forecast rows, got {len(forecast)}")
    if not yhat.notna().all():
        failures.append("forecast yhat contains non-finite values")
    if yhat.abs().max() > 10_000 or yhat.abs().min() <= 0:
        failures.append("forecast yhat outside release smoke bounds")
    if trust.empty or pd.to_numeric(trust["trust_score_0_100"], errors="coerce").isna().any():
        failures.append("trust_summary missing numeric trust score")
    if not diagnostics.get("executive_headline", {}).get("paragraph"):
        failures.append("diagnostics executive headline missing")
    failures.extend(interval_checks)
    return (
        ReleaseGateResult(
            gate="quick_forecast_numeric",
            status="passed" if not failures else "failed",
            details={
                "forecast_rows": int(len(forecast)),
                "selected_models": sorted(forecast["model"].dropna().astype(str).unique().tolist()),
                "yhat_sum": float(yhat.sum()) if yhat.notna().all() else None,
                "trust_scores": trust["trust_score_0_100"].tolist() if "trust_score_0_100" in trust else [],
                "interval_checks": interval_checks,
                "failures": failures,
            },
            artifact=str(run_dir),
            remediation=_remediation_for("quick_forecast_numeric") if failures else "",
        ),
        run_dir,
    )


def _artifact_hygiene_gate(run_dir: Path) -> ReleaseGateResult:
    missing = [path for path in REQUIRED_RUN_OUTPUTS if not (run_dir / path).exists()]
    manifest_missing: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.get("outputs", {})
        for rel_path in outputs.values():
            if not (run_dir / str(rel_path)).exists():
                manifest_missing.append(str(rel_path))
    bad_suffixes = {".pyc", ".tmp", ".bak"}
    bad_files = [
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.name == "__pycache__"
        or path.suffix.lower() in bad_suffixes
        or path.name.startswith("failure_diagnostics")
    ]
    failures = []
    if missing:
        failures.append(f"missing required outputs: {', '.join(missing)}")
    if manifest_missing:
        failures.append(f"manifest outputs missing: {', '.join(manifest_missing)}")
    if bad_files:
        failures.append(f"unexpected generated artifacts: {', '.join(bad_files)}")
    return ReleaseGateResult(
        gate="artifact_hygiene",
        status="passed" if not failures else "failed",
        details={"run_dir": str(run_dir), "missing": missing, "manifest_missing": manifest_missing, "bad_files": bad_files, "failures": failures},
        artifact=str(run_dir),
        remediation=_remediation_for("artifact_hygiene") if failures else "",
    )


def _optional_extras_gate(required_optional: tuple[str, ...]) -> ReleaseGateResult:
    unknown = [extra for extra in required_optional if extra not in OPTIONAL_EXTRAS]
    rows: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    if unknown:
        failures.append(f"unknown optional extra(s): {', '.join(unknown)}")
    for extra, modules in OPTIONAL_EXTRAS.items():
        missing = [module for module in modules if importlib.util.find_spec(module) is None]
        rows[extra] = {
            "modules": list(modules),
            "installed": not missing,
            "missing": missing,
            "required": extra in required_optional,
        }
        if extra in required_optional and missing:
            failures.append(f"{extra} missing module(s): {', '.join(missing)}")
    return ReleaseGateResult(
        gate="optional_extras",
        status="passed" if not failures else "failed",
        details={"extras": rows, "failures": failures},
        remediation=_remediation_for("optional_extras") if failures else "",
    )


def _workbench_qa_gate(output_dir: Path, *, model_policy: str, app_test: bool, app_test_timeout_seconds: int) -> ReleaseGateResult:
    qa_dir = output_dir / "workbench_qa"
    _remove_owned_path(qa_dir)
    payload = run_workbench_qa(
        output_dir=qa_dir,
        model_policy=model_policy,
        app_test=app_test,
        app_test_timeout_seconds=app_test_timeout_seconds,
    )
    summary = payload["summary"]
    failures: list[str] = []
    if summary["failed"]:
        failures.append(f"{summary['failed']} workbench scenario(s) failed")
    if summary["min_usability_score"] < 90:
        failures.append(f"min usability below 90: {summary['min_usability_score']}")
    return ReleaseGateResult(
        gate="workbench_qa",
        status="passed" if not failures else "failed",
        details={"summary": summary, "failures": failures},
        artifact=str(qa_dir),
        remediation=_remediation_for("workbench_qa") if failures else "",
    )


def _live_streamlit_gate(run_dir: Path, *, timeout_seconds: int) -> ReleaseGateResult:
    app_path = run_dir / "streamlit_app.py"
    if not app_path.exists():
        return _failed("live_streamlit", {"reason": "streamlit_app.py missing", "run_dir": str(run_dir)})
    port = _free_port()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path.name),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(command, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{port}"
    health_url = f"{url}/_stcore/health"
    status_code: int | None = None
    health_status_code: int | None = None
    error: str | None = None
    deadline = time.time() + timeout_seconds
    try:
        while time.time() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                error = f"streamlit process exited early with code {exit_code}"
                break
            try:
                with urlopen(health_url, timeout=5) as response:
                    health_status_code = int(response.status)
                with urlopen(url, timeout=5) as response:
                    status_code = int(response.status)
                break
            except Exception as exc:  # pragma: no cover - timing dependent
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(1)
        if status_code != 200 or health_status_code != 200:
            stdout, stderr = _terminate_process(proc)
            return _failed(
                "live_streamlit",
                {
                    "url": url,
                    "health_url": health_url,
                    "status_code": status_code,
                    "health_status_code": health_status_code,
                    "last_error": error,
                    "timeout_seconds": timeout_seconds,
                    "environment": "current_python_environment",
                    "python_executable": sys.executable,
                    "process_returncode": proc.returncode,
                    "stdout_head": _head(stdout),
                    "stdout_tail": _tail(stdout),
                    "stderr_head": _head(stderr),
                    "stderr_tail": _tail(stderr),
                },
                artifact=str(app_path),
            )
        return ReleaseGateResult(
            gate="live_streamlit",
            status="passed",
            details={
                "url": url,
                "health_url": health_url,
                "status_code": status_code,
                "health_status_code": health_status_code,
                "timeout_seconds": timeout_seconds,
                "environment": "current_python_environment",
                "python_executable": sys.executable,
            },
            artifact=str(app_path),
        )
    finally:
        if proc.poll() is None:
            _terminate_process(proc)


def _release_smoke_frame() -> pd.DataFrame:
    t = list(range(24))
    return pd.DataFrame(
        {
            "unique_id": ["ReleaseSmoke"] * 24,
            "ds": pd.date_range("2024-01-31", periods=24, freq="ME"),
            "y": [120 + idx * 2.5 + (8 if idx % 12 in {10, 11} else 0) for idx in t],
        }
    )


def _inspect_package_artifacts(wheels: list[Path], sdists: list[Path]) -> dict[str, Any]:
    missing: dict[str, list[str]] = {}
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        absent = [member for member in REQUIRED_PACKAGE_MEMBERS if not any(name.endswith(member) for name in names)]
        if absent:
            missing[str(wheel)] = absent
    for sdist in sdists:
        with tarfile.open(sdist, "r:gz") as archive:
            names = archive.getnames()
        absent = [
            member
            for member in REQUIRED_PACKAGE_MEMBERS
            if not any(name.endswith("src/" + member) or name.endswith(member) for name in names)
        ]
        if absent:
            missing[str(sdist)] = absent
    return {"required_members": list(REQUIRED_PACKAGE_MEMBERS), "missing": missing}


def _interval_sanity_failures(forecast: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    yhat = pd.to_numeric(forecast["yhat"], errors="coerce") if "yhat" in forecast else pd.Series(dtype=float)
    for column in forecast.columns:
        if not column.startswith("yhat_lo_"):
            continue
        level = column.removeprefix("yhat_lo_")
        hi_col = f"yhat_hi_{level}"
        if hi_col not in forecast:
            failures.append(f"interval level {level} missing {hi_col}")
            continue
        lo = pd.to_numeric(forecast[column], errors="coerce")
        hi = pd.to_numeric(forecast[hi_col], errors="coerce")
        present = lo.notna() | hi.notna()
        if not present.any():
            continue
        if lo[present].isna().any() or hi[present].isna().any():
            failures.append(f"interval level {level} has one-sided bounds")
            continue
        if (lo[present] > hi[present]).any():
            failures.append(f"interval level {level} has lower bound above upper bound")
        if not yhat.empty and ((lo[present] > yhat[present]) | (hi[present] < yhat[present])).any():
            failures.append(f"interval level {level} does not contain yhat")
    return failures


def _run_command(command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout_head": _head(completed.stdout),
            "stdout_tail": _tail(completed.stdout),
            "stderr_head": _head(completed.stderr),
            "stderr_tail": _tail(completed.stderr),
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout_head": "",
            "stdout_tail": "",
            "stderr_head": f"{type(exc).__name__}: {exc}",
            "stderr_tail": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        stdout = _process_text(exc.stdout)
        stderr = _process_text(exc.stderr)
        message = f"{type(exc).__name__}: {exc}"
        stderr_with_message = f"{stderr}\n{message}".strip()
        return {
            "command": command,
            "returncode": 127,
            "stdout_head": _head(stdout),
            "stdout_tail": _tail(stdout),
            "stderr_head": _head(stderr_with_message),
            "stderr_tail": _tail(stderr_with_message),
        }


def _remove_owned_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _results_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload["results"]:
        rows.append(
            {
                "gate": row.get("gate"),
                "status": row.get("status"),
                "duration_seconds": row.get("duration_seconds"),
                "reason": _primary_reason_from_row(row),
                "remediation": row.get("remediation") or "",
                "artifact": row.get("artifact") or "",
                "details_json": json.dumps(row.get("details") or {}, default=str, sort_keys=True),
            }
        )
    return rows


def _payload(
    results: list[ReleaseGateResult],
    *,
    output_dir: Path,
    options: dict[str, Any],
    total_duration_seconds: float,
) -> dict[str, Any]:
    rows = [asdict(result) for result in results]
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    passed = sum(1 for result in results if result.status == "passed")
    failed_gates = [result.gate for result in results if result.status == "failed"]
    skipped_gates = [result.gate for result in results if result.status == "skipped"]
    git_sha, git_sha_unavailable_reason = _git_sha_with_reason()
    provenance_warnings = [f"git SHA unavailable: {git_sha_unavailable_reason}"] if git_sha_unavailable_reason else []
    failure_rollup = [
        {
            "gate": result.gate,
            "reason": _primary_failure_reason(result),
            "remediation": result.remediation or _remediation_for(result.gate),
            "artifact": result.artifact,
        }
        for result in results
        if result.status == "failed"
    ]
    status = "failed" if failed else "passed"
    summary = {
        "status": status,
        "headline": _release_headline(status, passed=passed, failed=failed, skipped=skipped, count=len(results), failed_gates=failed_gates),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "count": len(results),
        "failed_gates": failed_gates,
        "skipped_gates": skipped_gates,
        "failure_rollup": failure_rollup,
        "output_dir": str(output_dir),
        "artifacts": {
            "json": str(output_dir / "release_gate_summary.json"),
            "csv": str(output_dir / "release_gate_results.csv"),
            "markdown": str(output_dir / "release_gate_summary.md"),
        },
        "options": options,
        "required_optional_groups": list(options.get("require_optional", [])),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": total_duration_seconds,
        "package_version": _package_version(),
        "git_sha": git_sha,
        "git_sha_unavailable_reason": git_sha_unavailable_reason,
        "provenance_warnings": provenance_warnings,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }
    return {"summary": summary, "results": rows, "output_dir": str(output_dir)}


def format_release_gate_console_summary(payload: dict[str, Any]) -> str:
    """Return a compact release-gate summary suitable for analyst/agent terminals."""

    summary = payload["summary"]
    lines = [
        str(summary["headline"]),
        f"Version: {summary.get('package_version')} | Python: {summary.get('python_version')} | Duration: {summary.get('duration_seconds')}s",
        f"Output: {summary['output_dir']}",
        "Artifacts: "
        + ", ".join(
            [
                str(summary["artifacts"]["markdown"]),
                str(summary["artifacts"]["json"]),
                str(summary["artifacts"]["csv"]),
            ]
        ),
    ]
    if summary.get("failure_rollup"):
        lines.append("Failures:")
        for item in summary["failure_rollup"]:
            lines.append(f"- {item['gate']}: {item['reason']}")
            if item.get("remediation"):
                lines.append(f"  Remediation: {item['remediation']}")
    else:
        lines.append("All required release gates passed.")
    skipped = summary.get("skipped_gates") or []
    if skipped:
        skip_reasons = _skip_reasons(payload)
        lines.append("Skipped: " + ", ".join(f"{gate} ({reason})" for gate, reason in skip_reasons.items()))
    if summary.get("provenance_warnings"):
        lines.append("Provenance warnings: " + "; ".join(str(item) for item in summary["provenance_warnings"]))
    lines.append("Use --json to print the full machine-readable payload.")
    return "\n".join(lines) + "\n"


def format_release_gate_markdown(payload: dict[str, Any]) -> str:
    """Return a human-readable release-gate summary artifact."""

    summary = payload["summary"]
    lines = [
        "# Release gate summary",
        "",
        f"**Verdict:** {summary['headline']}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Generated UTC | {_md_cell(summary.get('generated_at_utc'))} |",
        f"| Package version | {_md_cell(summary.get('package_version'))} |",
        f"| Git SHA | {_md_cell(summary.get('git_sha') or 'unavailable')} |",
        f"| Git SHA unavailable reason | {_md_cell(summary.get('git_sha_unavailable_reason') or '')} |",
        f"| Python | {_md_cell(summary.get('python_version'))} on {_md_cell(summary.get('platform'))} |",
        f"| Duration | {_md_cell(summary.get('duration_seconds'))} seconds |",
        f"| Output directory | `{_md_cell(summary.get('output_dir'))}` |",
        f"| Required optional groups | {_md_cell(', '.join(summary.get('required_optional_groups') or []) or 'none')} |",
        "",
        "## Gate results",
        "",
        "| Gate | Status | Duration (s) | Reason | Remediation | Artifact |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in payload["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_cell(row.get('gate'))}`",
                    _md_cell(str(row.get("status", "")).upper()),
                    _md_cell(row.get("duration_seconds", "")),
                    _md_cell(_primary_reason_from_row(row)),
                    _md_cell(row.get("remediation") or ""),
                    _md_artifact(row.get("artifact")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Scenario archetype scores",
            "",
            *_scenario_archetype_lines(payload),
            "",
            "## Thresholds and exit codes",
            "",
            "- `scenario_lab_numeric`: zero crashes, at least `count - 2` scenarios pass, composite score >= 70, validity >= 85, explainability >= 80.",
            "- `quick_forecast_numeric`: expected horizon rows, finite bounded point forecasts, valid interval containment when interval columns exist, numeric trust score, and an executive headline.",
            "- `artifact_hygiene`: required forecast artifacts must exist; `__pycache__`, `.pyc`, `.tmp`, `.bak`, and `failure_diagnostics*` files fail the gate.",
            "- `workbench_qa`: no failed golden scenarios and minimum usability score >= 90.",
            "- `live_streamlit`: launches the generated dashboard with the current Python environment and requires HTTP 200 from `/` and `/_stcore/health`.",
            "- CLI exit codes: `0` means all required gates passed; `1` means one or more gates failed; `2` means the CLI hit an unhandled runtime/configuration error.",
            "",
            "## How to act on failures",
            "",
        ]
    )
    if summary.get("failure_rollup"):
        for item in summary["failure_rollup"]:
            lines.append(f"- **`{item['gate']}`**: {item['reason']} Remediation: {item['remediation']}")
    else:
        lines.append("- No required gates failed.")
    return "\n".join(lines) + "\n"


def _skip_reasons(payload: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in payload["results"]:
        if row.get("status") == "skipped":
            details = row.get("details") or {}
            reasons[str(row.get("gate"))] = str(details.get("reason") or "skipped")
    return reasons


def _scenario_archetype_lines(payload: dict[str, Any]) -> list[str]:
    scenario = next((row for row in payload["results"] if row.get("gate") == "scenario_lab_numeric"), None)
    if not scenario:
        return ["- Scenario-lab gate did not run."]
    summary = (scenario.get("details") or {}).get("summary") or {}
    archetypes = summary.get("lowest_scoring_archetypes") or {}
    if not archetypes:
        return ["- No scenario archetype score details were reported."]
    return [f"- `{_md_cell(name)}`: {_md_cell(score)}" for name, score in archetypes.items()]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _head(text: str, *, limit: int = 4000) -> str:
    text = text or ""
    return text[:limit]


def _tail(text: str, *, limit: int = 4000) -> str:
    text = text or ""
    return text[-limit:]


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _skipped(gate: str, reason: str) -> ReleaseGateResult:
    return ReleaseGateResult(gate=gate, status="skipped", details={"reason": reason})


def _failed(gate: str, details: dict[str, Any], artifact: str | None = None) -> ReleaseGateResult:
    return ReleaseGateResult(gate=gate, status="failed", details=details, artifact=artifact, remediation=_remediation_for(gate))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _with_duration(result: ReleaseGateResult, started_at: float) -> ReleaseGateResult:
    return replace(result, duration_seconds=round(max(time.perf_counter() - started_at, 0.0), 3))


def _command_output_contains(result: dict[str, Any], needle: str) -> bool:
    return needle in str(result.get("stdout_head", "")) or needle in str(result.get("stdout_tail", ""))


def _terminate_process(proc: subprocess.Popen[str]) -> tuple[str, str]:
    if proc.poll() is None:
        proc.terminate()
    try:
        return proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
        proc.kill()
        return proc.communicate(timeout=10)


def _package_version() -> str:
    try:
        return importlib.metadata.version("nixtla-scaffold")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_sha_with_reason() -> tuple[str | None, str | None]:
    result = _run_command(["git", "rev-parse", "--short", "HEAD"], cwd=_repo_root(), timeout=30)
    if result["returncode"] != 0:
        reason = str(result.get("stderr_tail") or result.get("stderr_head") or "git command failed").strip()
        return None, reason
    sha = str(result.get("stdout_tail", "")).strip()
    if not sha:
        return None, "git returned an empty SHA"
    return sha, None


def _git_sha() -> str | None:
    sha, _ = _git_sha_with_reason()
    return sha


def _release_headline(status: str, *, passed: int, failed: int, skipped: int, count: int, failed_gates: list[str]) -> str:
    prefix = "PASSED" if status == "passed" else "FAILED"
    if failed_gates:
        gate_text = ", ".join(failed_gates)
        return f"{prefix}: {failed} of {count} gates failed ({gate_text}); {passed} passed, {skipped} skipped"
    return f"{prefix}: {passed} of {count} gates passed; {skipped} skipped"


def _primary_failure_reason(result: ReleaseGateResult) -> str:
    return _primary_reason_from_row(asdict(result))


def _primary_reason_from_row(row: dict[str, Any]) -> str:
    status = str(row.get("status", ""))
    details = row.get("details") or {}
    if status == "passed":
        return "Passed"
    if status == "skipped":
        return str(details.get("reason") or "Skipped")
    failures = details.get("failures")
    if isinstance(failures, list) and failures:
        return "; ".join(str(item) for item in failures)
    reason = details.get("reason")
    if reason:
        return str(reason)
    missing = details.get("missing")
    if missing:
        return f"missing: {missing}"
    return "See gate details"


def _remediation_for(gate: str) -> str:
    return {
        "package_metadata": "Check pyproject.toml for project metadata, console script, and optional dependency groups.",
        "build": "Run uv build from the repository root and inspect the captured stdout/stderr head and tail.",
        "install_smoke": "Rebuild the wheel, reinstall it into an isolated venv, and verify both public imports and the installed nixtla-scaffold console script.",
        "scenario_lab_numeric": "Open the scenario_lab artifacts, inspect failed scenarios and recommendations, then fix validity/explainability regressions before rerunning.",
        "quick_forecast_numeric": "Open the quick_forecast diagnostics, forecast.csv, and trust_summary.csv; restore finite selected forecasts and headline/trust artifacts.",
        "artifact_hygiene": "Restore missing required outputs and remove temporary, bytecode, backup, or failure_diagnostics artifacts from generated runs.",
        "optional_extras": "Install required extras in the active environment, for example uv pip install -e .[ml,hierarchy], or do not require that extra for this release.",
        "workbench_qa": "Open workbench_qa_summary.csv/json, then fix missing artifacts, Streamlit compile/AppTest failures, or usability score regressions.",
        "live_streamlit": "Run uv run streamlit run streamlit_app.py from the quick_forecast folder and inspect the captured Streamlit logs.",
    }.get(gate, "Inspect this gate's details and rerun release-gates after fixing the failure.")


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _md_artifact(value: Any) -> str:
    if not value:
        return ""
    return f"`{_md_cell(value)}`"
