from __future__ import annotations

import json
import sys
import tarfile
import zipfile

from nixtla_scaffold.cli import main
import nixtla_scaffold.release_gates as release_gates
from nixtla_scaffold.release_gates import (
    _artifact_hygiene_gate,
    _inspect_package_artifacts,
    _interval_sanity_failures,
    _optional_extras_gate,
    _package_metadata_gate,
    _run_command,
    run_release_gates,
)


def test_release_gates_fast_path_writes_summary(tmp_path) -> None:
    payload = run_release_gates(
        output_dir=tmp_path / "release",
        build=False,
        install_smoke=False,
        scenario_count=2,
        workbench_qa=False,
        live_streamlit=False,
        workbench_app_test_timeout_seconds=33,
    )

    assert payload["summary"]["status"] == "passed"
    assert payload["summary"]["headline"].startswith("PASSED:")
    assert payload["summary"]["failed_gates"] == []
    assert payload["summary"]["failure_rollup"] == []
    assert "duration_seconds" in payload["summary"]
    assert "git_sha_unavailable_reason" in payload["summary"]
    assert payload["summary"]["skipped"] == 4
    assert payload["summary"]["options"]["workbench_app_test_timeout_seconds"] == 33
    gates = {row["gate"]: row for row in payload["results"]}
    assert gates["package_metadata"]["status"] == "passed"
    assert gates["scenario_lab_numeric"]["status"] == "passed"
    assert gates["quick_forecast_numeric"]["status"] == "passed"
    assert gates["artifact_hygiene"]["status"] == "passed"
    assert (tmp_path / "release" / "release_gate_summary.json").exists()
    assert (tmp_path / "release" / "release_gate_results.csv").exists()
    assert (tmp_path / "release" / "release_gate_summary.md").exists()
    persisted = json.loads((tmp_path / "release" / "release_gate_summary.json").read_text(encoding="utf-8"))
    assert persisted["summary"]["status"] == "passed"
    assert len(persisted["results"]) == len(payload["results"])
    markdown = (tmp_path / "release" / "release_gate_summary.md").read_text(encoding="utf-8")
    for gate in gates:
        assert f"`{gate}`" in markdown
    assert "## Scenario archetype scores" in markdown
    assert "## Thresholds and exit codes" in markdown
    assert "CLI exit codes" in markdown
    assert "valid interval containment" in markdown
    csv_text = (tmp_path / "release" / "release_gate_results.csv").read_text(encoding="utf-8")
    assert "reason,remediation,artifact,details_json" in csv_text


def test_package_metadata_gate_requires_public_publish_metadata() -> None:
    result = _package_metadata_gate()

    assert result.status == "passed"
    assert result.details["missing"] == []


def test_build_inspection_requires_bundled_skill_in_wheel_and_sdist(tmp_path) -> None:
    wheel = tmp_path / "nixtla_scaffold-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in release_gates.REQUIRED_WHEEL_MEMBERS:
            if member == "nixtla_scaffold/skills/nixtla-forecast/SKILL.md":
                continue
            archive.writestr(member, "")

    sdist = tmp_path / "nixtla_scaffold-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for member in release_gates.REQUIRED_SDIST_MEMBERS:
            if member == "skills/nixtla-forecast/SKILL.md":
                continue
            path = tmp_path / member.replace("/", "_")
            path.write_text("", encoding="utf-8")
            archive.add(path, arcname=f"nixtla_scaffold-0.1.0/src/{member}" if member.startswith("nixtla_scaffold/") else f"nixtla_scaffold-0.1.0/{member}")

    inspection = _inspect_package_artifacts([wheel], [sdist])

    assert inspection["missing"][str(wheel)] == ["nixtla_scaffold/skills/nixtla-forecast/SKILL.md"]
    assert inspection["missing"][str(sdist)] == ["skills/nixtla-forecast/SKILL.md"]


def test_release_gates_cli_compact_success_is_low_noise(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "release-gates",
            "--output",
            str(tmp_path / "release"),
            "--no-build",
            "--no-install-smoke",
            "--scenario-count",
            "1",
            "--no-workbench-qa",
            "--no-live-streamlit",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "PASSED:" in output
    assert "Version:" in output
    assert "Duration:" in output
    assert "Skipped: build (build disabled)" in output
    assert '"results":' not in output


def test_release_gates_cli_compact_output_returns_nonzero_for_failed_gate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "release-gates",
            "--output",
            str(tmp_path / "release"),
            "--no-build",
            "--no-install-smoke",
            "--scenario-count",
            "1",
            "--no-workbench-qa",
            "--no-live-streamlit",
            "--require-optional",
            "not-a-real-extra",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAILED:" in output
    assert "optional_extras" in output
    assert "release_gate_summary.md" in output
    assert '"command":' not in output


def test_release_gates_cli_json_outputs_full_payload_for_agents(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "release-gates",
            "--output",
            str(tmp_path / "release"),
            "--no-build",
            "--no-install-smoke",
            "--scenario-count",
            "1",
            "--no-workbench-qa",
            "--no-live-streamlit",
            "--require-optional",
            "not-a-real-extra",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    optional = payload["results"][[row["gate"] for row in payload["results"]].index("optional_extras")]
    assert exit_code == 1
    assert payload["summary"]["status"] == "failed"
    assert payload["summary"]["failed_gates"] == ["optional_extras"]
    assert payload["summary"]["required_optional_groups"] == ["not-a-real-extra"]
    assert payload["summary"]["failure_rollup"][0]["remediation"]
    assert "unknown optional extra" in optional["details"]["failures"][0]


def test_artifact_hygiene_flags_unexpected_files(tmp_path) -> None:
    bad = tmp_path / "quick_forecast"
    bad.mkdir()
    (bad / "failure_diagnostics.json").write_text("{}", encoding="utf-8")
    (bad / "debug.tmp").write_text("temporary", encoding="utf-8")

    result = _artifact_hygiene_gate(bad)

    assert result.status == "failed"
    assert "failure_diagnostics.json" in result.details["bad_files"]
    assert "debug.tmp" in result.details["bad_files"]


def test_optional_extras_gate_reports_unknown_required_group() -> None:
    result = _optional_extras_gate(("not-a-real-extra",))

    assert result.status == "failed"
    assert "unknown optional extra" in result.details["failures"][0]


def test_optional_extras_include_datasetsforecast_group() -> None:
    result = _optional_extras_gate(())

    assert release_gates.OPTIONAL_EXTRAS["datasets"] == ("datasetsforecast",)
    assert "datasets" in result.details["extras"]
    assert result.details["extras"]["datasets"]["modules"] == ["datasetsforecast"]
    assert result.details["extras"]["datasets"]["required"] is False


def test_optional_extras_gate_can_require_datasets_group(monkeypatch) -> None:
    monkeypatch.setattr(release_gates.importlib.util, "find_spec", lambda name: None if name == "datasetsforecast" else object())

    result = _optional_extras_gate(("datasets",))

    assert result.status == "failed"
    assert "datasets missing module(s): datasetsforecast" in result.details["failures"]


def test_extended_release_mode_sets_stricter_options(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        release_gates,
        "_scenario_lab_gate",
        lambda output_dir, count, model_policy: release_gates.ReleaseGateResult(
            gate="scenario_lab_numeric",
            status="passed",
            details={
                "summary": {
                    "lowest_scoring_archetypes": {"monthly_trend": 99.0},
                    "count": count,
                    "model_policy": model_policy,
                },
                "failures": [],
            },
        ),
    )
    monkeypatch.setattr(
        release_gates,
        "_quick_forecast_gate",
        lambda output_dir: (
            release_gates.ReleaseGateResult(gate="quick_forecast_numeric", status="passed", details={"failures": []}),
            tmp_path / "quick",
        ),
    )
    monkeypatch.setattr(
        release_gates,
        "_artifact_hygiene_gate",
        lambda run_dir: release_gates.ReleaseGateResult(gate="artifact_hygiene", status="passed", details={"failures": []}),
    )
    monkeypatch.setattr(
        release_gates,
        "_optional_extras_gate",
        lambda required_optional: release_gates.ReleaseGateResult(
            gate="optional_extras",
            status="passed",
            details={"required_optional": list(required_optional), "failures": []},
        ),
    )

    payload = run_release_gates(
        output_dir=tmp_path / "release",
        extended=True,
        build=False,
        install_smoke=False,
        scenario_count=1,
        workbench_qa=False,
        live_streamlit=False,
    )

    assert payload["summary"]["options"]["extended"] is True
    assert payload["summary"]["options"]["scenario_count"] == 20
    assert payload["summary"]["options"]["scenario_model_policy"] == "auto"
    assert payload["summary"]["required_optional_groups"] == ["ml", "hierarchy"]


def test_interval_sanity_flags_invalid_bounds() -> None:
    frame = release_gates.pd.DataFrame(
        {
            "yhat": [10.0, 20.0],
            "yhat_lo_80": [9.0, 21.0],
            "yhat_hi_80": [11.0, 19.0],
        }
    )

    failures = _interval_sanity_failures(frame)

    assert "interval level 80 has lower bound above upper bound" in failures
    assert "interval level 80 does not contain yhat" in failures


def test_run_command_timeout_captures_head_and_tail(tmp_path) -> None:
    code = (
        "import sys, time; "
        "print('STDOUT_HEAD', flush=True); "
        "print('x' * 5000, flush=True); "
        "sys.stderr.write('STDERR_HEAD\\n' + 'e' * 5000); "
        "sys.stderr.flush(); "
        "time.sleep(2)"
    )

    result = _run_command([sys.executable, "-c", code], cwd=tmp_path, timeout=1)

    assert result["returncode"] == 127
    assert "STDOUT_HEAD" in result["stdout_head"]
    assert "x" * 100 in result["stdout_tail"]
    assert "STDERR_HEAD" in result["stderr_head"]
    assert "TimeoutExpired" in result["stderr_tail"]


def test_live_streamlit_failure_captures_logs(tmp_path, monkeypatch) -> None:
    app = tmp_path / "streamlit_app.py"
    app.write_text("import streamlit as st\nst.write('x')\n", encoding="utf-8")

    class FakeProc:
        returncode = 1

        def poll(self) -> int:
            return 1

        def communicate(self, timeout: int) -> tuple[str, str]:
            return "fake stdout", "fake stderr"

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(release_gates.subprocess, "Popen", lambda *args, **kwargs: FakeProc())

    result = release_gates._live_streamlit_gate(tmp_path, timeout_seconds=1)

    assert result.status == "failed"
    assert result.details["process_returncode"] == 1
    assert result.details["stdout_head"] == "fake stdout"
    assert result.details["stderr_tail"] == "fake stderr"
