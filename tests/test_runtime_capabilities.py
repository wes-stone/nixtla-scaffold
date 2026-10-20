import json
import subprocess
from importlib import metadata
from types import SimpleNamespace

import pytest

from nixtla_scaffold import runtime
from nixtla_scaffold.cli import main
from nixtla_scaffold.workbench import dispatch


def test_distribution_excludes_neural_dependencies():
    info = metadata.metadata("nixtla-scaffold")
    assert "neural" not in info.get_all("Provides-Extra", [])
    requirements = metadata.requires("nixtla-scaffold")
    assert requirements
    assert not any(requirement.lower().startswith(("neuralforecast", "torch"))
                   for requirement in requirements)


def test_catalog_is_separate_from_executed_models(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_version", lambda name: "1.1.0" if name == "smooth" else None)
    monkeypatch.setattr(runtime, "_probe", lambda modules: {"status": "importable", "error": ""})
    result = dispatch(tmp_path / "does-not-exist", "runtime_capabilities", {})
    assert not (tmp_path / "does-not-exist").exists()
    assert result["registered_recipes"] == 53
    assert result["importable_recipes"] == 3
    smooth = next(engine for engine in result["engines"] if engine["package"] == "smooth")
    assert smooth["compatibility"] == "version_matched"
    assert all(engine["execution"] == "not_attempted" for engine in result["engines"])
    assert all(engine["data_eligibility"] == "not_evaluated" for engine in result["engines"])
    assert "neuralforecast" not in {engine["package"] for engine in result["engines"]}
    assert {"neuralforecast", "torch"}.isdisjoint(result["identity"]["runtime_versions"])
    assert "conformalforecast" in result["identity"]["runtime_versions"]


@pytest.mark.parametrize("failure,expected", [
    (subprocess.TimeoutExpired("probe", 8), "timeout"),
    (OSError("denied"), "probe_failed"),
    (SimpleNamespace(returncode=1, stdout="", stderr="ImportError: broken binary"), "import_failed"),
    (SimpleNamespace(returncode=0, stdout="", stderr=""), "importable"),
])
def test_probe_outcomes_are_explicit(monkeypatch, failure, expected):
    def run(*args, **kwargs):
        assert kwargs["timeout"] == 30
        assert "-B" in args[0]
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime._probe(("smooth",))["status"] == expected


def test_cli_uses_only_selected_runtime(monkeypatch, capsys):
    selected = []
    monkeypatch.setattr(runtime, "inspect_runtime", lambda python: selected.append(python) or {"scope": "imports only"})
    assert main(["runtime", "--python", "explicit-python"]) == 0
    assert selected == ["explicit-python"]
    assert json.loads(capsys.readouterr().out)["scope"] == "imports only"


def test_explicit_runtime_timeout_is_actionable(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("runtime", 55)
    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="55 seconds"):
        runtime.inspect_runtime("explicit-python")
