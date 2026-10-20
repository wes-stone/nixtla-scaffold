from __future__ import annotations

import hashlib
import json
import sys

import pytest

from nixtla_scaffold.cli import main
from nixtla_scaffold.citations import FPPY_CITATION
from nixtla_scaffold.knowledge import (
    check_agent_skill,
    format_knowledge,
    load_agent_skill,
    load_skill_manifest,
    search_knowledge,
    sync_agent_skill,
)
from nixtla_scaffold.mcp_contracts import describe_contract


def test_guide_search_returns_readable_sections() -> None:
    rendered = format_knowledge(search_knowledge("intervals"))

    assert "## Intervals need evidence" in rendered
    assert "\n\n- guidance:" in rendered
    assert FPPY_CITATION in rendered


def test_guide_skill_prints_bundled_agent_skill(capsys) -> None:
    assert "name: nixtla-forecast" in load_agent_skill()

    exit_code = main(["guide", "skill"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output == load_agent_skill().replace("\r\n", "\n") + "\n"
    assert "name: nixtla-forecast" in output
    assert "End-to-end FPPy-aligned time series forecasting" in output


def test_skill_manifest_pins_canonical_skill_hash() -> None:
    manifest = load_skill_manifest()
    digest = hashlib.sha256(load_agent_skill().encode("utf-8")).hexdigest()

    assert manifest["skill_version"] == "2.6.0"
    assert manifest["source_sha256"] == digest
    skill = load_agent_skill()
    assert "ANALYZE BEFORE ASKING" in skill
    assert "never require every field" in skill
    assert "NEVER skip the intake questions" not in skill
    assert "USE THE FORECAST WORKBENCH" in skill
    assert "DO NOT STITCH CROSS-RUN WINNERS" in skill
    assert "DO NOT FORCE EXCEL AUTOMATION" in skill
    assert "STOP ON SKILL/PACKAGE DRIFT" in skill
    assert "configuration_diagnostics" in skill
    assert "Configuration review before more model search" in skill
    assert 'AnomalySpec(mode="detect")' in skill
    assert "custom holidays require `known_as_of`" in skill
    assert "native_fitted_points.csv" in skill
    assert 'configuration_guard="exclude"' in skill
    assert skill.count("### Paired ensemble procedure") == 1
    assert "NeuralForecast and Torch are intentionally excluded" in skill
    for obsolete in (
        "NEVER skip the intake questions",
        "ALL of these, every time",
        "Ask the user the feature questions",
        "Only proceed after the user confirms",
        "**Always ask**",
        "Before touching ANY data",
        "Proceed? [yes/adjust]",
    ):
        assert obsolete not in skill
    for required in (
        "installed_runtime.python",
        "--research-max-variants-per-iteration 1",
        "`compare-models` executes a new forecast",
    ):
        assert required.lower() in " ".join(skill.lower().split())
    for required in (
        "one variant each", "one focused", "provisional", "inspection-only",
        "previously viewed", "earlier observable outcomes", "blend-residual",
        "resolved_candidate_fingerprint", "RMSSE", "unavailable",
        "leave unknown context fields empty", "not preempt an individual fit",
    ):
        assert required.lower() in " ".join(skill.lower().split())
    assert len(skill.splitlines()) <= 450


def test_skill_check_and_confirmed_sync_preserve_backup(tmp_path, capsys) -> None:
    target = tmp_path / "nixtla-forecast"
    target.mkdir()
    (target / "SKILL.md").write_text("stale skill\n", encoding="utf-8")

    before = check_agent_skill(target)
    assert before["status"] == "drift"

    synced = sync_agent_skill(target, confirmed=True)
    assert synced["synced"] is True
    assert synced["backup_path"]
    assert (target / "skill_manifest.json").exists()
    assert (target / "SKILL.md").read_bytes() == load_agent_skill().encode("utf-8")

    exit_code = main(["skill", "check", "--target", str(target)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "in_sync"
    assert payload["installed_manifest_matches"]
    assert payload["installed_runtime"]["python"] == sys.executable
    assert payload["installed_runtime"]["package_version"] == payload["package_version"]


def test_skill_check_detects_stale_manifest_even_when_text_matches(tmp_path) -> None:
    sync_agent_skill(tmp_path, confirmed=True)
    manifest_path = tmp_path / "skill_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skill_version"] = "old"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert check_agent_skill(tmp_path)["status"] == "drift"
    assert not check_agent_skill(tmp_path)["installed_manifest_matches"]
    assert sync_agent_skill(tmp_path, confirmed=True)["in_sync"]


def test_skill_sync_refuses_incompatible_runtime_and_unapproved_replacement(tmp_path, monkeypatch) -> None:
    from nixtla_scaffold import knowledge

    target = tmp_path / "SKILL.md"
    target.write_text("user's existing skill", encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation"):
        sync_agent_skill(tmp_path)
    monkeypatch.setattr(knowledge, "_package_version", lambda: "0.2.3")
    with pytest.raises(RuntimeError, match="compatible installed package"):
        sync_agent_skill(tmp_path, confirmed=True)
    assert target.read_text(encoding="utf-8") == "user's existing skill"


def test_skill_check_exit_code_rejects_incompatible_runtime(tmp_path, monkeypatch, capsys) -> None:
    from nixtla_scaffold import knowledge

    sync_agent_skill(tmp_path, confirmed=True)
    monkeypatch.setattr(knowledge, "_package_version", lambda: "0.2.3")
    assert main(["skill", "check", "--target", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["in_sync"] and not payload["package_compatible"]


def test_guide_search_includes_agent_recipes() -> None:
    rendered = format_knowledge(search_knowledge("DAX"))
    setup_rendered = format_knowledge(search_knowledge("setup"))

    assert "## DAX MCP handoff" in rendered
    assert "dax-query-server" in rendered
    assert "Analyst-first setup" in setup_rendered
    assert "mcp-regressor-search" in setup_rendered
    assert "mlforecast" in setup_rendered.lower()
    assert "hierarchicalforecast" in setup_rendered.lower()


def test_guide_search_includes_nixtla_source_maps() -> None:
    rendered = format_knowledge(search_knowledge("python/statsforecast"))
    utils_rendered = format_knowledge(search_knowledge("utilsforecast/preprocessing.py"))
    notebook_rendered = format_knowledge(search_knowledge("IntermittentData"))

    assert "## StatsForecast" in rendered
    assert "python/statsforecast/core.py" in rendered
    assert "['" not in rendered
    assert "utilsforecast/preprocessing.py" in utils_rendered
    assert "StatsForecast intermittent demand tutorial" in notebook_rendered
    assert "nbs/docs/tutorials/IntermittentData.ipynb" in notebook_rendered


def test_guide_search_includes_weighted_and_diagnostics_guidance() -> None:
    rendered = format_knowledge(search_knowledge("model_weights"))

    assert "Weighted ensemble" in rendered
    assert "model_weights.csv" in rendered
    assert "diagnostics.json" in format_knowledge(search_knowledge("diagnostics"))


def test_mcp_contract_describes_required_columns_and_recipes() -> None:
    contract = describe_contract()

    assert set(contract["required_columns"]) == {"unique_id", "ds", "y"}
    assert contract["recipes"]
