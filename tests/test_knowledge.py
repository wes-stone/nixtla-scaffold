from __future__ import annotations

import hashlib
import json

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
    assert "name: nixtla-forecast" in output
    assert "End-to-end FPPy-aligned time series forecasting" in output


def test_skill_manifest_pins_canonical_skill_hash() -> None:
    manifest = load_skill_manifest()
    digest = hashlib.sha256(load_agent_skill().encode("utf-8")).hexdigest()

    assert manifest["skill_version"] == "1.3.0"
    assert manifest["source_sha256"] == digest


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


def test_guide_search_includes_agent_recipes() -> None:
    rendered = format_knowledge(search_knowledge("DAX"))
    setup_rendered = format_knowledge(search_knowledge("setup"))

    assert "## DAX MCP handoff" in rendered
    assert "dax-query-server" in rendered
    assert "Agent setup intake" in setup_rendered
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
