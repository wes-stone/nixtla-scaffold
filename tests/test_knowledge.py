from __future__ import annotations

from nixtla_scaffold.cli import main
from nixtla_scaffold.citations import FPPY_CITATION
from nixtla_scaffold.knowledge import format_knowledge, load_agent_skill, search_knowledge
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

