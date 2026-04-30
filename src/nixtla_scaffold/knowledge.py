from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from nixtla_scaffold.citations import FPPY_CITATION

FPPY_URL_MARKERS = ("otexts.com/fpppy", "OTexts.com/fpppy")


def load_knowledge() -> dict[str, Any]:
    with resources.files("nixtla_scaffold").joinpath("knowledge_base.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_agent_skill() -> str:
    """Return the bundled nixtla-forecast agent skill text."""

    package_skill = resources.files("nixtla_scaffold").joinpath("skills/nixtla-forecast/SKILL.md")
    if package_skill.is_file():
        return package_skill.read_text(encoding="utf-8")
    repo_skill = Path(__file__).resolve().parents[2] / "skills" / "nixtla-forecast" / "SKILL.md"
    if repo_skill.exists():
        return repo_skill.read_text(encoding="utf-8")
    raise FileNotFoundError("bundled nixtla-forecast skill was not found in the package or source checkout")


def search_knowledge(query: str | None = None) -> list[dict[str, str]]:
    kb = load_knowledge()
    entries: list[dict[str, str]] = []
    for value in kb.values():
        if isinstance(value, list):
            entries.extend(entry for entry in value if isinstance(entry, dict))
    if not query:
        return entries
    needle = query.casefold()
    return [
        entry
        for entry in entries
        if needle in " ".join(str(value) for value in entry.values()).casefold()
    ]


def format_knowledge(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "No matching guidance found.\n"
    lines: list[str] = []
    for entry in entries:
        title = entry.get("name") or entry.get("title") or entry.get("check")
        lines.append(f"## {title}")
        for key, value in entry.items():
            if key in {"name", "title", "check"}:
                continue
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            if _contains_fppy_source(value):
                value = f"{value} | citation: {FPPY_CITATION}"
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n\n".join(lines).strip() + "\n"


def _contains_fppy_source(value: Any) -> bool:
    text = str(value)
    return FPPY_CITATION not in text and any(marker in text for marker in FPPY_URL_MARKERS)

