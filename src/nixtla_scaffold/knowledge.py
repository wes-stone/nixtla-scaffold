from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from importlib import resources
from pathlib import Path
import shutil
from typing import Any

from nixtla_scaffold.citations import FPPY_CITATION

FPPY_URL_MARKERS = ("otexts.com/fpppy", "OTexts.com/fpppy")
SKILL_NAME = "nixtla-forecast"
SKILL_FILE = "SKILL.md"
SKILL_MANIFEST_FILE = "skill_manifest.json"


def load_knowledge() -> dict[str, Any]:
    with resources.files("nixtla_scaffold").joinpath("knowledge_base.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_agent_skill() -> str:
    """Return the bundled nixtla-forecast agent skill text."""

    return _canonical_skill_bytes(SKILL_FILE).decode("utf-8")


def load_skill_manifest() -> dict[str, Any]:
    """Return metadata that pins the canonical skill text to a deterministic hash."""

    return json.loads(_canonical_skill_bytes(SKILL_MANIFEST_FILE).decode("utf-8"))


def check_agent_skill(target: str | Path | None = None) -> dict[str, Any]:
    """Compare an installed skill with the repository/package source of truth."""

    skill_path, manifest_path = _skill_target_paths(target)
    canonical = _canonical_skill_bytes(SKILL_FILE)
    manifest = load_skill_manifest()
    expected_hash = str(manifest.get("source_sha256", ""))
    canonical_hash = _sha256_bytes(canonical)
    installed_hash = _sha256_path(skill_path) if skill_path.is_file() else None
    canonical_valid = bool(expected_hash) and canonical_hash == expected_hash
    if not canonical_valid:
        status = "canonical_manifest_mismatch"
    elif installed_hash is None:
        status = "missing"
    elif installed_hash == canonical_hash:
        status = "in_sync"
    else:
        status = "drift"
    package_version = _package_version()
    return {
        "skill_name": SKILL_NAME,
        "skill_version": manifest.get("skill_version"),
        "compatible_package": manifest.get("compatible_package"),
        "package_version": package_version,
        "package_compatible": _package_version_compatible(package_version, manifest),
        "status": status,
        "in_sync": status == "in_sync",
        "canonical_valid": canonical_valid,
        "canonical_sha256": canonical_hash,
        "expected_sha256": expected_hash,
        "installed_sha256": installed_hash,
        "skill_path": str(skill_path),
        "manifest_path": str(manifest_path),
    }


def sync_agent_skill(
    target: str | Path | None = None,
    *,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Install the canonical skill after explicit confirmation, preserving a backup."""

    if not confirmed:
        raise ValueError("skill sync requires explicit confirmation; rerun with --yes")
    before = check_agent_skill(target)
    if not before["canonical_valid"]:
        raise RuntimeError("canonical skill hash does not match skill_manifest.json; refusing to sync")
    skill_path, manifest_path = _skill_target_paths(target)
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if skill_path.exists() and not before["in_sync"]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = skill_path.with_name(f"{skill_path.name}.backup-{timestamp}")
        shutil.copy2(skill_path, backup_path)
        backup = str(backup_path)
    skill_path.write_bytes(_canonical_skill_bytes(SKILL_FILE))
    manifest_path.write_bytes(_canonical_skill_bytes(SKILL_MANIFEST_FILE))
    after = check_agent_skill(target)
    return {
        **after,
        "previous_status": before["status"],
        "backup_path": backup,
        "synced": after["in_sync"],
    }


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


def _canonical_skill_bytes(filename: str) -> bytes:
    repo_file = Path(__file__).resolve().parents[2] / "skills" / SKILL_NAME / filename
    if repo_file.is_file():
        return repo_file.read_bytes()
    package_file = resources.files("nixtla_scaffold").joinpath(f"skills/{SKILL_NAME}/{filename}")
    if package_file.is_file():
        return package_file.read_bytes()
    raise FileNotFoundError(f"bundled {SKILL_NAME} {filename} was not found in the package or source checkout")


def _skill_target_paths(target: str | Path | None) -> tuple[Path, Path]:
    base = Path(target).expanduser() if target is not None else Path.home() / ".copilot" / "skills" / SKILL_NAME
    if base.name.casefold() == SKILL_FILE.casefold():
        return base, base.with_name(SKILL_MANIFEST_FILE)
    return base / SKILL_FILE, base / SKILL_MANIFEST_FILE


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return importlib_metadata.version("nixtla-scaffold")
    except importlib_metadata.PackageNotFoundError:
        return "source"


def _package_version_compatible(version: str, manifest: dict[str, Any]) -> bool | None:
    parsed = _version_tuple(version)
    minimum = _version_tuple(str(manifest.get("minimum_package_version", "")))
    maximum = _version_tuple(str(manifest.get("maximum_package_version_exclusive", "")))
    if parsed is None or minimum is None or maximum is None:
        return None
    return minimum <= parsed < maximum


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    parts = value.split(".", maxsplit=3)[:3]
    if len(parts) != 3:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None
