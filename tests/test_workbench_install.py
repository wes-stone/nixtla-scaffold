import json
from pathlib import Path

import pytest

from nixtla_scaffold.workbench_install import install_workbench


def test_installer_requires_approval_and_preserves_unmanaged_directory(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        install_workbench(tmp_path / "canvas")
    target = tmp_path / "user-extension"
    target.mkdir()
    (target / "keep.txt").write_text("user content", encoding="utf-8")
    with pytest.raises(ValueError, match="not a managed"):
        install_workbench(target, confirmed=True)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "user content"


def test_installer_supports_clean_outside_checkout_and_safe_update(tmp_path):
    target = tmp_path / "ordinary-workspace" / ".github" / "extensions" / "nixtla-workbench"
    first = install_workbench(target, confirmed=True)
    assert first["reloadRequired"] and not first["executedModels"]
    assert (target / "ui" / "dist" / "index.html").is_file()
    assert (target / "third-party" / "echarts-LICENSE").is_file()
    assert Path(json.loads((target / "runtime.json").read_text())["python"]).is_file()
    assert "compare_versions" in json.loads((target / "workbench_contract.json").read_text())["operations"]
    assert {"evidence", "evidence_image", "export_evidence"} <= json.loads((target / "workbench_contract.json").read_text())["operations"].keys()
    assert {"explore_catalog", "explore_schema", "explore_preview", "explore_queries", "explore_save",
            "explore_export", "explore_download"} <= json.loads((target / "workbench_contract.json").read_text())["operations"].keys()
    assert {"calendar_preview", "calendar_export"} <= json.loads((target / "workbench_contract.json").read_text())["operations"].keys()
    assert {"judgment_source", "judgment_import", "judgment_save", "judgment_export"} <= json.loads((target / "workbench_contract.json").read_text())["operations"].keys()
    second = install_workbench(target, confirmed=True)
    assert Path(second["backup"]).is_dir()
    assert not Path(second["backup"]).is_relative_to(target.parent)
    (target / "extension.mjs").write_text("user edit", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        install_workbench(target, confirmed=True)
    assert (target / "extension.mjs").read_text() == "user edit"
