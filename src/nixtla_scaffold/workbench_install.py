"""Explicit, version-matched installation of the optional local canvas."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from importlib.resources import files
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
import zipfile


def install_workbench(target: str | Path, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise ValueError("Canvas installation requires explicit --yes approval.")
    destination = Path(target).absolute()
    if destination.is_symlink():
        raise ValueError("The extension destination cannot be a symbolic link.")
    bundle = files("nixtla_scaffold").joinpath("workbench_assets.zip")
    if not bundle.is_file():
        raise ValueError("This package has no built canvas bundle. Build the UI and run tools\\build_workbench_bundle.py before packaging.")
    data = bundle.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("bundle.json"))
        if manifest["packageVersion"] != version("nixtla-scaffold"):
            raise ValueError("Canvas bundle and installed package versions differ.")
        names = set(archive.namelist()) - {"bundle.json"}
        if names != set(manifest["files"]):
            raise ValueError("Canvas bundle file manifest is incomplete.")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or ":" in name or "\\" in name:
                raise ValueError("Canvas bundle contains an unsafe path.")
            if hashlib.sha256(archive.read(name)).hexdigest() != manifest["files"][name]:
                raise ValueError("Canvas bundle artifact hash mismatch.")
        backup = None
        if destination.exists():
            receipt_path = destination / ".nixtla-install.json"
            if not receipt_path.is_file():
                raise ValueError("Destination is not a managed Nixtla canvas installation; choose an empty destination.")
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = set(previous["files"]) | {".nixtla-install.json", "runtime.json"}
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
            if actual != expected:
                raise ValueError("Installed extension has added or missing files. Preserve your edits before updating.")
            for name, sha in previous["files"].items():
                path = destination / name
                if path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                    raise ValueError("Installed extension was modified; preserve your edits before updating.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".nixtla-install-", dir=destination.parent))
        try:
            for name in names:
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
            (stage / "runtime.json").write_text(json.dumps({"python": sys.executable, "packageVersion": manifest["packageVersion"]}), encoding="utf-8")
            (stage / ".nixtla-install.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if destination.exists():
                backups = destination.parent.parent / "nixtla-extension-backups"
                backups.mkdir(exist_ok=True)
                backup = backups / str(uuid.uuid4())
                destination.rename(backup)
            try:
                stage.rename(destination)
            except OSError:
                if backup is not None:
                    backup.rename(destination)
                raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {"path": str(destination), "packageVersion": manifest["packageVersion"],
            "backup": str(backup) if backup else None, "reloadRequired": True, "executedModels": False}
