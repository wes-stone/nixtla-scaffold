"""Build the deterministic package resource from an already-built canvas."""
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

root = Path(__file__).resolve().parents[1]
extension = root / ".github" / "extensions" / "nixtla-workbench"
dist = extension / "ui" / "dist"
if not (dist / "index.html").is_file():
    raise SystemExit("Build the UI before bundling: npm --prefix .github\\extensions\\nixtla-workbench\\ui run build")
blobs = {name: (extension / name).read_bytes() for name in ("extension.mjs", "server.mjs")}
blobs["workbench_contract.json"] = (root / "src" / "nixtla_scaffold" / "workbench_contract.json").read_bytes()
for path in dist.rglob("*"):
    if path.is_file() and not path.name.endswith(".map"):
        blobs["ui/dist/" + path.relative_to(dist).as_posix()] = path.read_bytes()
for package in ("react", "react-dom", "scheduler", "echarts", "zrender", "zod", "lucide-react", "tslib", "decimal.js"):
    package_dir = extension / "ui" / "node_modules" / package
    notices = [path for path in package_dir.iterdir() if path.is_file()
               and path.name.lower().startswith(("license", "licence", "notice", "copyrightnotice"))]
    if not notices:
        raise SystemExit(f"Missing distribution license for {package}; install the locked UI dependencies first.")
    for path in notices:
        blobs[f"third-party/{package}-{path.name}"] = path.read_bytes()
manifest = {
    "packageVersion": tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"],
    "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(blobs.items())},
}
blobs["bundle.json"] = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
output = root / "src" / "nixtla_scaffold" / "workbench_assets.zip"
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for name, data in sorted(blobs.items()):
        info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, data)
print(f"Built {output.name}: {output.stat().st_size} bytes; package {manifest['packageVersion']}")
