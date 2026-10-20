"""Owned workbench state, alongside the existing forecast ledger."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from nixtla_scaffold.ledger import init_ledger, register_run

SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=str)


def _filesystem_path(value: str | Path) -> Path:
    path = Path(value).absolute()
    if os.name != "nt" or str(path).startswith("\\\\?\\"):
        return path
    text = str(path)
    return Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)


def _database_uri(path: Path) -> str:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        path = Path("\\\\" + text[8:])
    elif text.startswith("\\\\?\\"):
        path = Path(text[4:])
    return path.as_uri()


def _publish_snapshot(stage: Path, destination: Path) -> None:
    for attempt in range(5):
        try:
            stage.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            logging.getLogger(__name__).warning(
                "Windows blocked snapshot publication; retrying (%d/4).", attempt + 1,
            )
            time.sleep(0.05 * 2 ** attempt)


class WorkbenchError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class WorkbenchStore:
    def __init__(self, workspace: str | Path):
        self.workspace = _filesystem_path(workspace).resolve(strict=True)
        self.root = self.workspace / ".nixtla-workbench"
        self.ledger = self.root / "ledger"
        self.database = self.ledger / "ledger.sqlite"
        if self.root.is_symlink() or not self.root.resolve().is_relative_to(self.workspace):
            raise WorkbenchError("unsafe_path", "Workbench storage cannot link outside the selected workspace.")

    @property
    def initialized(self) -> bool:
        if not self.database.exists():
            return False
        with self.connect(readonly=True, require_ready=False) as conn:
            table = conn.execute("SELECT name FROM sqlite_master WHERE name='wb_schema'").fetchone()
            if not table:
                return False
            row = conn.execute("SELECT version FROM wb_schema").fetchone()
            if not row or row[0] != SCHEMA_VERSION:
                raise WorkbenchError("schema_mismatch", "This workspace requires an explicit compatible migration.")
        return True

    def initialize(self) -> dict:
        if self.initialized:
            return {"initialized": True}
        if self.database.exists():
            raise WorkbenchError("migration_required", "Existing ledger needs an explicit migration; opening never migrates it.")
        self.root.mkdir(exist_ok=True)
        init_ledger(self.ledger)
        with self.connect(require_ready=False) as conn:
            conn.executescript("""
                CREATE TABLE wb_schema(version INTEGER NOT NULL);
                INSERT INTO wb_schema VALUES (1);
                CREATE TABLE wb_runs(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
                    path TEXT NOT NULL, hashes TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE wb_sources(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE wb_jobs(
                    id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
                    status TEXT NOT NULL, input TEXT NOT NULL, result TEXT,
                    error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE wb_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    status TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE wb_state(
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL);
                CREATE TABLE wb_notes(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    body TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE wb_blends(
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, recipe TEXT NOT NULL,
                    path TEXT NOT NULL, receipt TEXT NOT NULL);
            """)
        return {"initialized": True}

    @contextmanager
    def connect(self, *, readonly: bool = False, require_ready: bool = True):
        if not self.database.resolve().is_relative_to(self.root.resolve()):
            raise WorkbenchError("unsafe_path", "The ledger path is outside owned workbench storage.")
        if require_ready and not self.database.exists():
            raise WorkbenchError("not_initialized", "Create this local workspace before saving or running analysis.")
        if readonly:
            conn = sqlite3.connect(_database_uri(self.database) + "?mode=ro", uri=True, timeout=10)
        else:
            conn = sqlite3.connect(self.database, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            if not readonly:
                conn.commit()
        except BaseException:
            if not readonly:
                conn.rollback()
            raise
        finally:
            conn.close()

    def owned(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise WorkbenchError("unsafe_path", "Path is outside owned workbench storage.")
        return path

    @contextmanager
    def snapshot_lock(self):
        with self.owned("snapshot.lock").open("a+b") as lock:
            if not lock.tell():
                lock.write(b"0")
                lock.flush()
            deadline = time.monotonic() + 10
            while True:
                lock.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {11, 13, 35, 36}:
                        raise
                    if time.monotonic() >= deadline:
                        raise WorkbenchError("snapshot_busy", "Another process is publishing evidence. Retry after it finishes.") from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def source(self, source_id: str) -> tuple[Path, dict]:
        with self.connect(readonly=True) as conn:
            row = conn.execute("SELECT * FROM wb_sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise WorkbenchError("not_found", "Source snapshot no longer exists.")
        path = self.owned(row["path"])
        if digest(path.read_bytes()) != row["fingerprint"]:
            raise WorkbenchError("evidence_changed", "Source snapshot changed. Import a new snapshot.")
        return path, json.loads(row["metadata"])

    def save_source(self, name: str, csv: bytes, metadata: dict) -> str:
        if len(csv) > MAX_ARTIFACT_BYTES:
            raise WorkbenchError("too_large", "Source exceeds the local snapshot byte limit.")
        fingerprint = digest(csv)
        source_id = digest(csv + canonical_json(metadata).encode())
        path = self.owned(f"sources/{source_id}.csv")
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_bytes() != csv:
            raise WorkbenchError("evidence_changed", "An existing source snapshot has changed.")
        if not path.exists():
            with path.open("xb") as file:
                file.write(csv)
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO wb_sources VALUES(?,?,?,?,?)", (
                source_id, name, str(path.relative_to(self.root)), fingerprint, canonical_json(metadata),
            ))
        return source_id

    def run(self, run_id: str) -> tuple[Path, dict]:
        with self.connect(readonly=True) as conn:
            row = conn.execute("SELECT * FROM wb_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise WorkbenchError("not_found", "Forecast version no longer exists.")
        return self.owned(row["path"]), dict(row)

    def read_artifact(self, run_id: str, name: str) -> bytes:
        root, run = self.run(run_id)
        hashes = json.loads(run["hashes"])
        if name not in hashes:
            raise WorkbenchError("not_found", "Artifact is not part of this version.")
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise WorkbenchError("unsafe_path", "Artifact path escaped its version.")
        data = path.read_bytes()
        if digest(data) != hashes[name]:
            raise WorkbenchError("evidence_changed", "Artifact hash changed; this evidence cannot be used.")
        return data

    def import_run(self, directory: str | Path, *, name: str = "", metadata: dict | None = None) -> dict:
        with self.snapshot_lock():
            return self._import_run(directory, name=name, metadata=metadata)

    def _import_run(self, directory: str | Path, *, name: str = "", metadata: dict | None = None) -> dict:
        source = _filesystem_path(directory).resolve(strict=True)
        if not source.is_dir() or not (source / "manifest.json").is_file():
            raise WorkbenchError("invalid_run", "Select a completed run folder containing manifest.json.")
        allowed = {".json", ".jsonl", ".csv", ".md", ".html", ".xlsx", ".py", ".ps1", ".cmd", ".txt", ".yaml", ".png", ".jpg", ".jpeg"}
        files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
        if len(files) > 500 or any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in files):
            raise WorkbenchError("unsafe_run", "Run has too many artifacts or linked files.")
        if sum(p.stat().st_size for p in files) > MAX_ARTIFACT_BYTES:
            raise WorkbenchError("too_large", "Run exceeds the 128 MiB snapshot limit.")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("spec"), dict):
            raise WorkbenchError("invalid_run", "Run manifest has no forecast spec.")
        if manifest.get("purpose") == "judgment_scenario":
            judgment = manifest.get("judgment", {})
            try:
                metadata = {**(metadata or {}), "purpose": "judgment_scenario", "adopted": False, "planningReady": False,
                            "parentRunId": judgment["parentRunId"], "judgmentBaseRunId": judgment["baseRunId"],
                            "judgmentSeries": judgment["series"], "judgmentLabel": judgment["settings"]["label"],
                            "judgmentModel": judgment["settings"]["model"]}
            except (KeyError, TypeError) as exc:
                raise WorkbenchError("invalid_run", "Judgment version has incomplete lineage metadata.") from exc
        blobs = {p.relative_to(source).as_posix(): p.read_bytes() for p in files}
        hashes = {key: digest(value) for key, value in blobs.items()}
        run_id = digest(canonical_json(hashes).encode())
        with self.connect(readonly=True) as conn:
            existing = conn.execute("SELECT id FROM wb_runs WHERE id=?", (run_id,)).fetchone()
        if existing:
            for key in json.loads(self.run(run_id)[1]["hashes"]):
                self.read_artifact(run_id, key)
            return {"id": run_id, "reused": True}
        runs = self.owned("versions")
        runs.mkdir(exist_ok=True)
        destination = runs / run_id
        stage = Path(tempfile.mkdtemp(prefix="import-", dir=runs))
        try:
            for relative, data in blobs.items():
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            if not destination.exists():
                _publish_snapshot(stage, destination)
            else:
                existing_names = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
                if existing_names - {"ledger_context.json"} != set(blobs) - {"ledger_context.json"}:
                    raise WorkbenchError("import_conflict", "Interrupted snapshot has unexpected or missing artifacts; no files were overwritten.")
                for relative, data in blobs.items():
                    if relative == "ledger_context.json":
                        continue
                    if (destination / relative).read_bytes() != data:
                        raise WorkbenchError("import_conflict", "Incomplete snapshot differs; reconcile it before retrying.")
            # Use content identity as the legacy version label, so changed recipes
            # cannot collide under the legacy forecast-key identity.
            register_run(
                self.ledger, destination, forecast_key=name or source.name,
                version_label=run_id, export=False,
            )
            hashes = {
                p.relative_to(destination).as_posix(): digest(p.read_bytes())
                for p in destination.rglob("*") if p.is_file()
            }
            with self.connect() as conn:
                conn.execute("INSERT INTO wb_runs VALUES(?,?,?,?,?,?)", (
                    run_id, name or source.name, timestamp(),
                    str(destination.relative_to(self.root)), canonical_json(hashes),
                    canonical_json(metadata or {}),
                ))
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return {"id": run_id, "reused": False}

    def event(self, job_id: str, status: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO wb_events(job_id,status,message,created_at) VALUES(?,?,?,?)",
                         (job_id, status, message, timestamp()))

    def job(self, job_id: str) -> dict:
        with self.connect(readonly=True) as conn:
            row = conn.execute("SELECT * FROM wb_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise WorkbenchError("not_found", "Job no longer exists.")
        result = dict(row)
        result["input"] = json.loads(result["input"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    def finish_job(self, job_id: str, status: str, *, result=None, error: str | None = None) -> dict:
        if status not in {"completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("invalid terminal job status")
        with self.connect() as conn:
            from nixtla_scaffold.workbench_jobs import settle_budget
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT input,status FROM wb_jobs WHERE id=?", (job_id,)).fetchone()
            if row is not None and row["status"] in {"queued", "running", "cancel_requested"}:
                conn.execute("UPDATE wb_jobs SET input=? WHERE id=?", (canonical_json(settle_budget(json.loads(row["input"]))), job_id))
            conn.execute(
                "UPDATE wb_jobs SET status=?,result=COALESCE(?,result),error=?,updated_at=? WHERE id=? AND status IN ('queued','running','cancel_requested')",
                (status, canonical_json(result) if result is not None else None, error, timestamp(), job_id),
            )
        return self.job(job_id)

    def checkpoint_job(self, job_id: str, result: dict) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE wb_jobs SET result=?,updated_at=? WHERE id=? AND status IN ('running','cancel_requested')",
                         (canonical_json(result), timestamp(), job_id))
