"""Local, scope-bound records of target exposure and one-use confirmation."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from nixtla_scaffold.schema import ForecastSpec


def scope_directory(spec: ForecastSpec) -> Path:
    configured = spec.evaluation.evidence_directory or os.environ.get("NIXTLA_EVIDENCE_DIRECTORY")
    return Path(configured or ".nixtla-evidence").expanduser().resolve()


@contextmanager
def _transaction(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "exposure.sqlite3", timeout=30)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reservations "
            "(token TEXT PRIMARY KEY, snapshot TEXT NOT NULL, state TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS observations "
            "(key TEXT PRIMARY KEY, exposed INTEGER NOT NULL DEFAULT 0, "
            "reservation TEXT REFERENCES reservations(token))"
        )
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _keys(frame: pd.DataFrame) -> tuple[str, ...]:
    pairs = frame[["unique_id", "ds"]].copy()
    pairs["ds"] = pd.to_datetime(pairs["ds"])
    if pairs.isna().any().any():
        raise ValueError("Exposure records require non-null series and observation dates")
    return tuple(sorted({
        sha256(json.dumps([str(uid), pd.Timestamp(date).isoformat()], separators=(",", ":")).encode()).hexdigest()
        for uid, date in pairs.itertuples(index=False, name=None)
    }))


def _snapshot(frame: pd.DataFrame) -> str:
    canonical = frame.sort_values(["unique_id", "ds"])[sorted(frame.columns)]
    return sha256(canonical.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def record_exposure(frame: pd.DataFrame, spec: ForecastSpec) -> None:
    keys = _keys(frame.loc[frame["y"].notna()])
    if not keys:
        return
    with _transaction(scope_directory(spec)) as connection:
        connection.executemany(
            "INSERT INTO observations(key,exposed) VALUES(?,1) "
            "ON CONFLICT(key) DO UPDATE SET exposed=1 WHERE exposed=0",
            ((key,) for key in keys),
        )


def reserve_confirmation(frame: pd.DataFrame, spec: ForecastSpec) -> dict:
    directory = scope_directory(spec)
    receipt = {
        "scope": str(directory), "scope_kind": "recorded_runtime_workspace",
        "limitation": "No claim about exposure outside this recorded scope.",
        "status": "unavailable", "token": None, "snapshot": _snapshot(frame),
    }
    keys = _keys(frame)
    if not keys:
        return receipt
    if spec.evaluation.known_target_exposure == "exposed":
        record_exposure(frame, spec)
    with _transaction(directory) as connection:
        connection.executemany("INSERT OR IGNORE INTO observations(key) VALUES(?)", ((key,) for key in keys))
        unavailable = any(
            connection.execute("SELECT exposed,reservation FROM observations WHERE key=?", (key,)).fetchone()
            != (0, None) for key in keys
        )
        if unavailable:
            return {**receipt, "status": "previously_exposed_or_reserved", "observations": len(keys)}
        token = uuid4().hex
        connection.execute("INSERT INTO reservations VALUES(?,?,'reserved')", (token, receipt["snapshot"]))
        connection.executemany("UPDATE observations SET reservation=? WHERE key=?", ((token, key) for key in keys))
    return {**receipt, "status": "reserved", "token": token, "observations": len(keys)}


def consume_confirmation(receipt: dict, frame: pd.DataFrame, spec: ForecastSpec) -> dict:
    if receipt.get("status") != "reserved":
        return {**receipt, "passed": False}
    directory = scope_directory(spec)
    if receipt.get("scope") != str(directory) or receipt.get("snapshot") != _snapshot(frame):
        raise ValueError("Confirmation scope or source snapshot changed after reservation")
    if spec.evaluation.known_target_exposure == "exposed":
        record_exposure(frame, spec)
    keys = _keys(frame)
    with _transaction(directory) as connection:
        stored = connection.execute(
            "SELECT snapshot,state FROM reservations WHERE token=?", (receipt["token"],)
        ).fetchone()
        if stored != (receipt["snapshot"], "reserved"):
            return {**receipt, "status": "confirmation_consumed", "passed": False}
        if any(connection.execute(
            "SELECT exposed,reservation FROM observations WHERE key=?", (key,)
        ).fetchone() != (0, receipt["token"]) for key in keys):
            connection.execute("UPDATE reservations SET state='invalidated' WHERE token=?", (receipt["token"],))
            return {**receipt, "status": "exposed_after_reservation", "passed": False}
        connection.execute("UPDATE reservations SET state='consumed' WHERE token=?", (receipt["token"],))
        connection.executemany("UPDATE observations SET exposed=1 WHERE key=?", ((key,) for key in keys))
    return {**receipt, "status": "consumed_for_frozen_confirmation"}
