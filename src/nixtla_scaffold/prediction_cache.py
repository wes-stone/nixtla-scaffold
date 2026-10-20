"""Transactional native-fit cache; predictions only, never approval or exposure."""
from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from nixtla_scaffold.models import ModelResult, _with_resolved_candidate_identity


class CacheError(RuntimeError):
    pass


def _encode(value):
    if isinstance(value, pd.DataFrame):
        return {"$frame": _encode(value.to_dict("split")),
                "dtypes": [str(dtype) for dtype in value.dtypes], "attrs": _encode(value.attrs)}
    if value is pd.NA or value is pd.NaT:
        return {"$missing": True}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return {"$timestamp": pd.Timestamp(value).isoformat()}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$frame" in value:
        frame = pd.DataFrame(**_decode(value["$frame"]))
        for column, dtype in zip(frame, value["dtypes"]):
            frame[column] = frame[column].astype(dtype)
        frame.attrs = _decode(value["attrs"])
        return frame
    if "$timestamp" in value:
        return pd.Timestamp(value["$timestamp"])
    if "$float" in value:
        return float(value["$float"])
    if "$missing" in value:
        return pd.NA
    if "$tuple" in value:
        return tuple(_decode(item) for item in value["$tuple"])
    return {key: _decode(item) for key, item in value.items()}


def _bytes(value):
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class PredictionCache:
    """Group fits stay coherent (including shared ML preprocessing).

    BEGIN IMMEDIATE serializes misses so concurrent callers cannot fit the same
    unit twice. A crashed worker rolls back. No executable serialization is used.
    The bound is on live payload bytes; SQLite retains only bounded high-water space.
    """

    def __init__(self, directory: str | Path, max_bytes: int):
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.path, self.max_bytes = root / "native_predictions.sqlite", max_bytes
        self.runtime = _with_resolved_candidate_identity({}, resolved_candidates=[])["resolved_candidate_identity"]
        self.hits, self.misses, self.uncached, self.failed = 0, 0, 0, 0
        self.seconds = 0.
        with self.connect() as conn:
            conn.execute("PRAGMA auto_vacuum=FULL")
            conn.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, payload BLOB NOT NULL, "
                         "checksum TEXT NOT NULL, size INTEGER NOT NULL, created REAL NOT NULL)")
            conn.execute("BEGIN IMMEDIATE")
            size = conn.execute("SELECT COALESCE(SUM(size),0) FROM entries").fetchone()[0]
            for key, entry_size in conn.execute("SELECT key,size FROM entries ORDER BY created,key").fetchall():
                if size <= self.max_bytes:
                    break
                conn.execute("DELETE FROM entries WHERE key=?", (key,))
                size -= entry_size

    def connect(self):
        return sqlite3.connect(self.path, timeout=900)

    def key(self, training, profile, spec, fit):
        settings = spec.to_dict()
        for name in ("cache_directory", "cache_max_bytes", "evidence_directory", "known_target_exposure"):
            settings["evaluation"].pop(name, None)
        # These govern reporting/admission, not the native computation.
        settings["context"] = None
        return sha256(_bytes({
            "version": 1, "runtime": self.runtime, "training": training,
            "profile": profile.to_dict(), "spec": settings,
            "fit": f"{fit.__module__}.{fit.__qualname__}",
        })).hexdigest()

    def get_or_fit(self, key, compute):
        started = time.perf_counter()
        try:
            with self.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT payload,checksum FROM entries WHERE key=?", (key,)).fetchone()
                if row:
                    payload, checksum = row
                    if sha256(payload).hexdigest() != checksum:
                        raise CacheError(f"Corrupt native prediction cache entry {key}; remove this cache explicitly.")
                    try:
                        result = ModelResult(**_decode(json.loads(payload)))
                    except (TypeError, ValueError, KeyError) as exc:
                        raise CacheError(f"Unreadable native prediction cache entry {key}: {exc}") from exc
                    self.hits += 1
                    return result
                result = compute()
                self.misses += 1
                fits = result.model_policy_resolution.get("candidate_evidence", {}).get("fits", [])
                if any(row.get("status") == "failed" for row in fits):
                    self.failed += 1
                    return result
                payload = _bytes({field.name: getattr(result, field.name) for field in fields(ModelResult)})
                if len(payload) > self.max_bytes:
                    self.uncached += 1
                    return result
                size = conn.execute("SELECT COALESCE(SUM(size),0) FROM entries").fetchone()[0]
                for old_key, old_size in conn.execute("SELECT key,size FROM entries ORDER BY created,key").fetchall():
                    if size + len(payload) <= self.max_bytes:
                        break
                    conn.execute("DELETE FROM entries WHERE key=?", (old_key,))
                    size -= old_size
                conn.execute("INSERT INTO entries VALUES(?,?,?,?,?)",
                             (key, payload, sha256(payload).hexdigest(), len(payload), time.time()))
                return result
        finally:
            self.seconds += time.perf_counter() - started

    def receipt(self):
        return {"version": 1, "unit": "native_origin_cohort", "hits": self.hits, "misses": self.misses,
                "oversized_uncached": self.uncached, "max_payload_bytes": self.max_bytes,
                "failed_fit_units_uncached": self.failed,
                "native_fits": self.misses, "lookup_and_compute_seconds": self.seconds,
                "scope": "Native predictions only; scoring, exposure, calibration and approval rerun."}
