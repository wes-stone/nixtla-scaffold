"""Read only explicitly granted snapshots and reconstruct origin-local features."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from dataclasses import replace
import re

import numpy as np
import pandas as pd

from nixtla_scaffold.schema import DriverSnapshotGrant, ForecastSpec, KnownFutureRegressor


MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_ROWS = 250_000


class DriverSourceError(ValueError):
    """A source, timing or join contract failed; no substitute source is allowed."""


def read_granted_snapshot(grant: DriverSnapshotGrant) -> bytes:
    path = Path(grant.path)
    raw = str(path)
    local_extended = re.match(r"^\\\\\?\\[a-zA-Z]:\\", raw) is not None
    if raw.startswith(("\\\\", "//")) and not local_extended:
        raise DriverSourceError("Driver research accepts materialized local CSV snapshots, not network sources")
    from nixtla_scaffold.workbench_store import _filesystem_path
    path = _filesystem_path(path)
    resolved = str(path.resolve())
    local_extended = re.match(r"^\\\\\?\\[a-zA-Z]:\\", resolved) is not None
    if path.suffix.lower() != ".csv" or (resolved.startswith(("\\\\", "//")) and not local_extended):
        raise DriverSourceError("Driver research accepts materialized local CSV snapshots, not network sources")
    with path.open("rb") as file:
        data = file.read(MAX_SNAPSHOT_BYTES + 1)
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise DriverSourceError("Driver snapshot exceeds the 64 MiB read bound")
    if sha256(data).hexdigest() != grant.sha256:
        raise DriverSourceError("Driver snapshot changed; refresh requires a new explicitly granted vintage")
    return data


def persist_driver_snapshots(spec: ForecastSpec, output: Path) -> dict:
    """Save only referenced grants; failures are durable, never a replacement source."""
    from nixtla_scaffold.workbench_store import _filesystem_path
    output = _filesystem_path(output)
    referenced = {item.source.source_id for item in spec.regressors if item.source}
    receipts = {}
    for source_id in sorted(referenced):
        grant = next((item for item in spec.driver_snapshots if item.source_id == source_id), None)
        if grant is None:
            receipts[source_id] = {"status": "unavailable", "reason": "No explicit snapshot grant; no file was read."}
            continue
        try:
            data = read_granted_snapshot(grant)
        except (DriverSourceError, OSError) as exc:
            receipts[source_id] = {"status": "unavailable", "reason": str(exc), "sha256": grant.sha256}
            continue
        relative = f"driver_sources/{grant.sha256}.csv"
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise DriverSourceError("An existing saved driver snapshot differs; no artifact was overwritten")
        target.write_bytes(data)
        receipts[source_id] = {"status": "frozen", "artifact": relative, "sha256": grant.sha256,
                               "series_ids": list(grant.series_ids)}
    return receipts


def bind_saved_driver_snapshots(spec: ForecastSpec, root: Path, manifest: dict) -> ForecastSpec:
    """Resolve content-addressed local artifacts without changing any saved recipe."""
    from nixtla_scaffold.workbench_store import _filesystem_path
    root = _filesystem_path(root)
    if "driver_snapshot_artifacts" not in manifest:
        return spec
    records = manifest["driver_snapshot_artifacts"]
    grants = list(spec.driver_snapshots)
    referenced = {item.source.source_id for item in spec.regressors if item.source}
    for source_id in referenced:
        record = records.get(source_id, {})
        index = next((i for i, grant in enumerate(grants) if grant.source_id == source_id), None)
        if index is None or record.get("status") != "frozen":
            raise DriverSourceError("Saved driver source is unavailable; an explicitly approved new snapshot is required")
        grant = grants[index]
        if record.get("sha256") != grant.sha256 or record.get("series_ids") != list(grant.series_ids):
            raise DriverSourceError("Saved driver artifact scope differs from its original grant")
        path = root / record["artifact"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise DriverSourceError("Saved driver artifact escapes the selected run")
        grants[index] = replace(grant, path=str(path.resolve()))
        read_granted_snapshot(grants[index])
    return replace(spec, driver_snapshots=tuple(grants))


def source_features(
    history: pd.DataFrame, requirements: pd.DataFrame, spec: ForecastSpec,
    regressor: KnownFutureRegressor, *, freq: str, origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    contract = regressor.source
    if contract is None:
        raise DriverSourceError("A source contract is required")
    grant = next((item for item in spec.driver_snapshots if item.source_id == contract.source_id), None)
    if grant is None:
        raise DriverSourceError(f"Source {contract.source_id!r} has no explicit snapshot grant; no file was read")
    requested_ids = set(history["unique_id"].astype(str)) | set(requirements["unique_id"].astype(str))
    if not requested_ids.issubset(grant.series_ids):
        raise DriverSourceError("Requested target series exceed the granted source scope")
    try:
        expected_offset = pd.tseries.frequencies.to_offset(freq)
        source_offset = pd.tseries.frequencies.to_offset(contract.grain)
    except ValueError as exc:
        raise DriverSourceError("Invalid driver calendar/grain") from exc
    if source_offset != expected_offset:
        raise DriverSourceError("Driver grain differs from the target; implicit aggregation is forbidden")
    frame = pd.read_csv(BytesIO(read_granted_snapshot(grant)), keep_default_na=False,
                        dtype={contract.entity_col: str, contract.vintage_col: str},
                        nrows=MAX_SNAPSHOT_ROWS + 1, float_precision="round_trip")
    if len(frame) > MAX_SNAPSHOT_ROWS:
        raise DriverSourceError("Driver snapshot exceeds the 250,000-row bound")
    names = {
        contract.entity_col: "unique_id", contract.time_col: "ds", contract.value_col: "value",
        contract.release_col: "released", contract.vintage_col: "vintage",
        contract.period_end_col: "period_end", contract.units_col: "units",
    }
    missing = set(names) - set(frame.columns)
    if missing:
        raise DriverSourceError(f"Driver snapshot is missing required columns: {sorted(missing)}")
    frame = frame[list(names)].rename(columns=names)
    if frame.empty:
        raise DriverSourceError("Driver snapshot is empty")
    if not set(frame.unique_id).issubset(grant.series_ids) or frame.unique_id.eq("").any():
        raise DriverSourceError("Driver rows exceed granted entity scope; broadcasting is forbidden")
    if not frame.units.eq(contract.units).all():
        raise DriverSourceError("Driver snapshot has mixed or mismatched units")
    for column in ("ds", "released", "period_end"):
        try:
            frame[column] = pd.to_datetime(frame[column], errors="raise", format="mixed")
        except (ValueError, TypeError) as exc:
            raise DriverSourceError(f"Driver snapshot has invalid {column} timestamps") from exc
        if not pd.api.types.is_datetime64_dtype(frame[column].dtype) or frame[column].isna().any():
            raise DriverSourceError("Driver timestamps must be complete, timezone-normalized local calendar values")
    if frame.vintage.eq("").any() or frame.duplicated(["unique_id", "ds", "vintage"]).any():
        raise DriverSourceError("Driver snapshot has missing or duplicate entity/period/vintage keys")
    if frame.duplicated(["unique_id", "ds", "released"]).any():
        raise DriverSourceError("Multiple driver revisions have the same publication time")
    if frame.period_end.lt(frame.ds).any():
        raise DriverSourceError("Driver observation period ends before it starts")
    if frame.period_end.ge(frame.ds + expected_offset).any():
        raise DriverSourceError("Driver observation period spans multiple declared grain periods")
    if not frame.ds.map(expected_offset.is_on_offset).all():
        raise DriverSourceError("Driver observation dates do not match the declared target calendar")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if not np.isfinite(frame["value"]).all():
        raise DriverSourceError("Driver snapshot contains missing or nonfinite values")
    origin = pd.Timestamp(origin)
    cutoffs = history.groupby("unique_id", sort=False)["ds"].max()
    cutoffs.index = cutoffs.index.astype(str)
    row_cutoffs = frame.unique_id.map(cutoffs).clip(upper=origin)
    available = frame.released.le(row_cutoffs)
    if regressor.availability == "historical_only":
        available &= frame.period_end.le(row_cutoffs)
        if regressor.refresh_latency_days is not None:
            available &= (frame.period_end + pd.Timedelta(days=regressor.refresh_latency_days)).le(row_cutoffs)
    elif regressor.availability == "forecasted":
        raise DriverSourceError("Forecasted drivers require a paired upstream/downstream forecast; realized futures are not admissible")
    selected = (frame.loc[available & frame.unique_id.isin(requested_ids)]
                .sort_values("released").drop_duplicates(["unique_id", "ds"], keep="last"))
    value_col = regressor.value_col or regressor.name
    if value_col in {"unique_id", "ds", "y", "available_at"}:
        raise DriverSourceError("Driver value cannot overwrite target or calendar keys")
    values = selected[["unique_id", "ds", "value", "released"]].rename(
        columns={"value": value_col, "released": regressor.known_as_of_col})
    historical = history.drop(columns=[value_col], errors="ignore").copy()
    historical["unique_id"] = historical["unique_id"].astype(str)
    historical = historical.merge(values[["unique_id", "ds", value_col]],
                                  on=["unique_id", "ds"], how="left", validate="one_to_one")
    future = requirements.copy()
    future["unique_id"] = future["unique_id"].astype(str)
    future = future.merge(values, on=["unique_id", "ds"], how="left", validate="one_to_one")
    receipt = {
        "source_id": grant.source_id, "snapshot_sha256": grant.sha256, "query_ref": contract.query_ref,
        "origin": origin.isoformat(), "grain": contract.grain, "units": contract.units,
        "series_origins": {str(uid): min(pd.Timestamp(cutoff), origin).isoformat() for uid, cutoff in cutoffs.items()},
        "measure_kind": contract.measure_kind, "aggregation": "none", "join": "one_to_one_by_entity_and_period",
        "rows_in_snapshot": len(frame), "rows_available": len(selected),
        "rows_withheld_by_time": int((~available).sum()),
        "selected_vintages": sorted(selected.vintage.unique().tolist()),
        "historical_missing": int(historical[value_col].isna().sum()),
        "future_missing": int(future[value_col].isna().sum()),
        "availability": regressor.availability,
        "limitations": "Publication dates and business semantics are supplied metadata, not provider-certified facts.",
    }
    return historical, future, receipt
