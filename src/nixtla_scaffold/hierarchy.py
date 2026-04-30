from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nixtla_scaffold.data import read_tabular_source


def aggregate_hierarchy_frame(
    data: str | Path | pd.DataFrame,
    *,
    hierarchy_cols: Sequence[str],
    sheet: str | int | None = None,
    time_col: str = "ds",
    target_col: str = "y",
    total_label: str = "Total",
) -> pd.DataFrame:
    """Aggregate raw leaf-level data into canonical long-form hierarchy nodes."""

    raw = read_tabular_source(data, sheet=sheet)
    hierarchy_cols = tuple(hierarchy_cols)
    _validate_hierarchy_input(raw, hierarchy_cols, time_col, target_col)

    frame = raw.copy()
    frame["ds"] = pd.to_datetime(frame[time_col], errors="coerce")
    frame["y"] = pd.to_numeric(frame[target_col], errors="coerce")
    if frame["ds"].isna().any():
        raise ValueError(f"{int(frame['ds'].isna().sum())} rows have invalid dates in '{time_col}'")
    invalid_y = int(frame["y"].isna().sum() - raw[target_col].isna().sum())
    if invalid_y:
        raise ValueError(f"{invalid_y} rows have non-numeric values in '{target_col}'")
    if frame.empty:
        raise ValueError("hierarchy input is empty; at least one row is required")
    _reject_reserved_hierarchy_values(frame, hierarchy_cols)

    nodes = [_total_node(frame, hierarchy_cols, total_label)]
    for depth in range(1, len(hierarchy_cols) + 1):
        prefix = hierarchy_cols[:depth]
        grouped = frame.groupby(["ds", *prefix], dropna=False, as_index=False)["y"].sum()
        if grouped[list(prefix)].isna().any().any():
            raise ValueError(f"hierarchy columns cannot contain null values: {list(prefix)}")
        grouped["unique_id"] = grouped.apply(lambda row: _node_id(prefix, row), axis=1)
        grouped["hierarchy_level"] = "/".join(prefix)
        grouped["hierarchy_depth"] = depth
        for col in hierarchy_cols[depth:]:
            grouped[col] = None
        nodes.append(grouped[["unique_id", "ds", "y", "hierarchy_level", "hierarchy_depth", *hierarchy_cols]])

    out = pd.concat(nodes, ignore_index=True)
    return out.sort_values(["hierarchy_depth", "unique_id", "ds"]).reset_index(drop=True)


def hierarchy_summary(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"unique_id", "ds", "y", "hierarchy_level", "hierarchy_depth"}
    if not required.issubset(frame.columns):
        raise ValueError(f"expected hierarchy columns {required}, got {set(frame.columns)}")

    levels = (
        frame.groupby(["hierarchy_depth", "hierarchy_level"], as_index=False)
        .agg(series_count=("unique_id", "nunique"), rows=("unique_id", "size"), total_y=("y", "sum"))
        .sort_values(["hierarchy_depth", "hierarchy_level"])
    )
    return {
        "rows": int(len(frame)),
        "series_count": int(frame["unique_id"].nunique()),
        "levels": levels.to_dict("records"),
    }


def hierarchy_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hierarchy node for joining metadata onto forecasts."""

    if "hierarchy_level" not in frame.columns or "hierarchy_depth" not in frame.columns:
        return pd.DataFrame()
    metadata_cols = ["unique_id"]
    for col in frame.columns:
        if col in {"unique_id", "ds", "y"}:
            continue
        if col.startswith("hierarchy_") or frame.groupby("unique_id")[col].nunique(dropna=False).max() <= 1:
            metadata_cols.append(col)
    return frame[metadata_cols].drop_duplicates("unique_id").reset_index(drop=True)


def hierarchy_coherence(frame: pd.DataFrame, *, value_cols: Sequence[str] = ("yhat", "yhat_scenario")) -> pd.DataFrame:
    """Compare parent forecasts with sums of their immediate children."""

    required = {"unique_id", "ds", "hierarchy_depth"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    work["hierarchy_depth"] = pd.to_numeric(work["hierarchy_depth"], errors="coerce")
    checks: list[pd.DataFrame] = []
    for value_col in value_cols:
        if value_col not in work.columns:
            continue
        child = work[work["hierarchy_depth"] > 0].copy()
        root_ids = work.loc[work["hierarchy_depth"] == 0, "unique_id"].dropna().astype(str).unique()
        root_id = root_ids[0] if len(root_ids) else "Total"
        child["parent_unique_id"] = child.apply(lambda row: _parent_id(row, root_id), axis=1)
        child_sum = child.groupby(["parent_unique_id", "ds"], as_index=False)[value_col].sum()
        child_sum = child_sum.rename(columns={value_col: "child_sum"})
        parent = work[["unique_id", "ds", value_col]].rename(columns={"unique_id": "parent_unique_id", value_col: "parent_value"})
        check = child_sum.merge(parent, on=["parent_unique_id", "ds"], how="left")
        check["value_col"] = value_col
        check["gap"] = check["parent_value"] - check["child_sum"]
        check["gap_pct"] = check["gap"] / check["parent_value"].abs().where(check["parent_value"].abs() > 0)
        checks.append(check[["ds", "value_col", "parent_unique_id", "parent_value", "child_sum", "gap", "gap_pct"]])
    if not checks:
        return pd.DataFrame()
    return pd.concat(checks, ignore_index=True).sort_values(["value_col", "parent_unique_id", "ds"]).reset_index(drop=True)


def reconcile_hierarchy_forecast(
    frame: pd.DataFrame,
    *,
    method: str = "bottom_up",
    value_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return a coherent hierarchy forecast plus reconciliation audit rows."""

    if method == "none":
        return frame.copy(), pd.DataFrame(), []
    if "hierarchy_depth" not in frame.columns:
        return frame.copy(), pd.DataFrame(), [f"hierarchy reconciliation '{method}' requested, but hierarchy metadata is absent"]

    value_cols = tuple(value_cols or _reconciliation_value_columns(frame))
    if not value_cols:
        return frame.copy(), pd.DataFrame(), [f"hierarchy reconciliation '{method}' requested, but no numeric forecast columns were available"]

    out = frame.copy()
    pre = hierarchy_coherence(out, value_cols=value_cols)
    applied_methods: dict[str, str] = {}
    warnings: list[str] = []
    for value_col in value_cols:
        if method == "bottom_up":
            out[value_col] = _bottom_up_reconciled_values(out, value_col)
            applied_methods[value_col] = "bottom_up"
            continue
        try:
            out[value_col] = _hierarchicalforecast_reconciled_values(out, value_col, method=method)
            applied_methods[value_col] = method
        except Exception as exc:
            out[value_col] = _bottom_up_reconciled_values(out, value_col)
            applied_methods[value_col] = "bottom_up_fallback"
            warnings.append(
                f"HierarchicalForecast reconciliation '{method}' failed for {value_col}; "
                f"used bottom_up fallback ({exc})"
            )

    _recompute_event_adjustment(out)
    _ensure_reconciled_interval_containment(out)
    post = hierarchy_coherence(out, value_cols=value_cols)
    summary = _reconciliation_summary(
        requested_method=method,
        applied_methods=applied_methods,
        pre=pre,
        post=post,
    )
    max_post_gap = _max_abs_gap_pct(post)
    warnings.append(
        f"hierarchy reconciliation applied using {method}; "
        f"post-reconciliation max absolute gap_pct={max_post_gap:.4%}"
    )
    return out, summary, warnings


def hierarchy_structure(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build the S matrix and tags expected by HierarchicalForecast."""

    required = {"unique_id", "hierarchy_depth", "hierarchy_level"}
    if not required.issubset(frame.columns):
        raise ValueError(f"hierarchy structure requires columns {required}")
    nodes = (
        frame[["unique_id", "hierarchy_depth", "hierarchy_level"]]
        .drop_duplicates("unique_id")
        .assign(hierarchy_depth=lambda df: pd.to_numeric(df["hierarchy_depth"], errors="coerce"))
        .sort_values(["hierarchy_depth", "unique_id"])
        .reset_index(drop=True)
    )
    if nodes["hierarchy_depth"].isna().any():
        raise ValueError("hierarchy_depth contains non-numeric values")
    max_depth = int(nodes["hierarchy_depth"].max())
    bottom_ids = nodes.loc[nodes["hierarchy_depth"] == max_depth, "unique_id"].astype(str).tolist()
    if not bottom_ids:
        raise ValueError("hierarchy structure requires at least one bottom-level series")
    root_ids = nodes.loc[nodes["hierarchy_depth"] == 0, "unique_id"].astype(str).tolist()
    root_id = root_ids[0] if root_ids else "Total"

    rows: list[dict[str, Any]] = []
    for node in nodes.to_dict("records"):
        node_id = str(node["unique_id"])
        row: dict[str, Any] = {"unique_id": node_id}
        descendants = set(_descendant_bottom_ids(node_id, bottom_ids, root_id=root_id))
        for bottom_id in bottom_ids:
            row[bottom_id] = 1.0 if bottom_id in descendants else 0.0
        rows.append(row)
    s_df = pd.DataFrame(rows)
    tags = {
        str(level): group["unique_id"].astype(str).to_numpy()
        for level, group in nodes.groupby("hierarchy_level", sort=True)
    }
    tags["bottom"] = np.asarray(bottom_ids)
    return s_df, tags


def _validate_hierarchy_input(
    frame: pd.DataFrame,
    hierarchy_cols: tuple[str, ...],
    time_col: str,
    target_col: str,
) -> None:
    if not hierarchy_cols:
        raise ValueError("at least one hierarchy column is required")
    missing = [col for col in (time_col, target_col, *hierarchy_cols) if col not in frame.columns]
    if missing:
        raise ValueError(f"missing hierarchy input columns {missing}; detected columns: {list(frame.columns)}")


def _reject_reserved_hierarchy_values(frame: pd.DataFrame, hierarchy_cols: tuple[str, ...]) -> None:
    reserved = ("|", "=")
    for col in hierarchy_cols:
        values = frame[col].dropna().astype(str)
        bad = values[values.str.contains(r"[|=]", regex=True)]
        if not bad.empty:
            example = bad.iloc[0]
            raise ValueError(
                f"hierarchy column '{col}' contains reserved delimiter characters '|' or '=' "
                f"(example: {example!r}); clean or map labels before aggregation"
            )


def _total_node(frame: pd.DataFrame, hierarchy_cols: tuple[str, ...], total_label: str) -> pd.DataFrame:
    total = frame.groupby("ds", as_index=False)["y"].sum()
    total["unique_id"] = total_label
    total["hierarchy_level"] = "total"
    total["hierarchy_depth"] = 0
    for col in hierarchy_cols:
        total[col] = None
    return total[["unique_id", "ds", "y", "hierarchy_level", "hierarchy_depth", *hierarchy_cols]]


def _node_id(cols: tuple[str, ...], row: pd.Series) -> str:
    return "|".join(f"{col}={row[col]}" for col in cols)


def _parent_id(row: pd.Series, root_id: str) -> str:
    if int(row["hierarchy_depth"]) == 1:
        return root_id
    return str(row["unique_id"]).rsplit("|", 1)[0]


def _reconciliation_value_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("yhat", "yhat_scenario")
    columns: list[str] = []
    for col in frame.columns:
        if col == "event_adjustment":
            continue
        if col in prefixes or col.startswith("yhat_lo_") or col.startswith("yhat_hi_"):
            columns.append(col)
        elif col.startswith("yhat_scenario_lo_") or col.startswith("yhat_scenario_hi_"):
            columns.append(col)
    return [col for col in columns if pd.api.types.is_numeric_dtype(frame[col])]


def _bottom_up_reconciled_values(frame: pd.DataFrame, value_col: str) -> pd.Series:
    work = frame[["unique_id", "ds", "hierarchy_depth", value_col]].copy()
    work["hierarchy_depth"] = pd.to_numeric(work["hierarchy_depth"], errors="coerce")
    max_depth = int(work["hierarchy_depth"].max())
    bottom_ids = work.loc[work["hierarchy_depth"] == max_depth, "unique_id"].dropna().astype(str).unique().tolist()
    root_ids = work.loc[work["hierarchy_depth"] == 0, "unique_id"].dropna().astype(str).unique().tolist()
    root_id = root_ids[0] if root_ids else "Total"

    out = pd.to_numeric(work[value_col], errors="coerce").astype("float64")
    bottom = work[work["unique_id"].astype(str).isin(bottom_ids)].copy()
    for node_id in work["unique_id"].dropna().astype(str).unique():
        descendants = _descendant_bottom_ids(node_id, bottom_ids, root_id=root_id)
        if not descendants:
            continue
        sums = (
            bottom[bottom["unique_id"].astype(str).isin(descendants)]
            .groupby("ds", sort=False)[value_col]
            .sum(min_count=1)
        )
        mask = work["unique_id"].astype(str).eq(node_id)
        out.loc[mask] = work.loc[mask, "ds"].map(sums).to_numpy(dtype="float64")
    return out


def _hierarchicalforecast_reconciled_values(frame: pd.DataFrame, value_col: str, *, method: str) -> pd.Series:
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import MinTrace

    method_map = {
        "mint_ols": "ols",
        "mint_wls_struct": "wls_struct",
    }
    if method not in method_map:
        raise ValueError(f"unsupported HierarchicalForecast reconciliation method: {method}")
    s_df, tags = hierarchy_structure(frame)
    y_hat = frame[["unique_id", "ds", value_col]].rename(columns={value_col: "yhat"})
    reconciler = HierarchicalReconciliation([MinTrace(method=method_map[method])])
    reconciled = reconciler.reconcile(Y_hat_df=y_hat, S_df=s_df, tags=tags)
    rec_cols = [col for col in reconciled.columns if col.startswith("yhat/")]
    if not rec_cols:
        raise ValueError("HierarchicalForecast did not return a reconciled yhat column")
    values = reconciled[["unique_id", "ds", rec_cols[0]]].rename(columns={rec_cols[0]: value_col})
    merged = frame[["unique_id", "ds"]].merge(values, on=["unique_id", "ds"], how="left")
    return pd.to_numeric(merged[value_col], errors="coerce").astype("float64")


def _descendant_bottom_ids(node_id: str, bottom_ids: Sequence[str], *, root_id: str) -> list[str]:
    if node_id == root_id:
        return list(bottom_ids)
    return [bottom_id for bottom_id in bottom_ids if bottom_id == node_id or bottom_id.startswith(f"{node_id}|")]


def _recompute_event_adjustment(frame: pd.DataFrame) -> None:
    if {"yhat", "yhat_scenario"}.issubset(frame.columns):
        frame["event_adjustment"] = (
            pd.to_numeric(frame["yhat_scenario"], errors="coerce") - pd.to_numeric(frame["yhat"], errors="coerce")
        ).round(10)


def _ensure_reconciled_interval_containment(frame: pd.DataFrame) -> None:
    for prefix in ("yhat", "yhat_scenario"):
        point_col = prefix
        if point_col not in frame.columns:
            continue
        point = pd.to_numeric(frame[point_col], errors="coerce")
        for lo_col, hi_col in _interval_column_pairs(frame, prefix):
            lo = pd.to_numeric(frame[lo_col], errors="coerce")
            hi = pd.to_numeric(frame[hi_col], errors="coerce")
            lower = pd.concat([lo, hi, point], axis=1).min(axis=1, skipna=True)
            upper = pd.concat([lo, hi, point], axis=1).max(axis=1, skipna=True)
            valid = lo.notna() & hi.notna() & point.notna()
            frame.loc[valid, lo_col] = lower.loc[valid]
            frame.loc[valid, hi_col] = upper.loc[valid]


def _interval_column_pairs(frame: pd.DataFrame, prefix: str) -> list[tuple[str, str]]:
    marker = f"{prefix}_lo_"
    pairs: list[tuple[str, str]] = []
    for col in frame.columns:
        if not col.startswith(marker):
            continue
        level = col.rsplit("_", 1)[-1]
        hi_col = f"{prefix}_hi_{level}"
        if hi_col in frame.columns:
            pairs.append((col, hi_col))
    return pairs


def _reconciliation_summary(
    *,
    requested_method: str,
    applied_methods: dict[str, str],
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for value_col, applied_method in applied_methods.items():
        rows.append(
            {
                "value_col": value_col,
                "requested_method": requested_method,
                "applied_method": applied_method,
                "pre_max_abs_gap_pct": _max_abs_gap_pct(pre[pre["value_col"] == value_col] if not pre.empty else pre),
                "post_max_abs_gap_pct": _max_abs_gap_pct(post[post["value_col"] == value_col] if not post.empty else post),
                "pre_gap_rows": int(len(pre[pre["value_col"] == value_col])) if not pre.empty else 0,
                "post_gap_rows": int(len(post[post["value_col"] == value_col])) if not post.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _max_abs_gap_pct(coherence: pd.DataFrame) -> float:
    if coherence.empty or "gap_pct" not in coherence.columns:
        return 0.0
    gaps = pd.to_numeric(coherence["gap_pct"], errors="coerce").abs().dropna()
    if gaps.empty:
        return 0.0
    return float(gaps.max())
