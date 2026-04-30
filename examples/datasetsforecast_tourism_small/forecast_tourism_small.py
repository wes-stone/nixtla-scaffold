from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from nixtla_scaffold import ForecastSpec, aggregate_hierarchy_frame, run_forecast


ROOT = Path(__file__).resolve().parent
DATASET_GROUP = "TourismSmall"
BOTTOM_TAG = "Country/Purpose/State/CityNonCity"
HIERARCHY_COLS = ("purpose", "state", "city_noncity")
EXPECTED_SOURCE_SERIES = 89
EXPECTED_SOURCE_ROWS = 3204
HORIZON = 4
SEASON_LENGTH = 4


def parse_tourism_small_bottom_id(unique_id: str) -> dict[str, str]:
    """Parse DatasetsForecast TourismSmall bottom IDs like nsw-hol-city."""

    parts = str(unique_id).split("-")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "TourismSmall bottom IDs must use '<state>-<purpose>-<city_noncity>' "
            f"format; got {unique_id!r}"
        )
    state, purpose, city_noncity = parts
    if city_noncity not in {"city", "noncity"}:
        raise ValueError(f"TourismSmall city/noncity label must be 'city' or 'noncity'; got {unique_id!r}")
    return {"state": state, "purpose": purpose, "city_noncity": city_noncity}


def tourism_small_leaf_frame(y_df: pd.DataFrame, tags: Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    """Return bottom-level TourismSmall rows with explicit hierarchy columns."""

    required = {"unique_id", "ds", "y"}
    missing = required.difference(y_df.columns)
    if missing:
        raise ValueError(f"TourismSmall Y_df is missing required columns: {sorted(missing)}")
    if BOTTOM_TAG not in tags:
        raise ValueError(f"TourismSmall tags are missing bottom-level tag {BOTTOM_TAG!r}")

    bottom_ids = [str(value) for value in tags[BOTTOM_TAG]]
    if not bottom_ids:
        raise ValueError(f"TourismSmall bottom-level tag {BOTTOM_TAG!r} is empty")

    source = y_df.copy()
    source["source_unique_id"] = source["unique_id"].astype(str)
    leaf = source.loc[source["source_unique_id"].isin(bottom_ids), ["source_unique_id", "ds", "y"]].copy()
    observed = set(leaf["source_unique_id"].unique())
    missing_bottom = sorted(set(bottom_ids).difference(observed))
    if missing_bottom:
        sample = ", ".join(missing_bottom[:5])
        raise ValueError(f"TourismSmall Y_df is missing {len(missing_bottom)} bottom-level series from tags: {sample}")
    if leaf.empty:
        raise ValueError("TourismSmall bottom-level frame is empty")

    leaf["ds"] = pd.to_datetime(leaf["ds"], errors="raise")
    leaf["y"] = pd.to_numeric(leaf["y"], errors="raise").astype("float64")
    duplicates = leaf.duplicated(["source_unique_id", "ds"])
    if duplicates.any():
        example = leaf.loc[duplicates, ["source_unique_id", "ds"]].iloc[0].to_dict()
        raise ValueError(f"TourismSmall bottom-level rows contain duplicate source_unique_id/ds values: {example}")

    parsed = pd.DataFrame(
        [parse_tourism_small_bottom_id(value) for value in leaf["source_unique_id"]],
        index=leaf.index,
    )
    return pd.concat([leaf[["source_unique_id", "ds", "y"]], parsed], axis=1).reset_index(drop=True)


def build_tourism_small_hierarchy(
    y_df: pd.DataFrame,
    tags: Mapping[str, Sequence[Any]],
    *,
    s_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bridge DatasetsForecast TourismSmall into scaffold hierarchy nodes."""

    leaf = tourism_small_leaf_frame(y_df, tags)
    nodes = aggregate_hierarchy_frame(
        leaf,
        hierarchy_cols=HIERARCHY_COLS,
        time_col="ds",
        target_col="y",
    )
    if s_df is not None:
        expected_nodes = len(s_df)
        observed_nodes = int(nodes["unique_id"].nunique())
        if observed_nodes != expected_nodes:
            raise ValueError(
                "TourismSmall scaffold hierarchy node count does not match DatasetsForecast S matrix rows: "
                f"observed {observed_nodes}, expected {expected_nodes}"
            )
    _validate_total_matches_source(nodes, y_df)
    return nodes


def tourism_small_cache_exists(cache_dir: str | Path) -> bool:
    """Return whether the DatasetsForecast TourismSmall cache appears present."""

    cache = Path(cache_dir)
    return (
        (cache / "hierarchical" / DATASET_GROUP / "data.csv").exists()
        and (cache / "hierarchical" / DATASET_GROUP / "agg_mat.csv").exists()
    ) or (cache / "hierarchical" / f"{DATASET_GROUP}.p").exists()


def load_tourism_small_dataset(
    cache_dir: str | Path,
    *,
    allow_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Sequence[Any]]]:
    """Load TourismSmall, requiring explicit download permission when uncached."""

    hierarchical_data = _load_hierarchical_data_class()
    cache = Path(cache_dir)
    if not allow_download and not tourism_small_cache_exists(cache):
        raise RuntimeError(
            "DatasetsForecast TourismSmall cache was not found. Re-run with --allow-download "
            "to permit the one-time public dataset download, or pass --cache-dir to an existing cache."
        )
    y_df, s_df, tags = hierarchical_data.load(directory=str(cache), group=DATASET_GROUP)
    return y_df, s_df, tags


def run_example(
    output_dir: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
    model_policy: str = "baseline",
) -> Path:
    output = Path(output_dir) if output_dir is not None else ROOT / "runs" / "tourism_small"
    cache = Path(cache_dir) if cache_dir is not None else ROOT / "runs" / "datasetsforecast_cache"
    y_df, s_df, tags = load_tourism_small_dataset(cache, allow_download=allow_download)
    nodes = build_tourism_small_hierarchy(y_df, tags, s_df=s_df)
    _validate_real_tourism_small_shape(y_df, s_df, nodes)

    spec = ForecastSpec(
        horizon=HORIZON,
        freq="QE",
        season_length=SEASON_LENGTH,
        model_policy=model_policy,
        hierarchy_reconciliation="bottom_up",
        weighted_ensemble=False,
        unit_label="trips",
        verbose=False,
    )
    run = run_forecast(nodes, spec)
    return run.to_directory(output)


def _load_hierarchical_data_class():
    try:
        from datasetsforecast.hierarchical import HierarchicalData
    except ModuleNotFoundError as exc:
        if "datasetsforecast" in str(exc):
            raise RuntimeError(
                "The optional DatasetsForecast dependency is not installed. Install it with "
                "`uv pip install -e .[datasets]` or run commands with `uv run --extra datasets ...`."
            ) from exc
        raise
    return HierarchicalData


def _validate_total_matches_source(nodes: pd.DataFrame, source_y: pd.DataFrame) -> None:
    source = source_y.copy()
    source["source_unique_id"] = source["unique_id"].astype(str)
    source_total = source.loc[source["source_unique_id"].str.lower() == "total", ["ds", "y"]].copy()
    if source_total.empty:
        raise ValueError("TourismSmall source data must include a 'total' series for Total-node validation")
    source_total["ds"] = pd.to_datetime(source_total["ds"], errors="raise")
    source_total["source_total_y"] = pd.to_numeric(source_total["y"], errors="raise").astype("float64")
    source_total = source_total.groupby("ds", as_index=False)["source_total_y"].sum()

    generated_total = nodes.loc[nodes["unique_id"].astype(str) == "Total", ["ds", "y"]].copy()
    generated_total["ds"] = pd.to_datetime(generated_total["ds"], errors="raise")
    generated_total = generated_total.rename(columns={"y": "generated_total_y"})
    merged = source_total.merge(generated_total, on="ds", how="outer")
    if merged[["source_total_y", "generated_total_y"]].isna().any().any():
        raise ValueError("TourismSmall generated Total node does not cover the same dates as the source total")
    max_gap = (merged["source_total_y"] - merged["generated_total_y"]).abs().max()
    if float(max_gap) > 1e-6:
        raise ValueError(f"TourismSmall generated Total node does not match source total; max gap={max_gap}")


def _validate_real_tourism_small_shape(y_df: pd.DataFrame, s_df: pd.DataFrame, nodes: pd.DataFrame) -> None:
    source_series = int(y_df["unique_id"].nunique())
    source_rows = int(len(y_df))
    node_count = int(nodes["unique_id"].nunique())
    if source_series != EXPECTED_SOURCE_SERIES or source_rows != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            "Unexpected TourismSmall source shape; expected "
            f"{EXPECTED_SOURCE_SERIES} series/{EXPECTED_SOURCE_ROWS} rows, got {source_series} series/{source_rows} rows"
        )
    if len(s_df) != EXPECTED_SOURCE_SERIES or node_count != EXPECTED_SOURCE_SERIES:
        raise ValueError(
            "Unexpected TourismSmall hierarchy shape; expected "
            f"{EXPECTED_SOURCE_SERIES} S rows and scaffold nodes, got {len(s_df)} S rows/{node_count} nodes"
        )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opt-in real-data validation forecast on Nixtla DatasetsForecast TourismSmall. "
            "This may download public data only when --allow-download is supplied."
        )
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "tourism_small")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "datasetsforecast_cache")
    parser.add_argument("--allow-download", action="store_true", help="Permit DatasetsForecast to download public data")
    parser.add_argument(
        "--model-policy",
        choices=["baseline", "statsforecast", "auto", "all"],
        default="baseline",
        help="Use baseline for fast validation; try statsforecast/auto for deeper local experiments.",
    )
    args = parser.parse_args(argv)
    try:
        print(
            run_example(
                args.output,
                cache_dir=args.cache_dir,
                allow_download=args.allow_download,
                model_policy=args.model_policy,
            )
        )
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
