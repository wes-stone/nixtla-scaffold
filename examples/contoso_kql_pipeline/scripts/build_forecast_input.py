from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _clean_category(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip()


def _month_end(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert(None)
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Contoso KQL extracts into unique_id, ds, y forecast input.")
    parser.add_argument("--revenue", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    revenue = pd.read_csv(args.revenue)
    products = pd.read_csv(args.products)
    _require_columns(revenue, {"Month", "ProductCategoryName", "Revenue"}, name="revenue extract")
    _require_columns(products, {"ProductCategoryName"}, name="product extract")

    revenue = revenue.copy()
    products = products.copy()
    revenue["ProductCategoryName"] = _clean_category(revenue["ProductCategoryName"])
    products["ProductCategoryName"] = _clean_category(products["ProductCategoryName"])

    known_categories = set(products["ProductCategoryName"].dropna().unique())
    missing_categories = sorted(set(revenue["ProductCategoryName"].dropna().unique()) - known_categories)
    if missing_categories:
        raise ValueError(f"revenue extract has categories missing from product extract: {', '.join(missing_categories)}")

    canonical = (
        revenue.assign(ds=_month_end(revenue["Month"]), y=pd.to_numeric(revenue["Revenue"], errors="raise"))
        .groupby(["ProductCategoryName", "ds"], as_index=False)["y"]
        .sum()
        .rename(columns={"ProductCategoryName": "unique_id"})
        .sort_values(["unique_id", "ds"])
    )
    canonical["ds"] = canonical["ds"].dt.date.astype(str)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical[["unique_id", "ds", "y"]].to_csv(output, index=False)


if __name__ == "__main__":
    main()
