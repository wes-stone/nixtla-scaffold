from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _normalise_sales_extract(sales: pd.DataFrame) -> pd.DataFrame:
    if {"FiscalMonth", "SalesAmount", "ProductKey"}.issubset(sales.columns):
        return sales.copy()
    if {"DateKey", "Amount", "ProductKey"}.issubset(sales.columns):
        frame = sales.copy()
        date_key = frame["DateKey"].astype(str).str.replace(r"\.0$", "", regex=True)
        frame["FiscalMonth"] = pd.to_datetime(date_key, format="%Y%m%d").dt.to_period("M").dt.to_timestamp("M")
        frame["SalesAmount"] = pd.to_numeric(frame["Amount"], errors="raise")
        if "Quantity" in frame.columns:
            frame["SalesQuantity"] = pd.to_numeric(frame["Quantity"], errors="raise")
        return frame
    return sales


def _normalise_product_extract(products: pd.DataFrame) -> pd.DataFrame:
    if "ProductCategory" in products.columns:
        return products.copy()
    if "Category" in products.columns:
        return products.rename(columns={"Category": "ProductCategory"})
    return products


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Contoso DAX extracts into unique_id, ds, y forecast input.")
    parser.add_argument("--sales", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sales = _normalise_sales_extract(pd.read_csv(args.sales))
    products = _normalise_product_extract(pd.read_csv(args.products))
    _require_columns(sales, {"FiscalMonth", "ProductKey", "SalesAmount"}, name="sales extract")
    _require_columns(products, {"ProductKey", "ProductCategory"}, name="product extract")

    joined = sales.merge(products[["ProductKey", "ProductCategory"]], on="ProductKey", how="left", validate="many_to_one")
    if joined["ProductCategory"].isna().any():
        missing = sorted(joined.loc[joined["ProductCategory"].isna(), "ProductKey"].astype(str).unique())
        raise ValueError(f"sales extract has ProductKey values missing from product extract: {', '.join(missing)}")

    canonical = (
        joined.assign(ds=pd.to_datetime(joined["FiscalMonth"]), y=pd.to_numeric(joined["SalesAmount"], errors="raise"))
        .groupby(["ProductCategory", "ds"], as_index=False)["y"]
        .sum()
        .rename(columns={"ProductCategory": "unique_id"})
        .sort_values(["unique_id", "ds"])
    )
    canonical["ds"] = canonical["ds"].dt.date.astype(str)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical[["unique_id", "ds", "y"]].to_csv(output, index=False)


if __name__ == "__main__":
    main()
