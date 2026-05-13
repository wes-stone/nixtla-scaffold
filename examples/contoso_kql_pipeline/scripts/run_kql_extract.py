from __future__ import annotations

import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CLUSTER_URL = "https://help.kusto.windows.net"
DEFAULT_DATABASE = "ContosoSales"
DEFAULT_MODE = "mock"


def execute_kql(
    query: str,
    *,
    cluster_url: str = DEFAULT_CLUSTER_URL,
    database: str = DEFAULT_DATABASE,
    mode: str = DEFAULT_MODE,
    interactive_auth: bool = False,
) -> pd.DataFrame:
    mode = mode.lower().strip()
    if mode == "mock":
        return _normalize_dataframe(_mock_to_pandas(query))
    if mode == "live":
        return _normalize_dataframe(
            _live_to_pandas(
                query,
                cluster_url=cluster_url,
                database=database,
                interactive_auth=interactive_auth,
            )
        )
    raise ValueError("mode must be 'mock' or 'live'")


def _live_to_pandas(
    query: str,
    *,
    cluster_url: str,
    database: str,
    interactive_auth: bool,
) -> pd.DataFrame:
    try:
        from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
        from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
        from azure.kusto.data.helpers import dataframe_from_result_table
    except ImportError as exc:
        raise RuntimeError(
            "Live Kusto extracts require azure-kusto-data and azure-identity. "
            "Install those packages or use the default KUSTO_MODE=mock."
        ) from exc

    if not cluster_url.startswith("https://"):
        raise ValueError("cluster_url must start with https://")
    if not database:
        raise ValueError("database is required")

    credential = InteractiveBrowserCredential() if interactive_auth else DefaultAzureCredential()
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(cluster_url, credential)
    response = KustoClient(kcsb).execute(database, query)
    if not response.primary_results:
        return pd.DataFrame()
    return dataframe_from_result_table(response.primary_results[0])


def _normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = [str(column).strip().replace(" ", "_") for column in dataframe.columns]
    for column in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce").dt.tz_localize(None)
    return dataframe


def _categories() -> list[tuple[str, float, float]]:
    return [
        ("Audio", 800_000, 0.55),
        ("Cameras and camcorders ", 19_000_000, 0.39),
        ("Cell phones", 1_700_000, 0.53),
        ("Computers", 6_700_000, 0.52),
        ("Games and Toys", 620_000, 0.50),
        ("Home Appliances", 9_100_000, 0.54),
        ("Music, Movies and Audio Books", 540_000, 0.48),
        ("TV and Video", 22_000_000, 0.84),
    ]


def _mock_products() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    product_key = 1_000
    for category, _, _ in _categories():
        clean = category.strip()
        for index in range(2):
            rows.append(
                {
                    "ProductKey": product_key,
                    "ProductName": f"{clean} Product {index + 1}",
                    "ProductCategoryName": category,
                    "ProductSubcategoryName": f"{clean} Subcategory",
                    "Manufacturer": "Contoso",
                }
            )
            product_key += 1
    return pd.DataFrame(rows)


def _mock_revenue_by_month() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month_index, month in enumerate(pd.date_range("2007-01-01", "2009-12-01", freq="MS")):
        trend = 1 + (month_index * 0.008)
        season = 1 + (0.10 * math.sin((month.month - 1) / 12 * 2 * math.pi))
        for category, base_revenue, cost_ratio in _categories():
            revenue = round(base_revenue * trend * season, 2)
            rows.append(
                {
                    "Month": month.to_pydatetime(),
                    "ProductCategoryName": category,
                    "Revenue": revenue,
                    "TotalCost": round(revenue * cost_ratio, 2),
                    "Transactions": int(max(revenue / 600, 1)),
                }
            )
    return pd.DataFrame(rows)


def _mock_to_pandas(query: str) -> pd.DataFrame:
    upper = re.sub(r"\s+", " ", query.upper())
    if re.search(r"\bPRODUCTS\b", upper) and "SALESFACT" not in upper:
        return _mock_products()
    return _mock_revenue_by_month()


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one Contoso KQL extract to CSV.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-url", default=os.getenv("KUSTO_CLUSTER_URL", DEFAULT_CLUSTER_URL))
    parser.add_argument("--database", default=os.getenv("KUSTO_DATABASE", DEFAULT_DATABASE))
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default=os.getenv("KUSTO_MODE", DEFAULT_MODE),
        help="Use 'mock' for deterministic offline output or 'live' for Azure Data Explorer.",
    )
    parser.add_argument("--interactive-auth", action="store_true")
    args = parser.parse_args()

    query_file = Path(args.query_file)
    query = query_file.read_text(encoding="utf-8")
    frame = execute_kql(
        query,
        cluster_url=args.cluster_url,
        database=args.database,
        mode=args.mode,
        interactive_auth=args.interactive_auth,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
