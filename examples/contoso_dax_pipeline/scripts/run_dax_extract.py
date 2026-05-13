from __future__ import annotations

import argparse
import os
import random
import re
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONNECTION_STRING = "MOCK://contoso"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def execute_dax(
    dax_query: str,
    connection_string: str,
    *,
    connection_timeout: int = 300,
    command_timeout: int = 1800,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if _is_mock_connection(connection_string):
        return _normalize_dataframe(_mock_to_pandas(dax_query, max_rows=max_rows))
    return _normalize_dataframe(
        _adodb_to_pandas(
            dax_query,
            connection_string,
            connection_timeout=connection_timeout,
            command_timeout=command_timeout,
            max_rows=max_rows,
        )
    )


def _adodb_to_pandas(
    dax_query: str,
    connection_string: str,
    *,
    connection_timeout: int,
    command_timeout: int,
    max_rows: int | None,
) -> pd.DataFrame:
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "Live MSOLAP DAX extracts require pywin32 plus the Microsoft Analysis "
            "Services OLE DB Provider (MSOLAP). Install MSOLAP, install pywin32 in "
            "this Python environment, or use the default MOCK://contoso connection."
        ) from exc

    conn = None
    cmd = None
    recordset = None
    try:
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.ConnectionTimeout = connection_timeout
        conn.CommandTimeout = command_timeout
        conn.Open(connection_string)

        cmd = win32com.client.Dispatch("ADODB.Command")
        cmd.ActiveConnection = conn
        cmd.CommandText = dax_query
        cmd.CommandTimeout = command_timeout

        recordset = cmd.Execute()[0]
        fields = [recordset.Fields(i).Name for i in range(recordset.Fields.Count)]
        rows = recordset.GetRows(max_rows) if max_rows else recordset.GetRows()
    finally:
        if cmd is not None:
            with suppress(Exception):
                cmd.ActiveConnection = None
        for obj in (recordset, conn):
            close = getattr(obj, "Close", None)
            if callable(close):
                with suppress(Exception):
                    close()

    data: dict[str, list[Any]] = {}
    for i, name in enumerate(fields):
        values = [_strip_tz(value) for value in rows[i]] if rows and i < len(rows) else []
        data[str(name)] = list(values)
    return pd.DataFrame(data)


def _normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = [_clean_column_name(str(col)) for col in dataframe.columns]
    return dataframe


def _clean_column_name(name: str) -> str:
    name = _ANSI_RE.sub("", name)
    if "[" in name and "]" in name:
        name = name[name.find("[") + 1 : name.find("]")]
    return name.replace(" ", "_")


def _strip_tz(value: object) -> object:
    if isinstance(value, datetime) and getattr(value, "tzinfo", None) is not None:
        return value.replace(tzinfo=None)
    return value


def _is_mock_connection(connection_string: str) -> bool:
    return connection_string.strip().upper().startswith("MOCK://")


def _mock_products() -> list[dict[str, Any]]:
    return [
        {"ProductKey": 1, "ProductName": "Mountain Bike", "Category": "Bikes", "Price": 1500.00},
        {"ProductKey": 2, "ProductName": "Road Bike", "Category": "Bikes", "Price": 1200.00},
        {"ProductKey": 3, "ProductName": "Helmet", "Category": "Accessories", "Price": 50.00},
        {"ProductKey": 4, "ProductName": "Gloves", "Category": "Accessories", "Price": 25.00},
        {"ProductKey": 5, "ProductName": "Water Bottle", "Category": "Accessories", "Price": 10.00},
    ]


def _mock_calendar() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        for day in range(1, 29):
            current = date(2025, month, day)
            rows.append(
                {
                    "DateKey": int(current.strftime("%Y%m%d")),
                    "Date": datetime(current.year, current.month, current.day),
                    "Month": current.strftime("%B"),
                    "MonthNum": month,
                    "Year": 2025,
                    "Weekday": current.strftime("%A"),
                }
            )
    return rows


def _mock_sales() -> list[dict[str, Any]]:
    products = _mock_products()
    calendar = _mock_calendar()
    random.seed(42)
    rows: list[dict[str, Any]] = []
    for index in range(100):
        product = random.choice(products)
        calendar_row = random.choice(calendar)
        quantity = random.randint(1, 5)
        rows.append(
            {
                "SalesKey": index + 1,
                "ProductKey": product["ProductKey"],
                "DateKey": calendar_row["DateKey"],
                "Quantity": quantity,
                "Amount": round(product["Price"] * quantity, 2),
            }
        )
    return rows


def _mock_to_pandas(dax_query: str, *, max_rows: int | None) -> pd.DataFrame:
    upper = dax_query.upper().strip()
    if re.search(r"\bPRODUCTS\b", upper) and "SALES" not in upper:
        dataframe = pd.DataFrame(_mock_products())
    elif re.search(r"\bCALENDAR\b", upper) and "SALES" not in upper:
        dataframe = pd.DataFrame(_mock_calendar())
    else:
        dataframe = pd.DataFrame(_mock_sales())
    return dataframe.head(max_rows) if max_rows is not None else dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one Contoso DAX extract to CSV.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--connection-string",
        default=os.getenv("DAX_CONNECTION_STRING", DEFAULT_CONNECTION_STRING),
        help="DAX connection string. Defaults to DAX_CONNECTION_STRING or MOCK://contoso.",
    )
    parser.add_argument("--connection-timeout-seconds", type=int, default=300)
    parser.add_argument("--command-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    query_file = Path(args.query_file)
    dax_query = query_file.read_text(encoding="utf-8")
    frame = execute_dax(
        dax_query,
        args.connection_string,
        connection_timeout=args.connection_timeout_seconds,
        command_timeout=args.command_timeout_seconds,
        max_rows=args.max_rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
