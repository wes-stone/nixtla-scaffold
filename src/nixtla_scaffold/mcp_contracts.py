"""Documented MCP handoff contracts.

MCP tools should hand this package either:
1. a CSV/Excel file with unique_id, ds, y columns; or
2. JSON-like records passed to dataframe_from_records().

Keep connector-specific logic outside the forecasting engine until repeated
workflows prove a local MCP server is necessary.
"""

CANONICAL_COLUMNS = {
    "unique_id": "Series identifier at the forecast grain.",
    "ds": "Date or timestamp.",
    "y": "Numeric target value.",
}

OPTIONAL_COLUMNS = {
    "hierarchy_level": "Hierarchy level label emitted by the hierarchy command.",
    "hierarchy_depth": "Hierarchy depth emitted by the hierarchy command.",
    "y_adjusted": "Price/definition-normalized target for finance comparability.",
    "anomaly_label": "Boolean marker for one-time shocks or outliers.",
}

MCP_RECIPES = [
    {
        "name": "Excel workbook to forecast",
        "source": "excel-mcp",
        "handoff": "Export or save a worksheet with unique_id/ds/y (or map columns with --id-col/--time-col/--target-col).",
        "command": "nixtla-scaffold forecast --input workbook.xlsx --sheet Data --id-col Product --time-col Month --target-col Revenue --horizon 6 --output runs\\excel_forecast",
    },
    {
        "name": "DAX query to forecast",
        "source": "dax-query-server",
        "handoff": "Run a DAX query at the desired grain, export to CSV or JSON, ingest it into canonical unique_id/ds/y, then forecast. Keep fiscal period and metric definitions in the query file.",
        "command": "nixtla-scaffold ingest --input dax_export.csv --source dax --query-file query.dax --id-col Product --time-col FiscalMonth --target-col ARR --output runs\\dax_input.csv --forecast-output runs\\dax_forecast --freq ME --horizon 6",
    },
    {
        "name": "Kusto query to forecast",
        "source": "data-query or azure-kusto",
        "handoff": "Aggregate telemetry or revenue data in KQL, export the result, then ingest with explicit time/target/id mapping. Do not pass event-level logs directly to the forecaster.",
        "command": "nixtla-scaffold ingest --input kusto_export.json --source kusto --query-file query.kql --id-value \"Premium Overage ARR\" --time-col day_dt --target-col ARR_30day_avg --output runs\\premium_overage_arr_input.csv --forecast-output runs\\premium_overage_arr_demo --freq ME --horizon 6",
    },
]


def describe_contract() -> dict[str, object]:
    return {
        "required_columns": CANONICAL_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "recipes": MCP_RECIPES,
    }

