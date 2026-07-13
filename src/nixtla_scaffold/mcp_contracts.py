"""Documented MCP handoff contracts.

MCP tools should hand this package either:
1. a CSV/Excel file with unique_id, ds, y columns; or
2. JSON-like records passed to dataframe_from_records(); or
3. script-backed source pipeline extracts that collapse many queries into one canonical input.

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

SIGNAL_CAPABILITY_ROUTES = (
    {
        "route_id": "target-history",
        "signal_families": ("target_integrity", "structural_break"),
        "source_families": ("target_extract", "operational_history", "finance_actuals"),
        "capabilities": ("schema_discovery", "bounded_profile", "time_series_aggregate"),
        "probe_sequence": ("schema", "count", "sample", "aggregate"),
    },
    {
        "route_id": "calendar-exposure",
        "signal_families": ("calendar_exposure", "seasonality"),
        "source_families": ("calendar_reference", "operating_calendar"),
        "capabilities": ("calendar_lookup", "bounded_time_alignment"),
        "probe_sequence": ("schema", "sample", "aggregate"),
    },
    {
        "route_id": "plan-and-benchmark",
        "signal_families": ("plan_benchmark", "financial_driver"),
        "source_families": ("semantic_model", "planning_workbook", "finance_table"),
        "capabilities": ("measure_discovery", "bounded_period_aggregate", "plan_actual_alignment"),
        "probe_sequence": ("schema", "count", "sample", "aggregate"),
    },
    {
        "route_id": "operational-driver",
        "signal_families": ("operational_driver", "demand_driver", "capacity_driver"),
        "source_families": ("telemetry_store", "operational_table", "product_metrics"),
        "capabilities": ("schema_discovery", "bounded_time_series_aggregate", "entity_time_alignment"),
        "probe_sequence": ("schema", "count", "sample", "aggregate"),
    },
    {
        "route_id": "commercial-commitment",
        "signal_families": ("pipeline_driver", "renewal_driver", "contract_driver"),
        "source_families": ("commercial_system", "contract_store", "renewal_schedule"),
        "capabilities": ("entity_discovery", "bounded_stage_aggregate", "known_date_extract"),
        "probe_sequence": ("schema", "count", "sample", "aggregate"),
    },
    {
        "route_id": "known-change",
        "signal_families": ("known_change", "event_scenario", "pricing_driver"),
        "source_families": ("launch_calendar", "pricing_schedule", "capacity_plan", "headcount_plan"),
        "capabilities": ("known_date_lookup", "bounded_scenario_extract", "owner_validation"),
        "probe_sequence": ("schema", "sample", "aggregate"),
    },
)

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
        "handoff": "Run a DAX query at the desired grain, export to CSV or JSON, ingest it into canonical unique_id/ds/y, then forecast. Live Power BI/Analysis Services connections require the Microsoft Analysis Services OLE DB Provider (MSOLAP); the Mock Contoso connection does not. Keep fiscal period and metric definitions in the query file.",
        "command": "nixtla-scaffold ingest --input dax_export.csv --source dax --query-file query.dax --id-col Product --time-col FiscalMonth --target-col ARR --output runs\\dax_input.csv --forecast-output runs\\dax_forecast --freq ME --horizon 6",
    },
    {
        "name": "Multiple query extracts to one forecast",
        "source": "dax-query-server, data-query, or azure-kusto",
        "handoff": "Use a pipeline YAML when several DAX/Kusto/SQL/Python extracts feed one forecast. The bundled Contoso DAX example executes .dax files through a run_query.py-style scaffold against MOCK://contoso by default; live DAX extracts require MSOLAP and pywin32 installed. The bundled Contoso KQL example executes .kql files through a small Kusto scaffold against deterministic ContosoSales-shaped data by default; live Kusto extracts require azure-kusto-data and azure-identity. Each script writes declared outputs, an optional transform builds the canonical input, and the forecast run receives source-pipeline provenance.",
        "command": "nixtla-scaffold pipeline run --config examples\\contoso_dax_pipeline\\pipeline.yaml --output runs\\contoso_dax_pipeline",
    },
    {
        "name": "Kusto query to forecast",
        "source": "data-query or azure-kusto",
        "handoff": "Aggregate telemetry or revenue data in KQL, export the result, then ingest with explicit time/target/id mapping. Do not pass event-level logs directly to the forecaster.",
        "command": "nixtla-scaffold ingest --input kusto_export.json --source kusto --query-file query.kql --id-value \"Usage Overage ARR\" --time-col day_dt --target-col ARR_30day_avg --output runs\\usage_overage_arr_input.csv --forecast-output runs\\usage_overage_arr_demo --freq ME --horizon 6",
    },
    {
        "name": "Multiple Kusto queries to one forecast",
        "source": "data-query or azure-kusto",
        "handoff": "Use the Contoso KQL pipeline shape when a forecast needs several KQL extracts. The checked-in example forecasts ContosoSales monthly Revenue by ProductCategoryName from SalesFact joined to Products, stays offline by default, and can run live with KUSTO_MODE=live when azure-kusto-data and azure-identity are installed.",
        "command": "nixtla-scaffold pipeline run --config examples\\contoso_kql_pipeline\\pipeline.yaml --output runs\\contoso_kql_pipeline",
    },
]


def describe_contract() -> dict[str, object]:
    return {
        "required_columns": CANONICAL_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "recipes": MCP_RECIPES,
        "signal_capability_routes": list(SIGNAL_CAPABILITY_ROUTES),
    }


def signal_routes_for_family(signal_family: str) -> tuple[dict[str, object], ...]:
    """Return generic capability routes without choosing a concrete MCP or tool."""

    family = str(signal_family).strip().lower()
    return tuple(
        route
        for route in SIGNAL_CAPABILITY_ROUTES
        if family in route["signal_families"]
    )


def signal_capabilities_for_family(signal_family: str) -> tuple[str, ...]:
    capabilities = {
        str(capability)
        for route in signal_routes_for_family(signal_family)
        for capability in route["capabilities"]
    }
    return tuple(sorted(capabilities))
