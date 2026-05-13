from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUP_COLS = ["ProductGroup", "ProductLine", "Product"]
DATES = pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"])
PRODUCTS = [
    ("Cloud", "Workspace", "Seats", 100.0),
    ("Cloud", "Workspace", "Usage", 45.0),
    ("Platform", "Automation", "Jobs", 70.0),
]
SCENARIOS = {"Base": 1.00, "Bull": 1.10, "Bear": 0.92}


def _scenario_sheet(multiplier: float, *, cutoff: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_group, product_line, product, base in PRODUCTS:
        for step, ds in enumerate(DATES, start=1):
            row = {
                "ProductGroup": product_group,
                "ProductLine": product_line,
                "Product": product,
                "ds": ds,
                "yhat": round(base * multiplier * (1 + 0.035 * step), 2),
                "owner": "fpna",
                "currency": "USD",
                "unit_label": "$",
            }
            if cutoff is not None:
                row["cutoff"] = cutoff
            rows.append(row)
    return pd.DataFrame(rows)


def _write_multisheet_workbook(path: Path, *, cutoff: str | None = None) -> None:
    with pd.ExcelWriter(path) as writer:
        for scenario, multiplier in SCENARIOS.items():
            _scenario_sheet(multiplier, cutoff=cutoff).to_excel(writer, sheet_name=scenario, index=False)


def _node_id(row: pd.Series, depth: int = len(GROUP_COLS)) -> str:
    return "|".join(f"{column}={row[column]}" for column in GROUP_COLS[:depth])


def _scaffold_forecast() -> pd.DataFrame:
    base = _scenario_sheet(0.97)
    rows = []
    for _, row in base.iterrows():
        rows.append({"unique_id": _node_id(row), "ds": row["ds"], "yhat": row["yhat"], "model": "ScaffoldChampion"})
    leaves = pd.DataFrame(rows)
    product_groups = base.assign(unique_id=base.apply(lambda row: _node_id(row, 1), axis=1))
    group_rollup = product_groups.groupby(["unique_id", "ds"], as_index=False)["yhat"].sum()
    group_rollup["model"] = "ScaffoldChampion"
    total = base.groupby("ds", as_index=False)["yhat"].sum()
    total["unique_id"] = "Total"
    total["model"] = "ScaffoldChampion"
    return pd.concat([leaves, group_rollup, total], ignore_index=True)[["unique_id", "ds", "yhat", "model"]]


def _actuals() -> pd.DataFrame:
    actual = _scenario_sheet(0.95)
    actual["y"] = actual["yhat"]
    return actual[[*GROUP_COLS, "ds", "y"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small BYO Excel finance-model example inputs.")
    parser.add_argument("--output", default="runs\\byo_excel_example", help="Folder to write example inputs")
    args = parser.parse_args()
    output = Path(args.output)
    scaffold_run = output / "scaffold_run"
    output.mkdir(parents=True, exist_ok=True)
    scaffold_run.mkdir(parents=True, exist_ok=True)

    _write_multisheet_workbook(output / "finance_model.xlsx")
    _write_multisheet_workbook(output / "finance_snapshots.xlsx", cutoff="2025-12-31")
    _actuals().to_csv(output / "actuals.csv", index=False)
    _scaffold_forecast().to_csv(scaffold_run / "forecast.csv", index=False)

    print(f"Wrote BYO Excel example inputs to {output}")
    print("Next:")
    print(
        "uv run nixtla-scaffold byo-model compare "
        f"--run {scaffold_run} --file {output / 'finance_model.xlsx'} "
        "--sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --main-model-preference Base"
    )


if __name__ == "__main__":
    main()
