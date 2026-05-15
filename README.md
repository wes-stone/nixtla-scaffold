Trying to minimize issues with people vibe coding forecasts what could possibly go wrong!!!

# nixtla-scaffold

[![PyPI](https://img.shields.io/pypi/v/nixtla-scaffold?logo=pypi&label=PyPI)](https://pypi.org/project/nixtla-scaffold/)

Simple, explainable Nixtla forecasting scaffolding for finance users and AI agents.

**AI agent skill:** use the bundled [`nixtla-forecast` skill](skills/nixtla-forecast/SKILL.md) when asking an AI agent to run this workflow. It gives agents the intake questions, FPPy guardrails, Nixtla command map, model-selection rules, prediction-interval checks, hierarchy guidance, and post-forecast review checklist. From a source clone, copy `skills/nixtla-forecast/` into your agent's skills directory; from an installed wheel, run `nixtla-scaffold guide skill` to print the same packaged skill text.

Install from PyPI with `uv`:

```powershell
uv tool install nixtla-scaffold
# or, with optional MLForecast + hierarchy extras:
uv tool install "nixtla-scaffold[ml,hierarchy]"
```

If you want the Python API inside a `uv` project instead of a global CLI tool, use `uv add nixtla-scaffold` or `uv add "nixtla-scaffold[ml,hierarchy]"`.

For local development, use `uv sync --extra ml --extra hierarchy`, then run commands with `uv run nixtla-scaffold ...`.

Start with the smallest useful path. If you already have a CSV with `unique_id`, `ds`, and `y`, this is the five-minute flow:

```powershell
nixtla-scaffold profile --input examples\monthly_finance_csv\input.csv
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --levels 80 95 --output runs\demo
```

Open these first:

| File                                                                   | Why it matters                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPEN_ME_FIRST.html` or `output/forecast_review.xlsx`                 | Curated landing page and compact workbook. Use this when you want the clean forecast, decision summary, model leaderboard, watchouts, and file guide without all feeder/debug artifacts.                                                                                               |
| `appendix\trust_summary.csv`                                           | First-stop High/Medium/Low readiness, caveats, next actions, and whether the full requested horizon is validated.                                                                                                                                                                       |
| `appendix\borrowed_strength_advisor.csv`                               | Advisory sparse-series guidance for whether a series can stand alone or needs parent anchoring, reference-class review, or panel-pool review. It never changes the selected champion or `forecast.csv`.                                                                               |
| `report.html`                                                          | Readable decision summary, model review, horizon trust, intervals, seasonality, backtest evidence, and a ledger preview when `ledger_context.json` exists.                                                                                                                              |
| `llm_context.json`                                                     | One-attachment LLM handoff packet with the headline, trust/horizon/interval/residual/seasonality/hierarchy/driver context, artifact index, guardrails, and questions to ask.                                                                                                            |
| `forecast.csv`                                                         | Selected forecast only. Start with `row_horizon_status` to identify beyond-validated-horizon rows; `planning_eligible=True` means the row passed the horizon-validation gate, not global planning approval. Pair it with trust, interval, residual, hierarchy, and data-quality review. |
| `appendix\forecast_long.csv`                                           | Primary model-feed output with every future series/model/date row, selected-model flag, intervals when available, interval status, row-level horizon validation, and `planning_eligibility_scope`.                                                                                      |
| `appendix\run_receipt.json` / `appendix\validation_receipt.csv`        | Local operations receipts: reproducibility/provenance plus pass/warn/fail data and artifact checks for agents that need to rerun or inspect the run without opening Streamlit.                                                                                                        |

Trust rubric: **High >=75**, **Medium 40-74**, **Low <40**. A High score still means "statistical baseline with evidence," not a plan or guarantee. `planning_eligible=True` in `forecast.csv` / `appendix\forecast_long.csv` only means the row passed the horizon-validation gate (`planning_eligibility_scope=horizon_validation_only`); it does not override Low trust, interval issues, residual warnings, hierarchy tradeoffs, or data-quality caveats. Agents should quote `diagnostics.json.executive_headline.paragraph` verbatim rather than rewriting it into a stronger claim. The generated Streamlit app also shows a copy-safe headline code block with a copy icon so the deterministic headline can be pasted without paraphrasing.

## Refresh-first workflow

Keep routine forecasting boring: use the same input contract, same command, and a new output folder for each refresh. Advanced scenarios and guardrails exist to protect quality, but the main monthly loop should stay `profile -> forecast -> report`.

```powershell
nixtla-scaffold profile --input data.csv
nixtla-scaffold forecast --input data.csv --preset standard --horizon 6 --output runs\latest
nixtla-scaffold report --run runs\latest
```

For demos and QA, think in two layers:

| Layer                  | Scenarios                                                   | Use                                                                                                                                                         |
| ---------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flagship refresh demos | `monthly_basic`, `hierarchy_reconciled`                     | The first examples to show analysts: a normal monthly forecast and a reconciled parent/child finance rollup.                                                |
| Finance guardrails     | `limited_history_new_product`, `normalized_target_forecast` | Edge-case checks that keep the scaffold honest for new products/metrics with sparse history and pricing, FX, inflation, or definition-change normalization. |

Champion selection uses backtested **RMSE** when available because it penalizes large misses; trust and benchmark review also show **MAE, MASE, RMSSE, WAPE, and bias** so scale-free and business-readable checks are visible.

## Data contract

The core input contract is a long table with three required columns:

| Column      | Meaning                                                         |
| ----------- | --------------------------------------------------------------- |
| `unique_id` | Series name, product, SKU, account, or other forecasting grain. |
| `ds`        | Timestamp/date at the forecast grain.                           |
| `y`         | Numeric value to forecast.                                      |

Single-series files can omit `unique_id`; the loader creates `series_1`. Business-friendly column names can be mapped with `--id-col`, `--time-col`, and `--target-col`. Use `--unit-label` (for example `$`, `USD`, `seats`, or `ARR`) when you want executive headlines to show currency/units and signed absolute deltas alongside percentage direction.

## External finance forecast contract

Finance teams often have their own workbook, Python, or regression forecasts. Import them as **external forecasts** for comparison and triangulation, not as observed history or scaffold model candidates.

```python
from nixtla_scaffold import load_external_forecasts

external = load_external_forecasts(
    "finance_plan.xlsx",
    sheet="Forecast",
    id_col="Product",
    time_col="Month",
    value_col="Finance Forecast",
    model_col="Model",
)
```

Supported inputs:

| Format | Required shape                                                    | Notes                                                                                                                                                                                                                                           |
| ------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Long   | Date column plus forecast-value column (`ds` / `yhat` by default) | Include a model column or pass `model_name`; omit `unique_id` only for single-series files.                                                                                                                                                     |
| Wide   | One row per series/model with date-like forecast columns          | Pass `format="wide"` for finance workbooks; date columns are melted to `ds` and non-date columns stay as metadata. Prefer explicit `format=` for ambiguous sheets so incidental date-like metadata headers are not treated as forecast periods. |

The canonical output keeps analyst metadata (`owner`, `model_version`, `scenario_name`, `notes`, source file/sheet), sets `family="external"`, `is_external_forecast=True`, and uses explicit evidence labels:

| `comparison_evidence_status`         | Meaning                                                                                                                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `future_only_unscored`               | A future-only imported forecast. Use for directional comparison only; do not claim accuracy or backtest validity.                                                            |
| `historical_cutoff_labeled_unscored` | A `cutoff` / `forecast_origin` label exists and is before the target date, but the origin has not been independently verified and rows have not been scored against actuals. |

`is_backtested` is always `False` at import time. Backtest claims require a later scoring workflow that joins cutoff-labeled external forecasts to actuals, preserves chronology, and reports accuracy metrics. Do not concatenate external `yhat` rows into the core `unique_id`, `ds`, `y` training data.

After a scaffold run exists, compare an external forecast as an optional second step:

```powershell
nixtla-scaffold compare --run runs\latest --external finance_plan.xlsx --sheet Forecast --format wide --model-name "FP&A plan" --output runs\latest\comparison
```

This writes `forecast_comparison.csv`, `forecast_comparison_summary.csv`, `comparison_manifest.json`, `forecast_comparison.xlsx`, `comparison_report.html`, and `comparison_llm_context.json` in the comparison output folder. The comparison aligns on `unique_id` + `ds`, preserves external model/source/scenario lineage, and reports directional deltas (`external_yhat - scaffold_yhat`). These deltas are **not** residuals, accuracy metrics, prediction intervals, or model-selection evidence; `is_comparison_scoreable` remains `False` until a later scoring workflow explicitly joins historical cutoff-labeled forecasts to actuals.

Cutoff-labeled external forecasts stay conservative in the compare workflow. If external `cutoff` values do not match the scaffold forecast origin, or if the scaffold origin cannot be verified, those rows are kept as count-only diagnostics with no delta averages. Summary fields such as `avg_abs_delta_yhat_vs_scaffold` include only `comparison_status="aligned"` rows, and `comparison_manifest.json` reports separate `rows_aligned`, `rows_with_both_forecasts`, `mismatch_cutoff_rows`, and `unknown_cutoff_origin_rows`.

When you have historical forecast snapshots with `cutoff` / `forecast_origin` labels and realized actuals, score them separately from the directional compare workflow:

```powershell
nixtla-scaffold score-external --external finance_snapshots.csv --actuals actuals.csv --output runs\finance_scores --season-length 12 --horizon 6
```

This writes `external_backtest_long.csv`, `external_model_metrics.csv`, and `external_scoring_manifest.json`. Scoring joins on `unique_id` + `ds`, requires `cutoff < ds`, and fails closed for future-only forecasts, non-positive `--season-length` / `--horizon`, duplicate actual `unique_id`/`ds` rows, or zero matched actuals. Missing actuals are kept as `missing_actual` diagnostics in the long file but excluded from metrics. MASE/RMSSE scales are computed only from actual history available at each row's cutoff; rows disclose `scale_basis` / `effective_season_length`, and metrics include `scale_basis_distribution`. Bias is `sum(yhat - y_actual) / sum(abs(y_actual))`, so positive bias means the external model overstated actuals.

### BYO Excel finance models

Use `byo-model` when the external forecast lives in an Excel driver model with multiple versions such as Base/Bull/Bear and product rollups. BYO Excel forecasts are still imported forecast outputs for triangulation; they do not become training data and do not override `forecast.csv` or champion selection.

Preferred long-form sheet shape:

| Required | Recommended | Optional |
| -------- | ----------- | -------- |
| `ds`, `yhat` | ordered grouping columns such as `ProductGroup`, `ProductLine`, `Product` | `cutoff`, `owner`, `currency`, `unit_label`, `model_version`, `notes` |

If `unique_id` is absent and `--group-cols` is supplied, IDs are generated from the ordered grouping columns and explicit derived-sum rollups are added for `Total` and each prefix level. Workbook subtotal rows with labels such as `Total` or `Subtotal` are rejected so rollups are not double counted.

```powershell
# Create a small synthetic Base/Bull/Bear workbook, cutoff snapshots, actuals, and a minimal scaffold run.
uv run python examples\byo_excel_model\create_example_inputs.py --output runs\byo_excel_example

# Import one or more workbook sheets into the canonical external forecast contract.
uv run nixtla-scaffold byo-model ingest --file runs\byo_excel_example\finance_model.xlsx --sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --output runs\byo_excel_example\ingest

# Compare BYO versions side by side against a scaffold run. Defaults to <run>\byo_model when --output is omitted.
uv run nixtla-scaffold byo-model compare --run runs\byo_excel_example\scaffold_run --file runs\byo_excel_example\finance_model.xlsx --sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --main-model-preference Base

# Score only cutoff-labeled historical snapshots after actuals land.
uv run nixtla-scaffold byo-model score --file runs\byo_excel_example\finance_snapshots.xlsx --actuals runs\byo_excel_example\actuals.csv --sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --output runs\byo_excel_example\scores
```

The compare workflow writes `byo_model_forecasts.csv`, `byo_model_contract.csv`, `forecast_comparison.csv`, `forecast_comparison_summary.csv`, `byo_model_comparison_summary.csv`, and `byo_model_manifest.json`. Generated Streamlit reports discover a sibling `byo_model` folder and add a **BYO / Finance model** section with scenario filters, hierarchy-level filters, a scaffold-vs-BYO line chart, delta table, contract lineage, and score summaries when available.

Executable custom models are also supported as **optional challengers** in the normal forecast tournament. The default refresh path does not change; custom models only run when supplied explicitly:

```powershell
nixtla-scaffold forecast --input data.csv --preset standard --horizon 6 --custom-script models\finance_model.py --custom-model-name "Finance seasonality" --output runs\custom_challenger
```

The runnable template at `examples\custom_finance_model` shows the common finance pattern "recent average MoM growth -> annual run-rate -> monthly seasonality allocation." It can run as a Python example or as a direct script challenger:

```powershell
uv run python examples\custom_finance_model\forecast_custom.py
nixtla-scaffold forecast --input examples\custom_finance_model\input.csv --preset standard --horizon 6 --model-policy baseline --custom-script examples\custom_finance_model\finance_seasonality_model.py --custom-model-name "MoM Growth FY Seasonal" --custom-arg=--annual-target --custom-arg=2850 --output runs\custom_finance_target
```

The template uses complete calendar years for month-of-year seasonality when available, treats `--annual-target` as a calendar-year target, and falls back to recent median MoM growth when no target is supplied. Adapt the script for fiscal-year starts, business-specific caps, or plan constraints before treating it as a production finance model; v1 custom challengers remain point forecasts and are not blended into `WeightedEnsemble`.

Python API:

```python
import pandas as pd
from nixtla_scaffold import CustomModelSpec, ForecastSpec, run_forecast

def last_value_model(history: pd.DataFrame, *, horizon: int, freq: str, cutoff: pd.Timestamp, levels: tuple[int, ...], context: dict) -> pd.DataFrame:
    future_grid = pd.DataFrame(context["future_grid"])
    last = history.sort_values("ds").groupby("unique_id")["y"].last()
    future_grid["yhat"] = future_grid["unique_id"].astype(str).map(last)
    return future_grid[["unique_id", "ds", "yhat"]]

run = run_forecast(
    "data.csv",
    ForecastSpec(
        horizon=6,
        custom_models=(CustomModelSpec(name="Finance last value", callable=last_value_model),),
    ),
)
```

Custom model v1 is deliberately small: one custom challenger per run, raw `y` units only (no log/log1p target transforms or normalization factor columns), point forecasts only, and no automatic arbitrary regressor wiring. Each invocation receives canonical history (`unique_id`, `ds`, `y`) filtered to `ds <= cutoff` for the series in that invocation's future grid, plus `context["future_grid"]`, the exact `unique_id`/`ds` grid that must be returned. Output must contain finite numeric `yhat` rows for every grid row, with no duplicates, missing rows, extra rows, or interval columns. Custom challengers are scored and can be selected independently, but they are not blended into `WeightedEnsemble` in v1. `.py` scripts are invoked with the current Python and receive `--history`, `--future-grid`, `--context`, and `--output` file paths. Successful runs write `appendix\custom_model_contracts.csv` and `audit\custom_model_invocations.csv`; failed custom challengers are excluded from selection but audited and surfaced in warnings.

The public `ExecutiveHeadline` Python object supports stable direct attribute access in the 0.1.x package line. Its `to_dict()` output may append optional fields such as unit labels, absolute deltas, YoY deltas, and portfolio direction splits; consumers should ignore unknown keys.

## Agent-first setup

Use `setup` when the user has a broad request like "forecast this Kusto/DAX/workbook metric" and the agent needs to ask the right intake questions before touching models:

```powershell
nixtla-scaffold setup --workspace runs\usage_overage_setup --data-source kusto --preset standard --series-count single --target-name ARR_30day_avg --time-col day_dt --id-value "Usage Overage ARR" --freq ME --horizon 6 --intervals auto --model-families statsforecast mlforecast --exploration-mode --mcp-regressor-search --outputs all
```

The setup workspace includes:

- `forecast_setup.yaml`: reusable answers/config for the run.
- `questions.json`: structured intake questions covering data source, number of series, intervals and caveats, exploration mode, MCP regressor search, and requested outputs.
- `agent_brief.md`: an analyst/agent checklist with metric-definition checks, grain checks, exploration steps, and exact next commands.
- `queries\source.kql` / `.dax` / `.sql` template when the source is query-backed.
- `data\raw`, `data\canonical`, `outputs`, `reports`, and `notes` folders.

The core intake questions are:

1. Which preset should we start from: quick, standard, strict, or hierarchy?
2. Where is the data coming from: CSV, Excel, Kusto, DAX, SQL, DataFrame, or unknown?
3. How many different things are being forecast: single, few, many, or hierarchy?
4. Do we want point forecasts, prediction intervals, or automatic interval gating? Intervals are only shown when history/model support them.
5. Which model families should be considered: baseline, StatsForecast, MLForecast, HierarchicalForecast, or NeuralForecast research-only?
6. Should exploration mode run first?
7. Can the agent use MCPs to find candidate regressors/drivers? Regressors must have known future values and pass leakage checks.
8. Which outputs are needed: all, CSV, Excel, HTML, base64 HTML, Streamlit, diagnostics, model card?

## Forecast presets

Presets keep the first touch simple while still making strict and hierarchy workflows discoverable:

| Preset      | Use when                                       | Key defaults                                                                              |
| ----------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `quick`     | You need a fast first read or smoke run        | Baseline ladder, 6-period horizon, concise verbose-off execution                          |
| `standard`  | You need the standard serious finance forecast | Light model policy, trust/action artifacts, intervals, weighted ensemble, full diagnostics |
| `strict`    | The forecast feeds a high-stakes decision      | Requires backtests and full-horizon CV for champion selection                             |
| `hierarchy` | Parent/child planning totals must tie          | Standard defaults plus `bottom_up` reconciliation unless overridden                       |

CLI flags always win over preset defaults:

```powershell
nixtla-scaffold forecast --input data.csv --preset quick --horizon 3 --output runs\quick
nixtla-scaffold forecast --input data.csv --preset strict --no-require-backtest --horizon 6 --output runs\strict_override
nixtla-scaffold guide presets
```

Python users can start from the same presets:

```python
from nixtla_scaffold import forecast_spec_preset, run_forecast

spec = forecast_spec_preset("standard", horizon=6, freq="ME")
run = run_forecast("data.csv", spec)
run.to_directory("runs/standard_forecast")
```

## Five-minute onboarding paths

Use these when a new analyst or agent needs a concrete starting point:

| Path                                                  | What it teaches                                                                                             | Command                                                                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `examples\feature_tour\forecast_feature_tour.ipynb`   | Notebook walkthrough of the full feature map: profile, forecast, compare, hierarchy, experiments, workbench | `uv run --with jupyter jupyter lab examples\feature_tour\forecast_feature_tour.ipynb`                               |
| `examples\air_tourism_demo`                           | Demo path from AirPassengers baseline to Control Pane, driver audit, notebook, and optional TourismSmall hierarchy | `uv run python examples\air_tourism_demo\forecast_air_tourism_demo.py --output runs\air_tourism_demo`          |
| `examples\quickstart_csv`                             | Smallest CSV-to-forecast flow using `preset="quick"`                                                        | `uv run python examples\quickstart_csv\forecast_quick.py`                                                           |
| `examples\serious_finance_forecast`                   | Finance target normalization plus an auditable future event overlay                                         | `uv run python examples\serious_finance_forecast\forecast_finance.py`                                               |
| `examples\custom_finance_model`                       | Opt-in custom challenger: recent MoM growth to annual run-rate, allocated by historical monthly seasonality | `uv run python examples\custom_finance_model\forecast_custom.py`                                                    |
| `examples\byo_excel_model`                            | BYO Excel model outputs: Base/Bull/Bear sheets, product rollups, compare, and cutoff scoring                | `uv run python examples\byo_excel_model\create_example_inputs.py --output runs\byo_excel_example`                   |
| `examples\hierarchy_reconciliation`                   | Leaf data -> hierarchy nodes -> coherent bottom-up planning forecast                                        | `uv run python examples\hierarchy_reconciliation\forecast_hierarchy.py`                                             |
| `examples\contoso_dax_pipeline\pipeline.yaml`         | DAX-style multi-query source pipeline using the Mock Contoso shape: extracts + transform -> canonical forecast input + run provenance | `uv run nixtla-scaffold pipeline run --config examples\contoso_dax_pipeline\pipeline.yaml --output runs\contoso_dax_pipeline` |
| `examples\contoso_kql_pipeline\pipeline.yaml`         | KQL-style multi-query source pipeline using the public ContosoSales revenue shape: KQL files + transform -> canonical forecast input + provenance | `uv run nixtla-scaffold pipeline run --config examples\contoso_kql_pipeline\pipeline.yaml --output runs\contoso_kql_pipeline` |
| `examples\datasetsforecast_tourism_small`             | Optional public real-data validation using Nixtla DatasetsForecast TourismSmall quarterly hierarchy         | `uv run --extra datasets python examples\datasetsforecast_tourism_small\forecast_tourism_small.py --allow-download` |
| `examples\python_api_templates\dataframe_forecast.py` | Minimal DataFrame API template for agents and notebooks                                                     | import `run_example(...)` or adapt the function                                                                     |

For workbook users, put the same `unique_id`, `ds`, `y` columns in an Excel sheet and run:

```powershell
nixtla-scaffold forecast --input workbook.xlsx --sheet Data --preset standard --horizon 6 --output runs\workbook_demo
```

For MCP-backed DAX/Kusto/SQL users, first export the query result, then let `ingest` preserve source metadata and run the forecast:

```powershell
nixtla-scaffold ingest --input query_result.json --source kusto --query-file source.kql --id-value "Revenue" --time-col Month --target-col Revenue --output runs\revenue_input.csv --forecast-output runs\revenue_forecast --preset standard --horizon 6
```

Optional public real-data validation is available through Nixtla DatasetsForecast. It is not part of the default install or test path because it may download public data. Install/run it explicitly:

```powershell
uv run --extra datasets python examples\datasetsforecast_tourism_small\forecast_tourism_small.py --allow-download --output runs\tourism_small_real_data
uv run --extra datasets python examples\air_tourism_demo\forecast_air_tourism_demo.py --include-tourism --allow-download --output runs\air_tourism_demo_full
```

The example uses TourismSmall bottom-level public data, rebuilds scaffold-compatible `purpose -> state -> city/noncity` hierarchy nodes, validates the generated total against the source total, then runs a quarterly bottom-up baseline forecast. Open `appendix\trust_summary.csv`, `appendix\hierarchy_reconciliation.csv`, `model_card.md`, and `report.html` first; this is a real-data package smoke, not a finance-specific forecast.

Model-family posture:

- `baseline`: always-on sanity methods such as Naive, HistoricAverage, RandomWalkWithDrift, WindowAverage, and SeasonalNaive.
- `statsforecast`: active production classical engine with AutoARIMA, AutoETS, AutoTheta, MSTL, MSTL-driven AutoARIMA variants, per-series StatsForecast `SklearnModel` trend/Fourier regressions, MFLES/AutoMFLES when available, benchmarks, cross-validation, and intervals when supported. The runner retries candidate-by-candidate so one model failure does not collapse the full family.
- `mlforecast`: active ML engine when optional dependencies are installed; uses lag/date features across sklearn and LightGBM candidate families, with Nixtla `PredictionIntervals` conformal bands when history/horizon/lag requirements allow. The default feature policy is deliberately small; `--mlforecast-feature-policy rolling` adds a conservative rolling-transform set, and `--train-known-future-regressors` lets audited known-future drivers enter MLForecast only after leakage/future-value checks pass.
- `smooth`: optional ADAM family from the `smooth` package. Install with `uv sync --extra smooth` to let `standard` and `all` runs evaluate `SmoothADAM_ZXZ`, `SmoothADAM_CCC`, and `SmoothADAM_CustomPool`; `light` stays slim unless smooth is explicitly allowlisted. If the extra is absent, `model_policy_resolution` reports `not_installed_or_not_enabled` and the run continues for standard/all runs. The optional dependency is LGPL-2.1, so review those license terms before redistributing an environment that includes it.
- `hierarchicalforecast`: optional reconciliation path for hierarchy nodes; the scaffold can emit diagnostic-only independent forecasts or coherent planning forecasts with BottomUp / TopDown / MinTrace-style reconciliation when the optional dependency is installed. The built-in `top_down` method is local and transparent; MinTrace variants require the optional hierarchy extra.
- `neuralforecast_research`: research-only because dependency weight and explainability need extra scrutiny.

Model-policy semantics are explicit and audited in `manifest.json` under `model_policy_resolution`:

| Policy          | What it means                                                                                                                                         | Failure/skipping behavior                                                                                                                                                                                                         |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `standard`      | Serious default run: StatsForecast/classical models, feasible MLForecast, and optional smooth ADAM candidates when the `smooth` extra is installed. | MLForecast import/runtime/no-candidate failures are warnings; smooth is optional unless explicitly allowlisted. Skipped optional families are disclosed in `model_policy_resolution`, and the run continues with eligible classical/baseline models. |
| `light`         | Slim compatibility path: StatsForecast/classical models plus feasible MLForecast, without smooth by default. | MLForecast import/runtime/no-candidate failures are warnings; the run continues with eligible classical/baseline models. Smooth runs only when explicitly allowlisted, and allowlisted smooth failures raise clearly. |
| `all`           | Run every eligible open-source family. "All" means all families that are valid for the data, not every model regardless of history.                   | If MLForecast is eligible but unavailable, fails, or produces no candidates, the run raises instead of silently downgrading. Smooth remains optional: when the extra is absent it is reported as unavailable; when explicitly allowlisted, smooth failures raise. If history/horizon cannot support a family, the skip is disclosed and classical models continue. |
| `statsforecast` | Run the classical/open-source statistical ladder only.                                                                                                | StatsForecast failures are surfaced rather than hidden behind MLForecast.                                                                                                                                                         |
| `mlforecast`    | Run MLForecast only.                                                                                                                                  | Missing dependencies, runtime failures, or no candidates raise because this policy explicitly requested ML.                                                                                                                       |
| `baseline`      | Run simple benchmark models only.                                                                                                                     | Always available for usable history; useful for smoke tests and short-series fallbacks.                                                                                                                                           |

`auto` remains accepted as a legacy alias for `light`, and `finance` remains accepted as a legacy alias for the `standard` preset. New configs, manifests, and docs prefer `light` and `standard`.

Use `--model` when a team wants a favorite-model tournament instead of the full ladder. Friendly aliases are canonicalized, so `--model arima --model "arima mstl"` runs only `AutoARIMA` and `MSTL_AutoARIMA`; add `--model "arima mstl features"` to run the Nixtla `mstl_decomposition` pattern that feeds MSTL trend/seasonal features into AutoARIMA as exogenous regressors (`AutoARIMA_MSTLFeatures`). Add aliases such as `--model "stats sklearn ridge"` for the StatsForecast `SklearnModel` per-series trend/Fourier regressions; plain `--model ridge` remains the MLForecast lag/date `Ridge` candidate. Smooth aliases include `--model "smooth adam zxz"`, `--model "smooth adam ccc"`, and `--model "smooth adam custom pool"` after installing the optional `smooth` extra. The same values can be passed in one grouped flag with `--model-allowlist arima "arima mstl" "arima mstl features" "stats sklearn ridge"`. If weighted ensembles stay enabled, `WeightedEnsemble` is derived only from the allowlisted candidates; add `--no-weighted-ensemble` when the output should contain literal model columns only.

The two MSTL + AutoARIMA options are intentionally different. `MSTL_AutoARIMA` is a StatsForecast MSTL model whose trend forecaster is non-seasonal AutoARIMA. `AutoARIMA_MSTLFeatures` first decomposes each eligible series with MSTL, then passes the generated trend/seasonal feature columns into AutoARIMA as exogenous regressors. Default `light` / `statsforecast` runs include both when the data can support them; in mixed-history panels, long-enough series get the MSTL-feature candidate while shorter series are skipped with a warning instead of disabling the candidate globally.

The StatsForecast sklearn candidates are also intentionally separate from MLForecast sklearn models. `StatsSklearn_LinearRegression`, `StatsSklearn_Ridge`, and `StatsSklearn_Lasso` train one sklearn estimator per series through StatsForecast using trend plus Fourier features from `utilsforecast.feature_engineering`; MLForecast candidates train global lag/date models across series. Default `light` / `statsforecast` runs include the StatsForecast sklearn candidates when sklearn is installed and at least one seasonal series has enough rows; shorter mixed-panel children are skipped with explicit warnings.

The 30-observation gate remains for ML-only allowlist runs and opt-in known-future regressor training, because those paths need a deeper history contract. The normal `light`/`standard` path uses a smaller feasibility check instead: enough rows for at least one ML rolling-origin window and usable lag features. The CLI prints a compact `Model families:` line after each forecast, and the full audited details live in `manifest.json -> model_policy_resolution`.

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --model arima --model "arima mstl" --no-weighted-ensemble --output runs\arima_favorites
```

When StatsForecast and MLForecast both run, family-local `WeightedEnsemble` columns are removed before the combined tournament is scored. The scaffold recomputes one unioned `WeightedEnsemble`, learns weights only from common finite cross-validation support, and records each weighted model's `family` in `audit\model_weights.csv`.

Use a long table:

| unique_id | ds         | y      |
| --------- | ---------- | ------ |
| Revenue   | 2024-01-31 | 100000 |
| Revenue   | 2024-02-29 | 104000 |

- `unique_id`: series name, product, SKU, account, or other grain.
- `ds`: date.
- `y`: numeric value to forecast.

If `unique_id` is missing and the file has only one series, the loader creates `series_1`.

Finance exports often use business-friendly names. Map them without rewriting the file:

```powershell
nixtla-scaffold forecast --input plan.xlsx --sheet Data --id-col Product --time-col Month --target-col Revenue --horizon 6 --output runs\plan
```

For hierarchy work, first aggregate leaf data into auditable hierarchy nodes, then forecast the generated CSV:

```powershell
nixtla-scaffold hierarchy --input examples\hierarchy_generic\input.csv --hierarchy-cols region product --output runs\hierarchy_nodes.csv
nixtla-scaffold forecast --input runs\hierarchy_nodes.csv --horizon 3 --freq ME --output runs\hierarchy_demo

# When parent/child planning totals must tie, enable reconciliation.
nixtla-scaffold forecast --input runs\hierarchy_nodes.csv --horizon 3 --freq ME --hierarchy-reconciliation bottom_up --output runs\hierarchy_reconciled

# When you want both planning views, save bottom-up and top-down paths together.
nixtla-scaffold forecast --input runs\hierarchy_nodes.csv --horizon 3 --freq ME --hierarchy-reconciliation both --output runs\hierarchy_both
```

Hierarchy column order matters: `region product` creates Total -> region -> region/product. By default, the workflow forecasts each node independently and writes appendix-only hierarchy add-on outputs: `appendix\hierarchy_rollup.csv` for parent history/selected forecast versus immediate-child sums, `appendix\hierarchy_coherence.csv` for parent forecast versus child-sum gaps, and `appendix\hierarchy_contribution.csv` for child-level contribution storytelling. Add `--hierarchy-reconciliation bottom_up`, `--hierarchy-reconciliation top_down`, `--hierarchy-reconciliation both`, `--hierarchy-reconciliation mint_ols`, or `--hierarchy-reconciliation mint_wls_struct` when parent/child forecasts need to be coherent for planning. Reconciliation is transparent: the model tournament still evaluates the independent statistical forecasts, while the reconciled outputs enforce planning coherence and write pre/post gap audits. `both` keeps bottom-up as the primary forecast output and writes `appendix\hierarchy_reconciliation_comparison.csv` so reports and Streamlit can compare bottom-up and top-down paths side by side.

For Kusto, DAX, or other MCP-backed data, export the query result and let `ingest` create the forecast-ready input plus source metadata:

```powershell
nixtla-scaffold ingest --input kusto_export.json --source kusto --query-file usage_overage_arr.kql --id-value "Usage Overage ARR" --time-col day_dt --target-col ARR_30day_avg --output runs\usage_overage_arr_input.csv --forecast-output runs\usage_overage_arr_demo --freq ME --horizon 6
```

`ingest` accepts CSV, Excel, row-oriented JSON/JSONL, and MCP-style columnar JSON. It writes the canonical `unique_id,ds,y` CSV, a `.source.json` metadata file, and a copied `.kql`/`.dax` query artifact when provided.

When one forecast needs several source queries, use the lightweight source pipeline instead of hand-running a pile of scripts. A pipeline YAML declares extract scripts, query files, optional transform logic, canonical column mapping, and the forecast spec. The runner writes a `pipeline_manifest.json`, validates every declared output, hashes config/scripts/queries/results, writes a `pipeline_summary.md` with a Mermaid flowchart, and attaches `appendix\source_pipeline_manifest.json` plus `appendix\source_pipeline_summary.md` to the forecast run.

```powershell
uv run nixtla-scaffold pipeline run --config examples\contoso_dax_pipeline\pipeline.yaml --output runs\contoso_dax_pipeline
uv run nixtla-scaffold pipeline run --config examples\contoso_kql_pipeline\pipeline.yaml --output runs\contoso_kql_pipeline
uv run nixtla-scaffold pipeline run --config examples\contoso_dax_pipeline\pipeline.yaml --output runs\contoso_dax_prepared --no-forecast
uv run nixtla-scaffold pipeline refresh --config examples\contoso_dax_pipeline\pipeline.yaml --previous-run runs\contoso_dax_pipeline\forecast --output runs\contoso_dax_refresh
```

The Contoso DAX example follows the Mock Contoso Sales connection shape (`MOCK://contoso`) and actually executes the `.dax` files through `scripts\run_dax_extract.py`, a small Python scaffold that mirrors the DAX MCP `run_query.py` output pattern. It stays deterministic/offline by default, but a real DAX/Power BI workflow can point `DAX_CONNECTION_STRING` at a semantic model. Live Power BI/Analysis Services DAX extracts require the Microsoft Analysis Services OLE DB Provider (`MSOLAP`) and `pywin32` in the Python environment running the script. Keep the YAML contract and swap only the connection/query details so each extract still writes the same declared CSV outputs. This keeps many Kusto/DAX/SQL/Python steps collapsible into one canonical `unique_id,ds,y` input without adding a DAG engine.

The Contoso KQL example mirrors the same pipeline pattern for Azure Data Explorer. Its checked-in query targets the public help cluster shape (`https://help.kusto.windows.net`, database `ContosoSales`, tables `SalesFact` and `Products`) and forecasts monthly `Revenue` by `ProductCategoryName`. It runs offline by default through `scripts\run_kql_extract.py` with deterministic ContosoSales-shaped data; set `KUSTO_MODE=live` after installing `azure-kusto-data` and `azure-identity` to query the live help cluster or swap in your own `KUSTO_CLUSTER_URL` / `KUSTO_DATABASE`.

Add auditable event/scenario overlays when finance knows something the history cannot know:

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --event '{"name":"Product launch","start":"2026-03-31","effect":"multiplicative","magnitude":0.10}' --output runs\launch_scenario

# Reusable JSON/YAML/CSV assumption files are also supported.
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --event-file scenario_events.csv --output runs\launch_scenario_file
```

The statistical `yhat` remains in the output; scenario columns such as `yhat_scenario`, `event_adjustment`, and `event_names` show the overlay.

Declare external regressors when MCPs or planning files provide drivers that may become future model inputs. By default, the scaffold audits leakage and future availability, then writes the contract artifacts without training external regressors. This keeps the default workflow simple and fail-closed. When you want approved model-candidate drivers to enter MLForecast, opt in explicitly with `--train-known-future-regressors`; known-future rows must pass future-value coverage, known-as-of timing, and leakage checks before they are materialized as dynamic features. `availability="historical_only"` model candidates can enter only as safe lag features, and only when every forecast-horizon row can be built from driver values known at the forecast origin. The scaffold does not forecast regressors recursively.

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --regressor '{"name":"Seats plan","value_col":"seats_plan","availability":"plan","mode":"model_candidate","future_file":"future_seats.csv","source_system":"excel","owner":"finance"}' --output runs\driver_audit
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --regressor-file known_future_regressors.yaml --output runs\driver_audit_file

# Opt-in driver modeling stays explicit.
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --model-policy mlforecast --train-known-future-regressors --regressor '{"name":"Seats plan","value_col":"seats_plan","availability":"plan","mode":"model_candidate","future_file":"future_seats.csv"}' --output runs\driver_model

# Historical-only drivers use safe lag features; no future driver file is required.
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 3 --model-policy mlforecast --train-known-future-regressors --regressor '{"name":"Seats history","value_col":"seats","availability":"historical_only","mode":"model_candidate"}' --output runs\driver_lag_model

# Keep the ML feature ladder small unless a run needs the rolling-transform experiment.
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --model-policy mlforecast --mlforecast-feature-policy rolling --output runs\ml_rolling
```

Regressor declarations produce `appendix\known_future_regressors.csv`, `appendix\driver_availability_audit.csv`, and `appendix\driver_experiment_summary.csv`. Opt-in MLForecast driver modeling also writes `appendix\driver_model_features.csv`, `appendix\driver_model_cv_delta.csv`, and `appendix\feature_selection_receipts.csv`. The feature receipts are descriptive evidence only; they do not auto-prune features or override backtested model selection.

Run bounded advisory experiments when you want agent-friendly iteration without turning the scaffold into an AutoML model zoo. MCP-connected data sources make this much less manual: pull a candidate slice from Kusto, DAX/Power BI, Excel, SQL, or an API; canonicalize it with the target; then test one driver family or event hypothesis at a time.

```powershell
# Clean leaderboard over the existing model tournament; writes a normal run plus compare_models_leaderboard.csv.
nixtla-scaffold compare-models --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --output runs\compare_models_demo

# Data-aware bounded variants: baseline, all_models, plus events/regressors/hierarchy when inputs exist.
nixtla-scaffold experiment --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --output runs\experiment_demo

# Explicit variants stay capped unless you raise --max-variants.
nixtla-scaffold experiment --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --variants baseline all_models rolling_features --max-variants 3 --output runs\experiment_rolling
```

`compare-models` is a thin advisory surface over the standard run; it does not change champion selection or `forecast.csv`. `experiment` writes normal child run folders under `variants\`, plus `experiment_manifest.json`, `experiment_summary.csv`, `experiment_recommendation.md`, and `experiment_llm_context.json`. The recommendation includes an `autoresearch_next_iteration` block so agents can test one next hypothesis with one metric, one executor, and a fixed budget. MCPs make it cheap to run many experiments, but the design stays bounded: cap variants, change one assumption at a time, stop when metric/trust improvement stalls, and avoid broad feature stuffing. Keep/discard decisions are still human/agent review steps; experiment ranking never mutates child forecasts.

Experiments also inspect preserved extra input columns for undeclared numeric driver candidates such as `rolling_minutes`, `usage`, `seats`, `pipeline`, or `plan` signals. These detections are advisory only: they appear in `candidate_drivers`, `human_context_questions`, and `autoresearch_hypotheses` inside `experiment_llm_context.json` and in the recommendation markdown, but they are not trained unless you explicitly declare one as an audited regressor and opt in with `--train-known-future-regressors`. Candidate records include same-period correlation plus a within-series lag screen; positive `best_lag` means the driver leads the target by that many periods, and lag evidence is never computed by shifting across `unique_id` boundaries.

Use first-class target transforms when the modeling target should differ from the raw reported actuals:

```powershell
# Model on log1p(y), then inverse-transform forecasts/backtests for reporting.
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --target-transform log1p --output runs\log1p_demo

# Model "true underlying movement" after a positive price/FX/inflation factor.
# Outputs remain in normalized units unless you provide future factors externally.
nixtla-scaffold forecast --input plan.csv --horizon 6 --normalization-factor-col price_factor --normalization-label "FY26 pricing" --output runs\normalized_demo
```

The transform audit trail is explicit: `audit\target_transform_audit.csv` records raw `y`, the normalization factor, adjusted `y`, transformed/modeling values, output scale, and notes. Log/log1p forecasts are inverse-transformed for reporting; factor-normalized forecasts stay in normalized units by design so analysts do not confuse raw reported actuals with adjusted economics.

Weighted forecasts are on by default. When rolling-origin metrics exist, the run adds `WeightedEnsemble` as an auditable candidate model and writes `audit\model_weights.csv`. WeightedEnsemble is point-only in this slice: if no calibrated interval bounds exist for the ensemble, long forecast rows disclose that through `interval_status` instead of implying ensemble intervals.
Model selection uses backtested RMSE when available (MAE if RMSE is missing), tie-broken by MAE and absolute bias, so large misses are penalized more heavily than percentage-only scoring. MASE, RMSSE, WAPE, and bias are still reported in `appendix\series_summary.csv`, `appendix\model_audit.csv`, and `appendix\trust_summary.csv` for scale-free and business-readable review. Runs also write `appendix\model_tradeoff_scores.csv` and `appendix\model_pareto_frontier.csv` as a non-authoritative RMSE/MAE Pareto review lens when metrics disagree; WAPE stays diagnostic and non-dominated alternatives do not override the official selected model or `forecast.csv`.
Selection outputs also record the CV horizon contract: `selection_horizon`, `requested_horizon`, `cv_windows`, `cv_step_size`, and `cv_horizon_matches_requested`. If adaptive CV uses a shorter horizon than the requested forecast horizon, the run emits a warning, caps trust, and labels future rows after the validated horizon as directional. Use `--strict-cv-horizon` when a high-stakes decision requires champion selection to be validated at the full requested horizon.

Row-level horizon fields use `row_horizon_status` as the clearest per-row status. `horizon_trust_state` and `forecast_horizon_status` remain backward-compatible aliases in `forecast.csv` / `appendix\forecast_long.csv`. `planning_eligible=True` only means the row passes the horizon-validation gate (`planning_eligibility_scope=horizon_validation_only`); it is not a global approval to ignore Low trust, interval issues, residual warnings, hierarchy tradeoffs, or data-quality caveats.

`horizon_trust_state` glossary:

| Value                        | Where you'll see it                                                        | Meaning                                                                                                                                                                                                                                            |
| ---------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `full_horizon_validated`     | `appendix\trust_summary.csv`, `forecast.csv`, `appendix\forecast_long.csv`, report/workbench | Rolling-origin model selection evaluated the champion through the requested horizon. Check `full_horizon_claim_allowed`: fewer than two CV windows means the full horizon was evaluated but not strong enough for a planning-ready champion claim. |
| `partial_horizon_validated`  | `appendix\trust_summary.csv`, `forecast.csv`, `appendix\forecast_long.csv`, report/workbench | Model selection had rolling-origin evidence, but only through a shorter horizon. Steps beyond `validated_through_horizon` are directional and not planning-ready by themselves.                                                                    |
| `beyond_validated_horizon`   | row-level `forecast.csv` / `appendix\forecast_long.csv`                             | This specific forecast row is after the validated CV horizon; `planning_eligible=False`.                                                                                                                                                           |
| `no_rolling_origin_evidence` | `appendix\trust_summary.csv`, `forecast.csv`, `appendix\forecast_long.csv`                   | No usable rolling-origin evidence exists for the selected champion; trust is capped Low and the forecast should be treated as exploratory.                                                                                                         |

`interval_status` glossary:

| Value                       | Where you'll see it                                                       | Meaning                                                                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `calibrated`                | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`, from `appendix\interval_diagnostics.csv` | Future bands have matching rolling-origin interval coverage evidence for the model/horizon.                                                                                           |
| `calibration_warning`       | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`, from `appendix\interval_diagnostics.csv` | Empirical coverage exists but the gap is large enough to review before planning use.                                                                                                  |
| `calibration_fail`          | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`, from `appendix\interval_diagnostics.csv` | Empirical coverage indicates undercoverage risk.                                                                                                                                      |
| `future_only`               | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`                                  | Future interval bands exist, but matching rolling-origin interval coverage evidence is unavailable for that horizon.                                                                  |
| `adjusted_not_recalibrated` | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`                                  | Final selected bands were shifted/scaled/reconciled after model calibration, so they are planning aids rather than freshly calibrated uncertainty.                                    |
| `point_only_ensemble`       | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`                                  | `WeightedEnsemble` is point-only; use component-model intervals or select a single calibrated model if planning ranges matter. Model disagreement lines are not prediction intervals. |
| `unavailable`               | `appendix\forecast_long.csv`, `appendix\trust_summary.csv`                                  | No interval bands are available for that model row.                                                                                                                                   |
| `insufficient_observations` | `appendix\trust_summary.csv`, from `appendix\interval_diagnostics.csv`                      | Too few interval backtest observations exist to validate coverage.                                                                                                                    |

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --weighted-ensemble --output runs\weighted
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --strict-cv-horizon --output runs\strict_h6
```

Verbose diagnostics are also on by default. Successful runs write `diagnostics.json`, `diagnostics.md`, and `llm_context.json`; failed CLI runs write `failure_diagnostics.json` and `failure_diagnostics.md` in the requested output folder when possible. `llm_context.json` is the best single file to attach when asking another LLM to walk through the forecast because it bundles the deterministic headline, trust summary, horizon gates, interval status, residual and seasonality diagnostics, hierarchy/driver context, guardrails, artifact index, and recommended questions.

The manifest includes a reproducibility block with a SHA-256 hash of canonical history, forecast origin, frequency, season length, Python/platform details, package versions, and git SHA when available.

Forecast runs keep the output root intentionally small. The primary decision artifacts stay at the run root (`forecast.csv`, `forecast.xlsx`, `report.html`, `streamlit_app.py`, `run_streamlit.ps1`, `streamlit_requirements.txt`, `model_card.md`, diagnostics, and manifest files); detailed feeder tables live under `appendix\` and raw wide audit traces live under `audit\`.

Forecast runs now write three classes of artifacts:

1. **Curated outputs** for routine review:
   - `OPEN_ME_FIRST.html`: landing page with the executive headline, file map, and links to the clean workbook, static report, selected forecast, decision summary, Streamlit app, and LLM handoff packet.
   - `output/forecast_review.xlsx`: compact workbook with Start Here, Forecast, Decision Summary, Model Leaderboard, Watchouts, and File Guide sheets. This is intentionally smaller than `forecast.xlsx`.
   - `output/forecast_for_review.csv`: selected forecast rows only, stripped down for finance review while preserving horizon and interval guardrails.
   - `output/decision_summary.csv`: condensed trust/readiness, caveats, and next actions by series.
   - `output/appendix/model_leaderboard.csv`, `output/appendix/forecast_brief.csv`, and `output/appendix/artifact_guide.csv`: small supporting appendix tables.

2. **Core feeder outputs** for downstream models and analyst workflows:
   - `forecast.csv`: selected forecast rows only, enriched with `interval_status`, `interval_method`, `interval_evidence`, `row_horizon_status`, `horizon_trust_state`, `validated_through_horizon`, `planning_eligible`, `planning_eligibility_scope`, and CV horizon metadata so analyst-facing interval bounds and far-horizon rows carry their provenance.
   - `llm_context.json`: single LLM feeder packet with executive headline, run summary, per-series review, trust/horizon/interval/residual/seasonality/hierarchy/driver context, artifact index, guardrails, and recommended questions.
   - `forecast.xlsx`: a curated workbook with the consolidated outputs plus audit sheets.
   - `report.html` / `streamlit_app.py`: human review surfaces. Use `run_streamlit.ps1` or `streamlit_requirements.txt` to open the app from a standalone run folder.

3. **Appendix feeder outputs** for downstream models, analysts, and LLM evidence packs:
    - `appendix\forecast_long.csv`: future predictions in long format, one row per series/model/date with `family`, horizon step, `yhat`, intervals, model weight, `interval_status`, `interval_method`, row-level horizon validation, `planning_eligibility_reason`, CV horizon metadata, and selected-model flag.
    - `appendix\backtest_long.csv`: rolling-origin validation predictions in long format with actuals, cutoff, horizon step, forecast error, squared error, interval bounds, and coverage flags when available.
   - `appendix\series_summary.csv`: one row per series with selected model, RMSE/MAE/MASE/RMSSE/WAPE, CV horizon contract, seasonality, and top weighted alternatives.
   - `appendix\series_features.csv`: deterministic forecastability features such as zero fraction, nonzero observations, coefficient of variation, recent level change, trend/seasonal proxies, forecastability score, and the recommended next experiment step. It is advisory only and does not change model selection.
   - `appendix\borrowed_strength_advisor.csv`: advisory borrowing-strength labels for sparse, short-history, or low-trust series. Labels such as `parent_anchored`, `analog_candidate`, and `panel_pool_review_candidate` tell the analyst what to review next; they do not anchor, pool, reconcile, or override the official forecast.
    - `appendix\model_audit.csv`, `appendix\model_win_rates.csv`, `appendix\model_tradeoff_scores.csv`, `appendix\model_pareto_frontier.csv`, `appendix\feature_selection_receipts.csv`, and `appendix\model_window_metrics.csv`: model leaderboard, naive benchmark win rates, Pareto tradeoff review, descriptive feature evidence, and per-cutoff metrics.
   - `appendix\residual_diagnostics.csv`, `appendix\residual_tests.csv`, and `appendix\interval_diagnostics.csv`: residual, structural-break screening, and interval calibration evidence.
   - `appendix\trust_summary.csv`: first-stop decision artifact with per-series High/Medium/Low trust, score drivers, horizon trust, full-horizon claim gate, caveats, and recommended next actions. Rubric: High >=75, Medium 40-74, Low <40.
   - `appendix\model_explainability.csv`: MLForecast lag/calendar, rolling-transform, and opt-in driver feature importance or coefficient magnitudes when ML models run.
   - `appendix\scenario_assumptions.csv`, `appendix\scenario_forecast.csv`, `appendix\known_future_regressors.csv`, `appendix\driver_availability_audit.csv`, `appendix\driver_model_features.csv`, `appendix\driver_model_cv_delta.csv`, and `appendix\driver_experiment_summary.csv`: scenario and driver evidence.
   - `appendix\hierarchy_rollup.csv`, `appendix\hierarchy_reconciliation.csv`, `appendix\hierarchy_reconciliation_comparison.csv`, `appendix\hierarchy_contribution.csv`, and `appendix\hierarchy_coherence.csv`: hierarchy parent roll-ups, method summaries, bottom-up/top-down comparison, contribution, and coherence evidence.

4. **Audit/detail outputs** for transparency and debugging live under `audit\`: `all_models.csv`, `model_selection.csv`, `backtest_metrics.csv`, `backtest_predictions.csv`, `backtest_windows.csv`, `model_weights.csv`, `target_transform_audit.csv`, `seasonality_profile.csv`, `seasonality_summary.csv`, `seasonality_diagnostics.csv`, `seasonality_decomposition.csv`, `hierarchy_backtest_comparison.csv`, `hierarchy_unreconciled_forecast.csv`, `hierarchy_coherence_pre.csv`, `hierarchy_coherence_post.csv`, and `interpretation.json`. `model_weights.csv` includes `family` so agents can see whether the ensemble learned from baseline, StatsForecast, or MLForecast candidates. `target_transform_audit.csv` records raw, adjusted, transformed, and modeled target values whenever log/log1p or finance normalization is enabled. `seasonality_diagnostics.csv` records cycle counts, credibility labels, warnings, and trend/seasonal/remainder strength so the report does not overclaim annual or weekly seasonality without enough complete cycles. Scenario, driver, hierarchy summary, and best-practice receipt tables live in `appendix\` so the run root stays a small front door; raw reconciliation/pre-post/debug traces remain in `audit\`. The hierarchy comparison/pre/post artifacts preserve the independent forecast, compare selected rolling-origin errors before versus after reconciliation, show parent/child gaps before reconciliation, and confirm remaining gaps after reconciliation. `model_selection.csv` and `backtest_metrics.csv` include CV horizon metadata so agents can see whether validation matched the business horizon. StatsForecast and MLForecast use the same adaptive rolling-origin horizon/window/step contract where feasible; MLForecast lags are capped and disclosed rather than reducing its CV windows.

Visual reports are scaffolded automatically with each forecast run:

- `report.html`: a plain, professional forecast review with a decision summary, model-policy resolution, target transformation audit when relevant, assumptions/driver audit section when events or regressors are declared, forecast-ledger preview when `ledger_context.json` exists, all-model and selected/top-weighted forecast charts, Pareto tradeoff review, descriptive feature receipts, shaded future bands with interval-status caveats, seasonality credibility evidence, a fixed-axis rolling-origin backtest filmstrip with train/holdout shading, model leaderboard, and core output map.
- `report_base64.txt`: the same HTML report base64-encoded for MCPs, tickets, notebooks, or LLM handoffs that need text-only payloads.
- `streamlit_app.py`: an editable local dashboard/workbench with cached local artifact loading and polished sidebar **Workbench section** button tabs so every section stays easy to find while only the active heavy section renders on each rerun. The Forecast review section owns the styled executive headline card, copy-safe code block, decision/next-action cards, and operating loop: connect the refresh end to end, add drivers/regressors, and track forecast performance over time. It includes model-policy resolution from `manifest.json`, series selector, champion lens controls for best overall vs best StatsForecast/classical vs best MLForecast, explicit skipped/failed family reasons when a lens has no candidate, active-champion horizon trust and interval-status banners, a winner-metric selector with guidance for RMSE/MAE/WAPE/MASE/RMSSE/bias/weight tradeoffs, first-glance forecast charts that include interval bands for every displayed interval-bearing candidate model, a dedicated Model investigation section with manual model picking and Pareto tradeoff scatter, menus that show `#rank | model | engine`, a model-picker guide with rank, engine, model type, and role so StatsForecast/classical, MLForecast, baseline, ensemble, and custom candidates are readable, focused forecast/CV comparison charts whose interval ribbons use the same color as the owning model line, are named in the legend, and use the same `forecast_long.csv` model feed for point forecasts plus interval bounds, a fixed-axis CV window player with previous/next/slider controls, a dedicated Prediction intervals section with the top interval-bearing candidate model bands selected by default plus the same rank/engine picker guide, interval-width summary, calibration evidence, and interval row review, benchmark win-rate chart, residual horizon/time/histogram/ACF diagnostics with white-noise heuristic and outlier dates, selected-interval availability warnings, a Seasonality section with cycle-count credibility, configurable seasonal-year overlay that defaults to the active/best model forecast and lets users choose any candidate model while retaining actual-year context, and additive decomposition evidence, MLForecast feature importance, hierarchy roll-up/down diagnostics plus pre/post reconciliation gap review when enabled, an Assumptions & Drivers section for scenario overlays, known-future regressor audits, ML feature importance, and optional PNG/JPG regressor/driver/STL visuals discovered from the run root, `appendix\`, `audit\`, or `experiments\`, and consolidated output previews with `yhat` and interval bound columns kept adjacent. Set `NIXTLA_SCAFFOLD_STREAMLIT_PERF=1` before launching to show artifact load/row-count diagnostics in the sidebar. Run it from the output folder; the launcher uses `uv` plus the generated `streamlit_requirements.txt`, so it does not require the original package project virtual environment:

  The Model investigation and Assumptions & Drivers sections include a lightweight Nixtla-style MLForecast explainability guide inspired by https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/analyzing_models.html: start with `model_explainability.csv`, optionally reproduce `models_`/`preprocess`/`SaveFeatures`/SHAP offline for deeper dives, and keep default runs free of persisted model-object or SHAP artifacts.

```powershell
cd runs\demo
.\run_streamlit.ps1
# Fallback if PowerShell script execution is restricted:
uv run --with-requirements streamlit_requirements.txt streamlit run .\streamlit_app.py
```

### Forecast ledger: versions, official locks, actuals, and Power BI exports

Use `refresh` when new actuals land and you want to reuse a prior run's persisted setup without manually retyping the spec. Explicit CLI flags override the previous manifest; otherwise the prior horizon, frequency, model policy, model allowlist, events, regressors, transforms, hierarchy, and feature settings are reused. Refresh reruns the normal model tournament on the updated data and selects the current best champion under that same policy; it does not silently pin the old selected model. The refreshed run writes `refresh_manifest.json` and `appendix\refresh_delta.csv` with selected forecast, model-selection, trust, and spec changes so model drift and champion changes are visible.

```powershell
uv run nixtla-scaffold refresh --previous-run runs\actions_arr_march --input data_april.csv --output runs\actions_arr_april
```

Use the ledger when a forecast becomes operational and you need to remember what was submitted to leadership, compare later refreshes against that locked view, and track landed actuals without losing source-system revisions. The default folder is `runs\forecast_ledger`; it contains a lightweight `ledger.sqlite` plus stable CSV mirrors under `runs\forecast_ledger\exports` for Power Query / semantic-model ingestion.

```powershell
# Register a run as a version while forecasting, and optionally lock it as a submitted view.
uv run nixtla-scaffold forecast --input data.csv --preset standard --horizon 6 --output runs\product_arr_march --ledger runs\forecast_ledger --forecast-key "Product ARR" --version-label "March refresh" --lock-official --lock-label "March lock"

# Register an existing run folder later.
uv run nixtla-scaffold ledger register --ledger runs\forecast_ledger --run runs\product_arr_april --forecast-key "Product ARR" --version-label "April refresh"

# Append revised actuals from Kusto/DAX/Excel/CSV and preserve every refresh revision.
uv run nixtla-scaffold ledger actuals --ledger runs\forecast_ledger --input actuals.csv --forecast-key "Product ARR" --source-kind kusto --revision-label "April actuals"

# Add anomaly corrections, business-model normalization, or forward-looking regime-change audit rows.
uv run nixtla-scaffold ledger adjustments --ledger runs\forecast_ledger --input adjustments.csv --forecast-key "Product ARR"

# Compare the latest registered version against a submitted lock. Thresholds are user-owned.
uv run nixtla-scaffold ledger compare --ledger runs\forecast_ledger --forecast-key "Product ARR" --against-lock "March lock" --call-up-pct 0.10 --call-down-pct 0.10

# Refresh the folder CSVs Power BI can ingest.
uv run nixtla-scaffold ledger export --ledger runs\forecast_ledger --output runs\forecast_ledger\exports
```

The ledger writes `forecast_versions.csv`, `forecast_snapshot.csv`, `forecast_version_metrics.csv`, `official_forecast_locks.csv`, `forecast_actual_revisions.csv`, `forecast_actuals.csv`, `forecast_performance.csv`, `forecast_version_deltas.csv`, `forecast_adjustments.csv`, `corrected_actuals.csv`, and `regime_changes.csv`. Official locks are intentionally plural: a March lock and April lock can both remain as filterable submitted forecasts. Actuals are revisioned because operational systems can correct history. Corrections/normalizations are audited separately and require an explicit user action before corrected or normalized history feeds a forecast.

When a run has `ledger_context.json`, both generated review surfaces include the ledger: `report.html` embeds a static preview of versions, official locks, lock-vs-refresh deltas, performance, adjustments, corrected/normalized actuals, and regime changes; `streamlit_app.py` adds an interactive **Forecast ledger** sidebar section that starts with one clean line chart: latest actuals/history, official locks emphasized, and recent non-lock forecast versions as lighter lines. Raw deltas, performance rows, actual revisions, adjustments, corrected/normalized actuals, and regime changes stay collapsed in audit tables. Ledger registration and lock commands refresh the run report automatically, and you can rerun `nixtla-scaffold report --run <run>` after later actuals, adjustments, compares, or exports.

### Local operations: status, doctor, drift, and operate

The package now includes production-shaped local operations: receipts, health checks, run discovery, and drift rollups that run entirely on your machine. This is not a cloud pipeline, scheduler, daemon, or DAG framework. It is a disciplined local runbook with machine-readable evidence.

```powershell
# Find the latest local runs and their health without opening Streamlit.
uv run nixtla-scaffold status --runs runs --output runs\ops_status

# Inspect one run for missing artifacts, Low-trust series, horizon caveats, and ledger linkage.
uv run nixtla-scaffold doctor --run runs\actions_arr_april --output runs\actions_arr_april\ops

# Roll up refresh deltas and ledger performance once actuals land.
uv run nixtla-scaffold drift --ledger runs\forecast_ledger --refreshed-run runs\actions_arr_april --output runs\forecast_ledger\drift

# Optional: execute a small linear checklist from YAML. No DAGs, retries, parallelism, or scheduler.
uv run nixtla-scaffold operate --config forecast_operating_loop.yaml --output runs\monthly_operate
```

Every normal forecast run writes `appendix\run_receipt.json`, `appendix\run_receipt.md`, `appendix\validation_receipt.csv`, and `appendix\validation_receipt.json`. `status` scans existing `manifest.json` files. `doctor` is non-mutating and points to the next action. `drift` reads existing `appendix\refresh_delta.csv` and ledger exports rather than rerunning models. `operate` simply runs listed CLI steps in order and writes `operate_manifest.json`.

You can regenerate these report artifacts from an existing run:

```powershell
nixtla-scaffold report --run runs\demo
```

For product-quality validation, run the synthetic scenario lab. It creates 100 train/holdout scenarios with actuals, scores the scaffold on accuracy, validity, ease of use, and explainability, and writes recommendations:

```powershell
nixtla-scaffold scenario-lab --count 100 --model-policy light --output runs\scenario_lab_100
```

The lab writes `scenario_scores.csv`, `scenario_summary.json`, and `scenario_recommendations.json`. The default `light` policy exercises the real StatsForecast-first path when available.

For workbench-quality validation, run the golden QA harness. By default it generates the two flagship refresh demos (`monthly_basic`, `hierarchy_reconciled`) plus two finance guardrails (`limited_history_new_product`, `normalized_target_forecast`), checks required artifacts, compiles each generated `streamlit_app.py`, and runs Streamlit AppTest unless disabled. Legacy aliases `short_history` and `transform_normalized` are still accepted for existing scripts, but new docs and outputs use the clearer names:

```powershell
nixtla-scaffold workbench-qa --output runs\workbench_qa
nixtla-scaffold workbench-qa --scenarios monthly_basic hierarchy_reconciled --model-policy baseline --no-app-test --output runs\workbench_qa_refresh
nixtla-scaffold workbench-qa --scenarios limited_history_new_product normalized_target_forecast --model-policy baseline --no-app-test --output runs\workbench_qa_guardrails
nixtla-scaffold workbench-qa --output runs\workbench_qa --app-test-timeout 120
```

The harness writes `workbench_qa_summary.csv` and `workbench_qa_summary.json` with pass/fail status, usability score, selected models, Streamlit compile status, AppTest status, AppTest timeout seconds, missing artifacts, and missing workbench text. It also writes `workbench_perf_summary.csv` and `workbench_perf_summary.json` with compile/AppTest seconds, generated app size, CSV artifact row counts, and a performance status so dashboard snappiness regressions are visible. Use it before declaring dashboard/report changes done.

For local release readiness, run the release gates. This builds the package, installs the wheel into an isolated environment, smoke-tests public APIs/CLI, runs deterministic numeric goldens, validates generated workbenches, checks artifact hygiene, reports optional-extra status, and launches one generated Streamlit dashboard for an HTTP health smoke:

```powershell
nixtla-scaffold release-gates --output runs\release_gates
nixtla-scaffold release-gates --output runs\release_gates_fast --no-build --no-install-smoke --no-workbench-qa --no-live-streamlit --scenario-count 2
nixtla-scaffold release-gates --output runs\release_gates --json
nixtla-scaffold release-gates --extended --output runs\release_gates_extended
```

The CLI prints a compact one-glance verdict by default and writes `release_gate_summary.md`, `release_gate_summary.json`, and `release_gate_results.csv`. Use `--json` when an agent needs the full nested payload in stdout. A failed required gate makes the CLI return nonzero. Optional extras (`ml`, `hierarchy`, `neural`, `datasets`) are reported by default and can be required with `--require-optional ml hierarchy` or `--require-optional datasets`. Use `--extended` for a stricter local release profile: at least 20 light-policy scenarios, all-family workbench QA, and required `ml` plus `hierarchy` extras.

How to read the gate:

| Surface                     | What to check                                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compact CLI output          | Top-line `PASSED`/`FAILED`, version, Python, duration, failed gate names, remediation hints, skip reasons, provenance warnings, and artifact paths.                                       |
| `release_gate_summary.md`   | Readable green-light artifact with gate table, provenance, required optional groups, scenario archetype scores, thresholds, exit codes, and failure actions.                              |
| `release_gate_summary.json` | Full machine-readable payload with `summary.headline`, `failed_gates`, `failure_rollup`, provenance, git-SHA unavailable reason, options, per-gate durations, and nested command details. |
| `release_gate_results.csv`  | One-row-per-gate table with flattened status, reason, remediation, artifact, and details JSON for quick audit/filtering.                                                                  |

Thresholds are intentionally broad smoke gates rather than brittle exact-value snapshots: scenario lab requires zero crashes, at least `count - 2` scenarios passing, composite score >= 70, validity >= 85, and explainability >= 80; workbench QA requires no failed golden scenarios and minimum usability score >= 90; quick forecast requires finite bounded selected forecasts, valid interval containment when interval columns exist, a numeric trust score, and an executive headline. Exit codes are `0` for passed gates, `1` for gate failure, and `2` for CLI/runtime errors. The live Streamlit smoke proves the generated dashboard launches in the current Python environment.

### Automated releases

The GitHub Actions workflow in `.github\workflows\release.yml` runs on pull requests and on pushes to `main`. Pull requests run validation only. A push to `main` runs the same validation, reads `[project].version` from `pyproject.toml`, builds the package, creates a GitHub release tagged `v<version>`, and marks it latest.

Release rule: every merge to `main` that should publish a release must bump `pyproject.toml` to a new version before merging. If the tag already exists, the release job fails with a clear version-bump message instead of overwriting an existing release. If the repository has a `PYPI_API_TOKEN` secret configured, the workflow publishes the built wheel and source distribution to PyPI before creating the GitHub release; otherwise it creates the GitHub release only.

Local outputs are intentionally ignored by git. `.gitignore` excludes `.intern`, `.venv`, `runs`, temporary validation folders, query artifacts, and local reports because user tests often contain sensitive data.

## What the scaffold does

1. Profiles the data: grain, date range, inferred frequency, missing dates, duplicates, short histories, zeros, negatives, and freshness.
2. Repairs regular time series gaps using an explicit fill policy.
3. Runs simple baselines and, when available, a resilient StatsForecast ladder.
4. Backtests when enough history exists.
5. Selects a champion model per series.
6. Adds a Pareto tradeoff review for non-dominated RMSE/MAE alternatives without changing the official champion; WAPE and scale-free metrics remain diagnostic context only.
7. Adds an inverse-error weighted ensemble when validation data supports it.
8. Supports auditable finance target transforms: log/log1p plus positive factor normalization for pricing, FX, inflation, or definition-change analysis.
9. Supports auditable event/scenario overlays plus known-future and historical-only regressor contracts with leakage/future-availability checks; historical-only drivers can enter MLForecast only as safe lag features and regressors are never recursively forecast.
10. Writes descriptive feature-selection receipts for MLForecast evidence without auto-pruning or overriding champion selection.
11. Refreshes prior runs from the persisted manifest and writes compact appendix deltas.
12. Optionally reconciles hierarchy forecasts for coherent parent/child planning outputs while preserving unreconciled model-tournament evidence.
13. Writes consolidated feeder files, audit artifacts, JSON manifest, model card, diagnostics, HTML/base64/Streamlit reports, and Excel workbook.
14. Runs a scenario lab to score the scaffold like a forecaster against known holdout actuals.

StatsForecast execution is failure-isolated: the scaffold tries the full ladder first, then retries individual candidates if a model/interval/backtest path fails. Successful classical models stay in the run, failed candidates are disclosed as warnings, and interval-only failures keep the point forecast when possible.

## Simplicity rules

- One obvious path first: load, profile, forecast, explain, export.
- Advanced options live in config/CLI flags, not required setup.
- No local MCP server in the MVP; MCPs can provide data as CSV, Excel, DataFrames, or script-backed source pipeline extracts.
- TimeGPT is out of scope.
- NeuralForecast is optional research, not a core default.

## Embedded guidance

`nixtla-scaffold guide` searches the built-in knowledge base of Nixtla docs, Nixtla GitHub source files, and FPPy best practices. It is intentionally JSON-backed and compact so agents can cite and update it without creating a docs framework.

FPPy citation standard used by the scaffold: Hyndman, R.J., Athanasopoulos, G., Garza, A., Challu, C., Mergenthaler, M., & Olivares, K.G. (2025). Forecasting: Principles and Practice, the Pythonic Way. OTexts: Melbourne, Australia. Available at: OTexts.com/fpppy. Accessed on 28 April 2026.

The knowledge base now includes Nixtla source-code maps for the repos that matter most:

- `Nixtla/statsforecast`: `python/statsforecast/core.py`, `models.py`, `arima.py`, `ets.py`, `theta.py`, `mstl.py`
- `Nixtla/utilsforecast`: `preprocessing.py`, `evaluation.py`, `losses.py`, `plotting.py`, `model_selection.py`, `validation.py`
- `Nixtla/hierarchicalforecast`: `core.py`, `methods.py`, `probabilistic_methods.py`, `evaluation.py`, `utils.py`
- `Nixtla/mlforecast`: future driver/regressor path using `core.py`, `forecast.py`, `feature_engineering.py`, lag transforms, target transforms, AutoML, and cross-validation guides

It also links to high-value docs/notebooks such as StatsForecast complete getting started, StatsForecast intermittent demand, UtilsForecast fill/evaluate/plot examples, MLForecast cross-validation, and HierarchicalForecast tourism reconciliation examples.
