Trying to minimize issues with people vibe coding forecasts what could possibly go wrong!!!

# nixtla-scaffold

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
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --preset finance --horizon 6 --levels 80 95 --output runs\demo
```

Open these first:

| File                                                                   | Why it matters                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_card.md` top paragraph or `diagnostics.json.executive_headline` | One-paragraph deterministic forecast headline. Read or quote this first if you only have 60 seconds.                                                                                                                                                                                    |
| `trust_summary.csv`                                                    | First-stop High/Medium/Low readiness, caveats, next actions, and whether the full requested horizon is validated.                                                                                                                                                                       |
| `report.html`                                                          | Human-readable decision summary, model review, horizon trust, intervals, seasonality, and backtest evidence.                                                                                                                                                                            |
| `llm_context.json`                                                     | One-attachment LLM handoff packet with the headline, trust/horizon/interval/residual/seasonality/hierarchy/driver context, artifact index, guardrails, and questions to ask.                                                                                                            |
| `forecast.csv`                                                         | Selected forecast only. Start with `row_horizon_status` to identify beyond-validated-horizon rows; `planning_eligible=True` means the row passed the horizon-validation gate, not global planning approval. Pair it with trust, interval, residual, hierarchy, and data-quality review. |
| `forecast_long.csv`                                                    | Primary model-feed output with every future series/model/date row, selected-model flag, intervals when available, interval status, row-level horizon validation, and `planning_eligibility_scope`.                                                                                      |

Trust rubric: **High >=75**, **Medium 40-74**, **Low <40**. A High score still means "statistical baseline with evidence," not a plan or guarantee. `planning_eligible=True` in `forecast.csv` / `forecast_long.csv` only means the row passed the horizon-validation gate (`planning_eligibility_scope=horizon_validation_only`); it does not override Low trust, interval issues, residual warnings, hierarchy tradeoffs, or data-quality caveats. Agents should quote `diagnostics.json.executive_headline.paragraph` verbatim rather than rewriting it into a stronger claim. The generated Streamlit app also shows a copy-safe "Copy headline" text box so the deterministic headline can be pasted without paraphrasing.

## Refresh-first workflow

Keep routine forecasting boring: use the same input contract, same command, and a new output folder for each refresh. Advanced scenarios and guardrails exist to protect quality, but the main monthly loop should stay `profile -> forecast -> report`.

```powershell
nixtla-scaffold profile --input data.csv
nixtla-scaffold forecast --input data.csv --preset finance --horizon 6 --output runs\latest
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

Executable custom models are also supported as **optional challengers** in the normal forecast tournament. The default refresh path does not change; custom models only run when supplied explicitly:

```powershell
nixtla-scaffold forecast --input data.csv --preset finance --horizon 6 --custom-script models\finance_model.py --custom-model-name "Finance seasonality" --output runs\custom_challenger
```

The runnable template at `examples\custom_finance_model` shows the common finance pattern "recent average MoM growth -> annual run-rate -> monthly seasonality allocation." It can run as a Python example or as a direct script challenger:

```powershell
uv run python examples\custom_finance_model\forecast_custom.py
nixtla-scaffold forecast --input examples\custom_finance_model\input.csv --preset finance --horizon 6 --model-policy baseline --custom-script examples\custom_finance_model\finance_seasonality_model.py --custom-model-name "MoM Growth FY Seasonal" --custom-arg=--annual-target --custom-arg=2850 --output runs\custom_finance_target
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

Custom model v1 is deliberately small: one custom challenger per run, raw `y` units only (no log/log1p target transforms or normalization factor columns), point forecasts only, and no automatic arbitrary regressor wiring. Each invocation receives canonical history (`unique_id`, `ds`, `y`) filtered to `ds <= cutoff` for the series in that invocation's future grid, plus `context["future_grid"]`, the exact `unique_id`/`ds` grid that must be returned. Output must contain finite numeric `yhat` rows for every grid row, with no duplicates, missing rows, extra rows, or interval columns. Custom challengers are scored and can be selected independently, but they are not blended into `WeightedEnsemble` in v1. `.py` scripts are invoked with the current Python and receive `--history`, `--future-grid`, `--context`, and `--output` file paths. Successful runs write `custom_model_contracts.csv` and `audit\custom_model_invocations.csv`; failed custom challengers are excluded from selection but audited and surfaced in warnings.

The public `ExecutiveHeadline` Python object supports stable direct attribute access in the 0.1.x package line. Its `to_dict()` output may append optional fields such as unit labels, absolute deltas, YoY deltas, and portfolio direction splits; consumers should ignore unknown keys.

## Agent-first setup

Use `setup` when the user has a broad request like "forecast this Kusto/DAX/workbook metric" and the agent needs to ask the right intake questions before touching models:

```powershell
nixtla-scaffold setup --workspace runs\premium_overage_setup --data-source kusto --preset finance --series-count single --target-name ARR_30day_avg --time-col day_dt --id-value "Premium Overage ARR" --freq ME --horizon 6 --intervals auto --model-families statsforecast mlforecast --exploration-mode --mcp-regressor-search --outputs all
```

The setup workspace includes:

- `forecast_setup.yaml`: reusable answers/config for the run.
- `questions.json`: structured intake questions covering data source, number of series, intervals and caveats, exploration mode, MCP regressor search, and requested outputs.
- `agent_brief.md`: a human/agent checklist with metric-definition checks, grain checks, exploration steps, and exact next commands.
- `queries\source.kql` / `.dax` / `.sql` template when the source is query-backed.
- `data\raw`, `data\canonical`, `outputs`, `reports`, and `notes` folders.

The core intake questions are:

1. Which preset should we start from: quick, finance, strict, or hierarchy?
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
| `finance`   | You need the standard serious finance forecast | Auto model policy, trust/action artifacts, intervals, weighted ensemble, full diagnostics |
| `strict`    | The forecast feeds a high-stakes decision      | Requires backtests and full-horizon CV for champion selection                             |
| `hierarchy` | Parent/child planning totals must tie          | Finance defaults plus `bottom_up` reconciliation unless overridden                        |

CLI flags always win over preset defaults:

```powershell
nixtla-scaffold forecast --input data.csv --preset quick --horizon 3 --output runs\quick
nixtla-scaffold forecast --input data.csv --preset strict --no-require-backtest --horizon 6 --output runs\strict_override
nixtla-scaffold guide presets
```

Python users can start from the same presets:

```python
from nixtla_scaffold import forecast_spec_preset, run_forecast

spec = forecast_spec_preset("finance", horizon=6, freq="ME")
run = run_forecast("data.csv", spec)
run.to_directory("runs/finance_forecast")
```

## Five-minute onboarding paths

Use these when a new analyst or agent needs a concrete starting point:

| Path                                                  | What it teaches                                                                                             | Command                                                                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `examples\quickstart_csv`                             | Smallest CSV-to-forecast flow using `preset="quick"`                                                        | `uv run python examples\quickstart_csv\forecast_quick.py`                                                           |
| `examples\serious_finance_forecast`                   | Finance target normalization plus an auditable future event overlay                                         | `uv run python examples\serious_finance_forecast\forecast_finance.py`                                               |
| `examples\custom_finance_model`                       | Opt-in custom challenger: recent MoM growth to annual run-rate, allocated by historical monthly seasonality | `uv run python examples\custom_finance_model\forecast_custom.py`                                                    |
| `examples\hierarchy_reconciliation`                   | Leaf data -> hierarchy nodes -> coherent bottom-up planning forecast                                        | `uv run python examples\hierarchy_reconciliation\forecast_hierarchy.py`                                             |
| `examples\datasetsforecast_tourism_small`             | Optional public real-data validation using Nixtla DatasetsForecast TourismSmall quarterly hierarchy         | `uv run --extra datasets python examples\datasetsforecast_tourism_small\forecast_tourism_small.py --allow-download` |
| `examples\python_api_templates\dataframe_forecast.py` | Minimal DataFrame API template for agents and notebooks                                                     | import `run_example(...)` or adapt the function                                                                     |

For workbook users, put the same `unique_id`, `ds`, `y` columns in an Excel sheet and run:

```powershell
nixtla-scaffold forecast --input workbook.xlsx --sheet Data --preset finance --horizon 6 --output runs\workbook_demo
```

For MCP-backed DAX/Kusto/SQL users, first export the query result, then let `ingest` preserve source metadata and run the forecast:

```powershell
nixtla-scaffold ingest --input query_result.json --source kusto --query-file source.kql --id-value "Revenue" --time-col Month --target-col Revenue --output runs\revenue_input.csv --forecast-output runs\revenue_forecast --preset finance --horizon 6
```

Optional public real-data validation is available through Nixtla DatasetsForecast. It is not part of the default install or test path because it may download public data. Install/run it explicitly:

```powershell
uv run --extra datasets python examples\datasetsforecast_tourism_small\forecast_tourism_small.py --allow-download --output runs\tourism_small_real_data
```

The example uses TourismSmall bottom-level public data, rebuilds scaffold-compatible `purpose -> state -> city/noncity` hierarchy nodes, validates the generated total against the source total, then runs a quarterly bottom-up baseline forecast. Open `trust_summary.csv`, `hierarchy_reconciliation.csv`, `model_card.md`, and `report.html` first; this is a real-data package smoke, not a finance-specific forecast.

Model-family posture:

- `baseline`: always-on sanity methods such as Naive, HistoricAverage, RandomWalkWithDrift, WindowAverage, and SeasonalNaive.
- `statsforecast`: active production classical engine with AutoARIMA, AutoETS, AutoTheta, MSTL, MFLES/AutoMFLES when available, benchmarks, cross-validation, and intervals when supported. The runner retries candidate-by-candidate so one model failure does not collapse the full family.
- `mlforecast`: active ML engine when optional dependencies are installed; uses lag/date features across sklearn and LightGBM candidate families, with Nixtla `PredictionIntervals` conformal bands when history/horizon/lag requirements allow. Add known-future regressors only when they pass leakage and future-availability checks.
- `hierarchicalforecast`: optional reconciliation path for hierarchy nodes; the scaffold can emit diagnostic-only independent forecasts or coherent planning forecasts with BottomUp / MinTrace-style reconciliation when the optional dependency is installed.
- `neuralforecast_research`: research-only because dependency weight and explainability need extra scrutiny.

Model-policy semantics are explicit and audited in `manifest.json` under `model_policy_resolution`:

| Policy          | What it means                                                                                                                                         | Failure/skipping behavior                                                                                                                                                                                                         |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto`          | Run StatsForecast/classical models and add MLForecast when the panel has at least 30 observations per series and optional dependencies are available. | MLForecast import/runtime/no-candidate failures are warnings; the run continues with eligible classical/baseline models.                                                                                                          |
| `all`           | Run every eligible open-source family. "All" means all families that are valid for the data, not every model regardless of history.                   | If MLForecast is eligible but unavailable, fails, or produces no candidates, the run raises instead of silently downgrading. If history is below the 30-observation ML gate, the skip is disclosed and classical models continue. |
| `statsforecast` | Run the classical/open-source statistical ladder only.                                                                                                | StatsForecast failures are surfaced rather than hidden behind MLForecast.                                                                                                                                                         |
| `mlforecast`    | Run MLForecast only.                                                                                                                                  | Missing dependencies, runtime failures, or no candidates raise because this policy explicitly requested ML.                                                                                                                       |
| `baseline`      | Run simple benchmark models only.                                                                                                                     | Always available for usable history; useful for smoke tests and short-series fallbacks.                                                                                                                                           |

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
```

Hierarchy column order matters: `region product` creates Total -> region -> region/product. By default, the workflow forecasts each node independently and writes coherence diagnostics. Add `--hierarchy-reconciliation bottom_up`, `--hierarchy-reconciliation mint_ols`, or `--hierarchy-reconciliation mint_wls_struct` when parent/child forecasts need to be coherent for planning. Reconciliation is transparent: the model tournament still evaluates the independent statistical forecasts, while the reconciled outputs enforce planning coherence and write pre/post gap audits.

For Kusto, DAX, or other MCP-backed data, export the query result and let `ingest` create the forecast-ready input plus source metadata:

```powershell
nixtla-scaffold ingest --input kusto_export.json --source kusto --query-file premium_overage_arr.kql --id-value "Premium Overage ARR" --time-col day_dt --target-col ARR_30day_avg --output runs\premium_overage_arr_input.csv --forecast-output runs\premium_overage_arr_demo --freq ME --horizon 6
```

`ingest` accepts CSV, Excel, row-oriented JSON/JSONL, and MCP-style columnar JSON. It writes the canonical `unique_id,ds,y` CSV, a `.source.json` metadata file, and a copied `.kql`/`.dax` query artifact when provided.

Add auditable event/scenario overlays when finance knows something the history cannot know:

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --event '{"name":"Product launch","start":"2026-03-31","effect":"multiplicative","magnitude":0.10}' --output runs\launch_scenario

# Reusable JSON/YAML/CSV assumption files are also supported.
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --event-file scenario_events.csv --output runs\launch_scenario_file
```

The statistical `yhat` remains in the output; scenario columns such as `yhat_scenario`, `event_adjustment`, and `event_names` show the overlay.

Declare known-future regressors when MCPs or planning files provide drivers that may become future model inputs. This release audits leakage and future availability, then writes the contract artifacts; it does **not** automatically train arbitrary external regressors yet, so current MLForecast candidates still use lag/date features unless a later modeling slice explicitly wires exogenous variables.

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --regressor '{"name":"Seats plan","value_col":"seats_plan","availability":"plan","mode":"model_candidate","future_file":"future_seats.csv","source_system":"excel","owner":"finance"}' --output runs\driver_audit
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --regressor-file known_future_regressors.yaml --output runs\driver_audit_file
```

Regressor declarations produce `known_future_regressors.csv`, `driver_availability_audit.csv`, and `driver_experiment_summary.csv`. Model-candidate rows must pass future-value coverage, known-as-of timing, and target-leakage checks before any future exogenous-modeling experiment should use them.

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
Model selection uses backtested RMSE when available (MAE if RMSE is missing), tie-broken by MAE and absolute bias, so large misses are penalized more heavily than percentage-only scoring. MASE, RMSSE, WAPE, and bias are still reported in `series_summary.csv`, `model_audit.csv`, and `trust_summary.csv` for scale-free and business-readable review.
Selection outputs also record the CV horizon contract: `selection_horizon`, `requested_horizon`, `cv_windows`, `cv_step_size`, and `cv_horizon_matches_requested`. If adaptive CV uses a shorter horizon than the requested forecast horizon, the run emits a warning, caps trust, and labels future rows after the validated horizon as directional. Use `--strict-cv-horizon` when a high-stakes decision requires champion selection to be validated at the full requested horizon.

Row-level horizon fields use `row_horizon_status` as the clearest per-row status. `horizon_trust_state` and `forecast_horizon_status` remain backward-compatible aliases in `forecast.csv` / `forecast_long.csv`. `planning_eligible=True` only means the row passes the horizon-validation gate (`planning_eligibility_scope=horizon_validation_only`); it is not a global approval to ignore Low trust, interval issues, residual warnings, hierarchy tradeoffs, or data-quality caveats.

`horizon_trust_state` glossary:

| Value                        | Where you'll see it                                                        | Meaning                                                                                                                                                                                                                                            |
| ---------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `full_horizon_validated`     | `trust_summary.csv`, `forecast.csv`, `forecast_long.csv`, report/workbench | Rolling-origin model selection evaluated the champion through the requested horizon. Check `full_horizon_claim_allowed`: fewer than two CV windows means the full horizon was evaluated but not strong enough for a planning-ready champion claim. |
| `partial_horizon_validated`  | `trust_summary.csv`, `forecast.csv`, `forecast_long.csv`, report/workbench | Model selection had rolling-origin evidence, but only through a shorter horizon. Steps beyond `validated_through_horizon` are directional and not planning-ready by themselves.                                                                    |
| `beyond_validated_horizon`   | row-level `forecast.csv` / `forecast_long.csv`                             | This specific forecast row is after the validated CV horizon; `planning_eligible=False`.                                                                                                                                                           |
| `no_rolling_origin_evidence` | `trust_summary.csv`, `forecast.csv`, `forecast_long.csv`                   | No usable rolling-origin evidence exists for the selected champion; trust is capped Low and the forecast should be treated as exploratory.                                                                                                         |

`interval_status` glossary:

| Value                       | Where you'll see it                                                       | Meaning                                                                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `calibrated`                | `forecast_long.csv`, `trust_summary.csv`, from `interval_diagnostics.csv` | Future bands have matching rolling-origin interval coverage evidence for the model/horizon.                                                                                           |
| `calibration_warning`       | `forecast_long.csv`, `trust_summary.csv`, from `interval_diagnostics.csv` | Empirical coverage exists but the gap is large enough to review before planning use.                                                                                                  |
| `calibration_fail`          | `forecast_long.csv`, `trust_summary.csv`, from `interval_diagnostics.csv` | Empirical coverage indicates undercoverage risk.                                                                                                                                      |
| `future_only`               | `forecast_long.csv`, `trust_summary.csv`                                  | Future interval bands exist, but matching rolling-origin interval coverage evidence is unavailable for that horizon.                                                                  |
| `adjusted_not_recalibrated` | `forecast_long.csv`, `trust_summary.csv`                                  | Final selected bands were shifted/scaled/reconciled after model calibration, so they are planning aids rather than freshly calibrated uncertainty.                                    |
| `point_only_ensemble`       | `forecast_long.csv`, `trust_summary.csv`                                  | `WeightedEnsemble` is point-only; use component-model intervals or select a single calibrated model if planning ranges matter. Model disagreement lines are not prediction intervals. |
| `unavailable`               | `forecast_long.csv`, `trust_summary.csv`                                  | No interval bands are available for that model row.                                                                                                                                   |
| `insufficient_observations` | `trust_summary.csv`, from `interval_diagnostics.csv`                      | Too few interval backtest observations exist to validate coverage.                                                                                                                    |

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --weighted-ensemble --output runs\weighted
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --strict-cv-horizon --output runs\strict_h6
```

Verbose diagnostics are also on by default. Successful runs write `diagnostics.json`, `diagnostics.md`, and `llm_context.json`; failed CLI runs write `failure_diagnostics.json` and `failure_diagnostics.md` in the requested output folder when possible. `llm_context.json` is the best single file to attach when asking another LLM to walk through the forecast because it bundles the deterministic headline, trust summary, horizon gates, interval status, residual and seasonality diagnostics, hierarchy/driver context, guardrails, artifact index, and recommended questions.

The manifest includes a reproducibility block with a SHA-256 hash of canonical history, forecast origin, frequency, season length, Python/platform details, package versions, and git SHA when available.

Forecast runs now write two classes of artifacts:

1. **Core feeder outputs** for downstream models and analyst workflows:
   - `forecast.csv`: selected forecast rows only, enriched with `interval_status`, `interval_method`, `interval_evidence`, `row_horizon_status`, `horizon_trust_state`, `validated_through_horizon`, `planning_eligible`, `planning_eligibility_scope`, and CV horizon metadata so analyst-facing interval bounds and far-horizon rows carry their provenance.
   - `forecast_long.csv`: future predictions in long format, one row per series/model/date with `family`, horizon step, `yhat`, intervals, model weight, `interval_status`, `interval_method`, row-level horizon validation, `planning_eligibility_reason`, CV horizon metadata, and selected-model flag.
   - `backtest_long.csv`: rolling-origin validation predictions in long format with actuals, cutoff, horizon step, forecast error, squared error, interval bounds, and coverage flags when available.
   - `series_summary.csv`: one row per series with selected model, RMSE/MAE/MASE/RMSSE/WAPE, CV horizon contract, seasonality, and top weighted alternatives.
   - `model_audit.csv`: model leaderboard enriched with `family`, weights, and selected/challenger flags.
   - `model_win_rates.csv`: model win rates against SeasonalNaive or Naive for cross-series benchmark review.
   - `model_window_metrics.csv`: per-cutoff RMSE/MAE/MASE/RMSSE/WAPE/bias so rolling-origin windows can be reviewed one at a time.
   - `residual_diagnostics.csv`: horizon-step residual diagnostics for checking whether errors worsen further into the forecast horizon.
   - `residual_tests.csv`: heuristic residual bias, one-step autocorrelation, outlier, and early/late structural-break checks over rolling-origin residuals. These are diagnostic screening tools, not formal model adequacy tests; small samples are directional only.
   - `interval_diagnostics.csv`: empirical prediction-interval coverage, width, conformal method label, horizon metadata, and calibration status when interval backtests are available.
   - `trust_summary.csv`: first-stop decision artifact with per-series High/Medium/Low trust, score drivers, horizon trust, full-horizon claim gate, caveats, and recommended next actions. Rubric: High >=75, Medium 40-74, Low <40.
   - `model_explainability.csv`: MLForecast lag/calendar feature importance or coefficient magnitudes when ML models run.
   - `scenario_assumptions.csv`: event/scenario overlay assumptions when `--event` or `--event-file` is supplied.
   - `scenario_forecast.csv`: baseline `yhat` beside `yhat_scenario`, `event_adjustment`, and `event_names` for scenario review.
   - `known_future_regressors.csv`: declared known-future driver contracts from `--regressor` or `--regressor-file`.
   - `driver_availability_audit.csv`: leakage, future-coverage, known-as-of, audit-status, and modeling-decision checks for declared regressors.
   - `driver_experiment_summary.csv`: one summary table covering event overlays and known-future regressor audit outcomes.
   - `llm_context.json`: single LLM feeder packet with executive headline, run summary, per-series review, trust/horizon/interval/residual/seasonality/hierarchy/driver context, artifact index, guardrails, and recommended questions.
   - `forecast.xlsx`: a curated workbook with the consolidated outputs plus audit sheets.
   - `hierarchy_reconciliation.csv`: reconciliation method summary and pre/post max gap when hierarchy reconciliation is enabled. Reconciliation enforces parent = sum(children) coherence; node-level accuracy may decrease, so inspect `audit\hierarchy_backtest_comparison.csv`.
   - `hierarchy_contribution.csv`: parent/child contribution and gap attribution for hierarchy storytelling. Gap contributions are allocation heuristics, not reconciliation algorithm outputs.

2. **Audit/detail outputs** for transparency and debugging live under `audit\`: `all_models.csv`, `model_selection.csv`, `backtest_metrics.csv`, `backtest_predictions.csv`, `backtest_windows.csv`, `model_weights.csv`, `target_transform_audit.csv`, `seasonality_profile.csv`, `seasonality_summary.csv`, `seasonality_diagnostics.csv`, `seasonality_decomposition.csv`, `hierarchy_backtest_comparison.csv`, `hierarchy_unreconciled_forecast.csv`, `hierarchy_coherence_pre.csv`, `hierarchy_coherence_post.csv`, and `interpretation.json`. `model_weights.csv` includes `family` so agents can see whether the ensemble learned from baseline, StatsForecast, or MLForecast candidates. `target_transform_audit.csv` records raw, adjusted, transformed, and modeled target values whenever log/log1p or finance normalization is enabled. `seasonality_diagnostics.csv` records cycle counts, credibility labels, warnings, and trend/seasonal/remainder strength so the report does not overclaim annual or weekly seasonality without enough complete cycles. Driver contract files stay at the root because analysts and agents need to review assumptions before using scenario-adjusted rows or driver candidates. The hierarchy comparison/pre/post artifacts preserve the independent forecast, compare selected rolling-origin errors before versus after reconciliation, show parent/child gaps before reconciliation, and confirm remaining gaps after reconciliation. `model_selection.csv` and `backtest_metrics.csv` include CV horizon metadata so agents can see whether validation matched the business horizon. StatsForecast and MLForecast use the same adaptive rolling-origin horizon/window/step contract where feasible; MLForecast lags are capped and disclosed rather than reducing its CV windows. `best_practice_receipts.csv` stays at the root because it is a compact compliance summary and now includes a horizon-claim receipt.

Visual reports are scaffolded automatically with each forecast run:

- `report.html`: a plain, professional forecast review with a decision summary, model-policy resolution, target transformation audit when relevant, assumptions/driver audit section when events or regressors are declared, all-model and selected/top-weighted forecast charts, shaded future bands with interval-status caveats, seasonality credibility evidence, a fixed-axis rolling-origin backtest filmstrip with train/holdout shading, model leaderboard, and core output map.
- `report_base64.txt`: the same HTML report base64-encoded for MCPs, tickets, notebooks, or LLM handoffs that need text-only payloads.
- `streamlit_app.py`: an editable local dashboard/workbench with a top-level trust/action decision summary, model-policy resolution from `manifest.json`, series selector, champion lens controls for best overall vs best StatsForecast/classical vs best MLForecast, explicit skipped/failed family reasons when a lens has no candidate, active-champion horizon trust and interval-status banners, a winner-metric selector with guidance for RMSE/MAE/WAPE/MASE/RMSSE/bias/weight tradeoffs, first-glance forecast charts that include interval bands for every displayed interval-bearing candidate model, a dedicated Model investigation tab with manual model picking whose menus show `#rank | model | engine`, a model-picker guide with rank, engine, model type, and role so StatsForecast/classical, MLForecast, baseline, ensemble, and custom candidates are readable, focused forecast/CV comparison charts whose interval ribbons use the same color as the owning model line, are named in the legend, and use the same `forecast_long.csv` model feed for point forecasts plus interval bounds, a fixed-axis CV window player with previous/next/slider controls, a dedicated Prediction intervals tab with all interval-bearing candidate model bands selected by default plus the same rank/engine picker guide, interval-width summary, calibration evidence, and interval row review, benchmark win-rate chart, residual horizon/time/histogram/ACF diagnostics with white-noise heuristic and outlier dates, selected-interval availability warnings, a Seasonality tab with cycle-count credibility, configurable seasonal-year overlay, and additive decomposition evidence, MLForecast feature importance, hierarchy roll-up/down diagnostics plus pre/post reconciliation gap review when enabled, an Assumptions & Drivers tab for scenario overlays and known-future regressor audits, and consolidated output previews with `yhat` and interval bound columns kept adjacent. Run it from the output folder:

```powershell
cd runs\demo
uv run streamlit run streamlit_app.py
```

You can regenerate these report artifacts from an existing run:

```powershell
nixtla-scaffold report --run runs\demo
```

For product-quality validation, run the synthetic scenario lab. It creates 100 train/holdout scenarios with actuals, scores the scaffold on accuracy, validity, ease of use, and explainability, and writes recommendations:

```powershell
nixtla-scaffold scenario-lab --count 100 --model-policy auto --output runs\scenario_lab_100
```

The lab writes `scenario_scores.csv`, `scenario_summary.json`, and `scenario_recommendations.json`. The default `auto` policy exercises the real StatsForecast-first path when available.

For workbench-quality validation, run the golden QA harness. By default it generates the two flagship refresh demos (`monthly_basic`, `hierarchy_reconciled`) plus two finance guardrails (`limited_history_new_product`, `normalized_target_forecast`), checks required artifacts, compiles each generated `streamlit_app.py`, and runs Streamlit AppTest unless disabled. Legacy aliases `short_history` and `transform_normalized` are still accepted for existing scripts, but new docs and outputs use the clearer names:

```powershell
nixtla-scaffold workbench-qa --output runs\workbench_qa
nixtla-scaffold workbench-qa --scenarios monthly_basic hierarchy_reconciled --model-policy baseline --no-app-test --output runs\workbench_qa_refresh
nixtla-scaffold workbench-qa --scenarios limited_history_new_product normalized_target_forecast --model-policy baseline --no-app-test --output runs\workbench_qa_guardrails
nixtla-scaffold workbench-qa --output runs\workbench_qa --app-test-timeout 120
```

The harness writes `workbench_qa_summary.csv` and `workbench_qa_summary.json` with pass/fail status, usability score, selected models, Streamlit compile status, AppTest status, AppTest timeout seconds, missing artifacts, and missing workbench text. Use it before declaring dashboard/report changes done.

For local release readiness, run the release gates. This builds the package, installs the wheel into an isolated environment, smoke-tests public APIs/CLI, runs deterministic numeric goldens, validates generated workbenches, checks artifact hygiene, reports optional-extra status, and launches one generated Streamlit dashboard for an HTTP health smoke:

```powershell
nixtla-scaffold release-gates --output runs\release_gates
nixtla-scaffold release-gates --output runs\release_gates_fast --no-build --no-install-smoke --no-workbench-qa --no-live-streamlit --scenario-count 2
nixtla-scaffold release-gates --output runs\release_gates --json
nixtla-scaffold release-gates --extended --output runs\release_gates_extended
```

The CLI prints a compact one-glance verdict by default and writes `release_gate_summary.md`, `release_gate_summary.json`, and `release_gate_results.csv`. Use `--json` when an agent needs the full nested payload in stdout. A failed required gate makes the CLI return nonzero. Optional extras (`ml`, `hierarchy`, `neural`, `datasets`) are reported by default and can be required with `--require-optional ml hierarchy` or `--require-optional datasets`. Use `--extended` for a stricter local release profile: at least 20 auto-policy scenarios, all-family workbench QA, and required `ml` plus `hierarchy` extras.

How to read the gate:

| Surface                     | What to check                                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compact CLI output          | Top-line `PASSED`/`FAILED`, version, Python, duration, failed gate names, remediation hints, skip reasons, provenance warnings, and artifact paths.                                       |
| `release_gate_summary.md`   | Human-readable green-light artifact with gate table, provenance, required optional groups, scenario archetype scores, thresholds, exit codes, and failure actions.                        |
| `release_gate_summary.json` | Full machine-readable payload with `summary.headline`, `failed_gates`, `failure_rollup`, provenance, git-SHA unavailable reason, options, per-gate durations, and nested command details. |
| `release_gate_results.csv`  | One-row-per-gate table with flattened status, reason, remediation, artifact, and details JSON for quick audit/filtering.                                                                  |

Thresholds are intentionally broad smoke gates rather than brittle exact-value snapshots: scenario lab requires zero crashes, at least `count - 2` scenarios passing, composite score >= 70, validity >= 85, and explainability >= 80; workbench QA requires no failed golden scenarios and minimum usability score >= 90; quick forecast requires finite bounded selected forecasts, valid interval containment when interval columns exist, a numeric trust score, and an executive headline. Exit codes are `0` for passed gates, `1` for gate failure, and `2` for CLI/runtime errors. The live Streamlit smoke proves the generated dashboard launches in the current Python environment.

Local outputs are intentionally ignored by git. `.gitignore` excludes `.intern`, `.venv`, `runs`, temporary validation folders, query artifacts, and local reports because user tests often contain sensitive data.

## What the scaffold does

1. Profiles the data: grain, date range, inferred frequency, missing dates, duplicates, short histories, zeros, negatives, and freshness.
2. Repairs regular time series gaps using an explicit fill policy.
3. Runs simple baselines and, when available, a resilient StatsForecast ladder.
4. Backtests when enough history exists.
5. Selects a champion model per series.
6. Adds an inverse-error weighted ensemble when validation data supports it.
7. Supports auditable finance target transforms: log/log1p plus positive factor normalization for pricing, FX, inflation, or definition-change analysis.
8. Supports auditable event/scenario overlays and known-future regressor contracts with leakage/future-availability checks, while keeping arbitrary external regressor training disabled until explicitly implemented.
9. Optionally reconciles hierarchy forecasts for coherent parent/child planning outputs while preserving unreconciled model-tournament evidence.
10. Writes consolidated feeder files, audit artifacts, JSON manifest, model card, diagnostics, HTML/base64/Streamlit reports, and Excel workbook.
11. Runs a scenario lab to score the scaffold like a forecaster against known holdout actuals.

StatsForecast execution is failure-isolated: the scaffold tries the full ladder first, then retries individual candidates if a model/interval/backtest path fails. Successful classical models stay in the run, failed candidates are disclosed as warnings, and interval-only failures keep the point forecast when possible.

## Simplicity rules

- One obvious path first: load, profile, forecast, explain, export.
- Advanced options live in config/CLI flags, not required setup.
- No local MCP server in the MVP; MCPs can provide data as CSV, Excel, or DataFrames.
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
