# Agent overview and instructions

This file is the detailed handoff for AI agents, maintainers, and power users. Keep the README focused on the product story; put workflow mechanics, operating rules, and release details here.

## Agent priority

When asked to forecast a metric, keep the run explainable and conservative:

1. Confirm the target, grain, time column, forecast horizon, and whether the output is exploratory or planning-facing.
2. Normalize the input to `unique_id`, `ds`, and `y`.
3. Start with `profile`, then run `forecast` with an explicit preset and output folder.
4. Open or cite the curated artifacts first: `OPEN_ME_FIRST.html`, `output\forecast_review.xlsx`, `appendix\trust_summary.csv`, `forecast.csv`, `report.html`, and `llm_context.json`.
5. Treat `planning_eligible=True` as a horizon-validation flag only. It does not override Low trust, interval failures, residual warnings, hierarchy gaps, data-quality caveats, or business review.
6. Quote the deterministic executive headline from `diagnostics.json.executive_headline.paragraph` or `llm_context.json`; do not paraphrase it into a stronger claim.

The bundled skill at `skills\nixtla-forecast\SKILL.md` is the most complete agent playbook. From an installed wheel, run:

```powershell
nixtla-scaffold guide skill
```

## Install and local development

Install from PyPI:

```powershell
uv tool install nixtla-scaffold
uv tool install "nixtla-scaffold[ml,hierarchy]"
```

Use the Python API inside a project:

```powershell
uv add nixtla-scaffold
uv add "nixtla-scaffold[ml,hierarchy]"
```

For local repository development:

```powershell
uv sync --extra ml --extra hierarchy
uv run nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --preset quick --output runs\dev_smoke
```

## Core command flow

```powershell
nixtla-scaffold profile --input data.csv
nixtla-scaffold forecast --input data.csv --preset standard --horizon 6 --output runs\latest
nixtla-scaffold report --run runs\latest
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--id-col`, `--time-col`, `--target-col` | Map business-friendly source columns. |
| `--unit-label` | Add units or currency to executive summaries. |
| `--levels 80 95` | Request prediction interval levels when the model/history support them. |
| `--model-policy baseline|light|standard|all|statsforecast|mlforecast` | Control model family breadth. |
| `--strict-cv-horizon` | Require champion validation at the requested horizon. |
| `--hierarchy-reconciliation bottom_up|top_down|both|mint_ols|mint_wls_struct` | Produce coherent hierarchy planning outputs. |

## Intake questions

Use `setup` when the request is broad, the source is query-backed, or the agent needs a reusable workspace:

```powershell
nixtla-scaffold setup --workspace runs\usage_overage_setup --data-source kusto --preset standard --series-count single --target-name ARR_30day_avg --time-col day_dt --id-value "Usage Overage ARR" --freq ME --horizon 6 --intervals auto --model-families statsforecast mlforecast --exploration-mode --mcp-regressor-search --outputs all
```

The setup workspace includes `forecast_setup.yaml`, `questions.json`, `agent_brief.md`, source-query templates, and folders for raw data, canonical data, outputs, reports, and notes.

Ask these before modeling:

1. Which preset should we start from: `quick`, `standard`, `strict`, or `hierarchy`?
2. Where is the data coming from: CSV, Excel, Kusto, DAX, SQL, DataFrame, or unknown?
3. How many things are being forecast: single series, a few series, many series, or a hierarchy?
4. Are intervals required, optional, or automatic?
5. Which model families are allowed: baseline, StatsForecast, MLForecast, HierarchicalForecast, or research-only NeuralForecast?
6. Should exploration run before forecasting?
7. Can the agent use MCPs to find candidate drivers or regressors?
8. Which outputs are needed: clean review pack, CSV, Excel, HTML, Streamlit, diagnostics, model card, or all?

## Data contract

The required long-table contract is:

| Column | Meaning |
| --- | --- |
| `unique_id` | Series, product, SKU, account, or other forecasting grain. |
| `ds` | Timestamp/date at the forecast grain. |
| `y` | Numeric value to forecast. |

Single-series files may omit `unique_id`; the loader creates `series_1`.

Do not concatenate external forecast `yhat` rows into the training data. External forecasts are comparison artifacts, not observed history.

## Presets and model policies

Presets:

| Preset | Typical use |
| --- | --- |
| `quick` | Baseline ladder, short smoke run, first read. |
| `standard` | Serious default: trust artifacts, intervals, weighted ensemble, diagnostics. |
| `strict` | Higher-stakes decision support with stronger validation requirements. |
| `hierarchy` | Standard defaults plus coherent parent/child planning outputs. |

Model policies:

| Policy | Meaning |
| --- | --- |
| `baseline` | Always-available sanity methods. |
| `light` | StatsForecast/classical plus feasible MLForecast candidates; slim default path. |
| `standard` | Serious default: StatsForecast/classical, feasible MLForecast, optional smooth ADAM when installed. |
| `statsforecast` | Classical/open-source statistical ladder only. |
| `mlforecast` | MLForecast only; missing dependencies or no candidates raise clearly. |
| `all` | Every eligible open-source family available in the environment. |

`auto` remains a legacy alias for `light`; `finance` remains a legacy alias for the `standard` preset.

Champion selection uses rolling-origin RMSE when available, tie-broken by MAE and absolute bias. MAE, MASE, RMSSE, WAPE, and bias are included for review but do not silently override the champion.

## Trust and horizon rules

Trust rubric:

| Score | Label |
| --- | --- |
| `>=75` | High |
| `40-74` | Medium |
| `<40` | Low |

High trust means "statistical baseline with evidence," not a guarantee or plan approval.

Important horizon fields:

| Field/value | Meaning |
| --- | --- |
| `row_horizon_status` | Clearest per-row forecast-horizon status. |
| `full_horizon_validated` | Rolling-origin evidence covers the requested horizon. |
| `partial_horizon_validated` | Evidence exists but only through a shorter horizon. |
| `beyond_validated_horizon` | Row is beyond the validated horizon; directional only. |
| `no_rolling_origin_evidence` | No usable rolling-origin evidence; trust is capped Low. |
| `planning_eligible` | Row passed the horizon gate only; it is not global approval. |

Interval statuses:

| Status | Meaning |
| --- | --- |
| `calibrated` | Matching rolling-origin interval coverage exists. |
| `calibration_warning` | Evidence exists but coverage gap needs review. |
| `calibration_fail` | Undercoverage risk. |
| `future_only` | Future interval exists without matching rolling-origin evidence. |
| `adjusted_not_recalibrated` | Bands were shifted/scaled/reconciled after calibration. |
| `point_only_ensemble` | Weighted ensemble has no calibrated interval. |
| `unavailable` | No interval bands are available. |
| `insufficient_observations` | Too few observations to validate coverage. |

## Outputs

Curated outputs:

| File | Purpose |
| --- | --- |
| `OPEN_ME_FIRST.html` | Landing page and file map. |
| `output\forecast_review.xlsx` | Compact review workbook. |
| `output\forecast_for_review.csv` | Selected forecast rows for finance review. |
| `output\decision_summary.csv` | Trust/readiness, caveats, and next actions by series. |

Core feeder outputs:

| File | Purpose |
| --- | --- |
| `forecast.csv` | Selected forecast rows with horizon, interval, CV, and planning-gate metadata. |
| `forecast.xlsx` | Broader workbook with audit sheets. |
| `report.html` | Static human review report. |
| `streamlit_app.py` | Interactive local dashboard. |
| `llm_context.json` | Best single attachment for LLM review. |
| `manifest.json` | Reproducibility metadata and run configuration. |

Appendix and audit outputs:

| Folder/file | Purpose |
| --- | --- |
| `appendix\forecast_long.csv` | Future predictions by series/model/date. |
| `appendix\backtest_long.csv` | Rolling-origin validation rows. |
| `appendix\series_summary.csv` | Per-series model and metric summary. |
| `appendix\trust_summary.csv` | First-stop readiness and caveats. |
| `audit\model_weights.csv` | WeightedEnsemble evidence when available. |
| `audit\target_transform_audit.csv` | Transform and normalization provenance. |

## External and BYO finance forecasts

Use external forecasts for comparison and triangulation only:

```powershell
nixtla-scaffold compare --run runs\latest --external finance_plan.xlsx --sheet Forecast --format wide --model-name "FP&A plan" --output runs\latest\comparison
```

`compare` reports directional deltas against scaffold forecasts. These deltas are not residuals, prediction intervals, or model-selection evidence.

Score historical forecast snapshots separately after actuals land:

```powershell
nixtla-scaffold score-external --external finance_snapshots.csv --actuals actuals.csv --output runs\finance_scores --season-length 12 --horizon 6
```

Use `byo-model` when the source is a finance-owned Excel/Python model with Base/Bull/Bear, rollups, customer/SKU detail, purchase logic, renewals, or PxQ calculations:

```powershell
uv run nixtla-scaffold byo-model ingest --file runs\byo_excel_example\finance_model.xlsx --sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --output runs\byo_excel_example\ingest
uv run nixtla-scaffold byo-model compare --run runs\byo_excel_example\scaffold_run --file runs\byo_excel_example\finance_model.xlsx --sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --main-model-preference Base
uv run nixtla-scaffold byo-model score --file runs\byo_excel_example\finance_snapshots.xlsx --actuals runs\byo_excel_example\actuals.csv --sheet Base Bull Bear --group-cols ProductGroup ProductLine Product --output runs\byo_excel_example\scores
```

## Hierarchy workflow

Aggregate leaf data into hierarchy nodes, then forecast the generated file:

```powershell
nixtla-scaffold hierarchy --input examples\hierarchy_generic\input.csv --hierarchy-cols region product --output runs\hierarchy_nodes.csv
nixtla-scaffold forecast --input runs\hierarchy_nodes.csv --horizon 3 --freq ME --hierarchy-reconciliation bottom_up --output runs\hierarchy_reconciled
```

Hierarchy column order matters. `region product` creates `Total -> region -> region/product`. Reconciliation preserves model-tournament evidence while enforcing coherent planning outputs.

## Source pipelines, events, drivers, and experiments

For Kusto, DAX, SQL, or other query-backed sources, use `ingest` or a pipeline to preserve source metadata:

```powershell
nixtla-scaffold ingest --input query_result.json --source kusto --query-file source.kql --id-value "Revenue" --time-col Month --target-col Revenue --output runs\revenue_input.csv --forecast-output runs\revenue_forecast --preset standard --horizon 6
uv run nixtla-scaffold pipeline run --config examples\contoso_kql_pipeline\pipeline.yaml --output runs\contoso_kql_pipeline
```

Add business-known future events explicitly:

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --event '{"name":"Product launch","start":"2026-03-31","effect":"multiplicative","magnitude":0.10}' --output runs\launch_scenario
```

Declare regressors before training on them. By default, the scaffold audits leakage and future availability without training external drivers:

```powershell
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --regressor-file known_future_regressors.yaml --output runs\driver_audit
nixtla-scaffold forecast --input examples\monthly_finance_csv\input.csv --horizon 6 --model-policy mlforecast --train-known-future-regressors --regressor-file known_future_regressors.yaml --output runs\driver_model
```

Run bounded experiments when an agent needs iteration without turning the project into an open-ended AutoML search:

```powershell
nixtla-scaffold compare-models --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --output runs\compare_models_demo
nixtla-scaffold experiment --input examples\monthly_finance_csv\input.csv --preset standard --horizon 6 --variants baseline all_models rolling_features --max-variants 3 --output runs\experiment_rolling
```

## Refresh, ledger, and local operations

Refresh a prior run with new actuals while reusing the persisted setup:

```powershell
uv run nixtla-scaffold refresh --previous-run runs\actions_arr_march --input data_april.csv --output runs\actions_arr_april
```

Use the ledger when a forecast becomes operational:

```powershell
uv run nixtla-scaffold forecast --input data.csv --preset standard --horizon 6 --output runs\product_arr_march --ledger runs\forecast_ledger --forecast-key "Product ARR" --version-label "March refresh" --lock-official --lock-label "March lock"
uv run nixtla-scaffold ledger actuals --ledger runs\forecast_ledger --input actuals.csv --forecast-key "Product ARR" --source-kind kusto --revision-label "April actuals"
uv run nixtla-scaffold ledger compare --ledger runs\forecast_ledger --forecast-key "Product ARR" --against-lock "March lock" --call-up-pct 0.10 --call-down-pct 0.10
uv run nixtla-scaffold ledger export --ledger runs\forecast_ledger --output runs\forecast_ledger\exports
```

The generated Streamlit app includes a Ledger view when ledger artifacts are present, so reviewers can inspect version trends, official locks, landed actuals, drift, and call-up/call-down comparisons without digging through raw CSVs first.

Local operations:

```powershell
uv run nixtla-scaffold status --runs runs --output runs\ops_status
uv run nixtla-scaffold doctor --run runs\actions_arr_april --output runs\actions_arr_april\ops
uv run nixtla-scaffold drift --ledger runs\forecast_ledger --refreshed-run runs\actions_arr_april --output runs\forecast_ledger\drift
uv run nixtla-scaffold operate --config forecast_operating_loop.yaml --output runs\monthly_operate
```

## Validation and release operations

Scenario lab:

```powershell
nixtla-scaffold scenario-lab --count 100 --model-policy light --output runs\scenario_lab_100
```

Workbench QA:

```powershell
nixtla-scaffold workbench-qa --output runs\workbench_qa
nixtla-scaffold workbench-qa --scenarios monthly_basic hierarchy_reconciled --model-policy baseline --no-app-test --output runs\workbench_qa_refresh
nixtla-scaffold workbench-qa --scenarios limited_history_new_product normalized_target_forecast --model-policy baseline --no-app-test --output runs\workbench_qa_guardrails
```

Release gates:

```powershell
nixtla-scaffold release-gates --output runs\release_gates
nixtla-scaffold release-gates --output runs\release_gates_fast --no-build --no-install-smoke --no-workbench-qa --no-live-streamlit --scenario-count 2
nixtla-scaffold release-gates --output runs\release_gates --json
nixtla-scaffold release-gates --extended --output runs\release_gates_extended
```

Release workflow behavior:

- Pull requests run validation only.
- Pushes to `main` run validation, read `[project].version` from `pyproject.toml`, build the package, optionally publish to PyPI when `PYPI_API_TOKEN` exists, and create a GitHub release tagged `v<version>`.
- Every merge to `main` that should publish a release must bump `pyproject.toml` first.
- If the tag already exists, the release job fails with a version-bump message instead of overwriting an existing release.

Local outputs are intentionally ignored by git. `.gitignore` excludes local environments, `.intern`, run folders, temporary validation folders, query artifacts, local reports, workbooks, databases, and common data-export formats because user tests often contain sensitive or customer-shaped data.

## Embedded guidance and citation

`nixtla-scaffold guide` searches the built-in knowledge base of Nixtla docs, Nixtla GitHub source files, and FPPy best practices.

FPPy citation standard used by the scaffold:

> Hyndman, R.J., Athanasopoulos, G., Garza, A., Challu, C., Mergenthaler, M., & Olivares, K.G. (2025). Forecasting: Principles and Practice, the Pythonic Way. OTexts: Melbourne, Australia. Available at: OTexts.com/fpppy. Accessed on 28 April 2026.

The knowledge base includes source maps for:

- `Nixtla/statsforecast`
- `Nixtla/utilsforecast`
- `Nixtla/hierarchicalforecast`
- `Nixtla/mlforecast`

It also links to high-value docs and notebooks for StatsForecast, intermittent demand, UtilsForecast, MLForecast cross-validation, and HierarchicalForecast tourism reconciliation.
