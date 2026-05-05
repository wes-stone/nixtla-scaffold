---
name: nixtla-forecast
description: "End-to-end FPPy-aligned time series forecasting using the nixtla-scaffold package. Covers data ingestion from any MCP/query source, intake, profiling, model selection, rolling-origin backtesting, prediction intervals, event/driver overlays, MCP feature/regressor discovery, hierarchy, interpretation, and report generation. WHEN: forecast, time series, predict, project, trend, seasonal, ARIMA, ETS, backtest, prediction interval, scenario, nixtla, statsforecast, hierarchy forecast, intermittent demand, revenue forecast, ARR forecast, finance forecast, forecast this data, forecast from Kusto, forecast from Excel, forecast from DAX, find forecast drivers, forecast regressors."
---

# nixtla-forecast — AI-Driven Time Series Forecasting Skill

> One-stop-shop for finance-grade forecasting using the open-source Nixtla stack.
> Built for AI agents that need to go from raw data to explained, auditable forecasts.

## ⚠️ CRITICAL: Agent Behavior Rules

1. **NEVER skip the intake questions.** Even if the user says "just forecast this",
   walk them through Step 0 and Step 1. A forecast without context is dangerous.
2. **ALWAYS disclose limitations** before AND after the forecast.
3. **ALWAYS present an action plan** after the forecast — next steps for validation,
   enrichment, and operationalization.
4. **NEVER present a forecast as "the answer"** — it's a statistical baseline for
   human judgment. Say this explicitly.
5. **If the user pushes back on questions**, briefly explain WHY they matter:
   "These questions prevent the #1 cause of bad forecasts: wrong assumptions."
6. **ALWAYS ask about additional data that could become features/regressors.**
   The frontier workflow is not just "run a univariate forecast"; it is "build a
   statistical baseline, then search MCP-connected data sources for known-future
   drivers, events, hierarchy, plan/prior-year benchmarks, and normalization factors."
7. **ALWAYS separate baseline, scenario, and plan.** FPPy is explicit that a
   forecast is not a goal. Keep statistical yhat, event-adjusted scenario yhat,
   and business target/plan as distinct artifacts.

## FPPy Forecaster Standard

Every workflow should follow the Forecasting: Principles and Practice, the Pythonic Way
sequence; use the full FPPy citation in the Sources section whenever citing it.

1. **Frame**: What decision will this forecast support, and what future conditions
   could break "future resembles past"?
2. **Prepare**: canonical `unique_id`, `ds`, `y`; confirm grain, units, adjustments,
   missing timestamps, duplicates, outliers, zeros, and negatives.
3. **Visualize first**: inspect history, trend, seasonality, structural breaks,
   outliers, hierarchy rollups, and candidate driver relationships before trusting
   any model.
4. **Specify candidates**: simple baselines, ETS/ARIMA/Theta/MSTL, intermittent
   models, MLForecast when enough history/features exist, hierarchy when rollups
   matter.
5. **Estimate and evaluate**: chronological rolling-origin CV only; never shuffled
   K-fold. Evaluation horizon should match the business horizon when possible.
6. **Benchmark**: always compare to Naive, SeasonalNaive, HistoricAverage, and Drift.
7. **Diagnose**: residuals, bias, per-horizon errors, interval calibration, naive
   skill, and hierarchy coherence.
8. **Forecast with uncertainty**: point forecast plus 80%/95% intervals when history
   supports them; suppress or warn when intervals cannot be validated.
9. **Add judgment explicitly**: future events, pricing, launches, contracts, holidays,
   and driver scenarios must be auditable assumptions, not silent edits.
10. **Operationalize**: store query/config/manifest, refresh cadence, post-actual
    accuracy tracking, and next actions.

## Package Location

```
<your-local-clone>\nixtla-scaffold
```

From a local clone of this repository, `Set-Location` to the repo root before running commands.

---

## Quick Reference — The 8 Commands You Need

| Command | What it does | When to use |
|---------|-------------|-------------|
| `setup` | Creates a workspace with intake questions, config, and agent brief | First touch — user says "forecast this" |
| `profile` | Analyzes data quality: frequency, gaps, zeros, negatives, readiness | Before forecasting — check the data |
| `forecast` | Runs the full pipeline: profile → repair → model → backtest → select → output | The main event |
| `explain` | Generates model card from an existing run | After forecast — understand the results |
| `report` | Regenerates HTML/Streamlit report artifacts | When you need to share results |
| `ingest` | Converts MCP/query exports to canonical forecast input | When data comes from Kusto/DAX/SQL |
| `workbench-qa` | Runs golden forecast/workbench scenarios and validates generated artifacts + Streamlit apps | Before declaring dashboard/report changes production-ready |
| `release-gates` | Runs local package release checks: build/install smoke, numeric goldens, workbench QA, optional extras, artifact hygiene, and live Streamlit smoke | Before treating the package/workbench as release-ready |

Preset shortcut:

| Preset | Use when |
|--------|----------|
| `quick` | Fast first read / smoke run with baseline ladder |
| `finance` | Standard serious finance forecast with full audit/trust artifacts |
| `strict` | High-stakes run requiring backtests and full-horizon CV champion selection |
| `hierarchy` | Parent/child planning totals need coherent reconciliation |

Flags override presets:

```powershell
uv run nixtla-scaffold forecast --input data.csv --preset finance --horizon 6 --unit-label "$" --output runs\finance
uv run nixtla-scaffold forecast --input data.csv --preset hierarchy --hierarchy-reconciliation mint_ols --output runs\hierarchy_mint
uv run nixtla-scaffold guide presets
```

---

## WORKFLOW: How to Forecast Anything

### ⚠️ MANDATORY: Do NOT Skip the Intake

**Before touching ANY data or running ANY command**, you MUST walk the user through
this intake. Forecasting without understanding the context is malpractice. Use `ask_user`
to collect answers in a structured form.

### Step 0: Educate — What Forecasting Can and Cannot Do

Tell the user upfront:

> **What a statistical forecast IS**: A projection of historical patterns (trend,
> seasonality, noise) into the future. It assumes the future resembles the past
> unless you explicitly tell it otherwise.
>
> **What it is NOT**: It cannot predict unknown events, market shifts, competitive
> moves, or anything that hasn't happened before in the data. It's a starting
> point for judgment, not a replacement for it.
>
> **Key limitations to be aware of**:
> - Forecasts degrade with horizon — month 1 is much more reliable than month 12
> - Short history (<2 years for monthly) limits what models can learn
> - A forecast that "looks right" can still be wrong in important ways
> - Prediction intervals widen over time — uncertainty is real, not optional
> - All-else-equal assumption: the forecast does not know about your upcoming
>   product launch, pricing change, or market shift unless you add it as an event

### Step 1: Intake Questions (ALL of these, every time)

Use `ask_user` to collect structured answers. Group into one form when possible.

**1. Data source & shape**
- Where is the data? (CSV, Excel, Kusto, DAX, SQL, API, DataFrame)
- What are the column names for date, value, and series ID?
- What's the grain? (daily, weekly, monthly, quarterly)
- How much history? (rough: months or years)
- How many series? (1, handful, dozens, hundreds)
- Are there hierarchy columns available? (region, product, SKU, account, segment, business)
- Are there plan, budget, prior-year, or actuals-vs-plan tables available for benchmark storytelling?

**2. What we're forecasting**
- What metric is this? (revenue, costs, volume, headcount, usage...)
- Is this a rate, cumulative, or point-in-time metric?
- Are there known structural breaks? (acquisitions, product launches, pricing changes, COVID)
- Is the data adjusted for anything? (inflation, FX, pricing changes, one-time items)
- Are there seasonal patterns you expect? (Q4 budget flush, summer slowdown, monthly billing cycles)
- Is the metric bounded or constrained? (non-negative, capacity cap, saturation, contractual floor)

**3. Forecast requirements**
- How far ahead? State the **business need**, not just a number
- ⚠️ **Caveat**: Longer horizons = wider uncertainty. A 12-month forecast of a volatile
  series is a guess with structure, not a prediction. Be honest about this.
- Do you need prediction intervals? (Recommended: yes, at 80% and 95%)
- ⚠️ **Caveat**: Intervals assume the future has similar variability to the past.
  If a structural change is coming, intervals will be too narrow.
- Point forecast or range forecast for planning?

**4. Known future events & drivers**
- Are there ANY known future events that will affect this metric?
  (Product launches, pricing changes, market expansions, headcount changes, contracts)
- ⚠️ **Caveat**: Without events, the forecast is purely statistical — it assumes
  nothing changes. If you know something will change, you MUST add it as a
  DriverEvent overlay or the forecast will be wrong.
- Are there external drivers that correlate with this metric?
  (GDP, user growth, marketing spend, seasonality of a related metric)
- What MCP-accessible data sources might hold useful features?
  (Kusto telemetry, DAX/Power BI measures, Excel plans, SQL tables, CRM/contracts,
  headcount plans, marketing spend, product launch calendars, pricing tables, usage)
- Which candidate drivers have **known future values** or credible future scenarios?
- Which drivers may be leading indicators? At what lag? (e.g., usage leads revenue by 1 month)
- ⚠️ **If yes**: Search for regressors via MCPs, but validate future availability,
  leakage risk, data latency, and whether the relationship is causal/business-plausible.

**5. Quality & edge cases**
- Is the data complete? Any known gaps, system migrations, outages?
- Are there periods that should be excluded? (COVID quarters, data errors, one-time settlements)
- Are there zeros? (Intermittent demand patterns need special handling)
- Negative values? (Costs, net metrics)
- Very recent structural changes? (If the last 3 months look different from the prior 2 years,
  the model may overweight the old pattern)

**6. How will this be used?**
- Who is the audience? (CFO, board, planning team, operational)
- What decisions depend on this? (Budget setting, hiring, capacity planning)
- How often will this be refreshed? (One-time, monthly, weekly)
- What outputs do you need? (CSV, Excel, HTML report, Streamlit dashboard, all)
- What threshold changes the decision? (e.g., if forecast is ±5% of plan, no action; if >10% miss, escalate)
- Do parent and child series need to sum exactly for planning?

### Step 1.5: Summarize & Confirm Before Proceeding

After collecting answers, present a brief summary:

```
📋 Forecast Setup Summary
━━━━━━━━━━━━━━━━━━━━━━━━
Metric:     Premium Overage ARR (daily, annualized)
History:    309 daily observations (Jun 2025 – Apr 2026)
Grain:      Daily
Horizon:    90 days
Intervals:  80% and 95%
Events:     None specified
Drivers:    None specified
Feature data to search: Kusto usage, DAX bookings, Excel plan, pricing calendar
Hierarchy:  Product x Region available, reconciliation needed? yes/no

⚠️  Limitations for this forecast:
• 10 months of daily data — trend is captured but annual seasonality cannot be validated
• No events specified — forecast assumes current trajectory continues as-is
• 90-day horizon on daily data = last 30 days will have wide uncertainty bands
• Strong growth trend means the model may extrapolate aggressively
• No known-future regressors validated yet — MLForecast will use lags/calendar only

🔧 Recommended next steps AFTER the forecast:
1. Check if the trend extrapolation looks reasonable to domain experts
2. Consider adding known events (pricing changes, product launches)
3. Search for correlated regressors via Kusto/DAX MCPs
4. Compare against plan/budget/prior year for sanity check
5. Set up monthly refresh cadence if this becomes operational

Proceed? [yes/adjust]
```

Only proceed after the user confirms or adjusts.

### Step 2: Get the Data into Canonical Format

The scaffold expects: `unique_id`, `ds`, `y` columns.

**From CSV/Excel (direct):**
```powershell
uv run nixtla-scaffold profile --input data.csv
uv run nixtla-scaffold forecast --input data.csv --preset finance --horizon 6 --output runs\my_forecast
```

**From CSV/Excel with custom column names:**
```powershell
uv run nixtla-scaffold forecast --input plan.xlsx --sheet Data --id-col Product --time-col Month --target-col Revenue --preset finance --horizon 6 --output runs\plan
```

**Runnable onboarding examples:**
```powershell
uv run python examples\quickstart_csv\forecast_quick.py
uv run python examples\serious_finance_forecast\forecast_finance.py
uv run python examples\custom_finance_model\forecast_custom.py
uv run python examples\hierarchy_reconciliation\forecast_hierarchy.py
uv run --extra datasets python examples\datasetsforecast_tourism_small\forecast_tourism_small.py --allow-download
```

Use `examples\quickstart_csv` for the five-minute first forecast, `examples\serious_finance_forecast` for target normalization plus event overlays, `examples\custom_finance_model` when a finance-owned MoM/FY-seasonality model should enter the tournament as an audited custom challenger, `examples\hierarchy_reconciliation` when parent/child totals need to tie, and `examples\datasetsforecast_tourism_small` only for opt-in public real-data validation with DatasetsForecast.

**With finance target transforms / normalization:**
```powershell
# Model on log1p(y), then inverse-transform forecast/backtest outputs for reporting.
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --target-transform log1p --output runs\log1p_demo

# Model normalized units after a positive price/FX/inflation factor.
# Forecast outputs stay in normalized units unless future factors are supplied externally.
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --normalization-factor-col price_factor --normalization-label "FY26 pricing" --output runs\normalized_demo
```

**From Kusto/DAX/SQL query results:**
```powershell
# First, run your query via the appropriate MCP and save results to a file
# Then ingest it:
uv run nixtla-scaffold ingest --input query_result.json --source kusto --query-file my_query.kql --id-value "My Metric" --time-col day_dt --target-col revenue --output runs\input.csv --forecast-output runs\forecast_demo --preset finance --freq ME --horizon 6
```

**From a pandas DataFrame (in Python):**
```python
from nixtla_scaffold import forecast_spec_preset, run_forecast

run = run_forecast(df, forecast_spec_preset("finance", horizon=6))
run.to_directory("runs/my_forecast")
```

### Step 2.5: MCP Feature / Regressor Discovery Loop

This is mandatory for frontier forecasting enablement. A univariate model is the
baseline, not the ceiling.

**Current package reality:** MLForecast currently uses safe lag/calendar features
by default. It does **not** yet automatically train with arbitrary external
regressors from MCPs. The skill should still walk users through the feature
engineering discovery process so the next iteration is structured, auditable,
and leakage-safe instead of "throw every column into LightGBM."

Before or immediately after the baseline forecast, ask:

1. **What else could explain this metric?**
   - Revenue/ARR: seats, active users, usage, pipeline, renewals, pricing, discounts,
     overage, product launches, sales capacity, macro/industry seasonality.
   - Costs: usage volume, headcount, infrastructure units, committed spend, vendor
     contracts, seasonality, one-time migrations.
   - Demand/volume: customers, traffic, marketing, holidays, outages, launches,
     sales motions, capacity constraints.
2. **Where can we get it through MCPs?**
   - Excel MCP: plans, manually maintained event calendars, pricing tables.
   - DAX/Power BI MCP: finance measures, bookings, ARR, budget, active customers.
   - Kusto/data-query MCP: telemetry, usage, pipeline events, operational signals.
   - SQL/API MCPs: CRM, contracts, product catalog, headcount, marketing spend.
3. **Can it be used for forecasting?**
   - It must have known future values, a planned scenario, or a defensible forecast
     of its own.
   - It must not leak the target from the future.
   - It should make business sense, not only correlate historically.
   - It should be tested in rolling-origin CV against the univariate baseline.
4. **How to use it today if true exogenous modeling is not wired yet?**
   - Add known future events with `--event` or reusable JSON/YAML/CSV files via `--event-file`.
   - Create scenario overlays for launch/pricing/contract assumptions.
   - Declare candidate known-future regressors with `--regressor` or `--regressor-file` so the package writes leakage and future-availability audit artifacts.
   - Store candidate feature extracts next to the run for analyst review.
   - Use `model_explainability.csv` when MLForecast runs to inspect lag/date
     importance; do not over-interpret feature importance as causality.

If the user has no driver data, say that clearly: "This run is a statistical
baseline. The next frontier step is to search MCP-accessible sources for validated
drivers and future scenarios."

### Step 2.6: Feature Engineering Walkthrough

Use this section whenever the user asks "what features should we add?" or when
the baseline forecast needs improvement.

#### A. Separate feature types

| Feature class | Examples | Current status |
|---------------|----------|----------------|
| Safe target-history features | `lag1`-`lag13`, seasonal lag such as `lag12`, second seasonal lag such as `lag24` | Built into MLForecast |
| Calendar features | month, day of week for daily data | Built into MLForecast |
| Lag transforms | rolling mean/std, expanding mean, exponentially weighted mean | Roadmap / experiment design |
| Target transforms | `--target-transform log/log1p`; `--normalization-factor-col` for positive price/FX/inflation/definition factors | Built into the core audit pipeline for simple transforms; Box-Cox/differencing remain experiment design |
| Known-future events | launches, pricing changes, renewals, contract starts/ends | Use DriverEvent scenario overlays today |
| Known-future regressors | future price, future seats/users, plan, pipeline, bookings, spend, capacity | Discover via MCPs; only use when future values/scenarios exist |
| Benchmarks/context | prior year, plan/budget, actuals-vs-plan, targets | Use for storytelling and sanity checks; do not blend into yhat silently |

#### B. Ask the user the feature questions

Use `ask_user` or a concise structured prompt to collect:

1. Which business drivers plausibly move the target?
2. Which systems hold those drivers? Excel, DAX/Power BI, Kusto, SQL, CRM, planning workbook, pricing calendar?
3. Are future values known, planned, or scenario-able?
4. What lag should be tested? Same period, 1-period lead, 2-period lead, seasonal lag?
5. What data latency exists? Would the feature have been known at each historical forecast cutoff?
6. Is the feature causal/business-plausible, or just historically correlated?
7. Should the driver become a model regressor, an event overlay, or only a diagnostic/storytelling benchmark?

#### C. Leakage gate before modeling

Do **not** use a candidate feature unless all are true:

1. It is known at forecast time or has an explicit future scenario path.
2. It is timestamped at the same or earlier time than the forecast origin.
3. It does not contain future actuals, post-period adjustments, or the target rebuilt under a different name.
4. It is available at refresh cadence with acceptable latency.
5. It improves rolling-origin CV against the univariate lag/calendar baseline.

If any gate fails, keep it as a diagnostic or scenario note, not a regressor.

#### D. Recommended experiment ladder

Run feature work in this order:

1. **Baseline tournament**: StatsForecast + MLForecast lag/calendar candidates.
2. **Feature inventory**: MCP searches for candidate drivers and benchmark tables.
3. **Scenario/event overlay**: Add known future events first because assumptions are auditable.
4. **Lag-transform experiment**: Add rolling/expanding/EWM target features only if CV improves.
5. **External-regressor experiment**: Add known-future drivers one family at a time; compare by rolling-origin CV.
6. **Stress test**: Compare selected model vs naive, seasonal naive, plan, prior-year, and business judgment.
7. **Operationalize**: Save feature source query, refresh timing, future-value assumption, and leakage verdict.

#### E. How to communicate current limitations

Say this plainly:

> This run uses statistical and ML lag/calendar features. External business
> drivers are not automatically wired yet. The correct next step is to discover
> candidate drivers through MCPs, prove they are known in the future or can be
> scenario-modeled, and test them against the current baseline with rolling-origin
> CV before allowing them into production.

### Step 3: Review the Outputs

Every forecast run produces these artifacts:

| File | Purpose |
|------|---------|
| `OPEN_ME_FIRST.html` | Curated landing page with the executive headline, clean file map, and links to the compact workbook/report; open this first when the run folder feels noisy |
| `output/forecast_review.xlsx` | Compact review workbook with Start Here, Forecast, Decision Summary, Model Leaderboard, Watchouts, and File Guide sheets; smaller than the full audit workbook |
| `output/forecast_for_review.csv` | Selected forecast rows only, stripped down for finance review while retaining horizon and interval guardrails |
| `output/decision_summary.csv` | Condensed trust/readiness, caveats, and next actions by series |
| `output/appendix/model_leaderboard.csv` | Supporting top-model appendix table behind the workbook leaderboard |
| `output/appendix/forecast_brief.csv` | Supporting one-page run brief used by `OPEN_ME_FIRST.html` and the compact workbook |
| `output/appendix/artifact_guide.csv` | Supporting file guide for curated output, appendix, agent, and audit artifacts |
| `model_card.md` | Readable summary whose first paragraph is the deterministic executive forecast headline; quote it verbatim rather than strengthening the claim |
| `diagnostics.json` | Machine-readable run context with `executive_headline.paragraph`, trust/horizon distributions, warnings, next steps, unit labels, absolute deltas, YoY deltas when history supports them, and reproducibility metadata |
| `forecast_long.csv` | Primary model-feed output: one future row per series/model/date with yhat, intervals, weight, selected-model flag, interval status, and row-level horizon validation |
| `backtest_long.csv` | Primary validation-feed output: one row per cutoff/series/model/date with actuals, forecasts, errors, interval bounds, and coverage flags |
| `series_summary.csv` | One-row-per-series decision table with selected model, RMSE/MAE/MASE/RMSSE/WAPE, CV horizon contract, seasonality, and top alternatives |
| `model_audit.csv` | Model leaderboard enriched with weights and selected/challenger flags |
| `model_win_rates.csv` | Cross-series win rates versus SeasonalNaive or Naive benchmarks |
| `model_window_metrics.csv` | Per-cutoff RMSE/MAE/MASE/RMSSE/WAPE/bias for reviewing rolling-origin windows one at a time |
| `residual_diagnostics.csv` | Error diagnostics by model and horizon step |
| `residual_tests.csv` | Heuristic residual bias, one-step autocorrelation, outlier, and early/late structural-break checks over rolling-origin residuals; diagnostic screening only, not formal model adequacy certification; small samples are directional |
| `interval_diagnostics.csv` | Empirical prediction-interval coverage, width, method label, and pass/warn/fail calibration status when CV intervals are available |
| `trust_summary.csv` | First-stop decision artifact with per-series High/Medium/Low trust, score drivers, horizon trust, full-horizon claim gate, caveats, and recommended next actions |
| `model_explainability.csv` | MLForecast lag/date feature importance or coefficient magnitudes when ML models run |
| `forecast.csv` | Selected point forecasts + intervals (if supported), plus `row_horizon_status`, `horizon_trust_state`, `validated_through_horizon`, `planning_eligible`, and `planning_eligibility_scope`; `planning_eligible` is horizon-validation only, not a global planning approval |
| `forecast_comparison.csv` / `forecast_comparison.xlsx` | Optional `compare` output aligning scaffold forecasts to finance-owned external forecasts; directional deltas only, not residuals, accuracy metrics, or model-selection evidence |
| `comparison_report.html` / `comparison_llm_context.json` | Optional `compare` output for readable triangulation and LLM handoff; quote guardrails before discussing deltas |
| `scenario_assumptions.csv` | Event/scenario overlay assumptions from `--event` or `--event-file` |
| `scenario_forecast.csv` | Baseline `yhat` beside `yhat_scenario`, `event_adjustment`, and `event_names` for scenario review |
| `known_future_regressors.csv` | Declared known-future driver contracts from `--regressor` or `--regressor-file` |
| `driver_availability_audit.csv` | Leakage, future coverage, known-as-of timing, audit status, and modeling decision for declared regressors |
| `driver_experiment_summary.csv` | One summary table for event overlays and known-future regressor audit outcomes |
| `custom_model_contracts.csv` | Optional custom-challenger contract audit with source, invocation type, leakage guard, output contract, status, and error if excluded |
| `audit\all_models.csv` | Every model's future predictions for transparency |
| `audit\model_selection.csv` | Which model was chosen per series, why, selected CV horizon, requested horizon, window count, and whether the CV horizon matched the forecast horizon |
| `audit\backtest_metrics.csv` | Cross-validation accuracy metrics; RMSE is the primary selection metric, MASE/RMSSE are scale-free comparisons, and CV horizon metadata is included |
| `audit\backtest_predictions.csv` | Wide CV predictions for visual inspection |
| `audit\model_weights.csv` | Weighted ensemble breakdown |
| `audit\custom_model_invocations.csv` | Optional per-cutoff invocation audit for custom callable/script challengers, including history rows, future-grid rows, status, and script command metadata |
| `audit\target_transform_audit.csv` | Raw y, normalization factor, adjusted y, transformed/modeling values, output scale, and notes when log/log1p or factor normalization is enabled |
| `audit\seasonality_diagnostics.csv` | Cycle counts, complete cycles, seasonal/trend/remainder strength, credibility label, and warnings about insufficient seasonal evidence |
| `audit\seasonality_decomposition.csv` | Additive observed/trend/seasonal/remainder evidence when at least two complete seasonal cycles exist |
| `hierarchy_reconciliation.csv` | Reconciliation method summary and pre/post gap metrics when hierarchy reconciliation is enabled; coherence is prioritized and node-level accuracy may decrease |
| `hierarchy_contribution.csv` | Parent/child contribution and gap attribution for hierarchy storytelling; gap contributions are allocation heuristics, not reconciliation algorithm outputs |
| `audit\hierarchy_backtest_comparison.csv` | Selected-model rolling-origin errors before and after reconciliation for node-level accuracy/coherence tradeoff review |
| `audit\hierarchy_unreconciled_forecast.csv` | Independent model-tournament forecast preserved before reconciliation |
| `audit\hierarchy_coherence_pre.csv` | Parent/child coherence gaps before reconciliation |
| `audit\hierarchy_coherence_post.csv` | Parent/child coherence gaps after reconciliation |
| `interpretation.json` / `.md` | Backtest windows, seasonality, naive comparison |
| `diagnostics.md` | Readable diagnostics markdown with the same executive headline and next steps |
| `report.html` | Visual report with decision summary, charts, fixed-axis rolling-origin backtest filmstrip, and a static forecast-ledger preview when `ledger_context.json` exists |
| `report_base64.txt` | Same report, base64 for embedding |
| `streamlit_app.py` | Interactive dashboard with cached local artifact loading, polished sidebar **Workbench section** button tabs that keep every section visible while rendering only the active heavy section on rerun. Forecast review owns the styled executive headline card, copy-safe code block, decision/action cards for watchouts and current model next actions, and a forecast operating loop for connecting refreshes end to end, adding drivers/regressors, and tracking forecast performance over time. When `ledger_context.json` exists, a lazy Forecast ledger section opens with one clean line chart: latest actuals/history, official locks emphasized, and recent non-lock forecast versions as lighter lines before collapsing the raw ledger audit tables for deeper review. It also includes champion lens controls for best overall vs best StatsForecast/classical vs best MLForecast, active champion horizon/interval banners, winner-metric guidance, first-glance forecast charts, dedicated Model investigation, fixed-axis CV window player, Prediction intervals, Model audit, Seasonality, Hierarchy, Assumptions & Drivers, Feeder outputs, and pre/post reconciliation review when enabled. Set `NIXTLA_SCAFFOLD_STREAMLIT_PERF=1` before launching to show artifact-load diagnostics in the sidebar. |
| `ledger_context.json` | Optional pointer written when a run is registered in a forecast ledger; lets `report.html` and the Streamlit workbench discover ledger exports. |
| `runs\forecast_ledger\exports\*.csv` | Power BI-ready forecast ledger mirrors: versions, snapshots, official locks, actual revisions, forecast-vs-actuals, performance, selected-lock deltas, adjustments, corrected actuals, and regime changes. |
| `forecast.xlsx` | Excel workbook with all sheets |
| `best_practice_receipts.csv` | FPPy compliance audit trail |

Use `--unit-label` when the user cares about currency or business units in the executive headline. The generated headline includes signed absolute deltas and YoY comparisons only when a prior-year same-period actual exists; it should not invent YoY context. The generated Streamlit app includes a copy-safe headline code block with a copy icon so agents can paste the deterministic headline verbatim. The public `ExecutiveHeadline` object supports stable direct attribute access in the 0.1.x package line; its serialized dictionary may append optional fields, so ignore unknown keys.

### Step 4: Explain and Iterate

```powershell
uv run nixtla-scaffold explain --run runs\my_forecast
uv run nixtla-scaffold report --run runs\my_forecast
```

### Step 5: MANDATORY Post-Forecast Review & Action Plan

**Do NOT hand the forecast to the user and walk away.** Present a post-forecast review:

#### A. Sanity Checks (do all of these)

1. **Trust/action summary**: Open `trust_summary.csv` first. Use it to tell the user the per-series High/Medium/Low readiness, score drivers, `horizon_trust_state`, `full_horizon_claim_allowed`, caveats, and next actions. Do not present Low-trust or no-full-horizon-claim forecasts as planning-ready for the full requested horizon.
2. **Backtest accuracy**: Open `model_audit.csv` or `audit\backtest_metrics.csv`. RMSE is the default selection metric because it penalizes large misses more strongly than MAE/WAPE. Use MASE/RMSSE for scale-free cross-series comparisons and WAPE as a business-readable secondary metric, not the sole model selector.
3. **CV horizon contract**: Check `selection_horizon` vs `requested_horizon`, `cv_windows`, `full_horizon_claim_allowed`, `unvalidated_steps`, and `horizon_trust_score_cap`, then inspect `forecast.csv` row-level `row_horizon_status`, `planning_eligible`, and `planning_eligibility_scope`. If selection used a shorter horizon, disclose that steps after `validated_through_horizon` are directional, not validated planning rows. If full horizon was evaluated with only one CV window, disclose that it is still not a planning-ready champion claim. `planning_eligible=True` is horizon-validation only; still review trust, intervals, residuals, hierarchy, and data-quality caveats. Rerun with `--strict-cv-horizon` and/or add history when the decision requires full-horizon evidence.
4. **Naive comparison**: Check `interpretation.md` and `model_win_rates.csv` — does the selected model beat Naive/SeasonalNaive? If not, the data may be too noisy or have structural breaks. Be honest.
5. **Trend extrapolation**: Does the forecast trajectory make business sense? A model that extrapolates 40% monthly growth for 12 months may be mathematically correct but operationally useless.
6. **Interval width and validity**: Are the 95% intervals so wide they're uninformative? Does `lo <= yhat <= hi` hold? If not, fix before stakeholder use.
7. **Target transform / normalization audit**: If `audit\target_transform_audit.csv` has rows, explain the output scale before showing the forecast. Log/log1p outputs are inverse-transformed for reporting; factor-normalized forecasts are normalized units unless future factors are supplied externally.
8. **Seasonality credibility**: Check `audit\seasonality_diagnostics.csv` before trusting a seasonal story. If the credibility label is `low`, or complete cycles are fewer than 2, say the model cannot validate annual/weekly seasonality yet. Use `audit\seasonality_decomposition.csv` to inspect observed, trend, seasonal, and remainder components when available.
9. **Hierarchy coherence**: If the data has parent/child nodes, decide whether independent node forecasts are acceptable or whether planning requires `--hierarchy-reconciliation bottom_up`, `mint_ols`, or `mint_wls_struct`. Review `hierarchy_contribution.csv`, `audit\hierarchy_backtest_comparison.csv`, `hierarchy_reconciliation.csv`, and the pre/post coherence artifacts when reconciliation is enabled; reconciliation enforces planning coherence and can improve or worsen node-level accuracy.
10. **Driver opportunity**: Ask whether MCP sources can provide leading indicators, known future events, or normalization factors that should become features/scenarios.

#### B. Limitations Disclosure (present to user)

Always tell the user:
```
⚠️  What this forecast assumes:
• The future resembles the past (no structural changes)
• No events or drivers were added — this is a purely statistical baseline
• Prediction intervals assume similar volatility going forward
• The model was selected by backtesting on HISTORICAL data — it may not
  be the best model for the FUTURE if conditions have changed

⚠️  What could make this forecast wrong:
• Unknown future events (launches, pricing, regulation, competition)
• Market regime changes (growth → contraction, or vice versa)
• Data quality issues not caught by profiling
• Seasonality that hasn't completed a full cycle in the history
```

#### C. Action Plan (present as next steps)

```
📋 Recommended Next Steps
━━━━━━━━━━━━━━━━━━━━━━━━
1. VALIDATE: Compare forecast against plan/budget/prior year
   → Does it pass the "smell test" with domain experts?

2. ENRICH: Add known future events as DriverEvent overlays
   → Product launches, pricing changes, contract renewals
   → Each event gets a name, date window, and magnitude estimate
   → Re-run with --event flags to see the impact

3. REGRESSOR / FEATURE SEARCH: Use MCPs to find candidate drivers
   → Search Kusto/DAX/Excel/SQL for leading indicators and known future plans
   → Candidates: usage, seats, pipeline, bookings, spend, pricing, launches, contracts
   → ⚠️ Regressors need KNOWN FUTURE VALUES or scenario paths — not just history
   → Validate lag direction, leakage risk, refresh cadence, and business logic
   → Compare driver-enhanced experiments against the univariate baseline by rolling CV

4. REFINE: Iterate on the forecast
   → Try different horizons (shorter = more reliable)
   → Try different fill methods if gaps exist
   → Add hierarchy reconciliation if multiple series should sum coherently
   → Run scenario-lab to stress-test the model selection

5. OPERATIONALIZE: Set up refresh cadence
   → Monthly refresh for monthly forecasts
   → Store the query + setup config for reproducibility
   → Track forecast accuracy over time (forecast vs actual)
```

---

## EDGE CASES — What the Scaffold Handles

| Edge Case | How It's Handled |
|-----------|-----------------|
| **Short history (<6 obs)** | Falls back to baseline models (Naive, Drift, Average) |
| **Only 2 data points** | Baseline-only forecast with warnings |
| **Missing timestamps** | Detected in profile, repaired with configurable fill (ffill/interpolate/zero/drop) |
| **Intermittent demand (sparse zeros)** | ZeroForecast candidate added; scale-normalized error scoring |
| **Negative values (costs)** | Handled natively by all models |
| **Pricing/FX/inflation normalization** | `TransformSpec` / `--normalization-factor-col` divides y by a positive factor and writes `audit\target_transform_audit.csv` |
| **Log/log1p target transforms** | `--target-transform log/log1p` models transformed y and inverse-transforms outputs/backtests for reporting |
| **Outliers** | Models are robust; flagged in diagnostics |
| **Level shifts** | James-Stein shrinkage toward last actual reduces distribution shift |
| **Multiple series, mixed lengths** | Per-series model selection; short series get fallback |
| **Large panels (50+ series)** | Batch processing, tested up to 50 series |
| **Event/driver overlays** | `--event` / `--event-file` for known future scenario adjustments; `--regressor` / `--regressor-file` for known-future regressor contract audits |
| **No backtest possible** | Graceful fallback with warnings; `--require-backtest` for strict mode |

---

## COMMON PITFALLS — What Agents Must Watch For

These are the mistakes that make forecasts dangerous. Check for each one.

### 1. "Garbage In, Garbage Out" — Data Quality
- **Mixed grains**: Daily revenue mixed with monthly aggregates → wrong frequency detection
- **Duplicates**: Same date appears twice → inflated values
- **Adjusted vs raw**: Is the data inflation-adjusted? FX-adjusted? If not, the trend includes price changes, not volume growth
- **One-time items**: A $10M settlement in July will make the model think July is always big
- **Survivorship bias**: Only including currently-active products hides churn patterns

### 2. "All Else Equal" Trap
- Statistical forecasts assume nothing changes. If you KNOW something will change (launch, pricing, churn event), you MUST add it as a DriverEvent or the forecast will be wrong.
- Ask explicitly: "Is there anything happening in the next [horizon] that the history doesn't know about?"

### 3. Horizon Overreach
- Rule of thumb: reliable horizon ≈ 1/3 of history length for monthly data
- 12 months of history → 4-month forecast is solid, 12-month forecast is a story
- Daily data degrades faster — 90 days of history → 14-day forecast is strong, 60 days is speculative
- **Always tell the user** how much to trust near vs far horizons

### 4. Seasonality Mismatch
- Need ≥2 full seasonal cycles to detect seasonality reliably
- Monthly data needs ≥24 months for annual seasonality
- If you have 10 months, the model CANNOT confirm annual patterns — say so
- Quarterly fiscal patterns need ≥8 quarters (2 years)

### 5. Trend Extrapolation Danger
- Exponential growth models will happily project infinite growth
- A series growing 10% monthly will forecast 3x in 12 months — is that realistic?
- **Always ask**: "Does this growth rate / decline rate look sustainable for [horizon]?"

### 6. Intermittent Demand Illusion
- Sparse series (lots of zeros) produce misleading WAPE/MAE numbers
- A "good" model might just predict zero every period
- For intermittent demand, focus on whether the model captures the right *rate* of non-zero events

### 7. Multiple Series Without Hierarchy
- Forecasting Region A, Region B, and Total independently → they probably won't sum
- If coherence matters, use hierarchy forecasting with reconciliation
- Tell users: "Parent and child forecasts may not sum until reconciliation is added"

---

## MODEL SELECTION — How It Works

### Candidate Ladder (20+ models across 3 engines)

**Tier 1 — Always-on baselines** (any history):
Naive, HistoricAverage, RandomWalkWithDrift, WindowAverage, SES (Simple Exponential Smoothing)

**Tier 2 — Seasonal baselines** (need season > 1):
SeasonalNaive, SeasonalWindowAverage, SeasonalExpSmoothing

**Tier 3 — Intermittent demand** (≥35% zeros, non-negative):
CrostonClassic, CrostonOptimized, CrostonSBA, ADIDA, IMAPA, TSB, ZeroForecast

**Tier 4 — Classical state space** (≥6 obs):
AutoETS, AutoARIMA, Holt, HoltWinters (when ≥2 full seasons)

**Tier 5 — Theta family** (≥8 obs):
AutoTheta, Theta, OptimizedTheta, DynamicOptimizedTheta

**Tier 6 — Decomposition** (≥2 full seasons):
MSTL, MSTL_AutoARIMA (MSTL decomposition with non-seasonal AutoARIMA trend forecaster)

**Tier 7 — Boosted decomposition** (enough history for internal validation):
MFLES, AutoMFLES. AutoMFLES must be parameterized with a valid `test_size`; do not call it bare.

**Tier 8 — Machine learning** (≥30 obs, `--model-policy all` or `auto`):
MLForecast with lag/date features across sklearn and LightGBM families when installed: LinearRegression, Ridge, Ridge_Regularized, BayesianRidge, ElasticNet, Huber, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, KNeighbors, LightGBM, LightGBM_Conservative, LightGBM_Shallow, LightGBM_Robust.
MLForecast uses Nixtla `PredictionIntervals` for conformal future bands when the history, horizon, and lag plan can support at least two calibration windows. If long seasonal lags would make calibration impossible, the scaffold can drop only the interval-incompatible lags and discloses that in warnings.

### Selection Pipeline
1. **Adaptive backtest**: Per-series rolling-origin CV with window count, horizon, and step size adapted to history depth and seasonality — NOT one-size-fits-all
2. **Fair StatsForecast/MLForecast tournament**: when both engines run, they share the same rolling-origin `h`, `n_windows`, `step_size`, and cutoff dates where feasible. MLForecast lags are capped/disclosed rather than giving MLForecast fewer/easier windows.
3. **Selection**: Lowest backtested RMSE, tie-broken by MAE, WAPE, and bias
4. **Naive guard**: Selected model must beat naive by ≥5% margin or falls back
5. **Weighted ensemble**: Inverse-error ensemble when backtest supports it
6. **CV horizon contract**: `selection_horizon`, `requested_horizon`, `cv_windows`, and `cv_horizon_matches_requested` are written to selection/audit outputs; `trust_summary.csv`, `forecast.csv`, and `forecast_long.csv` also expose `horizon_trust_state`, `validated_through_horizon`, and row-level `planning_eligible`
7. **James-Stein shrinkage**: Post-processing that blends forecast toward last actual based on backtest residual variance while shifting selected-model intervals with the point forecast
8. **ZeroForecast**: Added for intermittent demand series (≥35% zeros)
9. **Candidate failure isolation**: StatsForecast and MLForecast try the full ladder first, then retry individual models when one candidate fails. A model that cannot produce intervals may still keep its point forecast; failed candidates are disclosed in warnings instead of collapsing the engine to baselines.

### Adaptive Backtest Windows

The backtest adapts per series instead of using fixed windows:

| History | Horizon | Windows | Step | Rationale |
|---------|---------|---------|------|-----------|
| 3 obs | 1 | 1 | 1 | Minimal: just enough to validate |
| 8 obs monthly | 3 | 2 | 2 | Short series: small h, 2 windows |
| 24 obs monthly | 6 | 2 | 6 | Standard: full horizon test |
| 48 obs monthly | 6 | 3 | 6 | More windows for reliability |
| 90 obs daily | 14 | 5 | 7 | Rich daily: seasonal step alignment |
| 365 obs daily | 30 | 5 | 7 | Deep history: max windows, weekly steps |

### Model Policy Options

| Policy | CLI flag | What runs |
|--------|----------|-----------|
| `auto` | `--model-policy auto` | StatsForecast + MLForecast when ≥30 obs |
| `all` | `--model-policy all` | StatsForecast + MLForecast always |
| `statsforecast` | `--model-policy statsforecast` | StatsForecast only |
| `mlforecast` | `--model-policy mlforecast` | MLForecast only |
| `baseline` | `--model-policy baseline` | Naive/Drift/Average only |

Favorite-model allowlist:

```powershell
uv run nixtla-scaffold forecast --input data.csv --preset finance --horizon 6 --model arima --model "arima mstl" --output runs\arima_favorites
uv run nixtla-scaffold forecast --input data.csv --preset finance --horizon 6 --model-allowlist arima "arima mstl" --no-weighted-ensemble --output runs\literal_arima_favorites
```

Aliases are canonicalized, so `arima` becomes `AutoARIMA` and `arima mstl` / `mstl arima` becomes `MSTL_AutoARIMA`. This is a real model tournament allowlist, not just a UI preference: non-allowlisted StatsForecast/MLForecast/baseline candidates are skipped and `manifest.json -> model_policy_resolution.model_allowlist` records the canonical set. If `--weighted-ensemble` remains enabled, `WeightedEnsemble` is derived only from the allowlisted candidates; use `--no-weighted-ensemble` when the user wants literal model outputs only.

Strict evaluation option:

```powershell
uv run nixtla-scaffold forecast --input data.csv --horizon 12 --strict-cv-horizon --output runs\strict_h12
```

Use this when the forecast will feed a high-stakes decision and the user prefers
"no full-horizon validation, no champion claim" over shorter-horizon adaptive CV.
If strict CV cannot be produced, disclose that more history or a shorter horizon is needed.

---

## EVENT/SCENARIO OVERLAYS

When the user knows something the history can't:

```powershell
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --event '{"name":"Product launch","start":"2026-03-31","effect":"multiplicative","magnitude":0.10}' --output runs\scenario
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --event-file scenario_events.csv --output runs\scenario_file
```

Event fields:
- `name`: descriptive label
- `start`/`end`: date window
- `effect`: "multiplicative" or "additive"
- `magnitude`: size of the effect (0.10 = 10% bump)
- `confidence`: 0-1 scaling factor (default 1.0)

Event interval rule: additive events shift interval bounds; multiplicative events
scale interval bounds. Keep statistical `yhat` and scenario `yhat_scenario` separate.

Known-future regressor declarations are also supported for audit-first driver work:

```powershell
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --regressor '{"name":"Seats plan","value_col":"seats_plan","availability":"plan","mode":"model_candidate","future_file":"future_seats.csv","source_system":"excel","owner":"finance"}' --output runs\driver_audit
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --regressor-file known_future_regressors.yaml --output runs\driver_audit_file
```

This writes `known_future_regressors.csv`, `driver_availability_audit.csv`, and
`driver_experiment_summary.csv`. A model-candidate regressor must pass future
coverage, known-as-of timing, and target-leakage checks before a future
exogenous-modeling experiment should use it. Current models still use lag/date
features only; declared external regressors are audited, not auto-trained.

---

## PREDICTION INTERVALS

Intervals use conformal prediction (distribution-free, no normality assumption).

```powershell
uv run nixtla-scaffold forecast --input data.csv --horizon 6 --levels 80 95 --output runs\with_intervals
```

Requirements: enough history for ≥2 conformal windows. StatsForecast uses conformal intervals directly. MLForecast uses Nixtla `PredictionIntervals(n_windows, h)` during fit and predict; this may cap lag choices so each calibration fold has enough rows. If insufficient, intervals are skipped with a warning.

FPPy review rule: intervals are not decoration. Check interval coverage, average width,
and whether the interval method is appropriate. If `selection_horizon < requested_horizon`,
the far horizon intervals are less validated.
If `interval_diagnostics.csv` is empty but `forecast.csv` has intervals, the future bands exist but the rolling-origin CV folds were too short to empirically calibrate coverage at that horizon; disclose this before stakeholder use.

---

## HIERARCHY FORECASTING

```powershell
# Step 1: Build hierarchy nodes
uv run nixtla-scaffold hierarchy --input raw_data.csv --hierarchy-cols region product --output runs\nodes.csv

# Step 2: Forecast all nodes
uv run nixtla-scaffold forecast --input runs\nodes.csv --horizon 3 --freq ME --output runs\hierarchy_demo

# Step 3: If parent/child planning totals must tie, rerun with reconciliation
uv run nixtla-scaffold forecast --input runs\nodes.csv --horizon 3 --freq ME --hierarchy-reconciliation bottom_up --output runs\hierarchy_reconciled
```

Default hierarchy forecasts are independent per node so the model tournament can judge each series on its own merits. When planning requires coherent rollups, use:

| Method | CLI value | When to use |
|--------|-----------|-------------|
| Diagnostics only | `none` | You only need to see parent/child gaps or want independent statistical forecasts |
| Bottom-up | `bottom_up` | Parent forecasts should equal the sum of bottom-level forecasts; simple and transparent |
| MinTrace OLS | `mint_ols` | You want HierarchicalForecast reconciliation without insample covariance requirements |
| MinTrace structural WLS | `mint_wls_struct` | You want a structure-weighted MinTrace variant when the hierarchy is reliable |

Review rule: reconciliation improves planning coherence, not necessarily independent predictive accuracy. Always inspect `hierarchy_contribution.csv`, `audit\hierarchy_backtest_comparison.csv`, the unreconciled model tournament, `audit\hierarchy_unreconciled_forecast.csv`, `audit\hierarchy_coherence_pre.csv`, `audit\hierarchy_coherence_post.csv`, and `hierarchy_reconciliation.csv` before stakeholder use.

---

## SETUP WIZARD (Agent-First)

When a user says "forecast this" without clear specs:

```powershell
uv run nixtla-scaffold setup --workspace runs\my_workspace --data-source kusto --preset finance --series-count single --target-name revenue --time-col date --horizon 6 --freq ME --model-families statsforecast --exploration-mode
```

This creates:
- `forecast_setup.yaml`: reusable config
- `agent_brief.md`: checklist with exact next commands
- `questions.json`: structured intake for programmatic use
- Query templates and folder structure

---

## KNOWLEDGE BASE

```powershell
uv run nixtla-scaffold guide intervals
uv run nixtla-scaffold guide statsforecast
uv run nixtla-scaffold guide seasonality
uv run nixtla-scaffold guide hierarchy
```

Sources: Nixtla docs, Nixtla GitHub source code for StatsForecast/UtilsForecast/HierarchicalForecast/MLForecast, and FPPy. When citing FPPy, use: Hyndman, R.J., Athanasopoulos, G., Garza, A., Challu, C., Mergenthaler, M., & Olivares, K.G. (2025). Forecasting: Principles and Practice, the Pythonic Way. OTexts: Melbourne, Australia. Available at: OTexts.com/fpppy. Accessed on 28 April 2026.

### FPPy Chapter Map for Agents

Use these as the doctrine map when explaining or improving the scaffold:

| Chapter | URL | Agent takeaway |
|---------|-----|----------------|
| 1 | https://otexts.com/fpppy/01-intro.html | Forecasts are not goals/plans; assess predictability and future similarity |
| 2 | https://otexts.com/fpppy/02-graphics.html | Plot first; visuals reveal patterns and breaks models can miss |
| 3 | https://otexts.com/fpppy/03-decomposition.html | Adjust/transform/decompose before blindly modeling |
| 5 | https://otexts.com/fpppy/05-toolbox.html | Prepare, visualize, specify, estimate, evaluate, forecast; benchmark and diagnose residuals |
| 6 | https://otexts.com/fpppy/06-judgmental.html | Judgment/events must be structured and auditable |
| 7 | https://otexts.com/fpppy/07-regression.html | Regression needs future predictor values |
| 10 | https://otexts.com/fpppy/10-dynamic-regression.html | Dynamic regressors require leakage checks and future scenarios |
| 11 | https://otexts.com/fpppy/11-hierarchical-forecasting.html | Parent/child forecasts should be coherent when used together |
| 13 | https://otexts.com/fpppy/13-practical.html | Weekly/daily/multiple seasonality, moving holidays, counts, and intermittent demand need special handling |

---

## PYTHON API

```python
from nixtla_scaffold import (
    ForecastSpec,
    DriverEvent,
    forecast_spec_preset,
    run_forecast,
    profile_dataset,
)

# Preset-based forecast
run = run_forecast("data.csv", forecast_spec_preset("finance", horizon=6, freq="ME"))

# Basic forecast
run = run_forecast("data.csv", ForecastSpec(horizon=6))

# With options
spec = ForecastSpec(
    horizon=6,
    freq="ME",
    levels=(80, 95),
    model_policy="auto",          # "auto", "baseline", "statsforecast"
    fill_method="ffill",          # "ffill", "interpolate", "zero", "drop"
    strict_cv_horizon=False,       # True = select only on CV h matching horizon
    weighted_ensemble=True,
    verbose=True,
    events=(
        DriverEvent(name="Launch", start="2026-03", effect="multiplicative", magnitude=0.15),
    ),
)
run = run_forecast(df, spec)

# Access results
run.forecast              # pd.DataFrame with yhat, model, intervals
run.all_models            # all candidate predictions
run.model_selection       # which model was chosen and why
run.backtest_metrics      # cross-validation metrics
run.profile               # DataProfile with data quality info
run.interpretation()      # backtest windows, seasonality, naive comparison
run.diagnostics()         # run status, warnings, next steps
run.explanation()         # model card markdown
run.manifest()            # full output manifest

# Write to directory
run.to_directory("runs/output")

# Write Excel workbook
run.to_excel("runs/output/forecast.xlsx")
```

---

## SCENARIO LAB (Validation)

Run 100 synthetic scenarios to validate the scaffold:

```powershell
uv run nixtla-scaffold scenario-lab --count 100 --model-policy auto --output runs\lab
```

Scores: validity (35%), accuracy (35%), ease (20%), explainability (10%).
Current best: **composite 95.537**, 100/100 passed, 0 crashes, 20 model candidates.

## WORKBENCH QA (Dashboard/Artifact Validation)

Run golden generated-workbench scenarios before claiming report or Streamlit changes are done:

```powershell
uv run nixtla-scaffold workbench-qa --output runs\workbench_qa
uv run nixtla-scaffold workbench-qa --scenarios hierarchy_reconciled transform_normalized --model-policy baseline --no-app-test --output runs\workbench_qa_fast
uv run nixtla-scaffold workbench-qa --output runs\workbench_qa --app-test-timeout 120
```

The harness generates representative runs for monthly, short-history, hierarchy reconciliation, and finance-normalization cases. It checks required artifacts, compiles each generated `streamlit_app.py`, runs Streamlit AppTest unless disabled, and writes `workbench_qa_summary.csv` plus `workbench_qa_summary.json` with the AppTest timeout seconds used for diagnostics. It also writes `workbench_perf_summary.csv` and `workbench_perf_summary.json` with compile/AppTest seconds, generated app size, CSV artifact row counts, and a performance status for dashboard snappiness regression tracking.

Rule: no dashboard/workbench feature is done until the relevant golden run passes this harness, and high-risk changes should still get a live HTTP smoke.

## RELEASE GATES (Package/Workbench Readiness)

Run local release gates before saying the package is ready for broader use:

```powershell
uv run nixtla-scaffold release-gates --output runs\release_gates
uv run nixtla-scaffold release-gates --output runs\release_gates_fast --no-build --no-install-smoke --no-workbench-qa --no-live-streamlit --scenario-count 2
uv run nixtla-scaffold release-gates --output runs\release_gates --json
uv run nixtla-scaffold release-gates --extended --output runs\release_gates_extended
uv run nixtla-scaffold release-gates --output runs\release_gates_datasets --no-build --no-install-smoke --no-workbench-qa --no-live-streamlit --scenario-count 2 --require-optional datasets
```

The CLI prints a compact verdict by default and writes `release_gate_summary.md`, `release_gate_summary.json`, and `release_gate_results.csv`. It checks package metadata, wheel/sdist contents, isolated wheel install smoke including the installed console script, deterministic scenario-lab numeric thresholds, a quick forecast golden with interval-containment checks when intervals exist, artifact hygiene, optional-extra availability, generated workbench QA, and one live Streamlit HTTP health smoke in the current Python environment. Use `--json` only when an agent needs the full nested payload in stdout. Use `--require-optional ml hierarchy` when a release must prove optional MLForecast or HierarchicalForecast extras are installed, or `--require-optional datasets` when validating the optional DatasetsForecast real-data smoke. Use `--extended` for a stricter local release profile that runs at least 20 auto-policy scenarios, uses all-family workbench QA defaults, and requires the `ml` plus `hierarchy` extras.

Read `release_gate_summary.md` first. It contains the one-glance headline, failed gates, remediation hints, provenance, git-SHA unavailable reason when relevant, scenario archetype scores, thresholds, and exit codes. Release-gate exit codes are `0` passed, `1` gate failure, and `2` CLI/runtime error.

---

## RESEARCH FINDINGS (from 10 autoresearch iterations)

Key facts verified by experiment:
1. Croston/ADIDA/IMAPA models don't help on Bernoulli-zero intermittent patterns (confirmed 2×)
2. WAPE is hostile to intermittent demand; MASE is too harsh with first-diff scale
3. Distribution shift causes 15-21 holdout underperformances (not model selection error)
4. James-Stein shrinkage toward last actual is the most effective single mitigation
5. 5% naive margin guard prevents overconfident complex model selection
6. ZeroForecast is WAPE-optimal when >50% of periods are zero
7. Weighted ensemble enabled is informational, not a warning

---

## TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| `failure_diagnostics.json` appears | Read it — it has the error, likely causes, and next steps |
| No intervals in output | History/horizon/lag plan cannot support two conformal windows; add history, shorten horizon, or reduce lag complexity |
| Model selected = Naive | The data may be too noisy for complex models to earn their keep |
| WeightedEnsemble selected | The ensemble of all models beat any single model — this is good |
| Warning: "prediction intervals skipped" | Normal for short series; not an error |
| `require_backtest=True` fails | Some series lack enough history; remove the flag or add data |
| `selection_horizon < requested_horizon` | Model was selected on a shorter CV horizon; disclose this, treat rows after `validated_through_horizon` as directional because `planning_eligible=False`, or rerun with `--strict-cv-horizon` |
| Need better forecast than univariate baseline | Search MCPs for known-future drivers/events, then test driver-enhanced experiments against baseline |

---

## DEPENDENCIES

```
numpy>=1.26, pandas>=2.0, statsforecast>=1.7, utilsforecast>=0.2, openpyxl>=3.1, pyyaml>=6.0
```

Reporting/dashboard dependency: `plotly>=6.0`.

Optional: `hierarchicalforecast>=1.0` (hierarchy), `lightgbm>=4.0 + mlforecast>=1.0` (ML drivers), `datasetsforecast>=1.0` (opt-in public real-data validation), `neuralforecast>=2.0` (research only).

