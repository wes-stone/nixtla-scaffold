# Forecast Workbench

## Product direction

An analyst's workstation, not a model-settings dashboard. The user's FINN
workbench is the interaction reference: a forecast explorer, independent scoped
documents, an optional evidence inspector and one existing Copilot conversation.
Nixtla remains the forecasting engine. This design does not imply affiliation,
predictive superiority or production readiness.

## Standalone engine boundary

The distributable Python engine and CLI own forecasting, evaluation and evidence.
They must remain usable without Finn, Azure credentials, Copilot, Node or a
particular user's filesystem. Agent skills and the canvas consume those same
interfaces; neither is a prerequisite for native execution.

Consumer-specific adapters own external schemas, authentication, paths, reporting
ledgers and publication approvals. Shared artifact contracts allow interoperability
without moving those dependencies into the engine. An adapter's supported subset
must not redefine the full package's capabilities or weaken its qualification gates.
Local runtime bindings are created at installation, never shipped as portable
configuration. Public distribution excludes private source data and saved runs.

## Rendered visual system

The canvas uses Segoe UI/system typography at a 14 px base, neutral bordered
surfaces, restrained blue selection, tabular numeric tables and a large ECharts
timeline. Light mode is white with a pale library; dark mode uses charcoal/navy
surfaces and light text. Appearance follows the system unless overridden.
There are no gradients, gold panels, KPI-card grids or embedded chatbot.

The top bar contains workspace identity and save state. The left explorer
contains version history, source snapshots and recoverable documents. Data, Explore data,
Forecast, Models, Backtests, Diagnostics, Compare, Adjust, Research and Pipeline organize the center. The inspector is
normally closed; on narrower canvases it becomes a dismissible overlay. The
activity strip shows persisted job states rather than simulated percentages.

Rendered browser journeys cover 1440 px and 620 px widths, light/dark themes,
real forecast evidence, paired ensemble replay and draft recovery. These checks
are not a WCAG certification. The design target is WCAG 2.2 AA: explicit control
names, visible keyboard focus, native dialogs with restored focus, exact-value
tables, period selection outside the chart, null-preserving plots and reduced
motion. Dark chart axes use explicit light labels, not ECharts' faint defaults.

## SQLite library explorer

The left explorer uses the existing SQLite ledger, not just a page of recent
results. Forecast families expand into versions and their original artifact
folders. Data, saved notes, activity and recoverable drafts have their own
branches. Children load on demand; search covers the complete selected library.
Counts and continuation controls make paging explicit. Folder expansion is
revisioned with workspace state; keyboard arrows traverse and expand folders.
Saved source and file tabs are read-only. Scripts and HTML display as text,
never execute. Source previews retain the saved tuning boundary; oversized or
binary files explain their preview limits. This is not a whole-machine browser.

## Interaction contracts

**The Declared, Observed, Fitted Rule.** Configuration is a read-only evidence
review within the existing workstation, not a settings editor. Its opening
decision table exposes the observation calendar, primary seasonal period and
basis, training-history scope, external target transform and outer forecast
selection objective. The review then proceeds through observed seasonal patterns,
native model behavior and settings, and a collapsed full-evidence/export section.
Constructor settings and actual selected native configuration remain separate;
internal transformations are not the external target transform, and native inner
search loss is not necessarily the outer backtest objective. Unreported fitted
structure is labeled missing, never inferred from a model name.

**The Scoped Seasonal Evidence Rule.** Configuration requires one series;
different series are not pooled into a diagnosis. The seasonal-period table and
paired early/recent phase bars retain original dimensionless ratios, independent
of display units. Each cycle is divided by its own mean; the early and recent
blocks each contain eight non-overlapping complete cycles. Detected periods,
strength and drift remain descriptive diagnostics, not automatic overrides or
significance tests. The next-cycle change and recent 5th–95th percentile change
range are a plausibility comparison, not a prediction interval or growth cap.
A review-required notice retains the forecast without clipping or replacing it.

**The Saved Configuration Evidence Rule.** Reload rereads saved evidence; it does
not reconstruct diagnostics, refit or adopt. Loading, failed reads, missing
artifacts and bounded previews remain explicit. Legacy versions retain their
original specification while explaining that fitted settings and seasonal-shape
evidence were not recorded. Model focus changes the inspected configuration,
not the saved selection; complete per-model receipts remain available in the
selected series' paged evidence tables and full-scope, full-precision exports.
Constructor/configuration parsing errors are shown rather than hidden.
Configuration changes belong to explicit frozen experiments; historical
comparisons confer neither adoption authority nor untouched confirmation.
The surface reuses neutral bordered tables, the existing chart, native controls,
notices and footnotes, without a new visual theme.

**The Modeling Copy Rule.** Calendar and anomaly review lives inside Configuration,
scoped to the selected series. Observed usage, a seasonal reference and the training
copy share the existing line chart; flags, proposed values and protection reasons
remain inspectable in exact-value tables. Scores and counts stay unscaled. The
reference is neither an interval nor a verified correction. Missing/off/unsupported
reviews are explicit, and opening or reloading the surface never applies changes.
Calendar features belong to supported native ML recipes, not a second global
multiplier. Full origin receipts and saved-unit exports retain the original as-of
scope. Automatic trials appear as separate unadopted versions in the library.

**The Native Fitted Evidence Rule.** The focused Configuration model exposes saved
native fitted values, 99% fitted bounds and flagged training targets in the inherited
chart, with dashed bounds and isolated flag markers. Counts and exact-value tables
stay unscaled; plot labels follow display units. These are in-sample diagnostics,
not future interval coverage or automatic actual revisions. Unsupported adapters
and older versions explain missing evidence without fitting. Configuration admission
is visible separately: excluded candidates remain inspectable, changing model focus
does not select them, and warning-only legacy policies are labeled explicitly.
Excluded scores stay visible but unranked; configuration exclusion is never labeled
as missing backtest support.

The model browser defaults to best backtest accuracy first, with an explicit
metric selector, numeric ranks and a persistent saved-winner label even while
another path is focused. Bias sorting uses absolute bias; deterministic ties use
MAE, absolute bias and model name. Only explicitly comparable components receive
ranks. Filters and alternative display metrics do not change the saved selection.
Forecast endpoints remain in the focused-model summary, not the accuracy column.

The native weighted ensemble is directly accessible from the browser and has a
visible evidence section in Forecast, Models and Compare. Its saved comparison
rescopes the component winner to the same later periods, exposes warm-up and
matched-row counts, and keeps incomplete comparisons unranked. Final component
weights are available separately from historical replay. The original tournament
and saved workbench blend experiments remain distinct; navigation never fits or
adopts a model.

Explore data uses an explicit local query builder, not an embedded warehouse SQL
console. Saved forecast CSVs, saved CSV file tabs and source snapshots share the
same rows/pivot surface. A searchable field rail sits beside row groups, one
column split, aggregation and AND filters; it stacks above the builder on narrow
canvases. The first table previews automatically. Subsequent query edits keep the
previous result labeled stale and disable export until Run query succeeds.

Pivots evaluate every matching previewable row before paging. Raw values retain
their original numeric text; sums use decimal arithmetic, averages use 34
significant digits, missing combinations remain empty and IDs remain text.
Explicit exploration bounds are 32 MiB, 250,000 rows, 200 fields / 2 million input
cells and 60 pivot columns / 100,000 pivot cells. Bounds produce errors, never
silently sampled aggregates. Protected sources reuse the saved tuning boundary
before schema inference, filtering, pivoting or export.

Named queries live in revisioned, namespaced SQLite state rows, independently of
document autosave, preserving compatibility with existing workspaces. They bind
the exact immutable source/run identity, fields, filters, pivot and sort; edits
detect concurrent-save conflicts. Exports persist the complete result CSV and a
JSON query/provenance receipt together. Browser download supports files up to
4 MiB; larger complete exports remain available at their explicit local paths.
Exploration does not fit, refresh, adopt or query external providers.

Adjust is the analyst judgment worksheet. A searchable full-name baseline selector
is independent of statistical winner selection; incomplete model paths cannot be
silently filled. The worksheet follows chosen model, optional calendar layer,
final tweaks and saved version in that order. Bulk signed percentages and
per-period absolute changes combine as adjustment base * (1 + percent / 100)
+ amount, where the adjustment base is the calendar output when enabled and
the original chosen model otherwise. Sparse row entries replace the bulk input;
a final override wins, including zero. Inputs always use original units while
display scaling affects only comparisons. A spreadsheet-style grid keeps period
labels, calculated values, horizon sum and running difference visible. A compact
comparison above the scrollable grid keeps model/adjusted totals and the delta
visible on narrow canvases; chart legends name the actual model baselines.
40-digit
decimal calculations match the immutable server output. Negative forecasts are
not silently floored. Horizon sums are clearly distinguished from stock/rate
endpoints.

The optional activity-day overlay uses existing monthly day counts and holiday
calendars. Effective days equal weighted weekdays/weekends less holiday loss;
explicit month losses replace, not compound with, dated holiday effects.
The raw overlay is model * effective days / reference effective days. Holding
the chosen model total is the default, with proportional allocation and exact
final rounding reconciliation; turning that option off allows total changes.
The layer is off until explicitly enabled. It is a model-path overlay, not the
independent constant-growth FINN scenario, a fit, or an automatic correction for
MSTL/ARIMA's existing seasonality; applying it can double-count those effects.
Weekdays have weight 1; weekend and holiday weights default to 1, with no holiday
calendar selected. Equal weights still reflect month length. No copied growth
or holiday-loss defaults and no learned coefficients are implied.
Reference actuals never change. Edited assumptions persist immediately; "Update
calendar layer" computes the combined change once, not a worker per keystroke.
Invalid inputs and stale layer responses block save/export; failed reads have
an explicit retry. Day-weight fields remain aligned in the existing two-column grid
and stack on mobile; optional monthly losses use a horizontally contained table.
The worksheet and chart expose original model, after-calendar and final values.

Named, reasoned saves create separate immutable selected-series forecast versions in the
existing SQLite library. Each retains its original baseline, model, parent,
recipe and hashes, freezing the inputs, calendar assumptions, dates, intermediate
values and stage deltas; edits branch from the original model, not an already
adjusted path. Opening another version requires a confirmation dialog only when the worksheet
differs from its saved or neutral settings: "Keep editing" preserves the draft;
"Discard edits and open version" replaces it. Adjusted values flow through Forecast, Explore data, saved version deltas,
CSV exports and the operating ledger. Baseline backtests and intervals are never
inherited as judgment evidence; saving confers neither planning approval nor an
official lock. Judgment maintenance offers landed-actuals recording, not direct
judgment refresh. "Open original baseline to refresh" hands off to that baseline;
using refreshed model output requires an explicit new adjustment version.
Actuals append a ledger vintage without changing the saved forecast or its
original historical inputs. CSV/TSV upload or Excel tab-separated paste
replaces row edits and clears bulk changes; duplicate, historical, foreign-series
and foreign-baseline rows fail visibly. Imports are bounded at 500 periods and
128 KiB, with an explicit error for oversized encoded requests. Downloaded worksheets and JSON recipes
retain the comparison and rationale.

Pipeline renders the saved finalization chain instead of implying that a
judgment version has new model/evaluation evidence. Every stage is inspectable;
the final judgment summary includes nonzero bulk percent and absolute change in
original units per period, the bulk period scope and row-input count when present.
Neutral settings read "No final changes."

**The Outcomes Before Factors Rule.** The primary Pipeline table leads with
chosen-model, after-calendar and final values, followed by calendar, final-tweak
and total deltas. Day counts, losses, month/day-mix/holiday factors and total
scaling belong in the separate "Calendar days & factor bridge" disclosure.
Both tables keep their period column sticky during horizontal scrolling;
factor display uses six decimals while raw copy and exports retain saved precision.
These remain neutral bordered evidence tables, not a new visual theme.

**The Pinned Replay Rule.** Replay checks the pinned original source fingerprint
and every comparison value without refitting, rebasing onto refreshed data or
writing a version. Saved recipe inspection works independently of the original
baseline; replay/editing requires that baseline. The displayed pipeline is
explicitly the saved version, not an unsaved worksheet draft. Refreshed baselines
need an explicit new adjustment version, never automatic carry-forward of dated
overrides.

**The Seasonal Projection Rule.** Forecast has adjacent Timeline / Seasonality
controls; the same seasonal plot is available in Diagnostics. Daily observations
overlay months at Monday-aligned slots, with a calendar-day alternative and
subtle weekend bands. Monthly observations overlay years by month with an
explicit year start; weekly observations retain ISO week-years and week 53;
calendar-quarter observations use four quarter positions. Other frequencies
explain their unsupported projection instead of being silently resampled.
The initial focus is the first saved forecast period, not the machine's current
date or the end of the horizon, with two preceding months/years for comparison.
Focus, alignment, comparison depth and actual visibility persist per document.

Period colors distinguish actual comparisons; dashed model paths continue from
an immediately adjacent observed point only. No line bridges a missing date or a
different comparison period. Full calendar positions remain visible beyond the
saved horizon, without invented values. Observed-only history is preferred;
legacy saved-history fallback and bounded reads are disclosed. Tooltips carry
original dates, and a keyboard-accessible exact-value table separates actuals
from forecasts and preserves raw copy. Model previews remain display-only.
Seasonal paths are point-only; saved interval controls remain on Timeline.
Neutral native controls, system typography and existing ECharts surfaces are
retained, with wrapping controls and legends on narrow canvases.

Diagnostics / Seasonality also retains its monthly calendar adjuster. It exposes
calendar days, weekdays/weekends, explicit holiday coverage and relative usage
weights, anchored to a selected complete observed month. Weekdays have weight 1;
weekends and holidays default to 1, with no holiday calendar selected. Holiday
weight replaces the ordinary day weight; weekend holidays are not counted twice.
US federal observed dates and explicit custom date lists are available, with
coverage caveats. Month length, day mix and holiday factors form a multiplicative
bridge relative to the reference month.

The adjuster normalizes historical totals and shows a separate constant-rate
calendar scenario beside the unchanged saved forecast. It never applies another
factor to model yhat, refits, claims learned weights or invents intervals. It is
only for complete calendar-month flows, not stocks/rates/partial months; already
normalized runs and ambiguous duplicate periods fail explicitly. Assumption
edits persist per document and label the old preview stale until updated.
Its handoff to Adjust copies day-weight assumptions, not constant-rate scenario
values; saved judgment versions instead link to their existing adjustment recipe.
CSV/JSON exports include the full calendar, normalized values, separate scenario,
source hashes, holiday dates and pinned reference assumptions. Seasonal
overlays, decomposition and seasonal credibility evidence remain below.

Model exploration is focus-first. Forecast, Models and Backtests share a
searchable run-scoped roster, previous/next navigation and a remembered comparison
of at most five paths. Legacy "all models" display state opens one focused path,
not a spaghetti chart. The complete tournament is still available below; chart
selection does not filter away scores or change the saved winner. Roster rows
show the final available forecast date/value and saved support, with absent
tournament scores explicitly distinguished from validated candidates.

At wide widths the roster and chart are side by side; narrow canvases stack
them with a bounded, scrollable roster. Full model names wrap rather than
disappearing into a select menu. Chart legends wrap below the plot, use stable
model colors and never paginate. Body-mounted, bounded HTML tooltips omit absent
predictions over history, escape model names and retain zero values. Diagnostic
tooltips show at most eight numeric rows with an explicit overflow count.
Candidate interval shading belongs only to the focused model. Model browsing,
comparison, reset and persistence never execute, adopt or stitch forecasts.

Display-unit help sits on a separate full-width toolbar row so neighboring
selects and actions align to their controls, not to the bottom of helper copy.
The shared scale control remains stacked in section headings. Wide review table
reveals the review and moves both scroll position and keyboard focus to it;
activating it again returns to the open table rather than hiding it offscreen.
Back to forecast closes the review and restores focus to its opening control.

- Inspect before configuring. A source path and horizon are enough when mapping
  and calendar are unambiguous. Optional business context is not fabricated.
- Opening or navigating cannot query a provider or run a model. Native jobs
  need an explicit forecast/research action and have a visible Stop control.
- Ordinary navigation reuses the active tab but retains its former draft.
  Explicit New tab creates an independent scope. Closing keeps a recoverable
  draft. Save conflicts stop writes; they never overwrite a concurrent edit.
- Clean panels synchronize shared tabs and revisions on focus, visibility
  restoration and every ten seconds while visible; active jobs use a shorter
  poll. Hidden panels do not poll. Pending saves and unsaved drafts are never
  replaced by background refresh. Explicit conflict recovery waits for queued
  saves and warns that local edits must be copied before reloading.
- Compare separates model/ensemble evidence from version-to-version changes.
  Version deltas are not accuracy rankings. Blend scores share finite rows;
  origin replay includes component weights and excludes unavailable outcomes.
- Native interval ribbons use saved lower/upper bounds. Blend ranges are not
  synthesized from component intervals. Retrospective blends are point-only;
  protected confirmation can add blend-residual calibration with explicit coverage
  caveats. Adoption regenerates the candidate's own reports and claim gate.
- Exports use source CSV numeric text rather than chart samples or a JSON float
  round-trip. Nulls remain empty. Formula-like text is escaped and counted.
- Chat handoffs contain scoped metadata and the user's question, not a private
  catalog dump or a new execution grant.

## Capability surfaces and boundaries

| Capability | Current surface |
| --- | --- |
| Classical/native models and installed eligible ML/smooth families | Local jobs through the existing engine |
| Hierarchy, normalization, drivers and scenarios | Optional native spec editor, existing engine, original evidence |
| Custom Python and optional FINN/finnts execution | Separate hash-pinned proposal and explicit approval; callable-only plugins retain the CLI path |
| External/BYO outputs | Explicit import, directional comparison, cutoff-contract scoring; no automatic cross-engine mixing |
| Equal, fixed, top-k, family-diverse and inverse-error ensembles | Editable recipes, matched replay, explicit immutable adoption |
| Blend-specific calibration and protected confirmation | 20 prior calibration windows plus one later confirmation; workspace exposure guards; no automatic promotion |
| Native chronological research | Existing optimizer with two iterations and one variant; previously consumed snapshots rejected |
| Saved notes, independent tabs and version deltas | Native canvas |
| Local pivots, full-result exports and saved export queries | Explore data on forecasts, source snapshots and CSV file tabs |
| Pipeline refresh, actuals ingestion and official locks | Native maintenance controls, operating ledger and reviewed script-backed authoring |
| Hard-crash reconciliation | Explicit recovery of verified orphan processes; no automatic replay |
| Larger datasets | Paged tables/catalog, scoped reads, recent-history chart bounds, full scoped exports; 1/50/1000-series fixtures |

## Streamlit analytical capability map

The source inventory includes Streamlit's persistent wrapper controls, not just
direct `st` widgets. A clean shell is not a substitute for these evidence views.

| Original review capability | Canvas surface |
| --- | --- |
| Forecast/champion lens, units, wide summary, actual/future periods and interval rows | Forecast; explicit display-only candidate selection; wide review table |
| Tournament, family/metric/focused-model lenses, weights and benchmark wins | Models; leaderboard plus scoped model-evidence explorer |
| Pareto frontier and dominance/selection alignment | Models; RMSE/MAE scatter and saved tradeoff table |
| Rolling-origin player, fixed axes, candidate bands and per-window scores | Backtests; previous/next windows, explicit evaluation dates, focused model, saved bands and error-over-origin chart |
| Interval candidates, widths and empirical calibration | Diagnostics / Intervals |
| Residuals by time, horizon, distribution, one-step ACF, outliers and heuristic tests | Diagnostics / Residuals |
| Year-start overlays, complete-cycle credibility and decomposition | Diagnostics / Seasonality; annual overlays for monthly data |
| Parent/child rollups, contributions, reconciliation methods, pre/post gaps and accuracy deltas | Diagnostics / Hierarchy; parent-scoped chartable tables |
| Scenarios, declared drivers, availability/leakage audits, feature gates, importance and CV deltas | Diagnostics / Drivers; paired scenario paths, evidence and review-only image previews |
| Vintages, forecast paths, locks, actual revisions, performance, adjustments, corrections and regimes | Diagnostics / Ledger; version selector and paged operating tables |
| FINN/BYO comparisons, scenarios, finance detail and comparability contracts | Diagnostics / Challengers; chartable saved evidence; Compare / External & BYO retains execution review |
| Consolidated feeder previews/downloads | Diagnostics / Feeders; complete registered CSV catalog and scoped exports |
| Control pane, pipeline, source/model specification, research and maintenance | Data, Pipeline, Research and existing reviewed operations |

Optional hierarchy/driver/challenger detail uses a common chartable evidence
explorer rather than claiming pixel-for-pixel Streamlit parity. Chart filters
are explicitly distinguished from table/export scope. Duplicate chart coordinates
are rejected rather than silently averaged. Missing optional artifacts remain
visible as explanatory empty states. Source files no longer crowd out analysis;
their complete catalog remains in the explorer and a collapsed source section.

Analytical presentation formats measurements to two decimals; Explore data shows
unrounded values. IDs, dates and counts retain
their meaning. Dimensionless diagnostics are not divided by display units.
Formatted copy is separate from raw copy and original-text CSV exports. Read-only
analysis retains immutable artifact hashes, series/model/origin filters and bounded
continuation; navigation cannot execute or adopt a forecast. Every chart preview
reports when it omits scoped rows. A real saved forecast can still be directional
and unfit for planning: provenance and readiness remain separate.

Time-series comparisons include saved actuals as a solid line. Forecast regions
are shaded gray independently of model visibility; the shading is a time window,
not an uncertainty claim. Backtests retain the full loaded date axis and actuals
trend across all window changes; only predictions and the shaded evaluation
window move. Later actuals remain explicitly separate from model inputs. The
value-scale lock controls only the Y axis. Other forecast views offer named
Full history / Focus forecast controls instead of unlabeled zoom sliders,
and backtest window navigation is discrete. Cached complete
evidence changes windows without a new read; paged reads retain the previous
coherent window with an explicit updating status until all replacement evidence
is ready. ECharts instances survive updates, including origin and appearance
changes. Detailed prediction/score tables load only when expanded.

Streamlit remains the compatibility viewer. Custom-script refresh requires another
review, rather than silently repeating executable code. Blend confirmation does not
accept normalized, reconciled or scenario-adjusted paths without matching deployed
evidence. Cloud publishing and scheduling are outside this local release.

Browser journeys include maintenance approvals, blocked locks and dialog focus
restoration. Native canvas action transport is checked separately; direct native
host pixel inspection is unavailable in this environment. Neither the browser
checks nor the action transport checks constitute accessibility certification.
