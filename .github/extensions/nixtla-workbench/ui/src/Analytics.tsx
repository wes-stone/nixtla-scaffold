import { useEffect, useMemo, useState } from "react";
import { call, failure, number, text } from "./api";
import type { Row } from "./api";
import { z } from "zod";
import { ErrorNotice, Table, Timeline } from "./components";
import { AnalysisPlot } from "./AnalysisPlot";
import { autocorrelation, histogram, paths, residuals } from "./analysis";
import { EvidenceStatus, EvidenceTable, ModelPicker, ScaleControl, useEvidence } from "./Evidence";
import type { AnalysisProps, Evidence } from "./Evidence";
import { chartNumber, measurementScale, scales } from "./format";
import { accuracyMetrics, compareAccuracy, metricLabel, modelSelection, savedModel, tournamentWinner } from "./modelSelection";
import { EnsembleReview } from "./EnsembleReview";
import { CalendarAdjuster } from "./CalendarAdjuster";
import { IntervalReview } from "./IntervalReview";
import { FinanceReview } from "./FinanceReview";
import { ConfigurationReview } from "./ConfigurationReview";
import { SeasonalForecast } from "./SeasonalForecast";

const A = "appendix/", B = `${A}backtest_long.csv`, F = `${A}forecast_long.csv`, H = `${A}history.csv`;
const metrics = accuracyMetrics;
function Metric({ document: doc, update }: AnalysisProps) {
  return <label>Diagnostic metric<select value={doc.analysis.metric} onChange={event => update({ analysis: { metric: event.target.value } })}>{metrics.map(metric => <option key={metric} value={metric}>{metricLabel(metric)}</option>)}</select></label>;
}
function Models(props: AnalysisProps) {
  const { document: doc, view, update, inspect, dark } = props;
  const names = [`${A}model_audit.csv`, `${A}model_pareto_frontier.csv`, F];
  const result = useEvidence(doc, names);
  const selection = modelSelection(doc, result.data[F]?.models ?? view.models, savedModel(view.rows));
  const savedPaths = result.data[F];
  const complete = savedPaths !== undefined && savedPaths.rows.length === savedPaths.total;
  const chart = useEvidence(doc, complete ? [] : [F], { models: selection.shown });
  const history = useEvidence(doc, [H], { tail: true });
  const historyRows = history.data[H]?.rows ?? [];
  const origin = historyRows.map(row => text(row.ds)).sort().at(-1);
  const audit = result.data[names[0]]?.rows ?? [], pareto = result.data[names[1]]?.rows ?? [];
  const families = [...new Set(audit.map(row => text(row.family)))];
  const filtered = audit.filter(row => !doc.analysis.family || row.family === doc.analysis.family);
  const sorted = [...filtered].sort((a, b) => compareAccuracy(a, b, doc.analysis.metric));
  const winner = view.run.metadata.purpose === "judgment_scenario" ? "" : tournamentWinner(audit);
  const comparable = sorted.filter(row => row.selection_eligible === true && row.family !== "ensemble")
    .map((row, index) => ({ ...row, rank: number(row[doc.analysis.metric]) === null ? null : index + 1, result: row.model === winner ? "Saved winner" : "" }));
  const diagnostic = sorted.filter(row => row.selection_eligible !== true || row.family === "ensemble");
  const scoreColumns = ["rank", "model", "result", "family", doc.analysis.metric, "rmse", "mae", "bias", "observations",
    "cv_windows", "cv_disjoint_windows", "comparison_support", "selection_eligible", "is_selected_model"]
    .filter((name, i, all) => all.indexOf(name) === i);
  const members = new Set(filtered.map(row => text(row.model)));
  const future = complete ? savedPaths.rows.filter(row => selection.shown.includes(text(row.model))) : chart.data[F]?.rows ?? [];
  const race = paths(future);
  const frontier = pareto.filter(row => members.has(text(row.model))).map(row => ({ ...row,
    frontier: row.is_pareto_optimal === true ? "Non-dominated" : row.pareto_status === "dominated_tradeoff" ? "Dominated" : "Not comparable" }));
  return <>
    <header className="section-heading"><div><h2>Model arena</h2><p>Explore one path, then build a small comparison. Every saved model remains in the tournament below.</p></div>
      <button onClick={() => update({ view: "Diagnostics", analysis: { section: "Configuration" } })}>Review configuration</button></header>
    <nav className="analysis-nav" aria-label="Model analyses"><button onClick={() => document.getElementById("model-leaderboard")?.scrollIntoView()}>Leaderboard</button><button onClick={() => document.getElementById("pareto-analysis")?.scrollIntoView()}>Pareto frontier</button><button onClick={() => document.getElementById("model-race")?.scrollIntoView()}>Forecast race</button></nav>
    <div className="model-workspace">
      <ModelPicker key={`${doc.id}:${doc.runId}`} {...props} models={result.data[F]?.models ?? view.models} audit={audit} forecasts={result.data[F]?.rows}/>
      <div id="model-race" className="model-stage"><h3>{doc.analysis.modelMode === "compare" ? `${selection.shown.length}-model comparison` : `Forecast · ${selection.focus}`}</h3>
        <EvidenceStatus result={history}/>{!complete && <EvidenceStatus result={chart}/>}
        {!!historyRows.length && <p className="footnote">{historyRows.length.toLocaleString()} saved actual observations through {origin?.slice(0, 10)}. The origin marker separates observed history from model forecasts.</p>}
        <Timeline rows={race.rows} historyRows={historyRows} fields={race.fields} focusField={selection.focus} origin={origin} dark={dark} inspect={inspect} scale={scales[doc.analysis.scale]}/>
        <p className="footnote">Paths are not ranked by their endpoint. Compare historical support in the tournament; missing backtests are not wins.</p>
      </div>
    </div>
    <EvidenceStatus result={result}/>
    <EnsembleReview {...props}/>
    <IntervalReview {...props}/>
    <FinanceReview {...props}/>
    <div className="analysis-controls"><Metric {...props}/><label>Model family<select value={doc.analysis.family} onChange={event => update({ analysis: { family: event.target.value } })}><option value="">All families</option>{families.map(family => <option key={family}>{family}</option>)}</select></label>
      <button onClick={() => inspect({ model_policy: view.spec.model_policy, installed_candidates: view.models, spec: view.spec })}>Inspect model policy</button></div>
    <div id="model-leaderboard"><Table title="Component tournament" rows={comparable} columns={scoreColumns} inspect={inspect} scale={scales[doc.analysis.scale]}
      rowActions={row => <button className="link" aria-label={`Explore ${text(row.model)}`} onClick={() => { update({ analysis: { previewModel: text(row.model), modelMode: "focus" } }); document.getElementById("model-race")?.scrollIntoView(); }}>Focus</button>}/></div>
    <p className="boundary-note">These are retrospective scores, not independent confirmation. Overlapping origins are not independent windows. Older versions without comparison metadata have unverified support.</p>
    {!!diagnostic.length && <details><summary>Diagnostic candidates and separate ensemble cohorts</summary>
      <p>Incomplete candidates cannot win the component tournament. Compare an ensemble with its rescored reference on the same later periods, not with full-history component scores.</p>
      <Table title="Separate-cohort diagnostics" rows={diagnostic} columns={scoreColumns} inspect={inspect} scale={scales[doc.analysis.scale]}
        rowActions={row => <button className="link" aria-label={`Explore ${text(row.model)}`} onClick={() => { update({ analysis: { previewModel: text(row.model), modelMode: "focus" } }); document.getElementById("model-race")?.scrollIntoView(); }}>Focus</button>}/>
    </details>}
    <details><summary>Fitted recipes and failed attempts</summary>
      <p>Each row records a future fit or historical origin. Successful fitting alone does not establish selection eligibility. Constructor settings and full calibration receipts are in the saved candidate evidence.</p>
      <EvidenceTable document={doc} name="audit/candidate_fits.csv" title="Actual fitted candidates" inspect={inspect}/>
    </details>
    <div id="pareto-analysis"><AnalysisPlot title="Pareto frontier · RMSE versus MAE" rows={frontier.filter(row => row.frontier !== "Not comparable")} x="rmse" fields={["mae"]} group="frontier" kind="scatter" dark={dark} scale={scales[doc.analysis.scale]}/></div>
    <p className="footnote">Saved non-dominated flags retain the pipeline's exact metric scope. This tradeoff view is advisory, not a new winner calculation.</p>
    <Table title="Pareto tradeoffs" rows={frontier} columns={["model", "frontier", "selection_alignment", "metrics_considered", "excluded_metrics", "rmse", "mae", "observations"]} inspect={inspect} scale={scales[doc.analysis.scale]}/>
    <SavedBrowser {...props} section="Models"/>
  </>;
}
type ReplayFrame = {
  scope: string; key: string; cutoff: string; rows: Row[]; source: Evidence;
  history: Evidence; metrics: Evidence; models?: string[];
};
function Backtests(props: AnalysisProps) {
  const { document: doc, update, inspect, dark, view } = props;
  const [refresh, setRefresh] = useState(0), [detailsOpen, setDetailsOpen] = useState(false);
  const [displayed, setDisplayed] = useState<ReplayFrame>();
  const roster = useEvidence(doc, [`${A}model_audit.csv`, F]);
  const models = roster.data[F]?.models ?? view.models;
  const filters = { models: modelSelection(doc, models, savedModel(view.rows)).shown };
  const scope = JSON.stringify([doc.runId, doc.series]), modelsKey = JSON.stringify(filters);
  const current = displayed?.scope === scope ? displayed : undefined;
  const result = useEvidence(doc, [B, `${A}model_window_metrics.csv`], filters, refresh);
  const history = useEvidence(doc, [H], { tail: true }, refresh);
  const saved = result.data[B], historical = history.data[H], metrics = result.data[`${A}model_window_metrics.csv`];
  const cutoffs = saved?.cutoffs ?? current?.source.cutoffs ?? [];
  const cutoff = cutoffs.includes(doc.analysis.cutoff) ? doc.analysis.cutoff : cutoffs.at(-1) ?? "";
  const key = JSON.stringify([scope, modelsKey, cutoff]);
  const complete = saved !== undefined && saved.rows.length === saved.total;
  const originResult = useEvidence(doc, saved && !complete && cutoff ? [B] : [],
    { ...filters, cutoff: cutoff || undefined }, refresh);
  const origin = originResult.data[B];
  const readyRows = useMemo(() => !saved ? undefined : complete
    ? saved.rows.filter(row => text(row.cutoff) === cutoff) : origin?.rows, [saved, complete, cutoff, origin]);
  useEffect(() => {
    if (saved && historical && metrics && readyRows) setDisplayed({
      scope, key, cutoff, rows: readyRows, source: saved, history: historical, metrics,
      models: filters.models,
    });
  }, [scope, key, cutoff, readyRows, saved, historical, metrics]);
  const error = result.error || history.error || originResult.error;
  const pending = current?.key !== key || result.loading || history.loading || (!complete && originResult.loading);
  const retry = <button onClick={() => setRefresh(value => value + 1)}>Reload saved backtests</button>;
  if (!error && saved && !saved.cutoffs.length) return <section>
    <h2>Rolling backtests</h2>
    <p className="empty-inline">No saved backtest windows are available for this series and model selection.</p>
    <ModelPicker {...props} models={models} audit={roster.data[`${A}model_audit.csv`]?.rows} forecasts={roster.data[F]?.rows}/>
    <EvidenceStatus result={result}/>
  </section>;
  if (!current) return <section className="replay-loading">{error ? <><ErrorNotice error={error}/>{retry}</> :
    <p role="status">Loading saved backtest windows and actuals...</p>}</section>;
  const race = paths(current.rows);
  const actuals = new Map(current.rows.map(row => [text(row.ds), row.y_actual]));
  const focus = modelSelection(doc, models, savedModel(view.rows)).focus;
  const focusAvailable = race.fields.includes(focus);
  const bands = new Map(current.rows.filter(row => text(row.model) === focus).map(row => [text(row.ds), row]));
  const replay = race.rows.map(row => ({ ...row, cutoff: current.cutoff, y: actuals.get(text(row.ds)),
    yhat_lo_80: bands.get(text(row.ds))?.yhat_lo_80, yhat_hi_80: bands.get(text(row.ds))?.yhat_hi_80,
    yhat_lo_95: bands.get(text(row.ds))?.yhat_lo_95, yhat_hi_95: bands.get(text(row.ds))?.yhat_hi_95 }));
  const dates = [...new Set([...current.history.rows, ...current.source.rows, ...current.rows].map(row => text(row.ds)).concat(current.source.cutoffs))].sort();
  const finite = [...Object.entries(current.source.bounds).filter(([key]) => ["yhat", "y_actual", "yhat_lo_80", "yhat_hi_80", "yhat_lo_95", "yhat_hi_95"].includes(key)).flatMap(([, range]) => [range.min, range.max]),
    ...current.history.rows.map(row => number(row.y))].filter((n): n is number => n !== null);
  const bounds = finite.length ? { min: Math.min(...finite), max: Math.max(...finite) } : undefined;
  const index = cutoffs.indexOf(cutoff), periods = race.rows.map(row => text(row.ds)).sort();
  const move = (position: number) => update({ analysis: { cutoff: cutoffs[position] } });
  const showEvidence = !pending && !error;
  return <>
    <header className="section-heading"><div><h2>Rolling backtests</h2><p>Choose when the forecast was made, then compare its predictions with what actually happened.</p></div></header>
    <div className="origin-player" role="group" aria-label="Backtest window navigation">
      <button disabled={pending || index <= 0} onClick={() => move(index - 1)}>Previous window</button>
      <label>Forecast made after<select aria-label="Backtest origin" disabled={pending && !error} value={cutoff} onChange={event => update({ analysis: { cutoff: event.target.value } })}>
        {cutoffs.map((item, i) => <option key={item} value={item}>Window {i + 1} · {item.slice(0, 10)}</option>)}</select></label>
      <button disabled={pending || index < 0 || index >= cutoffs.length - 1} onClick={() => move(index + 1)}>Next window</button>
      <span>Window {index + 1} of {cutoffs.length}</span>
    </div>
    {error && <><ErrorNotice error={error}/>{retry}</>}
    {pending && !error && <p className="replay-updating" role="status">Loading the selected window. The previous window remains visible until its replacement is ready.</p>}
    <div className="window-summary"><strong>Observed through {current.cutoff.slice(0, 10)}</strong>
      {!!periods.length && <span>Forecast evaluated {periods[0].slice(0, 10)} – {periods.at(-1)?.slice(0, 10)} · {periods.length} periods</span>}</div>
    <div className="model-workspace">
    <ModelPicker key={`${doc.id}:${doc.runId}`} {...props} models={models} audit={roster.data[`${A}model_audit.csv`]?.rows} forecasts={roster.data[F]?.rows}/>
    <div className="model-stage">
    <div className="analysis-controls replay-options">
      <label>Saved prediction interval<select aria-label="Backtest interval" value={doc.intervalLevel} onChange={event => update({ intervalLevel: Number(event.target.value) })}><option value={0}>No bands</option><option value={80}>80%</option><option value={95}>95%</option></select></label>
      <label className="inline"><input type="checkbox" checked={doc.analysis.fixedAxes} onChange={event => update({ analysis: { fixedAxes: event.target.checked } })}/>Keep value scale fixed</label>
    </div>
    <div id="backtest-replay" aria-busy={pending}>
      <Timeline rows={replay} historyRows={current.history.rows} fields={["y", ...race.fields]} intervalLevel={focusAvailable ? doc.intervalLevel : 0} focusField={focusAvailable ? focus : undefined}
        dark={dark} inspect={inspect} scale={scales[doc.analysis.scale]} origin={current.cutoff} fullHistoryOnly
        domainDates={dates} bounds={doc.analysis.fixedAxes ? bounds : undefined}/>
    </div>
    </div></div>
    {!pending && !focusAvailable && <p className="notice">No saved backtest predictions for {focus} in this window. Other comparison paths remain visible; no model has been substituted.</p>}
    <p className="footnote">Gray marks the selected forecast window, not a confidence interval. The full actuals trend stays visible; values after the origin were observed later and were not model inputs. The focused model is emphasized; other models remain available.</p>
    {doc.intervalLevel > 0 && !current.rows.some(row => text(row.model) === focus && number(row[`yhat_lo_${doc.intervalLevel}`]) !== null) &&
      <p className="notice">No saved {doc.intervalLevel}% bounds exist for {focus} in this window. No interval has been synthesized.</p>}
    {showEvidence && <><EvidenceStatus result={result}/><EvidenceStatus result={history}/><EvidenceStatus result={originResult}/></>}
    <Metric {...props}/><AnalysisPlot title="Model error over forecast origins" rows={current.metrics.rows} x="cutoff" fields={[doc.analysis.metric]} group="model" dark={dark} scale={["rmse", "mae"].includes(doc.analysis.metric) ? scales[doc.analysis.scale] : 1}/>
    <details onToggle={event => setDetailsOpen(event.currentTarget.open)}><summary>Detailed backtest evidence and exports</summary>
      {detailsOpen && <>
        <EvidenceTable document={doc} name={B} title="Selected-origin predictions and saved intervals" cutoff={current.cutoff || undefined} models={current.models} inspect={inspect}/>
        <EvidenceTable document={doc} name={`${A}model_window_metrics.csv`} title="Per-origin performance and support" models={current.models} inspect={inspect}/>
      </>}
    </details>
  </>;
}
function Intervals(props: AnalysisProps) {
  const { document: doc, update, inspect, dark } = props;
  const result = useEvidence(doc, [F, `${A}interval_diagnostics.csv`]);
  const rows = result.data[F]?.rows ?? [];
  const models = [...new Set(rows.filter(row => Object.keys(row).some(key => /^yhat_lo_/.test(key) && number(row[key]) !== null)).map(row => text(row.model)))];
  const model = models.includes(doc.analysis.previewModel) ? doc.analysis.previewModel : models[0] ?? "";
  const selected = rows.filter(row => text(row.model) === model);
  const level = doc.intervalLevel || 80;
  const widths = selected.map(row => ({ ds: row.ds, width: number(row[`yhat_hi_${level}`]) !== null && number(row[`yhat_lo_${level}`]) !== null ? Number(row[`yhat_hi_${level}`]) - Number(row[`yhat_lo_${level}`]) : null }));
  return <>
    <h2>Prediction intervals</h2><p>Saved model bands, future width and empirical coverage. Model spread is not a prediction interval.</p>
    <EvidenceStatus result={result}/>
    <div className="analysis-controls"><label>Interval model<select value={model} onChange={event => update({ analysis: { previewModel: event.target.value } })}>{models.map(item => <option key={item}>{item}</option>)}</select></label>
      <label>Interval level<select value={level} onChange={event => update({ intervalLevel: Number(event.target.value) })}><option value={80}>80%</option><option value={95}>95%</option></select></label></div>
    <Timeline rows={selected} intervalLevel={level} dark={dark} inspect={inspect} scale={scales[doc.analysis.scale]}/>
    <Table title="Interval method and calibration status" rows={selected} columns={["ds", "model", "yhat", `yhat_lo_${level}`, `yhat_hi_${level}`, "interval_status", "interval_method", "interval_evidence"]} inspect={inspect} scale={scales[doc.analysis.scale]}/>
    <AnalysisPlot title="Future interval width" rows={widths} x="ds" fields={["width"]} dark={dark} scale={scales[doc.analysis.scale]}/>
    <p className="boundary-note">Adjusted or future-only bands must not be described as calibrated coverage. Review observations and saved coverage status below; a requested 80% band is not proof of 80% empirical coverage.</p>
    <EvidenceTable document={doc} name={`${A}interval_diagnostics.csv`} title="Nominal versus empirical coverage by model and horizon" inspect={inspect}/>
  </>;
}
function Residuals(props: AnalysisProps) {
  const { document: doc, update, view, dark, inspect } = props;
  const model = doc.analysis.previewModel || view.models[0] || "";
  const result = useEvidence(doc, [B, `${A}residual_diagnostics.csv`], { models: [model] });
  const rows = residuals(result.data[B]?.rows ?? [], model), values = rows.map(row => number(row.residual)).filter((n): n is number => n !== null);
  const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  const std = values.length > 1 ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)) : 0;
  const outliers = rows.filter(row => std && number(row.residual) !== null && Math.abs((Number(row.residual) - mean) / std) >= 3);
  return <>
    <h2>Residual audit</h2><p>Residuals are actual minus predicted. Positive residuals mean underprediction. Tests are diagnostics, not certificates of correctness.</p>
    <label>Residual model<select value={model} onChange={event => update({ analysis: { previewModel: event.target.value } })}>{(result.data[B]?.models ?? view.models).map(item => <option key={item}>{item}</option>)}</select></label>
    <EvidenceStatus result={result}/>
    <AnalysisPlot title="Residuals over time, separated by origin" rows={rows} x="ds" fields={["residual"]} group="cutoff" dark={dark} scale={scales[doc.analysis.scale]}/>
    <AnalysisPlot title="Error by forecast horizon" rows={result.data[`${A}residual_diagnostics.csv`]?.rows ?? []} x="horizon_step" fields={["rmse", "mae"]} dark={dark} scale={scales[doc.analysis.scale]}/>
    <div className="analysis-grid"><AnalysisPlot title="Residual distribution" rows={histogram(rows, "residual").map(row => ({ ...row, bin: `${chartNumber(row.lower, scales[doc.analysis.scale])} to ${chartNumber(row.upper, scales[doc.analysis.scale])}` }))} x="bin" fields={["count"]} kind="bar" dark={dark}/>
      <AnalysisPlot title="One-step residual autocorrelation" rows={autocorrelation(rows)} x="lag" fields={["correlation", "heuristic_95"]} kind="bar" dark={dark}/></div>
    {autocorrelation(rows).length === 0 && <p className="notice">Insufficient nonconstant one-step residuals for autocorrelation (at least four required). Overlapping multi-step errors are not substituted.</p>}
    <Table title="Outliers · at least three sample standard deviations" rows={outliers} inspect={inspect} scale={scales[doc.analysis.scale]}/>
    <EvidenceTable document={doc} name={`${A}residual_tests.csv`} title="Saved bias, outlier, white-noise and structural-break tests" models={[model]} inspect={inspect}/>
  </>;
}
function Seasonality(props: AnalysisProps) {
  const { document: doc, dark, inspect } = props;
  const result = useEvidence(doc, ["forecast.csv", "audit/seasonality_decomposition.csv"]);
  const future = result.data["forecast.csv"]?.rows ?? [];
  const monthly = ["MS", "ME", "M"].includes(text(props.view.spec.freq));
  return <>
    <h2>Seasonality</h2><p>Review calendar exposure separately from learned annual patterns.</p>
    {monthly ? <CalendarAdjuster key={`${doc.runId}:${doc.series}`} {...props}/> : <p className="notice">Calendar adjustment requires complete monthly flow totals. This run keeps its original frequency; daily or fiscal periods are not silently converted.</p>}
    <h3>Saved seasonal patterns</h3><p>The model's original seasonality evidence remains unchanged.</p><EvidenceStatus result={result}/>
    <SeasonalForecast {...props} future={future} focusModel={savedModel(future)}/>
    <AnalysisPlot title="Observed, trend, seasonal and remainder" rows={result.data["audit/seasonality_decomposition.csv"]?.rows ?? []} x="ds" fields={["observed", "trend", "seasonal", "remainder"]} dark={dark} scale={scales[doc.analysis.scale]}/>
    <EvidenceTable document={doc} name="audit/seasonality_summary.csv" title="Seasonality credibility and complete cycles" inspect={inspect}/>
    <SavedBrowser {...props} section="Seasonality"/>
  </>;
}
const rules: Record<string, RegExp> = {
  Models: /model_audit|model_selection|model_weights|model_win_rates|model_tradeoff|target_transform|model_explainability|feature_selection|candidate_|ensemble_comparisons/,
  Seasonality: /seasonality/, Hierarchy: /hierarchy/,
  Drivers: /scenario|regressor|driver|feature_selection|model_explainability/,
  Challengers: /finn|byo_model|external_|challenger|custom_model/,
  Feeders: /\.csv$/,
};
const descriptions: Record<string, string> = {
  Hierarchy: "Parent/child paths, contributions, bottom-up/top-down reconciliation and pre/post coherence. Coherence can improve while node accuracy worsens.",
  Drivers: "Post-model scenario assumptions stay separate from baseline yhat. Regressor availability, leakage gates, feature importance and CV deltas show what actually entered training.",
  Challengers: "FINN and finance-owned BYO outputs remain advisory. Future alignment is directional; only exact matched cutoff support permits performance comparisons. Browse scenarios, bridge/detail tables and comparability receipts without running external code.",
  Feeders: "Every saved CSV is available with scoped pagination, chart controls and full-precision export. This is the complete feeder catalog, not just the selected forecast.",
};
export function SavedBrowser(props: AnalysisProps & { section: string }) {
  const { document: doc, view, update, inspect, section } = props;
  const files = view.artifacts.filter(name => /\.csv$/.test(name) && rules[section].test(name));
  const name = files.includes(doc.analysis.artifact) ? doc.analysis.artifact : files[0] ?? "";
  const receipts = view.artifacts.filter(name => /\.(json|md)$/.test(name) && rules[section].test(name));
  const [error, setError] = useState("");
  return <section className="saved-browser">
    <h3>{section === "Feeders" ? "Consolidated feeder files" : `${section} evidence explorer`}</h3>
    {descriptions[section] && <p>{descriptions[section]}</p>}
    {!files.length ? <p className="empty-inline">No {section.toLowerCase()} tables were produced in this saved version. This is absent evidence, not a hidden feature. No model or external provider is run by opening this view.</p> :
      <><label>Saved analytical table<select aria-label={`${section} evidence table`} value={name} onChange={event => update({ analysis: { artifact: event.target.value } })}>{files.map(file => <option key={file}>{file}</option>)}</select></label>
        <EvidenceBrowser key={`${doc.runId}:${doc.series}:${name}`} {...props} name={name}/></>}
    {!!receipts.length && <details><summary>Contracts, manifests and receipts</summary><div className="artifact-list">{receipts.map(name => <button key={name} onClick={() => {
      setError(""); void call("artifact", { runId: doc.runId, name }, z.object({ name: z.string(), text: z.string(), sha256: z.string() })).then(inspect).catch(error => setError(failure(error)));
    }}>{name}</button>)}</div></details>}
    {section === "Drivers" && <Images {...props}/>}
    {error && <ErrorNotice error={error}/>}
  </section>;
}
function EvidenceBrowser(props: AnalysisProps & { name: string }) {
  const { document: doc, name, dark, inspect } = props;
  const [scope, setScope] = useState(doc.series), [seriesColumn, setSeriesColumn] = useState("");
  const [field, setField] = useState(""), [value, setValue] = useState(""), [x, setX] = useState(""), [y, setY] = useState<string[]>([]);
  const result = useEvidence(doc, [name], { series: scope, ...(seriesColumn ? { seriesColumn } : {}) });
  const data = result.data[name], rows = data?.rows ?? [], columns = data?.columns ?? [];
  const numbers = columns.filter(column => rows.some(row => typeof row[column] === "number"));
  const xAxis = x || ["ds", "cutoff", "horizon_step", "feature", "regressor", "driver", "child_unique_id", "model", "parent_unique_id"].find(column => columns.includes(column)) || columns[0] || "";
  const defaultY = [["yhat", "yhat_scenario"], ["scaffold_yhat", "external_yhat"]].find(pair => pair.every(column => numbers.includes(column))) ??
    [["yhat", "parent_yhat", "scaffold_yhat", "rmse", "importance", "gap", "gap_pct"].find(column => numbers.includes(column)) || numbers[0] || ""];
  const yAxes = y.length ? y : defaultY;
  const filtered = field && value ? rows.filter(row => text(row[field]) === value) : rows;
  const groupColumns = ["model", "external_model", "scenario_name", "method", "applied_method", "child_unique_id", "unique_id"].filter(column => columns.includes(column) && column !== xAxis);
  const chart = filtered.map(row => ({ ...row, trace: groupColumns.map(column => text(row[column])).join(" · ") }));
  return <>
    <EvidenceStatus result={result}/>
    <div className="analysis-controls">
      <label>Table scope<select value={scope ? "series" : "all"} onChange={event => { setScope(event.target.value === "series" ? doc.series : ""); setValue(""); }}><option value="series">Selected series</option><option value="all">Entire saved run</option></select></label>
      <label>Series key<select value={seriesColumn} onChange={event => setSeriesColumn(event.target.value)}><option value="">Automatic (series / parent)</option>{columns.filter(column => /unique_id|parent|child/.test(column)).map(column => <option key={column}>{column}</option>)}</select></label>
      <label>Chart filter<select value={field} onChange={event => { setField(event.target.value); setValue(""); }}><option value="">None</option>{columns.map(column => <option key={column}>{column}</option>)}</select></label>
      {field && <label>Filter value<select value={value} onChange={event => setValue(event.target.value)}><option value="">All values</option>{[...new Set(rows.map(row => text(row[field])))].map(value => <option key={value}>{value}</option>)}</select></label>}
      <label>Chart X<select value={xAxis} onChange={event => setX(event.target.value)}>{columns.map(column => <option key={column}>{column}</option>)}</select></label>
      <label>Chart Y<select multiple size={3} value={yAxes} onChange={event => setY(Array.from(event.target.selectedOptions, option => option.value))}>{numbers.map(column => <option key={column}>{column}</option>)}</select></label>
    </div>
    {!!numbers.length && <AnalysisPlot title="Saved evidence chart" rows={chart} x={xAxis} fields={yAxes} group="trace" dark={dark} scale={yAxes.every(column => measurementScale(column, scales[doc.analysis.scale]) === scales[doc.analysis.scale]) ? scales[doc.analysis.scale] : 1}/>}
    {field && value && <p className="footnote">Chart filter applies to the bounded preview only. The table and export below retain the full selected series scope.</p>}
    <EvidenceTable document={doc} name={name} title={name.split("/").at(-1)?.replace(".csv", "").replaceAll("_", " ") ?? name} scope={scope} seriesColumn={seriesColumn || undefined} inspect={inspect}/>
  </>;
}
function Images({ document: doc, view }: AnalysisProps) {
  const files = view.artifacts.filter(name => /\.(png|jpg|jpeg)$/i.test(name));
  const [image, setImage] = useState<{ name: string; mediaType: string; base64: string }>(), [error, setError] = useState("");
  return <details><summary>Supplied visual evidence · {files.length} images</summary><p>Review-only images do not prove that a regressor entered the model.</p>
    {files.map(name => <button key={name} onClick={() => { setImage(undefined); setError(""); void call("evidence_image", { runId: doc.runId, name }, z.object({ name: z.string(), mediaType: z.string(), base64: z.string() })).then(setImage).catch(error => setError(failure(error))); }}>{name}</button>)}
    {error && <ErrorNotice error={error}/>}
    {image && <img className="evidence-image" src={`data:${image.mediaType};base64,${image.base64}`} alt={`Supplied review evidence: ${image.name}`}/>}
  </details>;
}
export function Analytics(props: AnalysisProps) {
  const { document: doc, update } = props;
  const section = doc.analysis.section;
  return <div className="analytics"><div className="analysis-controls"><ScaleControl {...props}/><p className="footnote">Read-only saved evidence · two-decimal presentation · full-precision exports</p></div>
    {doc.view === "Models" ? <Models {...props}/> : doc.view === "Backtests" ? <Backtests {...props}/> : <>
      <nav className="analysis-nav" aria-label="Diagnostic section">{["Configuration", "Intervals", "Residuals", "Seasonality", "Hierarchy", "Drivers", "Ledger", "Challengers", "Feeders"].map(item => <button key={item} aria-pressed={section === item} onClick={() => update({ analysis: { section: item as typeof section } })}>{item}</button>)}</nav>
      {section === "Configuration" ? <ConfigurationReview {...props}/> : section === "Intervals" ? <Intervals {...props}/> : section === "Residuals" ? <Residuals {...props}/> : section === "Seasonality" ? <Seasonality {...props}/> : section === "Ledger" ? <Ledger {...props}/> : <SavedBrowser {...props} section={section}/>}
    </>}
  </div>;
}
function Ledger(props: AnalysisProps) {
  const { document: doc, inspect, dark } = props;
  const schema = z.object({ rows: z.array(z.record(z.unknown())), total: z.number(), nextOffset: z.number().nullable() });
  const [kind, setKind] = useState("snapshot"), [result, setResult] = useState<z.infer<typeof schema>>(), [error, setError] = useState("");
  const [offset, setOffset] = useState(0), [versions, setVersions] = useState<Row[]>([]), [version, setVersion] = useState("");
  const versioned = ["snapshot", "metrics", "performance_summary", "performance", "locks", "versions", "source_refreshes"].includes(kind);
  useEffect(() => {
    const controller = new AbortController();
    setResult(undefined); setError("");
    void Promise.all([
      call("ledger_view", { runId: doc.runId, series: doc.series, kind, offset, limit: 500, ...(versioned && version ? { versions: [version] } : {}) }, schema, controller.signal),
      call("ledger_view", { runId: doc.runId, kind: "versions", limit: 2000 }, schema, controller.signal),
    ]).then(([result, versions]) => { if (!controller.signal.aborted) { setResult(result); setVersions(versions.rows); } })
      .catch(error => { if (!controller.signal.aborted) setError(failure(error)); });
    return () => controller.abort();
  }, [doc.runId, doc.series, kind, offset, version]);
  const rows = result?.rows ?? [];
  const fields = ["y", "yhat", "y_actual", "raw_y", "corrected_y", "normalized_y"].filter(field => rows.some(row => field in row));
  const chartRows = rows.filter(row => !("is_selected_model" in row) || row.is_selected_model === true || row.is_selected_model === 1);
  return <><h2>Forecast ledger</h2><p>Actual revisions, vintages, performance and official locks. Loading evidence does not create a lock or rewrite a vintage.</p>
    <div className="analysis-controls"><label>Ledger table<select value={kind} onChange={event => { setKind(event.target.value); setOffset(0); }}>{["snapshot", "versions", "metrics", "performance", "performance_summary", "actuals", "locks", "adjustments", "corrected_actuals", "regimes", "deltas", "source_refreshes"].map(item => <option key={item}>{item}</option>)}</select></label>
      {versioned && <label>Forecast vintage<select value={version} onChange={event => { setVersion(event.target.value); setOffset(0); }}><option value="">All vintages for this forecast</option>{versions.map(row => <option key={text(row.forecast_version_id)} value={text(row.forecast_version_id)}>{text(row.version_label)} · {text(row.forecast_origin)}</option>)}</select></label>}</div>
    {error && <ErrorNotice error={error}/>}
    {!result && !error && <p role="status">Reading forecast ledger...</p>}
    {rows.some(row => "ds" in row) && <AnalysisPlot title="Ledger paths over time" rows={chartRows} x="ds" fields={fields} group={kind === "actuals" ? "actual_revision_id" : "forecast_version_id"} dark={dark} scale={scales[doc.analysis.scale]}/>}
    <Table rows={rows} title="Ledger evidence · bounded to 500 rows" inspect={inspect} scale={scales[doc.analysis.scale]}/>
    {result && <div className="table-footer"><span>{rows.length} of {result.total} rows · chart reflects this page</span><button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 500))}>Previous ledger page</button><button disabled={result.nextOffset === null} onClick={() => setOffset(result.nextOffset ?? 0)}>Next ledger page</button></div>}
    <p className="footnote">Use Compare → Version changes for matching-period vintage deltas and Pipeline for reviewed actual imports, refresh and lock actions.</p>
  </>;
}
