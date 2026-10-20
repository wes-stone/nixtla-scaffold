import { useEffect, useId, useRef, useState } from "react";
import type { RefObject } from "react";
import { text } from "./api";
import type { Row } from "./api";
import type { AnalysisProps } from "./Evidence";
import { EvidenceStatus, ModelPicker, ScaleControl, useEvidence } from "./Evidence";
import { Table, Timeline } from "./components";
import { scales } from "./format";
import { modelSelection, savedModel, tournamentWinner } from "./modelSelection";
import { EnsembleReview } from "./EnsembleReview";
import { paths } from "./analysis";
import { neutralJudgment } from "./judgment";
import { IntervalReview } from "./IntervalReview";
import { FinanceReview } from "./FinanceReview";
import { SeasonalForecast } from "./SeasonalForecast";

export function ForecastReview(props: AnalysisProps & { historyRows: Row[] }) {
  const { document: doc, view, update, inspect, dark, historyRows } = props;
  const [wide, setWide] = useState(false);
  const wideId = useId(), wideRef = useRef<HTMLElement>(null), wideButton = useRef<HTMLButtonElement>(null);
  const jumpToWide = () => {
    wideRef.current?.focus({ preventScroll: true });
    wideRef.current?.scrollIntoView({ block: "start" });
  };
  useEffect(() => { if (wide) jumpToWide(); }, [wide]);
  const selected = [...new Set(view.rows.map(row => text(row.model)).filter(Boolean))].join(", ") || "Recorded in model selection";
  const roster = useEvidence(doc, ["appendix/model_audit.csv", "appendix/forecast_long.csv"]);
  const winner = view.run.metadata.purpose === "judgment_scenario" ? "" : tournamentWinner(roster.data["appendix/model_audit.csv"]?.rows);
  const models = roster.data["appendix/forecast_long.csv"]?.models ?? view.models;
  const { focus, shown } = modelSelection(doc, models, savedModel(view.rows));
  const savedSelection = !doc.analysis.previewModel && doc.analysis.modelMode === "focus";
  const savedPaths = roster.data["appendix/forecast_long.csv"];
  const complete = savedPaths !== undefined && savedPaths.rows.length === savedPaths.total;
  const result = useEvidence(doc, savedSelection || complete ? [] : ["appendix/forecast_long.csv"], { models: shown });
  const rows = savedSelection ? view.rows : complete ? savedPaths.rows.filter(row => shown.includes(text(row.model)))
    : result.data["appendix/forecast_long.csv"]?.rows ?? [];
  const race = paths(rows);
  const bands = new Map(rows.filter(row => text(row.model) === focus).map(row => [text(row.ds), row]));
  const chartRows = race.rows.map(row => ({ ...row,
    yhat_lo_80: bands.get(text(row.ds))?.yhat_lo_80, yhat_hi_80: bands.get(text(row.ds))?.yhat_hi_80,
    yhat_lo_95: bands.get(text(row.ds))?.yhat_lo_95, yhat_hi_95: bands.get(text(row.ds))?.yhat_hi_95 }));
  const level = doc.intervalLevel;
  const seasonal = doc.analysis.forecastLayout === "seasonal";
  const columns = ["ds", "model", "yhat", ...(level ? [`yhat_lo_${level}`, `yhat_hi_${level}`] : []), "interval_status", "interval_method"].filter(column => rows.some(row => column in row));
  return <>
    <p className="forecast-provenance"><strong>{winner ? `Saved tournament winner: ${winner}.` : `Saved selected forecast: ${selected}.`}</strong> Model browsing is a display-only preview, never adoption.</p>
    <button onClick={() => update({ view: "Adjust", judgment: view.run.metadata.purpose === "judgment_scenario" ? undefined : { ...(doc.judgment ?? neutralJudgment()), model: savedSelection ? "" : focus } })}>Adjust {savedSelection ? "saved forecast" : focus} & save a version</button>
    <div className="model-workspace">
    <ModelPicker key={`${doc.id}:${doc.runId}`} {...props} models={models} audit={roster.data["appendix/model_audit.csv"]?.rows} forecasts={roster.data["appendix/forecast_long.csv"]?.rows}/>
    <div className="model-stage">
    <div className="analysis-controls"><ScaleControl {...props}/>
      <label>Prediction interval<select aria-label="Prediction interval" disabled={seasonal} value={level} onChange={event => update({ intervalLevel: Number(event.target.value) })}><option value={0}>Point forecast</option>{view.columns.filter(column => /^yhat_lo_\d+$/.test(column)).map(column => { const value = Number(column.split("_").at(-1)); return <option key={value} value={value}>Saved {value}% interval</option>; })}</select></label>
      <button ref={wideButton} aria-expanded={wide} aria-controls={wideId} onClick={() => wide ? jumpToWide() : setWide(true)}>Wide review table</button></div>
    <h3>{savedSelection ? "Saved selected forecast" : doc.analysis.modelMode === "compare" ? `${shown.length}-model comparison` : `Forecast · ${focus}`}</h3>
    <p className="footnote">{savedSelection ? "Includes recorded pipeline adjustments." : "Candidate paths from this saved run; no recalculation."} {seasonal ? "Seasonality shows point paths; saved intervals remain on the timeline." : level > 0 ? `Saved ${level}% band belongs only to ${focus}; model spread is not an interval.` : "Point forecasts shown."}</p>
    {!savedSelection && !complete && <EvidenceStatus result={result}/>}
    <div className="timeline-range" role="group" aria-label="Forecast chart view"><span>Chart view</span>
      <button aria-pressed={!seasonal} onClick={() => update({ analysis: { forecastLayout: "timeline" } })}>Timeline</button>
      <button aria-pressed={seasonal} onClick={() => update({ analysis: { forecastLayout: "seasonal" } })}>Seasonality</button>
    </div>
    {seasonal ? <SeasonalForecast {...props} future={rows} focusModel={focus}/> :
      <Timeline rows={chartRows} fields={race.fields} focusField={focus} historyRows={historyRows} intervalLevel={level} dark={dark} inspect={inspect} scale={scales[doc.analysis.scale]}/>}
    </div></div>
    <EvidenceStatus result={roster}/>
    <EnsembleReview {...props}/>
    <IntervalReview {...props}/>
    <FinanceReview {...props}/>
    <Table rows={rows} columns={columns} title="Forecast values" inspect={inspect} scale={scales[doc.analysis.scale]}/>
    <details><summary>All saved forecast metadata</summary><Table rows={rows} title="Full forecast evidence" inspect={inspect}/></details>
    {wide && <WideReview {...props} reviewId={wideId} reviewRef={wideRef} close={() => {
      setWide(false); wideButton.current?.focus({ preventScroll: true }); wideButton.current?.scrollIntoView({ block: "center" });
    }}/>}
  </>;
}
function WideReview(props: AnalysisProps & { reviewId: string; reviewRef: RefObject<HTMLElement | null>; close: () => void }) {
  const { document: doc, update, inspect } = props;
  const result = useEvidence(doc, ["appendix/history.csv", "forecast.csv"], { series: doc.analysis.summaryScope === "all" ? "" : doc.series });
  const combined = new Map<string, Row>();
  for (const [name, field, kind, count] of [["appendix/history.csv", "y", "Actual", doc.analysis.historyPeriods], ["forecast.csv", "yhat", "Forecast", doc.analysis.forecastPeriods]] as const) {
    const source = result.data[name]?.rows ?? [];
    const ids = [...new Set(source.map(row => text(row.unique_id)))];
    for (const id of ids) {
      const scoped = source.filter(row => text(row.unique_id) === id).sort((a, b) => text(a.ds).localeCompare(text(b.ds)));
      const selected = kind === "Actual" ? count ? scoped.slice(-count) : [] : scoped.slice(0, count);
      const types = kind === "Forecast" && doc.intervalLevel ? [field, `yhat_lo_${doc.intervalLevel}`, `yhat_hi_${doc.intervalLevel}`] : [field];
      for (const measure of types) {
        const key = `${id}:${kind}:${measure}`, row: Row = { unique_id: id, values: measure === field ? kind : measure.includes("_lo_") ? `${doc.intervalLevel}% lower` : `${doc.intervalLevel}% upper` };
        selected.forEach(value => { row[text(value.ds)] = value[measure]; });
        combined.set(key, row);
      }
    }
  }
  const rows = [...combined.values()], periods = [...new Set(rows.flatMap(row => Object.keys(row).filter(key => /^\d{4}-/.test(key))))].sort();
  return <section id={props.reviewId} ref={props.reviewRef} className="wide-review" tabIndex={-1} aria-label="Wide forecast review">
    <header className="section-heading"><h3>Wide forecast review</h3><button onClick={props.close}>Back to forecast</button></header>
    <p>Selected forecast only. Rows distinguish actuals, forecast and requested saved bounds. Copy displayed for two-decimal paste; copy raw for original numeric precision.</p>
    <div className="analysis-controls"><label>Review rows<select value={doc.analysis.summaryScope} onChange={event => update({ analysis: { summaryScope: event.target.value as "series" | "all" } })}><option value="series">Selected series</option><option value="all">All series</option></select></label>
      <label>Actual periods<input type="number" min={0} max={2000} value={doc.analysis.historyPeriods} onChange={event => update({ analysis: { historyPeriods: Math.max(0, Math.min(2000, Number(event.target.value))) } })}/></label>
      <label>Forecast periods<input type="number" min={1} max={2000} value={doc.analysis.forecastPeriods} onChange={event => update({ analysis: { forecastPeriods: Math.max(1, Math.min(2000, Number(event.target.value))) } })}/></label></div>
    <EvidenceStatus result={result}/><Table rows={rows} columns={["unique_id", "values", ...periods]} title="Wide copy/paste summary" inspect={inspect} scale={scales[doc.analysis.scale]}/>
  </section>;
}
