import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import type { Row } from "./api";
import { failure, text } from "./api";
import { EvidenceStatus, useEvidence } from "./Evidence";
import type { AnalysisProps } from "./Evidence";
import { ErrorNotice, Table } from "./components";
import { scales } from "./format";
import { buildSeasonalOverlay, monthNames, seasonalChart, seasonalKind } from "./seasonal";
import type { SeasonalOverlay } from "./seasonal";

export function SeasonalForecast({ future, focusModel = "", ...props }: AnalysisProps & { future: Row[]; focusModel?: string }) {
  const { document: doc, view, update, dark, inspect } = props;
  const [refresh, setRefresh] = useState(0), [details, setDetails] = useState(false);
  const rawHistory = view.artifacts.some(name => ["appendix/observed_history.csv", "observed_history.csv"].includes(name));
  const historyName = rawHistory ? "appendix/observed_history.csv" : "appendix/history.csv";
  const result = useEvidence(doc, doc.series ? [historyName] : [], { tail: true }, refresh);
  const history = result.data[historyName];
  const kind = seasonalKind(text(view.spec.freq));
  const built = useMemo(() => {
    if (!history?.available) return {};
    try {
      return { overlay: buildSeasonalOverlay({
        history: history.rows, future, frequency: text(view.spec.freq),
        alignment: doc.analysis.seasonalAlignment, yearStart: doc.analysis.yearStart,
        focusPeriod: doc.analysis.seasonalFocus, periodCount: doc.analysis.seasonalPeriods,
        includeActuals: doc.analysis.includeActuals,
      }) };
    } catch (error) { return { error: failure(error) }; }
  }, [history, future, view.spec.freq, doc.analysis.seasonalAlignment, doc.analysis.yearStart,
    doc.analysis.seasonalFocus, doc.analysis.seasonalPeriods, doc.analysis.includeActuals]);
  const overlay = built.overlay;
  if (!doc.series) return <p className="notice">Select one series to compare its seasonal periods.</p>;
  return <section className="seasonal-forecast" aria-label="Seasonal forecast">
    <header className="section-heading"><h3>{overlay?.title ?? "Seasonal forecast"}</h3>
      <button className="quiet" onClick={() => setRefresh(value => value + 1)}>Reload seasonal evidence</button></header>
    <p className="footnote">Overlay saved periods, not seasonal averages. Actuals are solid; forecasts continue as dashed paths. No resampling, refit or source-cutoff changes.</p>
    <EvidenceStatus result={result}/>
    {!rawHistory && <p className="notice">This version has no observed-only history artifact. Showing saved history; it may include recorded preprocessing or repairs.</p>}
    {built.error && <ErrorNotice error={built.error}/>}
    {overlay && <>
      <div className="analysis-controls seasonal-controls">
        <label>Focus {kind === "daily" ? "month" : "year"}<select aria-label="Seasonal focus period" value={overlay.focus}
          onChange={event => update({ analysis: { seasonalFocus: event.target.value } })}>
          {!overlay.periods.length && <option value="">No saved periods</option>}
          {overlay.periods.map(period => <option key={period.key} value={period.key}>{period.label}</option>)}
        </select></label>
        <label>Compare<select aria-label="Seasonal comparison periods" value={doc.analysis.seasonalPeriods}
          onChange={event => {
            const count = ([1, 3, 6, 12] as const).find(value => String(value) === event.target.value);
            if (count !== undefined) update({ analysis: { seasonalPeriods: count } });
          }}>
          {[1, 3, 6, 12].map(count => <option key={count} value={count}>{count === 1 ? "Focus only" : `Focus + ${count - 1} prior ${kind === "daily" ? "months" : "years"}`}</option>)}
        </select></label>
        {kind === "daily" ? <label>Align by<select aria-label="Seasonal alignment" value={doc.analysis.seasonalAlignment}
          onChange={event => update({ analysis: { seasonalAlignment: event.target.value === "date" ? "date" : "weekday" } })}>
          <option value="weekday">Weekday · Monday aligned</option><option value="date">Calendar day of month</option>
        </select></label> : kind === "monthly" && <label>Year starts in<select value={doc.analysis.yearStart}
          onChange={event => update({ analysis: { yearStart: Number(event.target.value), seasonalFocus: "" } })}>
          {monthNames.map((month, i) => <option key={month} value={i + 1}>{month}</option>)}
        </select></label>}
        <label className="inline"><input type="checkbox" checked={doc.analysis.includeActuals}
          onChange={event => update({ analysis: { includeActuals: event.target.checked } })}/>Show actuals</label>
      </div>
      {overlay.missingPeriods.length > 0 && <p className="footnote">No saved values in this scope for {overlay.missingPeriods.join(", ")}.</p>}
      {overlay.lines.length ? <SeasonalPlot overlay={overlay} dark={dark} scale={scales[doc.analysis.scale]} focusModel={focusModel}/> :
        <p className="empty-inline">No saved paths for these periods. Choose another focus period or show actuals.</p>}
      <p className="boundary-note">Original dates and values are unchanged. Blanks remain gaps, including dates beyond the saved forecast horizon. Point paths only; use Timeline for saved prediction intervals. Source settlement is not certified by this view.</p>
      <details open={details} onToggle={event => setDetails(event.currentTarget.open)}>
        <summary>Exact seasonal values</summary>
        {details && <Table title="Seasonal dates and values" rows={overlay.points.map(point => ({
          ds: point.date, comparison_period: point.period, aligned_slot: point.slot,
          phase: point.phase, model: point.model, actual: point.phase === "Actual" ? point.value : null,
          forecast: point.phase === "Forecast" ? point.value : null,
        }))} columnLabels={{ comparison_period: "Comparison period", aligned_slot: "Aligned slot (zero-based)" }}
          inspect={row => inspect(overlay.points.find(point => point.date === row.ds && point.phase === row.phase && point.model === row.model)?.original ?? row)}
          scale={scales[doc.analysis.scale]}/>}
      </details>
    </>}
  </section>;
}

function SeasonalPlot({ overlay, dark, scale, focusModel }: { overlay: SeasonalOverlay; dark: boolean; scale: number; focusModel: string }) {
  const ref = useRef<HTMLDivElement>(null), chart = useRef<echarts.ECharts | undefined>(undefined);
  const presentation = useMemo(() => seasonalChart(overlay, dark, scale, focusModel), [overlay, dark, scale, focusModel]);
  useEffect(() => {
    if (!ref.current) return;
    const instance = echarts.init(ref.current);
    chart.current = instance;
    const resize = new ResizeObserver(() => instance.resize());
    resize.observe(ref.current);
    return () => { resize.disconnect(); instance.dispose(); chart.current = undefined; };
  }, []);
  useEffect(() => { chart.current?.setOption(presentation.option, { notMerge: true }); }, [presentation]);
  return <>
    <div className="chart" ref={ref} role="img" aria-label={presentation.description}/>
    <p className="seasonal-axis">{overlay.axisTitle}</p>
    <ul className="chart-legend seasonal-legend" aria-label="Seasonal chart series">
      {presentation.legend.map(item => <li key={item.id}>
        <i aria-hidden="true" style={{ borderColor: item.color, borderTopStyle: item.phase === "Forecast" ? "dashed" : "solid" }}/>
        <span>{item.name}</span>
      </li>)}
    </ul>
  </>;
}
