import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Copy, X, AlertCircle } from "lucide-react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import { AriaComponent, GridComponent, TooltipComponent, LegendComponent, MarkAreaComponent, MarkLineComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { text, failure } from "./api";
import type { Row } from "./api";
import { columnLabel, formatted, measured } from "./format";
import { buildTimeline } from "./timeline";
import type { TimelineInput } from "./timeline";

echarts.use([LineChart, AriaComponent, GridComponent, TooltipComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, CanvasRenderer]);

export function ErrorNotice({ error }: { error: string }) {
  return <div className="notice error" role="alert"><AlertCircle size={17}/><span>{error}</span></div>;
}

export function Modal({ title, close, children }: { title: string; close: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const previous = document.activeElement;
    ref.current?.showModal();
    return () => { ref.current?.close(); if (previous instanceof HTMLElement) previous.focus(); };
  }, []);
  return <dialog ref={ref} onCancel={event => { event.preventDefault(); close(); }}>
    <header><h2>{title}</h2><button className="icon" aria-label="Close dialog" onClick={close}><X size={18}/></button></header>
    {children}
  </dialog>;
}

export function Table({ rows, columns, inspect, title = "Values", scale = 1, rowActions, columnLabels = {}, raw = false, decimalPlaces = 2, stickyColumn }: {
  rows: Row[]; columns?: string[]; inspect?: (row: Row) => void; title?: string; scale?: number; rowActions?: (row: Row) => ReactNode;
  columnLabels?: Record<string, string>; raw?: boolean; decimalPlaces?: number; stickyColumn?: string;
}) {
  const cols = columns ?? Object.keys(rows[0] ?? {});
  const label = (column: string, fallback: string) => Object.hasOwn(columnLabels, column) ? columnLabels[column] : fallback;
  const [message, setMessage] = useState("");
  const copy = async (display = false) => {
    const escaped = (value: unknown) => {
      let v = text(value).replaceAll("\t", " ").replaceAll("\n", " ");
      if (typeof value === "string" && /^[=+\-@\r]/.test(v)) v = "'" + v;
      return v;
    };
    try {
      await navigator.clipboard.writeText([cols.map(col => escaped(label(col, col))).join("\t"), ...rows.map(row => cols.map(c => escaped(display && !raw ? formatted(row[c], c, scale, decimalPlaces) : row[c])).join("\t"))].join("\n"));
      setMessage(`Copied ${rows.length} rows ${display && !raw ? `as displayed (${decimalPlaces} decimals)` : "at original precision"}.`);
    } catch (error) { setMessage(`Copy failed: ${failure(error)} Select table text to copy manually.`); }
  };
  return <section className="values" aria-label={title}>
    <header className="section-heading"><h3>{title}</h3><div className="toolbar">
      <span className="muted">{rows.length.toLocaleString()} rows shown</span>
      <button className="quiet" onClick={() => void copy(true)} disabled={!rows.length}><Copy size={14}/> Copy displayed</button>
      <button className="quiet" onClick={() => void copy()} disabled={!rows.length}>Copy raw rows</button>
    </div></header>
    {message && <p role="status">{message}</p>}
    {!rows.length ? <p className="empty-inline">No evidence rows are available for this scope.</p> :
      <div className="table-scroll" tabIndex={0}><table>
        <thead><tr>{rowActions && <th>Explore</th>}{inspect && <th>Inspect</th>}{cols.map(col => <th key={col} className={col === stickyColumn ? "sticky-column" : ""}>{label(col, columnLabel(col))}</th>)}</tr></thead>
        <tbody>{rows.map((row, i) => <tr key={i}>
          {rowActions && <td>{rowActions(row)}</td>}
          {inspect && <td><button className="link" aria-label={`Inspect row ${i + 1}`} onClick={() => inspect(row)}>Details</button></td>}
          {cols.map(col => <td key={col} title={text(row[col])} className={`${measured(row[col], col) !== null ? "numeric" : ""} ${col === stickyColumn ? "sticky-column" : ""}`}>{(raw ? text(row[col]) : formatted(row[col], col, scale, decimalPlaces)) || <span className="missing" aria-label="Not available">—</span>}</td>)}
        </tr>)}</tbody>
      </table></div>}
  </section>;
}

export function Timeline({ inspect, defaultRange = "all", ...input }: TimelineInput & {
  inspect: (row: Row) => void; defaultRange?: "all" | "forecast";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | undefined>(undefined);
  const [period, setPeriod] = useState("");
  const [range, setRange] = useState(defaultRange);
  const { rows, historyRows, fields, intervalLevel, dark, scale, domainDates, bounds, origin, focusField, fullHistoryOnly } = input;
  const fieldKey = fields?.join("|");
  const timeline = useMemo(() => buildTimeline({ ...input, range }),
    [rows, historyRows, fieldKey, intervalLevel, dark, scale, domainDates, bounds?.min, bounds?.max, origin, focusField, range, fullHistoryOnly]);
  useEffect(() => {
    if (!ref.current) return;
    const instance = echarts.init(ref.current);
    chart.current = instance;
    const resize = new ResizeObserver(() => instance.resize());
    resize.observe(ref.current);
    return () => { resize.disconnect(); instance.dispose(); chart.current = undefined; };
  }, []);
  useEffect(() => { chart.current?.setOption(timeline.option, { replaceMerge: ["series", "xAxis", "yAxis"] }); }, [timeline]);
  return <section aria-label="Forecast timeline">
    {fullHistoryOnly && <p className="footnote">Full date axis: {timeline.axisDates[0]} – {timeline.axisDates.at(-1)}. Fixed across backtest windows.</p>}
    {timeline.window && !fullHistoryOnly && <div className="timeline-range" role="group" aria-label="Chart range">
      <span>Chart range</span><button aria-pressed={range === "all"} onClick={() => setRange("all")}>Full history</button>
      <button aria-pressed={range === "forecast"} onClick={() => setRange("forecast")}>Focus forecast</button>
      <small>{range === "forecast" ? "Up to 12 preceding actual periods plus this forecast window." : "All loaded history and forecast periods."}</small>
    </div>}
    <div ref={ref} className="chart" role="img" aria-label={timeline.description}/>
    <ul className="chart-legend" aria-label="Chart series">{timeline.legend.map(item => <li key={item.name}>
      <i aria-hidden="true" style={{ background: item.color }}/><span>{item.name}</span>
    </li>)}</ul>
    <div className="period-inspector"><label>Inspect a period <select value={timeline.dates.includes(period) ? period : ""} onChange={event => {
      setPeriod(event.target.value);
      inspect({ ...(timeline.actual.get(event.target.value) ?? {}), ...(timeline.forecast.get(event.target.value) ?? {}) });
    }}><option value="">Choose a period</option>{timeline.dates.map(date => <option key={date} value={date}>{date.slice(0, 10)}</option>)}</select></label>
    <span>Solid: actuals · Dashed: forecasts{timeline.window ? " · Gray: forecast window" : ""} · Gaps remain missing</span></div>
  </section>;
}
