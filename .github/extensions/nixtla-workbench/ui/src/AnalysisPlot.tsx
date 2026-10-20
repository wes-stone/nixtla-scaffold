import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, ScatterChart, LineChart } from "echarts/charts";
import { GridComponent, TooltipComponent, LegendComponent, AriaComponent, DataZoomComponent, MarkLineComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption, SeriesOption } from "echarts";
import type { Row } from "./api";
import { number, text } from "./api";
import { chartNumber, columnLabel } from "./format";
import { chartTooltip, tooltipStyle } from "./chartPresentation";

echarts.use([BarChart, ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, AriaComponent, DataZoomComponent, MarkLineComponent, CanvasRenderer]);

export function AnalysisPlot({ title, rows, x, fields, group, kind = "line", dark, scale = 1, dashedFields = [], pointFields = [], fieldLabels = {} }: {
  title: string; rows: Row[]; x: string; fields: string[]; group?: string;
  kind?: "line" | "bar" | "scatter"; dark: boolean; scale?: number; dashedFields?: string[]; pointFields?: string[]; fieldLabels?: Record<string, string>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const signature = fields.join("|");
  const dashedSignature = dashedFields.join("|");
  const pointSignature = pointFields.join("|");
  const labelSignature = JSON.stringify(fieldLabels);
  const keys = rows.map(row => `${group ? text(row[group]) : ""}\u0000${text(row[x])}`);
  const ambiguous = kind !== "scatter" && new Set(keys).size !== keys.length;
  useEffect(() => {
    if (!ref.current || ambiguous) return;
    const chart = echarts.init(ref.current);
    const groups = group ? [...new Set(rows.map(row => text(row[group])))] : [""];
    const categories = [...new Set(rows.map(row => text(row[x])))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const series: SeriesOption[] = [];
    groups.forEach(label => fields.forEach(field => {
      const selected = rows.filter(row => !group || text(row[group]) === label);
      const name = [label, fieldLabels[field] ?? columnLabel(field)].filter(Boolean).join(" · ");
      if (kind === "scatter") {
        series.push({ name, type: "scatter", symbolSize: 10, data: selected.map(row => ({ name: text(row.model), value: [number(row[x]), number(row[field])] })),
          emphasis: { label: { show: true, formatter: "{b}", position: "top" } } });
      } else {
        const byX = new Map(selected.map(row => [text(row[x]), row]));
        series.push({ name, type: kind, data: categories.map(key => number(byX.get(key)?.[field])), ...(kind === "line" ? {
          connectNulls: false, showSymbol: selected.length < 16,
          ...(dashedFields.includes(field) ? { lineStyle: { type: "dashed" as const }, symbol: "diamond" } : {}),
          ...(pointFields.includes(field) ? { showSymbol: true, symbolSize: 7, lineStyle: { opacity: 0 } } : {}),
        } : {}) });
      }
    }));
    const foreground = dark ? "#bdcad8" : "#526174";
    const unit = scale === 1e9 ? "billions" : scale === 1e6 ? "millions" : scale === 1e3 ? "thousands" : "";
    const option: EChartsOption = {
      animation: false, color: ["#0969da", "#25846a", "#8250df", "#bc6a23", "#bc4c69", "#758596"],
      textStyle: { color: foreground, fontFamily: "Segoe UI, system-ui, sans-serif" },
      grid: { top: 42, right: 35, bottom: 88, left: 65, containLabel: true },
      tooltip: { trigger: kind === "scatter" ? "item" : "axis", renderMode: "html", appendTo: () => document.body,
        className: "workbench-chart-tooltip", extraCssText: tooltipStyle, enterable: true,
        backgroundColor: dark ? "#232d39" : "#ffffff", textStyle: { color: foreground },
        formatter: params => chartTooltip(params, scale) },
      xAxis: kind === "scatter" ? { type: "value", name: `${columnLabel(x)}${unit ? ` (${unit})` : ""}`, nameLocation: "middle", nameGap: 28, scale: true, axisLabel: { formatter: value => chartNumber(value, scale), color: foreground } }
        : { type: "category", data: categories, axisLabel: { color: foreground } },
      yAxis: { type: "value", name: `${fields.length === 1 ? fieldLabels[fields[0]] ?? columnLabel(fields[0]) : "Value"}${unit ? ` (${unit})` : ""}`, scale: kind !== "bar", axisLabel: { formatter: value => chartNumber(value, scale), color: foreground }, splitLine: { lineStyle: { color: dark ? "#2d3948" : "#e9edf2" } } },
      legend: { bottom: 0, type: "scroll", textStyle: { color: foreground } },
      dataZoom: kind === "scatter" ? [] : [{ type: "inside" }, { type: "slider", height: 14, bottom: 35 }],
      series, aria: { enabled: true, description: `${title}. The companion evidence table provides the numeric values.` },
    };
    chart.setOption(option);
    const resize = new ResizeObserver(() => chart.resize()); resize.observe(ref.current);
    return () => { resize.disconnect(); chart.dispose(); };
  }, [rows, x, signature, dashedSignature, pointSignature, labelSignature, group, kind, dark, scale, title, ambiguous]);
  return <section className="analysis-plot" aria-label={title}><h3>{title}</h3>{!rows.length ? <p className="empty-inline">No saved values for this chart scope.</p> : ambiguous ? <p className="notice">Multiple rows share a chart coordinate. Narrow the model, scenario or series filter. Values are not silently averaged or overwritten; the evidence table remains available.</p> : <div className="chart" ref={ref} role="img" aria-label={title}/>}</section>;
}
