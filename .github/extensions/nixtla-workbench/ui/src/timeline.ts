import type { EChartsOption, LineSeriesOption } from "echarts";
import { number, text } from "./api";
import type { Row } from "./api";
import { chartNumber } from "./format";
import { chartTooltip, modelColor, tooltipStyle } from "./chartPresentation";

export type TimelineInput = {
  rows: Row[]; historyRows?: Row[]; fields?: string[]; intervalLevel?: number; dark: boolean;
  scale?: number; domainDates?: string[]; bounds?: { min: number; max: number };
  origin?: string; focusField?: string; range?: "all" | "forecast"; fullHistoryOnly?: boolean;
};

export function buildTimeline({
  rows, historyRows = [], fields = ["yhat"], intervalLevel = 0, dark, scale = 1,
  domainDates, bounds, origin, focusField, range = "all", fullHistoryOnly = false,
}: TimelineInput) {
  const actual = new Map(historyRows.map(row => [text(row.ds), row]));
  const forecast = new Map(rows.map(row => [text(row.ds), row]));
  const inspectDates = [...new Set([...actual.keys(), ...forecast.keys()])].sort();
  const futureDates = [...forecast.keys()].sort();
  const lastActual = [...actual.keys()].sort().at(-1);
  const cutoff = origin ?? (lastActual && futureDates[0] > lastActual ? lastActual : undefined);
  const end = futureDates.at(-1);
  const window = cutoff && end && end > cutoff ? { origin: cutoff, start: futureDates[0], end } : null;
  const fullDates = [...new Set([...(domainDates ?? []), ...inspectDates, ...(window ? [window.origin] : [])])].sort();
  const focused = !fullHistoryOnly && range === "forecast";
  const dates = focused && window
    ? fullDates.slice(Math.max(0, fullDates.indexOf(window.origin) - 11), fullDates.indexOf(window.end) + 1)
    : fullDates;
  const foreground = dark ? "#bdcad8" : "#526174";
  const colors = [dark ? "#79b8ff" : "#0969da", "#8250df", "#bc6a23", "#25846a"];
  const series: LineSeriesOption[] = [];
  if (historyRows.length) series.push({
    id: "observed-actuals", name: "Observed actuals", type: "line", showSymbol: false, connectNulls: false,
    data: dates.map(date => !window || date <= window.origin ? number(actual.get(date)?.y) : null),
    itemStyle: { color: dark ? "#c4d0dd" : "#44566c" }, lineStyle: { width: 2.5 },
  });
  fields.forEach(field => series.push({
    id: `path:${field}`, silent: false, tooltip: { show: true },
    name: field === "yhat" ? "Forecast" : field === "y" ? window ? "Actuals observed afterward" : "Actuals" : field,
    type: "line", showSymbol: false, connectNulls: false,
    data: dates.map(date => field === "y" && window && date >= window.origin && !forecast.has(date)
      ? number(actual.get(date)?.y) : number(forecast.get(date)?.[field])),
    itemStyle: { color: field === "y" ? (dark ? "#c4d0dd" : "#44566c") : modelColor(field, dark) },
    lineStyle: {
      width: field === "y" ? 2.5 : field === focusField || field === "Blend" ? 3 : 1.5,
      opacity: !focusField || field === focusField || field === "y" ? 1 : .45,
      type: field === "y" ? "solid" : "dashed",
    },
  }));
  const legend = series.map(item => ({ name: String(item.name), color: String(item.itemStyle?.color ?? colors[0]) }));
  if (window) series.push({
    id: "forecast-window", name: "Forecast window", type: "line", data: [], silent: true, z: 0, tooltip: { show: false },
    markArea: {
      silent: true, itemStyle: { color: dark ? "rgba(163,178,195,.14)" : "rgba(100,116,139,.12)" },
      label: { show: true, position: "insideTopRight", color: foreground },
      data: [[{ name: "Forecast window", xAxis: window.origin }, { xAxis: window.end }]],
    },
    markLine: { silent: true, symbol: "none", data: [{ xAxis: window.origin }], label: { formatter: "Forecast origin", color: foreground } },
  });
  if (intervalLevel) {
    const bands = dates.map(date => {
      const row = forecast.get(date);
      const lo = number(row?.[`yhat_lo_${intervalLevel}`]), hi = number(row?.[`yhat_hi_${intervalLevel}`]);
      return lo !== null && hi !== null && lo <= hi ? { lo, span: hi - lo } : null;
    });
    const label = `Saved ${intervalLevel}% interval${focusField ? ` · ${focusField}` : ""}`;
    if (bands.some(value => value !== null)) {
    series.push(
      { id: "interval-lower", name: "Interval lower bound", type: "line", stack: "interval", stackStrategy: "all",
        data: bands.map(value => value?.lo ?? null), showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false }, silent: true },
      { id: "interval-band", name: label, type: "line", stack: "interval", stackStrategy: "all",
        data: bands.map(value => value?.span ?? null), showSymbol: false, lineStyle: { opacity: 0 },
        areaStyle: { color: modelColor(focusField || "yhat", dark), opacity: .15 }, tooltip: { show: false }, silent: true },
    );
    legend.push({ name: label, color: modelColor(focusField || "yhat", dark) });
    }
  }
  const description = `Forecast timeline, ${dates[0] ?? "no dates"} to ${dates.at(-1) ?? "no dates"}; ` + (window
    ? `Observed history through ${window.origin}. Gray forecast window: ${window.start} to ${window.end}. Solid lines show actuals; dashed lines show forecasts.`
    : "Saved values. Exact values are available in the period selector and evidence tables.");
  const option: EChartsOption = {
    animation: false, color: colors, backgroundColor: "transparent",
    textStyle: { fontFamily: "Segoe UI, system-ui, sans-serif", color: foreground },
    grid: { top: 32, left: 12, right: 24, bottom: 20, containLabel: true },
    tooltip: { trigger: "axis", renderMode: "html", appendTo: () => document.body, enterable: true,
      className: "workbench-chart-tooltip", extraCssText: tooltipStyle,
      backgroundColor: dark ? "#232d39" : "#ffffff", textStyle: { color: foreground },
      formatter: params => chartTooltip(params, scale, focusField) },
    legend: { show: false, data: legend.map(item => item.name) },
    xAxis: { type: "category", data: dates, boundaryGap: false, axisLabel: { color: foreground, formatter: value => value.slice(0, 10) } },
    yAxis: { type: "value", scale: true, ...(bounds && !focused ? { min: bounds.min, max: bounds.max } : {}),
      axisLabel: { color: foreground, formatter: value => chartNumber(value, scale) },
      splitLine: { lineStyle: { color: dark ? "#2d3948" : "#e9edf2" } } },
    series,
    aria: { enabled: true, description },
  };
  return { option, legend, actual, forecast, dates: inspectDates, axisDates: dates, description, window };
}
