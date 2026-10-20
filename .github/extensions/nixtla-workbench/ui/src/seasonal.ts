import type { EChartsOption, LineSeriesOption } from "echarts";
import type { Row } from "./api";
import { number, text } from "./api";
import { chartNumber, formatted } from "./format";
import { escapeHtml, modelColor, tooltipStyle } from "./chartPresentation";

export type SeasonalKind = "daily" | "weekly" | "monthly" | "quarterly";
export type SeasonalAlignment = "weekday" | "date";
export type SeasonalPoint = {
  date: string; period: string; slot: number; phase: "Actual" | "Forecast";
  model: string; value: unknown; original: Row; bridge?: boolean;
};
export type SeasonalLine = {
  id: string; name: string; period: string; phase: SeasonalPoint["phase"];
  model: string; points: (SeasonalPoint | null)[];
};
export type SeasonalOverlay = {
  kind: SeasonalKind; alignment: SeasonalAlignment; focus: string;
  periods: { key: string; label: string }[]; shown: string[]; labels: string[];
  points: SeasonalPoint[]; lines: SeasonalLine[]; missingPeriods: string[];
  title: string; axisTitle: string;
};

const dayMs = 86400000;
export const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export function seasonalKind(frequency: string): SeasonalKind | null {
  const freq = frequency.replace(/^1(?=[A-Z])/, "");
  if (["D", "B"].includes(freq)) return "daily";
  if (["MS", "ME", "M"].includes(freq)) return "monthly";
  if (/^W(?:-(?:MON|TUE|WED|THU|FRI|SAT|SUN))?$/.test(freq)) return "weekly";
  if (["Q", "QE", "Q-DEC", "QE-DEC", "QS", "QS-JAN"].includes(freq)) return "quarterly";
  return null;
}

function parseDate(value: unknown): Date {
  const date = text(value).slice(0, 10);
  const parsed = new Date(`${date}T00:00:00Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== date)
    throw new Error(`Invalid saved period "${text(value)}". Seasonality requires unambiguous calendar dates.`);
  return parsed;
}

function isoWeek(date: Date) {
  const thursday = new Date(date);
  thursday.setUTCDate(date.getUTCDate() + 4 - (date.getUTCDay() || 7));
  const year = thursday.getUTCFullYear();
  return { year, week: Math.ceil(((thursday.getTime() - Date.UTC(year, 0, 1)) / dayMs + 1) / 7) };
}

function position(date: Date, kind: SeasonalKind, alignment: SeasonalAlignment, yearStart: number) {
  const year = date.getUTCFullYear(), month = date.getUTCMonth();
  if (kind === "daily") return {
    period: date.toISOString().slice(0, 7),
    slot: date.getUTCDate() - 1 + (alignment === "weekday" ? (new Date(Date.UTC(year, month, 1)).getUTCDay() + 6) % 7 : 0),
  };
  if (kind === "weekly") {
    const iso = isoWeek(date);
    return { period: String(iso.year), slot: iso.week - 1 };
  }
  if (kind === "quarterly") return { period: String(year), slot: Math.floor(month / 3) };
  const slot = (month - yearStart + 13) % 12;
  return { period: String(year - (month + 1 < yearStart ? 1 : 0)), slot };
}

export function buildSeasonalOverlay({ history, future, frequency, alignment = "weekday", yearStart = 1,
  focusPeriod = "", periodCount = 3, includeActuals = true }: {
  history: Row[]; future: Row[]; frequency: string; alignment?: SeasonalAlignment; yearStart?: number;
  focusPeriod?: string; periodCount?: number; includeActuals?: boolean;
}): SeasonalOverlay {
  const kind = seasonalKind(frequency);
  if (!kind) throw new Error(`Seasonality overlays are not defined for ${frequency || "an unspecified frequency"}. Use the timeline; no periods have been resampled.`);
  if (!Number.isInteger(yearStart) || yearStart < 1 || yearStart > 12 || !Number.isInteger(periodCount) || periodCount < 1 || periodCount > 12)
    throw new Error("Choose a valid year start and between 1 and 12 comparison periods.");
  const ids = new Set([...history, ...future].map(row => text(row.unique_id)).filter(Boolean));
  if (ids.size > 1) throw new Error("Select one series for seasonality. Different series are not pooled.");
  const points: SeasonalPoint[] = [], seen = new Set<string>();
  for (const [phase, rows] of [["Actual", history], ["Forecast", future]] as const) {
    for (const original of rows) {
      const date = parseDate(original.ds), model = phase === "Actual" ? "" : text(original.model) || "Saved forecast";
      const { period, slot } = position(date, kind, alignment, yearStart);
      const key = JSON.stringify([phase, model, period, slot]);
      if (seen.has(key)) throw new Error(`Multiple ${phase.toLowerCase()} values share ${period}, position ${slot + 1}${model ? ` for ${model}` : ""}. Narrow the series/scenario scope; values are not averaged.`);
      seen.add(key);
      points.push({ date: date.toISOString().slice(0, 10), period, slot, phase, model, value: original[phase === "Actual" ? "y" : "yhat"], original });
    }
  }
  points.sort((a, b) => a.date.localeCompare(b.date) || a.phase.localeCompare(b.phase) || a.model.localeCompare(b.model));
  const periodKeys = [...new Set(points.map(point => point.period))].sort();
  const label = (key: string) => kind !== "monthly" || yearStart === 1 ? key : `${key}/${String(Number(key) + 1).slice(-2)}`;
  const focus = periodKeys.includes(focusPeriod) ? focusPeriod : points.find(point => point.phase === "Forecast")?.period ?? periodKeys.at(-1) ?? "";
  const shown = focus ? Array.from({ length: periodCount }, (_, i) => {
    const offset = i - periodCount + 1;
    return kind === "daily" ? new Date(Date.UTC(Number(focus.slice(0, 4)), Number(focus.slice(5)) - 1 + offset, 1)).toISOString().slice(0, 7) : String(Number(focus) + offset);
  }) : [];
  const slotCount = kind === "daily" ? Math.max(1, ...shown.map(key => {
    const end = new Date(Date.UTC(Number(key.slice(0, 4)), Number(key.slice(5)), 0));
    return position(end, kind, alignment, yearStart).slot + 1;
  })) : kind === "weekly" ? Math.max(52, ...shown.map(key => isoWeek(new Date(Date.UTC(Number(key), 11, 28))).week)) : kind === "monthly" ? 12 : 4;
  const labels = Array.from({ length: slotCount }, (_, slot) => kind === "daily"
    ? alignment === "weekday" ? `${slot} · ${weekdays[slot % 7]}` : String(slot + 1)
    : kind === "monthly" ? monthNames[(yearStart - 1 + slot) % 12] : kind === "weekly" ? `W${String(slot + 1).padStart(2, "0")}` : `Q${slot + 1}`);
  const scoped = points.filter(point => shown.includes(point.period) && (includeActuals || point.phase !== "Actual"));
  const lines: SeasonalLine[] = [];
  for (const period of shown) {
    const members = scoped.filter(point => point.period === period);
    const actuals = new Map(members.filter(point => point.phase === "Actual").map(point => [point.slot, point]));
    const models = [...new Set(members.filter(point => point.phase === "Forecast").map(point => point.model))];
    for (const model of [...(actuals.size ? [""] : []), ...models]) {
      const phase = model ? "Forecast" : "Actual";
      const bySlot = new Map(members.filter(point => point.phase === phase && point.model === model).map(point => [point.slot, point]));
      const values = labels.map((_, slot) => bySlot.get(slot) ?? null);
      const first = Math.min(...bySlot.keys());
      const anchor = actuals.get(first - 1);
      // A dashed join uses only the immediately preceding observed period, never a gap or another month/year.
      if (phase === "Forecast" && number(bySlot.get(first)?.value) !== null && anchor && number(anchor.value) !== null)
        values[first - 1] = { ...anchor, bridge: true };
      lines.push({ id: JSON.stringify([period, phase, model]), name: `${label(period)} · ${phase === "Actual" ? "actual" : `${model} forecast`}`, period, phase, model, points: values });
    }
  }
  return {
    kind, alignment, focus, periods: periodKeys.map(key => ({ key, label: label(key) })), shown, labels,
    points: scoped, lines, missingPeriods: shown.filter(key => !scoped.some(point => point.period === key)),
    title: kind === "daily" ? alignment === "weekday" ? "Weekday-aligned months" : "Day-of-month comparison" : kind === "weekly" ? "Week-of-year comparison" : kind === "quarterly" ? "Quarter-of-year comparison" : "Month-of-year comparison",
    axisTitle: kind === "daily" ? alignment === "weekday" ? "Monday-aligned slot · 0 = Mon; 5/6 = weekend" : "Calendar day of month"
      : kind === "weekly" ? "ISO week · week 53 is not folded into week 52" : kind === "quarterly" ? "Calendar quarter · Q1 = Jan–Mar" : `Month position · year starts in ${monthNames[yearStart - 1]}`,
  };
}

export function seasonalTooltip(params: unknown, scale = 1) {
  const entries = (Array.isArray(params) ? params : [params]).flatMap(item => {
    if (!item || typeof item !== "object" || !("data" in item) || !item.data || typeof item.data !== "object") return [];
    const data = item.data as Record<string, unknown>;
    if (data.bridge || number(data.rawValue) === null) return [];
    return [{ name: "seriesName" in item ? text(item.seriesName) : "", date: text(data.date), value: data.rawValue }];
  });
  return '<div class="chart-tooltip-content">' + entries.slice(0, 8).map(entry =>
    `<div class="chart-tooltip-row"><span>${escapeHtml(entry.name)}<br/>${escapeHtml(entry.date)}</span><b>${escapeHtml(formatted(entry.value, "", scale))}</b></div>`).join("") +
    (entries.length > 8 ? `<small>${entries.length - 8} more paths. Use the exact-value table for all dates.</small>` : "") + "</div>";
}

export function seasonalChart(overlay: SeasonalOverlay, dark: boolean, scale = 1, focusModel = "") {
  const foreground = dark ? "#bdcad8" : "#526174";
  const historical = dark ? ["#edaf68", "#88c8b7", "#baafff", "#c4d0dd"] : ["#a85812", "#267563", "#7850b0", "#526174"];
  const colors = new Map(overlay.shown.map((period, i) => [period, period === overlay.focus ? dark ? "#79b8ff" : "#0969da" : historical[i % historical.length]]));
  const legend = overlay.lines.map(line => ({
    id: line.id, name: line.name, phase: line.phase,
    color: line.phase === "Forecast" && focusModel && line.model !== focusModel ? modelColor(line.model, dark) : colors.get(line.period)!,
  }));
  const series: LineSeriesOption[] = overlay.lines.map((line, i) => ({
    id: line.id, name: line.name, type: "line", connectNulls: false, showSymbol: true,
    symbol: line.phase === "Forecast" ? "diamond" : "circle", symbolSize: line.period === overlay.focus ? 5 : 4,
    lineStyle: { type: line.phase === "Forecast" ? "dashed" : "solid", width: line.period === overlay.focus ? 2.5 : 1.8 },
    itemStyle: { color: legend[i].color }, emphasis: { focus: "series" },
    data: line.points.map((point, slot) => ({
      value: [slot, point ? number(point.value) : null], date: point?.date ?? "", rawValue: point?.value,
      bridge: point?.bridge ?? false,
    })),
  }));
  if (series.length && overlay.kind === "daily" && overlay.alignment === "weekday") series[0].markArea = {
    silent: true, itemStyle: { color: dark ? "rgba(163,178,195,.09)" : "rgba(100,116,139,.07)" },
    data: overlay.labels.flatMap((_, slot) => slot % 7 === 5 ? [[{ xAxis: slot - .5 }, { xAxis: Math.min(slot + 1.5, overlay.labels.length - 1) }]] : []),
  };
  const description = `${overlay.title}. ${overlay.shown.join(", ")}. Solid actuals and dashed forecasts. Missing observations remain gaps. Exact original dates are available below.`;
  const option: EChartsOption = {
    animation: false, backgroundColor: "transparent",
    textStyle: { fontFamily: "Segoe UI, system-ui, sans-serif", color: foreground },
    grid: { top: 24, left: 12, right: 20, bottom: 18, containLabel: true },
    tooltip: { trigger: "axis", renderMode: "html", appendTo: () => document.body, enterable: true,
      className: "workbench-chart-tooltip", extraCssText: tooltipStyle,
      backgroundColor: dark ? "#232d39" : "#ffffff", textStyle: { color: foreground },
      formatter: params => seasonalTooltip(params, scale) },
    xAxis: { type: "value", min: 0, max: overlay.labels.length - 1,
      interval: overlay.kind === "daily" ? overlay.alignment === "weekday" ? 7 : 5 : overlay.kind === "weekly" ? 4 : 1,
      axisLabel: { color: foreground, hideOverlap: true, formatter: value => overlay.labels[value] ?? "" },
      splitLine: { show: false } },
    yAxis: { type: "value", min: value => Math.min(0, value.min),
      axisLabel: { color: foreground, formatter: value => chartNumber(value, scale) },
      splitLine: { lineStyle: { color: dark ? "#2d3948" : "#e9edf2" } } },
    series, aria: { enabled: true, description },
  };
  return { option, legend, description };
}
