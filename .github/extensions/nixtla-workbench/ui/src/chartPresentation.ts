import { number, text } from "./api";
import { formatted } from "./format";

export function modelColor(model: string, dark = false) {
  let hash = 0;
  for (const character of model) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  const hue = Math.round(((hash % 997) * 137.508) % 360);
  return `hsl(${hue},60%,${dark ? 70 : 36}%)`;
}

export function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
}

export function chartTooltip(params: unknown, scale = 1, focus = "") {
  const source = Array.isArray(params) ? params : [params];
  const entries = source.filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map(item => ({
      name: text(item.seriesName), date: text(item.axisValueLabel || item.name),
      value: number(Array.isArray(item.value) ? item.value.at(-1) : item.value),
    })).filter(item => item.value !== null);
  if (!entries.length) return "";
  const ordered = [...entries].sort((a, b) => Number(b.name === focus) - Number(a.name === focus));
  const visible = ordered.slice(0, 8);
  return `<div class="chart-tooltip-content"><strong>${escapeHtml(entries[0].date)}</strong>` +
    visible.map(item => `<div class="chart-tooltip-row"><span>${escapeHtml(item.name)}</span><b>${escapeHtml(formatted(item.value, "", scale))}</b></div>`).join("") +
    (ordered.length > visible.length ? `<small>${ordered.length - visible.length} more series. Narrow the comparison for details.</small>` : "") + "</div>";
}

export const tooltipStyle = "max-width:min(360px,calc(100vw - 32px));max-height:min(320px,45vh);overflow:auto;white-space:normal;z-index:1000;";
