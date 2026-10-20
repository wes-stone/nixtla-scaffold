import { text } from "./api";

const decimals = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const counts = /(^|_)(id|step|steps|windows|observations|rows|count|rank|year|depth|horizon|periods|lag|cutoffs)$/i;
const ratios = /(^|_)(wape|mase|rmsse|bias|weight|rate|share|coverage|strength|score|pct|correlation|ratio|stat|importance)(_|$)/i;
const measurements = /(^$|^y$|^y_actual$|^yhat(?:_|$)|^calendar_scenario$|^scenario_minus_saved$|^(?:cumulative_|calendar_|judgment_)?delta$|(?:^|_)(?:rmse|mae|error|residual|width|actual|forecast|observed|trend|seasonal|remainder|yhat)$|^\d{4}-\d{2}-\d{2})/i;
const identifiers = /(^|_)(id|date|ds|cutoff|model|version|name|parent|child)$/i;
export const scales = { raw: 1, thousands: 1e3, millions: 1e6, billions: 1e9 } as const;
export type Scale = keyof typeof scales;
export const measurementScale = (column: string, scale: number) => ["mean_interval_score", "weighted_interval_score", "point_total", "lower", "upper", "daily_bias", "prediction"].includes(column) ? scale : ratios.test(column) || counts.test(column) || !measurements.test(column) ? 1 : /squared/.test(column) ? scale ** 2 : scale;

export function measured(value: unknown, column = ""): number | null {
  if (identifiers.test(column)) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string" || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value)) return null;
  if (/^[+-]?0\d+/.test(value)) return null;
  const result = Number(value);
  return Number.isFinite(result) && Math.abs(result) <= Number.MAX_SAFE_INTEGER ? result : null;
}

export function formatted(value: unknown, column = "", scale = 1, decimalPlaces = 2): string {
  const numeric = measured(value, column);
  if (numeric === null) return text(value);
  if (counts.test(column)) return numeric.toLocaleString("en-US", { maximumFractionDigits: 0 });
  const divisor = measurementScale(column, scale);
  const resolved = numeric / divisor;
  const rounded = Math.abs(resolved) < 0.5 * 10 ** -decimalPlaces ? 0 : resolved;
  return decimalPlaces === 2 ? decimals.format(rounded) : rounded.toLocaleString("en-US", { minimumFractionDigits: decimalPlaces, maximumFractionDigits: decimalPlaces });
}

export function chartNumber(value: unknown, scale = 1): string {
  const candidate = Array.isArray(value) ? value.at(-1) : value;
  const numeric = measured(candidate);
  return numeric === null ? text(candidate) : scale === 1 ? compact.format(numeric) : decimals.format(numeric / scale);
}

export function columnLabel(column: string): string {
  const labels: Record<string, string> = {
    ds: "Period", y: "Actual", y_actual: "Actual", yhat: "Forecast",
    yhat_lo_80: "80% lower", yhat_hi_80: "80% upper", yhat_lo_95: "95% lower", yhat_hi_95: "95% upper",
    rmse: "RMSE", mae: "MAE", wape: "WAPE", mase: "MASE", rmsse: "RMSSE",
    landing_mape_pct: "Landing MAPE (%)", next28_mape_pct: "Next28 MAPE (%)", daily_wmape_pct: "Date-averaged WMAPE (%)",
    landing_bias_pct: "Landing bias (%)", landing_worst_pct: "Worst landing error (%)",
    daily_rmse: "Daily RMSE", daily_mae: "Daily MAE", daily_bias: "Daily bias",
  };
  return Object.hasOwn(labels, column) ? labels[column] : column.replaceAll("_", " ");
}
