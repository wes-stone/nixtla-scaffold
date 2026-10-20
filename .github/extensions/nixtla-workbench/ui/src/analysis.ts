import type { Row } from "./api";
import { number, text } from "./api";

export function paths(rows: Row[], x = "ds", group = "model", value = "yhat"): { rows: Row[]; fields: string[] } {
  const points = new Map<string, Row>(), fields = new Set<string>();
  for (const row of rows) {
    const date = text(row[x]), name = text(row[group]) || value;
    const point = points.get(date) ?? { ds: date };
    if (Object.hasOwn(point, name)) throw new Error(`Ambiguous chart: multiple ${name} values at ${date}. Narrow the series/scenario scope.`);
    point[name] = row[value];
    points.set(date, point); fields.add(name);
  }
  return { rows: [...points.values()].sort((a, b) => text(a.ds).localeCompare(text(b.ds), undefined, { numeric: true })), fields: [...fields] };
}

export function seasonalRows(history: Row[], future: Row[], yearStart: number) {
  const records: Row[] = [];
  for (const [source, rows] of [["Actual", history], ["Forecast", future]] as const) {
    for (const row of rows) {
      const date = new Date(`${text(row.ds).slice(0, 10)}T00:00:00Z`);
      if (!Number.isFinite(date.getTime())) continue;
      const month = date.getUTCMonth() + 1;
      const year = date.getUTCFullYear() - (month < yearStart ? 1 : 0);
      records.push({ ds: String((month - yearStart + 12) % 12 + 1).padStart(2, "0"), model: `${year} ${source}`, yhat: source === "Actual" ? row.y : row.yhat });
    }
  }
  return paths(records);
}

export function residuals(rows: Row[], model: string) {
  return rows.filter(row => text(row.model) === model).map(row => {
    const actual = number(row.y_actual), prediction = number(row.yhat);
    return { ...row, residual: actual === null || prediction === null ? null : actual - prediction };
  });
}

export function histogram(rows: Row[], key: string, bins = 12): Row[] {
  const values = rows.map(row => number(row[key])).filter((value): value is number => value !== null);
  if (!values.length) return [];
  const lo = Math.min(...values), hi = Math.max(...values), width = (hi - lo) / bins || 1;
  return Array.from({ length: hi === lo ? 1 : bins }, (_, index) => ({
    lower: lo + index * width, upper: hi === lo ? hi : lo + (index + 1) * width,
    count: values.filter(value => Math.min(bins - 1, Math.floor((value - lo) / width)) === index).length,
  }));
}

export function autocorrelation(rows: Row[]): Row[] {
  const values = rows.filter(row => number(row.horizon_step) === 1).sort((a, b) => text(a.cutoff).localeCompare(text(b.cutoff)))
    .map(row => number(row.residual)).filter((value): value is number => value !== null);
  if (values.length < 4) return [];
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const denominator = values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  if (!denominator) return [];
  return Array.from({ length: Math.min(12, values.length - 1) }, (_, i) => {
    const lag = i + 1;
    return { lag, correlation: values.slice(lag).reduce((sum, value, j) => sum + (value - mean) * (values[j] - mean), 0) / denominator,
      observations: values.length, heuristic_95: 1.96 / Math.sqrt(values.length) };
  });
}
