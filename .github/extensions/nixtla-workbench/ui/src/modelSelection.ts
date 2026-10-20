import { number, text } from "./api";
import type { Document, Row } from "./api";

export const MAX_COMPARE_MODELS = 5;

export const accuracyMetrics = ["rmse", "mae", "wape", "mase", "rmsse", "bias", "landing_mape_pct", "next28_mape_pct", "daily_wmape_pct"] as const;
export const metricLabel = (metric: string) => metric === "bias" ? "|Bias|"
  : metric === "landing_mape_pct" ? "Landing MAPE (%)" : metric === "next28_mape_pct" ? "Next28 MAPE (%)"
    : metric === "daily_wmape_pct" ? "Date-averaged WMAPE (%)" : metric.toUpperCase();

export function accuracyScore(row: Row | undefined, metric: string) {
  const score = number(row?.[metric]);
  return score === null ? null : metric === "bias" ? Math.abs(score) : score;
}

export function compareAccuracy(a: Row, b: Row, metric: string) {
  const ties = metric === "landing_mape_pct" ? ["next28_mape_pct", "daily_wmape_pct"] : ["mae", "bias"];
  for (const key of [...new Set([metric, ...ties])]) {
    const left = accuracyScore(a, key) ?? Infinity, right = accuracyScore(b, key) ?? Infinity;
    if (left !== right) return left < right ? -1 : 1;
  }
  return text(a.model).localeCompare(text(b.model));
}

export function modelRanking(models: string[], audit: Row[] = [], metric = "rmse") {
  const evidence = new Map(audit.map(row => [text(row.model), row]));
  const entries = [...new Set(models)].map(model => {
    const row = evidence.get(model), score = accuracyScore(row, metric);
    const ensemble = row?.family === "ensemble" || model === "WeightedEnsemble";
    const comparable = !ensemble && row?.selection_eligible === true && score !== null;
    return { model, evidence: row, score, comparable, ensemble };
  });
  entries.sort((a, b) => Number(b.comparable) - Number(a.comparable)
    || Number(a.ensemble) - Number(b.ensemble)
    || compareAccuracy({ ...a.evidence, model: a.model }, { ...b.evidence, model: b.model }, metric));
  let rank = 0;
  return entries.map(entry => ({ ...entry, rank: entry.comparable ? ++rank : null }));
}

export function tournamentWinner(audit: Row[] = []) {
  const selected = audit.filter(row => row.selection_eligible === true && row.family !== "ensemble"
    && (row.is_selected_model === true || (text(row.model) !== "" && row.model === row.selected_model)));
  return selected.length === 1 ? text(selected[0].model) : "";
}

export function modelSelection(doc: Document, models: string[], selected = "") {
  const available = [...new Set(models)];
  let focus = available.includes(doc.analysis.previewModel) ? doc.analysis.previewModel
    : available.includes(selected) ? selected : available[0] ?? "";
  const pinned = doc.analysis.models.filter(model => model !== "*" && available.includes(model));
  const shortlist = [...new Set(pinned.length ? pinned : focus ? [focus] : [])].slice(0, MAX_COMPARE_MODELS);
  if (doc.analysis.modelMode === "compare" && !shortlist.includes(focus)) focus = shortlist[0] ?? "";
  const shown = doc.analysis.modelMode === "compare" ? shortlist : focus ? [focus] : [];
  return { focus, shortlist, shown, unavailable: !!doc.analysis.previewModel && !available.includes(doc.analysis.previewModel) };
}

export function savedModel(rows: Row[]) {
  return text(rows.find(row => row.is_selected_model === true)?.model)
    || text(rows.find(row => row.selected_model)?.selected_model) || text(rows[0]?.model);
}

export function finalForecast(rows: Row[], model: string) {
  const scoped = rows.filter(row => text(row.model) === model).sort((a, b) => text(a.ds).localeCompare(text(b.ds)));
  const row = scoped.at(-1);
  return row ? { date: text(row.ds), value: number(row.yhat) } : undefined;
}

export function evidenceLabel(row?: Row) {
  if (!row) return "No saved tournament score";
  if (row.configuration_eligible === false) return `${text(row.cv_windows) || "Saved"} origins · configuration excluded`;
  if (row.family === "ensemble") return "Separate blend cohort";
  if (row.selection_eligible === true) return `${text(row.cv_windows) || "Saved"} origins · comparable`;
  if ((number(row.cv_windows) ?? 0) > 0) return `${text(row.cv_windows)} origins · partial support`;
  return "No full-horizon backtest";
}
