import Decimal from "decimal.js";
import { z } from "zod";
import { calendarSettingsSchema } from "./calendarSettings";

const D = Decimal.clone({ precision: 40, rounding: Decimal.ROUND_HALF_EVEN });
export const calendarLayerSchema = calendarSettingsSchema.extend({
  enabled: z.boolean().default(false), holdTotal: z.boolean().default(true),
  holidayLosses: z.array(z.object({ ds: z.string(), days: z.number().finite().min(0).max(31) })).max(501).default([]),
});
export type CalendarLayer = z.infer<typeof calendarLayerSchema>;
export const neutralCalendarLayer = () => calendarLayerSchema.parse({});
export const calendarReceiptSchema = z.object({
  settings: calendarLayerSchema, referencePeriod: z.string(), referenceExposure: z.string(), referenceActual: z.string(),
  modelTotal: z.string(), calendarTotal: z.string(), unscaledTotal: z.string(), scalingFactor: z.string(),
  roundingResidual: z.string(), method: z.string(), warnings: z.array(z.string()),
}).passthrough();
export const judgmentEditSchema = z.object({
  ds: z.string(), percent: z.string().max(80).default(""), amount: z.string().max(80).default(""),
  override: z.string().max(80).default(""), note: z.string().max(400).default(""),
});
export const judgmentDraftSchema = z.object({
  model: z.string().default(""), percent: z.string().max(80).default("0"), amount: z.string().max(80).default("0"),
  start: z.string().default(""), end: z.string().default(""), edits: z.array(judgmentEditSchema).max(500).default([]),
  label: z.string().max(120).default(""), reason: z.string().max(2000).default(""),
  calendar: calendarLayerSchema.optional(),
});
export type JudgmentDraft = z.infer<typeof judgmentDraftSchema>;
export type JudgmentEdit = z.infer<typeof judgmentEditSchema>;
export const neutralJudgment = () => judgmentDraftSchema.parse({});
export type ModelRow = { ds: string; model_yhat: string; selected_yhat: string; calendar_yhat?: string; calendar_delta?: string };
function numeric(value: string, name: string) {
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())) throw new Error(`${name}: enter a finite number.`);
  const number = new D(value);
  if (!number.isFinite() || (number.e !== undefined && Math.abs(number.e) > 100)) throw new Error(`${name}: number is outside the supported range.`);
  return number;
}
export function judgmentSummary(settings: JudgmentDraft, units = "original units") {
  const bulk = [
    ...(!numeric(settings.percent, "Percentage").isZero() ? [`${settings.percent}%`] : []),
    ...(!numeric(settings.amount, "Absolute change").isZero() ? [`${settings.amount} ${units || "original units"}/period`] : []),
  ];
  return [
    ...(bulk.length ? [...bulk, settings.start || settings.end ? `${settings.start || "First period"} to ${settings.end || "last period"}` : "Entire horizon"] : []),
    ...(settings.edits.length ? [`${settings.edits.length} row input${settings.edits.length === 1 ? "" : "s"}`] : []),
  ].join(" · ") || "No final changes";
}
export function calculateJudgment(rows: ModelRow[], settings: JudgmentDraft) {
  const keys = new Set(rows.map(row => row.ds));
  if ((settings.start && !keys.has(settings.start)) || (settings.end && !keys.has(settings.end)) || (settings.start && settings.end && settings.start > settings.end)) throw new Error("Choose a valid inclusive bulk period range.");
  const edits = new Map<string, JudgmentEdit>();
  for (const row of settings.edits) {
    if (!keys.has(row.ds) || edits.has(row.ds)) throw new Error("Edits must match unique future periods in this model.");
    edits.set(row.ds, row);
  }
  const bulkPercent = numeric(settings.percent, "Bulk percentage"), bulkAmount = numeric(settings.amount, "Bulk absolute change");
  let cumulative = new D(0);
  const result = rows.map(row => {
    const edit = edits.get(row.ds);
    const inside = (!settings.start || row.ds >= settings.start) && (!settings.end || row.ds <= settings.end);
    const percent = edit?.percent ? numeric(edit.percent, `${row.ds} percentage`) : inside ? bulkPercent : new D(0);
    const amount = edit?.amount ? numeric(edit.amount, `${row.ds} absolute change`) : inside ? bulkAmount : new D(0);
    const base = numeric(row.model_yhat, "Model forecast");
    const override = edit?.override ? numeric(edit.override, `${row.ds} final value`) : null;
    const adjustmentBase = numeric(row.calendar_yhat ?? row.model_yhat, "After calendar");
    const adjusted = override ?? adjustmentBase.times(new D(1).plus(percent.div(100))).plus(amount);
    numeric(adjusted.toString(), `${row.ds} adjusted result`);
    const delta = adjusted.minus(base); cumulative = cumulative.plus(delta);
    return { ...row, ...(row.calendar_yhat !== undefined ? { judgment_delta: adjusted.minus(adjustmentBase).toString() } : {}),
      percent: percent.toString(), amount: amount.toString(), override: override?.toString() ?? "",
      adjusted_yhat: adjusted.toString(), delta: delta.toString(), cumulative_delta: cumulative.toString(),
      delta_pct: base.isZero() ? null : delta.div(base.abs()).times(100).toString(), note: edit?.note ?? "" };
  });
  const sum = (key: "model_yhat" | "selected_yhat" | "adjusted_yhat" | "delta") => result.reduce((total, row) => total.plus(row[key]), new D(0));
  const base = sum("model_yhat"), delta = sum("delta");
  const calendarTotal = result.reduce((total, row) => total.plus(row.calendar_yhat ?? row.model_yhat), new D(0));
  return { rows: result, totals: { model_yhat: base.toString(), selected_yhat: sum("selected_yhat").toString(),
    ...(rows[0]?.calendar_yhat !== undefined ? { calendar_yhat: calendarTotal.toString(),
      calendar_delta: result.reduce((total, row) => total.plus(row.calendar_delta ?? "0"), new D(0)).toString(),
      judgment_delta: result.reduce((total, row) => total.plus(row.judgment_delta ?? "0"), new D(0)).toString() } : {}),
    adjusted_yhat: sum("adjusted_yhat").toString(), delta: delta.toString(),
    delta_pct: base.isZero() ? null : delta.div(base.abs()).times(100).toString() } };
}
