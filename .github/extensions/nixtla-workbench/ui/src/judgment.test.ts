import { describe, expect, it } from "vitest";
import { calculateJudgment, judgmentSummary, neutralCalendarLayer, neutralJudgment } from "./judgment";
import { documentSchema, stateSchema } from "./api";
import { closeDocument, reopenDocument } from "./domain";

const rows = [{ ds: "2026-10-01", model_yhat: "200", selected_yhat: "100" }, { ds: "2026-11-01", model_yhat: "300", selected_yhat: "120" }];
describe("exact analyst adjustments", () => {
  it("matches decimal backend haircuts and totals", () => {
    const result = calculateJudgment(rows, { ...neutralJudgment(), percent: "-10" });
    expect(result.rows.map(row => row.adjusted_yhat)).toEqual(["180", "270"]);
    expect(result.totals).toEqual({ model_yhat: "500", selected_yhat: "220", adjusted_yhat: "450", delta: "-50", delta_pct: "-10" });
    const decimal = calculateJudgment([{ ds: "2026-10-01", model_yhat: "0.1", selected_yhat: "0" }], { ...neutralJudgment(), percent: "-10", amount: "-0.1" });
    expect(decimal.rows[0].adjusted_yhat).toBe("-0.01");
  });
  it("replaces bulk changes and honors explicit zero without compounding", () => {
    const settings = { ...neutralJudgment(), percent: "-10", amount: "-5", edits: [
      { ds: "2026-10-01", percent: "20", amount: "", override: "", note: "Launch" },
      { ds: "2026-11-01", percent: "", amount: "", override: "0", note: "Closure" },
    ] };
    const result = calculateJudgment(rows, settings);
    expect(result.rows.map(row => row.adjusted_yhat)).toEqual(["235", "0"]);
    expect(result.totals.adjusted_yhat).toBe("235");
    expect(result.rows.at(-1)?.cumulative_delta).toBe("-265");
  });
  it("does not coerce invalid edits, range or zero denominators", () => {
    for (const amount of ["", "NaN", "Infinity", "1e999", "1_0"]) expect(() => calculateJudgment(rows, { ...neutralJudgment(), amount })).toThrow();
    expect(() => calculateJudgment(rows, { ...neutralJudgment(), start: "2028-01-01" })).toThrow();
    expect(calculateJudgment([{ ds: "2026-10-01", model_yhat: "0", selected_yhat: "0" }], neutralJudgment()).totals.delta_pct).toBeNull();
  });
  it("persists worksheet assumptions without bloating legacy documents", () => {
    expect(documentSchema.parse({ id: "legacy" }).judgment).toBeUndefined();
    const state = stateSchema.parse({ activeId: "a", documents: [{ id: "a", view: "Adjust", judgment: { ...neutralJudgment(), percent: "-10" } }] });
    const reopened = reopenDocument(closeDocument(state, "a"), "a");
    expect(reopened.documents[0].judgment?.percent).toBe("-10");
    expect(reopened.documents[0].view).toBe("Adjust");
    expect(reopened.documents[0].judgment?.calendar).toBeUndefined();
  });
  it("applies final tweaks after calendar without changing raw-model comparisons", () => {
    const staged = [
      { ...rows[0], calendar_yhat: "250", calendar_delta: "50" },
      { ...rows[1], calendar_yhat: "250", calendar_delta: "-50" },
    ];
    const result = calculateJudgment(staged, { ...neutralJudgment(), calendar: { ...neutralCalendarLayer(), enabled: true }, percent: "-10", amount: "-5" });
    expect(result.rows.map(row => row.adjusted_yhat)).toEqual(["220", "220"]);
    expect(result.rows.map(row => row.judgment_delta)).toEqual(["-30", "-30"]);
    expect(result.rows.map(row => row.delta)).toEqual(["20", "-80"]);
    expect(result.totals.calendar_yhat).toBe("500");
    expect(result.totals.calendar_delta).toBe("0");
    expect(result.totals.judgment_delta).toBe("-60");
    expect(result.totals.adjusted_yhat).toBe("440");
    const zero = calculateJudgment(staged, { ...neutralJudgment(), edits: [{ ds: rows[0].ds, percent: "", amount: "", override: "0", note: "" }] });
    expect(zero.rows[0].adjusted_yhat).toBe("0");
    expect(zero.rows[0].judgment_delta).toBe("-250");
  });
  it("persists optional layered recipes through document recovery", () => {
    const calendar = { ...neutralCalendarLayer(), enabled: true, weekendWeight: 0.4, holidayLosses: [{ ds: "2026-11-01", days: 2 }] };
    const state = stateSchema.parse({ activeId: "a", documents: [{ id: "a", view: "Adjust", judgment: { ...neutralJudgment(), calendar } }] });
    expect(reopenDocument(closeDocument(state, "a"), "a").documents[0].judgment?.calendar).toEqual(calendar);
  });
  it("summarizes absolute-only and period-restricted judgment without hiding changes", () => {
    expect(judgmentSummary(neutralJudgment())).toBe("No final changes");
    expect(judgmentSummary({ ...neutralJudgment(), amount: "-5" }, "minutes")).toBe("-5 minutes/period · Entire horizon");
    expect(judgmentSummary({ ...neutralJudgment(), percent: "-10", start: "2026-10-01", end: "2026-11-01" }))
      .toBe("-10% · 2026-10-01 to 2026-11-01");
    expect(judgmentSummary({ ...neutralJudgment(), edits: [{ ds: "2026-10-01", percent: "", amount: "", override: "0", note: "" }] }))
      .toBe("1 row input");
  });
});
