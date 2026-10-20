import { describe, expect, it } from "vitest";
import { documentSchema, number, stateSchema } from "./api";
import { closeDocument, openDocument, reopenDocument, updateDocument } from "./domain";

describe("independent, recoverable analysis documents", () => {
  it("preserves intake, recipes and notes across close and reload", () => {
    const doc = documentSchema.parse({ id: "one", intake: { horizon: 12, path: "C:\\data.csv" }, note: "Review seasonality" });
    let state = openDocument(stateSchema.parse({}), doc, true);
    state = updateDocument(state, "one", { series: "A" });
    state = closeDocument(state, "one");
    const recovered = reopenDocument(stateSchema.parse(JSON.parse(JSON.stringify(state))), "one");
    expect(recovered.documents[0]).toMatchObject({ intake: { horizon: 12 }, series: "A", note: "Review seasonality" });
  });
  it("ordinary navigation reuses the tab but archives its draft", () => {
    const first = documentSchema.parse({ id: "one", runId: "run-one", note: "Keep me" });
    let state = openDocument(stateSchema.parse({}), first, true);
    state = openDocument(state, documentSchema.parse({ id: "two", runId: "run-two" }));
    expect(state.documents).toHaveLength(1);
    expect(state.activeId).toBe("two");
    expect(state.closedDocuments[0].note).toBe("Keep me");
    state = updateDocument(state, "one", { note: "Late response for the old document" });
    expect(state.documents[0].note).toBe("");
  });
  it("two tabs keep independent scopes", () => {
    let state = openDocument(stateSchema.parse({}), documentSchema.parse({ id: "one", series: "A" }), true);
    state = openDocument(state, documentSchema.parse({ id: "two", series: "B" }), true);
    state = updateDocument(state, "two", { series: "C" });
    expect(state.documents.map(doc => doc.series)).toEqual(["A", "C"]);
  });
  it("defaults old documents to timelines and persists independent seasonal controls", () => {
    const first = documentSchema.parse({ id: "one" });
    expect(first.analysis).toMatchObject({ forecastLayout: "timeline", seasonalAlignment: "weekday", seasonalPeriods: 3 });
    let state = openDocument(stateSchema.parse({}), first, true);
    state = openDocument(state, documentSchema.parse({ id: "two" }), true);
    state = updateDocument(state, "one", { analysis: { forecastLayout: "seasonal", seasonalFocus: "2026-10", seasonalPeriods: 6, seasonalAlignment: "date" } });
    const restored = stateSchema.parse(JSON.parse(JSON.stringify(state)));
    expect(restored.documents[0].analysis).toMatchObject({ forecastLayout: "seasonal", seasonalFocus: "2026-10", seasonalPeriods: 6, seasonalAlignment: "date" });
    expect(restored.documents[1].analysis.forecastLayout).toBe("timeline");
  });
  it("an asynchronous source response cannot erase edits to other draft fields", () => {
    let state = openDocument(stateSchema.parse({}), documentSchema.parse({ id: "one" }), true);
    state = updateDocument(state, "one", { maintenance: { knownAsOf: "2025-04-01T00:00:00Z" } });
    state = updateDocument(state, "one", { maintenance: { sourceId: "profile-result", mapping: { freq: "MS" } } });
    expect(state.documents[0].maintenance).toMatchObject({ sourceId: "profile-result", knownAsOf: "2025-04-01T00:00:00Z" });
  });
});

it("keeps missing values distinct from zero in charts", () => {
  expect(number(null)).toBeNull();
  expect(number("")).toBeNull();
  expect(number("NaN")).toBeNull();
  expect(number(false)).toBeNull();
  expect(number(0)).toBe(0);
});
