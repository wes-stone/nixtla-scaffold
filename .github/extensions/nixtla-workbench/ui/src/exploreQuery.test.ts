import { expect, it } from "vitest";
import { documentSchema, stateSchema } from "./api";
import { closeDocument, openDocument, reopenDocument, updateDocument } from "./domain";
import { datasetKey, exploreDraftSchema, exploreQuerySchema, initialQuery } from "./exploreQuery";

it("keeps legacy documents small and accepts the Explore data view", () => {
  expect(documentSchema.parse({ id: "legacy" }).explore).toBeUndefined();
  expect(documentSchema.parse({ id: "new", view: "Explore data" }).view).toBe("Explore data");
});

it("retains a pinned query and saved revision across close/reload without changing forecast controls", () => {
  const query = { ...initialQuery({ kind: "artifact", runId: "immutable", name: "appendix/forecast_long.csv" }),
    mode: "pivot" as const, rows: ["ds"], column: "model", value: "yhat" };
  let state = openDocument(stateSchema.parse({}), documentSchema.parse({ id: "one", view: "Explore data",
    analysis: { previewModel: "AutoARIMA" }, explore: { query, name: "Monthly models", savedId: "saved", revision: 2 } }), true);
  state = updateDocument(state, "one", { note: "Unrelated edit" });
  state = reopenDocument(stateSchema.parse(JSON.parse(JSON.stringify(closeDocument(state, "one")))), "one");
  expect(state.documents[0].explore).toEqual({ query, name: "Monthly models", savedId: "saved", revision: 2 });
  expect(state.documents[0].analysis.previewModel).toBe("AutoARIMA");
});

it("normalizes a serialized recipe and preserves text filter values", () => {
  const query = initialQuery({ kind: "source", sourceId: "source" });
  query.filters = [{ column: "unique_id", operator: "eq", value: "001" }];
  expect(exploreDraftSchema.parse({ query }).query.filters[0].value).toBe("001");
  expect(datasetKey(exploreQuerySchema.parse(query).dataset)).toBe(datasetKey(query.dataset));
  expect(exploreQuerySchema.safeParse({ ...query, rows: ["a", "b", "c", "d"] }).success).toBe(false);
});
