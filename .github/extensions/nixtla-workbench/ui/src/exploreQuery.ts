import { z } from "zod";

export const datasetSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("artifact"), runId: z.string(), name: z.string() }),
  z.object({ kind: z.literal("source"), sourceId: z.string() }),
]);
export const exploreQuerySchema = z.object({
  version: z.literal(1), dataset: datasetSchema, mode: z.enum(["rows", "pivot"]),
  select: z.array(z.string()).max(200), rows: z.array(z.string()).max(3),
  column: z.string(), value: z.string(),
  aggregate: z.enum(["sum", "avg", "min", "max", "count", "count_distinct", "count_rows"]),
  filters: z.array(z.object({
    column: z.string(), operator: z.enum(["eq", "neq", "contains", "gte", "lte", "is_null", "not_null"]), value: z.string(),
  })).max(12),
  sort: z.object({ column: z.string(), direction: z.enum(["asc", "desc"]) }),
});
export const exploreDraftSchema = z.object({
  query: exploreQuerySchema, name: z.string().default(""), savedId: z.string().default(""), revision: z.number().default(0),
});
export type ExploreQuery = z.infer<typeof exploreQuerySchema>;
export type Dataset = z.infer<typeof datasetSchema>;
export const datasetKey = (dataset: Dataset) => JSON.stringify(dataset);
export const datasetLabel = (dataset: Dataset) => dataset.kind === "artifact" ? dataset.name : `Source ${dataset.sourceId.slice(0, 8)}`;
export const initialQuery = (dataset: Dataset): ExploreQuery => ({
  version: 1, dataset, mode: "rows", select: [], rows: [], column: "", value: "", aggregate: "sum",
  filters: [], sort: { column: "", direction: "asc" },
});
