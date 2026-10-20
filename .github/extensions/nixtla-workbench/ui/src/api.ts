import { z } from "zod";
import { exploreDraftSchema } from "./exploreQuery";
import { calendarSettingsSchema } from "./calendarSettings";
import { judgmentDraftSchema } from "./judgment";

export const rowSchema = z.record(z.unknown());
export type Row = z.infer<typeof rowSchema>;
export const runtimeSchema = z.object({
  schema_version: z.literal("nixtla_scaffold.runtime.v1"),
  probed_at: z.string(), python: z.string(), python_version: z.string(),
  cached: z.boolean().optional(), probe_seconds: z.number().optional(),
  package_version: z.string().nullable(), fingerprint: z.string(),
  registered_recipes: z.number(), importable_recipes: z.number(), scope: z.string(),
  engines: z.array(z.object({
    package: z.string(), version: z.string().nullable(), recipes: z.array(z.string()),
    integration: z.string(), compatibility: z.string(), tested_version: z.string().nullable(),
    status: z.enum(["not_installed", "importable", "import_failed", "probe_failed", "timeout"]),
    error: z.string(), data_eligibility: z.literal("not_evaluated"), execution: z.literal("not_attempted"),
    limitations: z.array(z.string()),
  })),
});
export type RuntimeCapabilities = z.infer<typeof runtimeSchema>;
export const runSchema = z.object({
  id: z.string(), name: z.string(), created_at: z.string(), metadata: rowSchema,
});
export const jobSchema = z.object({
  id: z.string(), kind: z.string(), status: z.string(), error: z.string().nullable(),
  result: rowSchema.nullable(), created_at: z.string(), updated_at: z.string(),
  remainingSeconds: z.number().nullable().optional(), resumeAvailable: z.boolean().default(false),
});
export const documentSchema = z.object({
  id: z.string(), runId: z.string().optional(), series: z.string().default(""),
  kind: z.enum(["analysis", "source", "artifact"]).default("analysis"),
  artifactName: z.string().default(""),
  view: z.enum(["Data", "Explore data", "Forecast", "Adjust", "Models", "Backtests", "Diagnostics", "Compare", "Research", "Pipeline"]).default("Forecast"),
  explore: exploreDraftSchema.optional(),
  calendar: calendarSettingsSchema.optional(),
  judgment: judgmentDraftSchema.optional(),
  title: z.string().default("New analysis"),
  intake: z.object({
    path: z.string().default(""), sourceId: z.string().default(""),
    mapping: z.record(z.string()).default({}), horizon: z.number().default(6),
    policy: z.string().default("standard"), repair: z.string().default(""),
    advanced: z.string().default(""), holdoutWindows: z.number().default(1),
  }).default({}),
  recipe: z.object({
    policy: z.string().default("equal"), members: z.array(z.string()).default([]),
    weights: z.record(z.string()).default({}),
  }).default({}),
  note: z.string().default(""),
  blendId: z.string().default(""),
  intervalLevel: z.number().default(0),
  analysis: z.object({
    section: z.enum(["Configuration", "Intervals", "Residuals", "Seasonality", "Hierarchy", "Drivers", "Ledger", "Challengers", "Feeders"]).default("Intervals"),
    models: z.array(z.string()).default(["*"]), family: z.string().default(""),
    metric: z.string().default("rmse"), cutoff: z.string().default(""),
    fixedAxes: z.boolean().default(true), scale: z.enum(["raw", "thousands", "millions", "billions"]).default("raw"),
    yearStart: z.number().min(1).max(12).default(1), artifact: z.string().default(""),
    previewModel: z.string().default(""), historyPeriods: z.number().int().min(0).max(2000).default(24),
    modelMode: z.enum(["focus", "compare"]).default("focus"),
    forecastPeriods: z.number().int().min(1).max(2000).default(24),
    includeActuals: z.boolean().default(true), summaryScope: z.enum(["series", "all"]).default("series"),
    forecastLayout: z.enum(["timeline", "seasonal"]).default("timeline"),
    seasonalAlignment: z.enum(["weekday", "date"]).default("weekday"),
    seasonalFocus: z.string().default(""),
    seasonalPeriods: z.union([z.literal(1), z.literal(3), z.literal(6), z.literal(12)]).default(3),
  }).default({}),
  comparison: z.enum(["models", "versions", "external"]).default("models"),
  externalPath: z.string().default(""),
  referenceRunId: z.string().default(""),
  maintenance: z.object({
    path: z.string().default(""), sourceId: z.string().default(""),
    knownAsOf: z.string().default(""), label: z.string().default(""),
    kind: z.enum(["refresh", "actuals", "lock"]).default("refresh"),
    mapping: z.record(z.string()).default({}),
  }).default({}),
  pipeline: z.object({
    configPath: z.string().default(""), script: z.string().default(""),
    query: z.string().default(""), transform: z.string().default(""),
    args: z.string().default('["{output:extract}"]'),
    dataOnly: z.boolean().default(false),
    frequency: z.string().default(""),
    proposalId: z.string().default(""), expectedReceipt: z.string().default(""),
  }).default({}),
});
export type Document = z.infer<typeof documentSchema>;
export type DocumentPatch = Omit<Partial<Document>, "intake" | "maintenance" | "pipeline" | "recipe" | "analysis"> & {
  intake?: Partial<Document["intake"]>; maintenance?: Partial<Document["maintenance"]>;
  pipeline?: Partial<Document["pipeline"]>; recipe?: Partial<Document["recipe"]>;
  analysis?: Partial<Document["analysis"]>;
};
export const stateSchema = z.object({
  documents: z.array(documentSchema).max(30).default([]),
  closedDocuments: z.array(documentSchema).max(200).default([]),
  activeId: z.string().default(""),
  theme: z.enum(["system", "light", "dark"]).default("system"),
  explorerExpanded: z.array(z.string()).max(500).default(["forecasts", "sources"]),
});
export type WorkspaceState = z.infer<typeof stateSchema>;
export const catalogSchema = z.object({
  initialized: z.boolean(), runs: z.array(runSchema), jobs: z.array(jobSchema),
  sources: z.array(z.object({ id: z.string(), name: z.string(), metadata: rowSchema })),
  state: rowSchema, revision: z.number(), catalogTruncated: z.boolean().optional(),
  nextOffset: z.number().nullable().optional(), offset: z.number().default(0), totalRuns: z.number().default(0),
  nextSourceOffset: z.number().nullable().default(null), sourceOffset: z.number().default(0), totalSources: z.number().default(0),
  totalNotes: z.number().default(0), totalJobs: z.number().default(0), workspace: z.string().default(""),
  totalForecasts: z.number().default(0),
});
export type Catalog = z.infer<typeof catalogSchema>;
export const libraryItemSchema = z.object({
  kind: z.enum(["family", "version", "source", "artifact", "note", "job"]),
  id: z.string(), name: z.string(), family: z.string().nullable().optional(),
  runId: z.string().nullable().optional(), sourceId: z.string().nullable().optional(),
  artifactName: z.string().nullable().optional(), createdAt: z.string().optional(),
  versionCount: z.number().optional(), status: z.string().optional(), rows: z.number().nullable().optional(),
  seriesCount: z.number().nullable().optional(), frequency: z.string().nullable().optional(),
  adopted: z.union([z.boolean(), z.number()]).nullable().optional(), purpose: z.string().nullable().optional(),
  versionLabel: z.string().nullable().optional(),
});
export type LibraryItem = z.infer<typeof libraryItemSchema>;
export const libraryPageSchema = z.object({
  items: z.array(libraryItemSchema), total: z.number(), offset: z.number(), nextOffset: z.number().nullable(),
});
export const sourceViewSchema = z.object({
  sourceId: z.string(), name: z.string(), mapping: z.record(z.string()), horizon: z.number(),
  profile: rowSchema, requiresRepair: z.boolean(), rows: z.array(rowSchema), columns: z.array(z.string()),
  total: z.number(), withheldRows: z.number(), offset: z.number(), nextOffset: z.number().nullable(), sha256: z.string(),
});
export const fileViewSchema = z.object({
  name: z.string(), path: z.string(), sha256: z.string(), bytes: z.number(),
  format: z.enum(["table", "text", "file"]), rows: z.array(rowSchema), columns: z.array(z.string()),
  total: z.number(), nextOffset: z.number().nullable(), text: z.string(), truncated: z.boolean(), notice: z.string().optional(),
});
export const viewSchema = z.object({
  rows: z.array(rowSchema), columns: z.array(z.string()), total: z.number(),
  nextOffset: z.number().nullable(),
  offset: z.number().default(0), cutoffs: z.array(z.string()).default([]),
  run: z.object({ id: z.string(), name: z.string(), metadata: rowSchema }),
  spec: rowSchema, profile: rowSchema, series: z.array(z.string()), seriesTruncated: z.boolean(),
  models: z.array(z.string()), artifacts: z.array(z.string()), gate: rowSchema,
  hypotheses: z.array(rowSchema).default([]),
  blends: z.array(z.object({ id: z.string(), recipe: rowSchema, receipt: rowSchema, receiptHash: z.string() })),
  notes: z.array(rowSchema),
});
export type View = z.infer<typeof viewSchema>;
export const profileSchema = z.object({
  columns: z.array(z.string()), mapping: z.record(z.string()),
  questions: z.array(z.string()), sourceId: z.string().nullable(), rows: z.number(),
  preview: z.array(rowSchema), findings: z.array(z.string()).optional(),
  assumptions: z.array(z.string()).optional(), profile: rowSchema.optional(),
});
export type Profile = z.infer<typeof profileSchema>;

const token = typeof location === "undefined" ? "" : location.hash.slice(1);
// The fragment is not sent in HTTP requests or referrers. Retain it for reload;
// never persist this short-lived local capability in browser storage.

export async function call<S extends z.ZodTypeAny>(
  operation: string, input: Record<string, unknown>, schema: S, signal?: AbortSignal,
): Promise<z.infer<S>> {
  const response = await fetch("/api", {
    method: "POST", headers: { "Content-Type": "application/json", "X-Workbench-Token": token },
    body: JSON.stringify({ operation, input }), signal,
  });
  const envelope = z.object({
    result: z.unknown().optional(),
    error: z.object({ code: z.string(), message: z.string() }).optional(),
  }).parse(await response.json());
  if (!response.ok || envelope.error) throw new Error(envelope.error?.message ?? "The local operation failed.");
  return schema.parse(envelope.result);
}

export function number(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value !== "number" && typeof value !== "string") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function text(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function failure(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected local error occurred.";
}

export const activeJob = (status: string) => ["queued", "running", "cancel_requested"].includes(status);
