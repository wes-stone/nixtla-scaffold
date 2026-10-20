import { useEffect, useId, useState } from "react";
import { z } from "zod";
import { call, failure, rowSchema, text } from "./api";
import type { Document, DocumentPatch, Row, View } from "./api";
import { ErrorNotice, Table } from "./components";
import { scales } from "./format";
import { intervalComparison, intervalScoreView, legacyIntervalWarning } from "./intervalScores";
export { ModelBrowser as ModelPicker } from "./ModelBrowser";

export const evidenceSchema = z.object({
  name: z.string(), available: z.boolean(), rows: z.array(rowSchema), columns: z.array(z.string()),
  total: z.number(), offset: z.number(), nextOffset: z.number().nullable(),
  models: z.array(z.string()), cutoffs: z.array(z.string()), seriesScope: z.string(),
  bounds: z.record(z.object({ min: z.number(), max: z.number() })), sha256: z.string().nullable(), message: z.string(),
});
export type Evidence = z.infer<typeof evidenceSchema>;
export type AnalysisProps = {
  document: Document; view: View; dark: boolean; update: (patch: DocumentPatch) => void; inspect: (row: Row) => void;
};
export function useEvidence(doc: Document, names: string[], options: Record<string, unknown> = {}, refresh = 0) {
  const request = JSON.stringify({ runId: doc.runId, series: doc.series, limit: 2000, ...options });
  const key = JSON.stringify([request, names, refresh]);
  const [state, setState] = useState<{ key: string; data: Record<string, Evidence>; error: string }>({ key: "", data: {}, error: "" });
  useEffect(() => {
    const controller = new AbortController();
    void Promise.all(names.map(async name => [name, await call("evidence", { ...JSON.parse(request), name }, evidenceSchema, controller.signal)] as const))
      .then(entries => { if (!controller.signal.aborted) setState({ key, data: Object.fromEntries(entries), error: "" }); })
      .catch(error => { if (!controller.signal.aborted) setState({ key, data: {}, error: failure(error) }); });
    return () => controller.abort();
  }, [key]);
  return { data: state.key === key ? state.data : {}, error: state.key === key ? state.error : "", loading: state.key !== key };
}
export function EvidenceStatus({ result, reportPartial = true }: { result: ReturnType<typeof useEvidence>; reportPartial?: boolean }) {
  if (result.error) return <ErrorNotice error={result.error}/>;
  if (result.loading) return <p role="status" className="loading">Reading saved analytical evidence...</p>;
  const partial = reportPartial ? Object.values(result.data).filter(item => item.total > item.rows.length) : [];
  const missing = reportPartial ? Object.values(result.data).filter(item => !item.available) : [];
  return <>{missing.map(item => <p key={item.name} className="empty-inline">{item.name}: {item.message}</p>)}
    {partial.length > 0 && <p className="notice" role="status">Bounded chart preview: {partial.map(item => `${item.name}: ${item.rows.length} of ${item.total} rows`).join("; ")}. Narrow the scope or use the paged table and full-scope export. Charts do not represent omitted rows.</p>}</>;
}
export function ScaleControl({ document: doc, update }: Pick<AnalysisProps, "document" | "update">) {
  const hintId = useId();
  return <div className="scale-control"><label>Display units<select aria-label="Display units" aria-describedby={hintId} value={doc.analysis.scale} onChange={event => update({ analysis: { scale: event.target.value as Document["analysis"]["scale"] } })}>
    {Object.keys(scales).map(scale => <option key={scale} value={scale}>{scale === "raw" ? "Original units" : scale[0].toUpperCase() + scale.slice(1)}</option>)}
  </select></label><small id={hintId}>Measurements only; ratios remain unscaled.</small></div>;
}
export function selectedRows(rows: Row[], doc: Document) {
  return doc.analysis.models.includes("*") ? rows : rows.filter(row => doc.analysis.models.includes(text(row.model)));
}

export function EvidenceTable({ document: doc, name, title, inspect, seriesColumn, scope, models, cutoff, decimalPlaces = 2, scale }: {
  document: Document; name: string; title: string; inspect: (row: Row) => void; seriesColumn?: string; scope?: string; models?: string[]; cutoff?: string; decimalPlaces?: number; scale?: number;
}) {
  const [offset, setOffset] = useState(0), [receipt, setReceipt] = useState(""), [error, setError] = useState("");
  const identity = JSON.stringify([doc.runId, doc.series, name, seriesColumn, scope, models, cutoff]);
  useEffect(() => { setOffset(0); setReceipt(""); setError(""); }, [identity]);
  const filters = { series: scope ?? doc.series, ...(seriesColumn ? { seriesColumn } : {}), ...(models?.length ? { models } : {}), ...(cutoff ? { cutoff } : {}) };
  const result = useEvidence(doc, [name], { ...filters, offset, limit: 100 });
  const data = result.data[name];
  const interval = name === intervalComparison ? intervalScoreView(data?.rows ?? []) : null;
  return <section className="evidence-table">
    <EvidenceStatus result={result} reportPartial={false}/>
    {data && (!data.available ? <p className="empty-inline">{title}: {data.message}</p> : <>
      {data.seriesScope === "run" && <p className="footnote">Run-wide evidence; this artifact has no applicable selected-series scope.</p>}
      {interval?.legacy && <p className="notice" role="status">{legacyIntervalWarning}</p>}
      <Table rows={interval?.rows ?? data.rows} columns={data.columns} title={title} inspect={inspect} scale={scale ?? scales[doc.analysis.scale]} decimalPlaces={decimalPlaces}/>
      <div className="table-footer"><span>{data.total ? data.offset + 1 : 0}–{data.offset + data.rows.length} of {data.total} · Original precision retained</span><div className="toolbar">
        <button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous evidence page</button>
        <button disabled={data.nextOffset === null} onClick={() => setOffset(data.nextOffset ?? 0)}>Next evidence page</button>
        <button onClick={() => { setError(""); void call("export_evidence", { runId: doc.runId, name, ...filters }, z.object({ path: z.string(), rows: z.number() }))
          .then(output => setReceipt(`Saved ${output.rows} full-precision rows to ${output.path}`)).catch(error => setError(failure(error))); }}>Export full scope</button>
      </div></div>
    </>)}
    {error && <ErrorNotice error={error}/>}
    {receipt && <p role="status" className="save-receipt">{receipt}</p>}
  </section>;
}
