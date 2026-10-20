import { useEffect, useState } from "react";
import { call, failure, fileViewSchema, sourceViewSchema, text } from "./api";
import type { Document, Row } from "./api";
import { ErrorNotice, Table } from "./components";
import type { z } from "zod";

export function SavedSource({ document: doc, inspect, useSource }: {
  document: Document; inspect: (row: Row) => void; useSource: (source: z.infer<typeof sourceViewSchema>) => void;
}) {
  const [view, setView] = useState<z.infer<typeof sourceViewSchema>>(), [offset, setOffset] = useState(0), [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setView(undefined); setError("");
    void call("source_view", { sourceId: doc.intake.sourceId, offset, limit: 100 }, sourceViewSchema, controller.signal)
      .then(setView).catch(err => { if (!controller.signal.aborted) setError(failure(err)); });
    return () => controller.abort();
  }, [doc.intake.sourceId, offset, retry]);
  if (error) return <><ErrorNotice error={error}/><button onClick={() => setRetry(retry + 1)}>Retry source preview</button></>;
  if (!view) return <p role="status" className="loading">Reading saved source…</p>;
  return <section>
    <div className="section-heading"><div><h2>Saved data snapshot</h2><p>{text(view.profile.series_count)} series · {view.mapping.freq} · {view.total.toLocaleString()} previewable observations</p></div>
      <button className="primary" onClick={() => useSource(view)}>Use for a forecast</button></div>
    <p className="footnote">Read-only. Opening this snapshot does not run a model or refresh a provider.
      {view.withheldRows > 0 && ` ${view.withheldRows.toLocaleString()} protected observations remain withheld; this preview uses only the saved tuning allocation.`}</p>
    {view.requiresRepair && <p className="notice">Missing observations need an explicit repair decision before modeling.</p>}
    <Table rows={view.rows} columns={view.columns} inspect={inspect} title="Saved source values"/>
    <div className="table-footer"><span>{offset + 1}–{offset + view.rows.length} of {view.total.toLocaleString()} previewable rows</span><div className="toolbar">
      <button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous data page</button>
      <button disabled={view.nextOffset === null} onClick={() => setOffset(view.nextOffset ?? 0)}>Next data page</button>
      <button onClick={() => inspect({ sourceId: view.sourceId, sha256: view.sha256, mapping: view.mapping, profile: view.profile })}>Source details</button>
    </div></div>
  </section>;
}

export function SavedFile({ document: doc, inspect }: { document: Document; inspect: (row: Row) => void }) {
  const [view, setView] = useState<z.infer<typeof fileViewSchema>>(), [offset, setOffset] = useState(0), [pages, setPages] = useState<number[]>([]);
  const [error, setError] = useState(""), [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setView(undefined); setError("");
    void call("file_view", { runId: doc.runId, name: doc.artifactName, offset, limit: 100 }, fileViewSchema, controller.signal)
      .then(setView).catch(err => { if (!controller.signal.aborted) setError(failure(err)); });
    return () => controller.abort();
  }, [doc.runId, doc.artifactName, offset, retry]);
  if (error) return <><ErrorNotice error={error}/><button onClick={() => setRetry(retry + 1)}>Retry file preview</button></>;
  if (!view) return <p className="loading" role="status">Reading verified file…</p>;
  return <section>
    <div className="section-heading"><h2>{doc.artifactName}</h2><button onClick={() => inspect({ path: view.path, sha256: view.sha256, bytes: view.bytes, runId: doc.runId })}>File details</button></div>
    <p className="footnote">Read-only · {(view.bytes / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} KiB · exact saved version. Scripts and HTML are never executed.</p>
    {view.notice && <p className="notice">{view.notice}</p>}
    {view.format === "table" && <><Table rows={view.rows} columns={view.columns} inspect={inspect} title="Saved file values"/>
      <div className="table-footer"><span>{view.total.toLocaleString()} rows in file</span><div className="toolbar">
        <button disabled={!pages.length} onClick={() => { setOffset(pages.at(-1) ?? 0); setPages(pages.slice(0, -1)); }}>Previous file page</button>
        <button disabled={view.nextOffset === null} onClick={() => { setPages([...pages, offset]); setOffset(view.nextOffset ?? 0); }}>Next file page</button>
      </div></div></>}
    {view.format === "text" && <pre className="file-preview" tabIndex={0}>{view.text}</pre>}
    {view.truncated && <p className="notice">Showing the first 64 KiB. The full file is unchanged at the verified local path.</p>}
    <details className="file-location"><summary>Verified local path</summary><p>{view.path}</p><p>SHA-256: {view.sha256}</p></details>
  </section>;
}
