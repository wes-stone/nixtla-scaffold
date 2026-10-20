import { useEffect, useRef, useState } from "react";
import { Download, Play, Plus, Save, X } from "lucide-react";
import { z } from "zod";
import { call, failure, rowSchema } from "./api";
import type { Document, DocumentPatch, Row } from "./api";
import { datasetKey, datasetLabel, datasetSchema, exploreDraftSchema, exploreQuerySchema, initialQuery } from "./exploreQuery";
import type { Dataset, ExploreQuery } from "./exploreQuery";
import { ErrorNotice, Table } from "./components";

const sourceSchema = z.object({
  dataset: datasetSchema, label: z.string(), sha256: z.string(), rows: z.number(), withheldRows: z.number(),
  columns: z.array(z.object({ name: z.string(), type: z.enum(["number", "text"]), nulls: z.number() })),
});
const resultSchema = z.object({
  rows: z.array(rowSchema), columns: z.array(z.string()), labels: z.record(z.string()), types: z.record(z.string()),
  total: z.number(), offset: z.number(), nextOffset: z.number().nullable(), matchedRows: z.number(),
  source: sourceSchema, query: exploreQuerySchema, queryHash: z.string(),
});
const savedSchema = z.object({
  id: z.string(), revision: z.number(), name: z.string(), query: exploreQuerySchema, updatedAt: z.string(),
  source: z.object({ label: z.string(), sha256: z.string() }),
});
const exportSchema = z.object({
  id: z.string(), path: z.string(), queryPath: z.string(), rows: z.number(), matchedRows: z.number(),
  sha256: z.string(), formulaEscapedCells: z.number(), bytes: z.number(),
});
const aggregateLabels: Record<ExploreQuery["aggregate"], string> = {
  sum: "Sum", avg: "Average", min: "Minimum", max: "Maximum",
  count: "Count nonempty", count_distinct: "Distinct count", count_rows: "Count rows",
};
const operatorLabels: Record<ExploreQuery["filters"][number]["operator"], string> = {
  eq: "Equals", neq: "Does not equal", contains: "Contains", gte: "At least / on or after",
  lte: "At most / on or before", is_null: "Is empty", not_null: "Is not empty",
};
type Result = z.infer<typeof resultSchema>;
type Saved = z.infer<typeof savedSchema>;

export function ExploreData({ document: doc, update, inspect }: {
  document: Document; update: (patch: DocumentPatch) => void; inspect: (row: Row) => void;
}) {
  const [tables, setTables] = useState<Array<{ label: string; dataset: Dataset }>>([]);
  const [saved, setSaved] = useState<Saved[]>([]);
  const [source, setSource] = useState<z.infer<typeof sourceSchema>>();
  const [result, setResult] = useState<Result>();
  const [pages, setPages] = useState<number[]>([]);
  const [receipt, setReceipt] = useState<z.infer<typeof exportSchema>>();
  const [error, setError] = useState(""), [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true), [busy, setBusy] = useState("");
  const [fieldSearch, setFieldSearch] = useState(""), [savedSearch, setSavedSearch] = useState("");
  const [showSaved, setShowSaved] = useState(false), [retry, setRetry] = useState(0);
  const [reopen, setReopen] = useState(0);
  const generation = useRef(0), busyRef = useRef(false);
  const draft = doc.explore;
  const query = draft?.query;
  const queryKey = JSON.stringify(query);
  const currentQueryKey = useRef(queryKey); currentQueryKey.current = queryKey;
  const sourceKey = query ? datasetKey(query.dataset) : "";
  const sameSource = source && query && datasetKey(source.dataset) === sourceKey;
  const stale = !!result && JSON.stringify(result.query) !== queryKey;
  const change = (patch: Partial<ExploreQuery>) => {
    if (draft) update({ explore: { ...draft, query: { ...draft.query, ...patch,
      ...(patch.filters && draft.query.mode === "pivot" ? { sort: { column: "", direction: "asc" } as const } : {}) } } });
    setReceipt(undefined); setMessage("");
  };
  const choose = (dataset: Dataset) => {
    update({ explore: exploreDraftSchema.parse({ query: initialQuery(dataset) }) });
    setReceipt(undefined); setResult(undefined); setFieldSearch(""); setMessage(""); setError("");
  };
  const loadSaved = async () => setSaved((await call("explore_queries", {}, z.object({ queries: z.array(savedSchema) }))).queries);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError("");
    void Promise.all([
      call("explore_catalog", { ...(doc.runId ? { runId: doc.runId } : {}), ...(doc.intake.sourceId ? { sourceId: doc.intake.sourceId } : {}) },
        z.object({ tables: z.array(z.object({ label: z.string(), dataset: datasetSchema })) }), controller.signal),
      call("explore_queries", {}, z.object({ queries: z.array(savedSchema) }), controller.signal),
    ]).then(([catalog, queries]) => {
      if (controller.signal.aborted) return;
      setTables(catalog.tables); setSaved(queries.queries);
      if (!doc.explore && catalog.tables.length) {
        const selected = catalog.tables.find(item => item.dataset.kind === "artifact" && item.dataset.name === doc.artifactName) ?? catalog.tables[0];
        choose(selected.dataset);
      }
    }).catch(err => { if (!controller.signal.aborted) setError(failure(err)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [doc.runId, doc.intake.sourceId, retry]);
  useEffect(() => {
    if (!query) return;
    const controller = new AbortController();
    const current = ++generation.current;
    setSource(undefined); setResult(undefined); setPages([]); setReceipt(undefined); setError("");
    void call("explore_schema", { dataset: query.dataset }, sourceSchema, controller.signal)
      .then(schema => {
        if (controller.signal.aborted || current !== generation.current) return;
        setSource(schema);
      }).catch(err => { if (!controller.signal.aborted) setError(failure(err)); });
    return () => { controller.abort(); generation.current++; };
  }, [sourceKey, retry]);
  const run = async (definition: ExploreQuery, offset = 0, direction: "reset" | "next" | "previous" = "reset") => {
    if (busyRef.current) return;
    busyRef.current = true; setBusy("query"); setError(""); setReceipt(undefined);
    const key = JSON.stringify(definition), current = generation.current;
    try {
      const next = await call("explore_preview", { query: definition, offset, limit: 100 }, resultSchema);
      if (current === generation.current && key === currentQueryKey.current) {
        setPages(previous => direction === "next" ? [...previous, result?.offset ?? 0] : direction === "previous" ? previous.slice(0, -1) : []);
        setResult(next); setMessage("");
      }
    } catch (err) { if (current === generation.current) setError(failure(err)); }
    finally { busyRef.current = false; setBusy(""); }
  };
  useEffect(() => {
    if (sameSource && query && !result && !busyRef.current) void run(query);
  }, [source, reopen]);
  const save = async (copy: boolean) => {
    if (!draft || busyRef.current) return;
    busyRef.current = true; setBusy("save"); setError("");
    try {
      const next = await call("explore_save", { name: draft.name, query: draft.query,
        ...(!copy && draft.savedId ? { id: draft.savedId, expectedRevision: draft.revision } : {}) }, savedSchema);
      update({ explore: { ...draft, savedId: next.id, revision: next.revision } });
      setMessage(`Saved "${next.name}". Its table, filters, pivot and sorting can be reopened from Saved queries.`);
      await loadSaved();
    } catch (err) { setError(failure(err)); }
    finally { busyRef.current = false; setBusy(""); }
  };
  const exportResult = async () => {
    if (!result || stale || busyRef.current) return;
    busyRef.current = true; setBusy("export"); setError("");
    try { setReceipt(await call("explore_export", { query: result.query }, exportSchema)); }
    catch (err) { setError(failure(err)); }
    finally { busyRef.current = false; setBusy(""); }
  };
  const download = async (kind: "csv" | "query") => {
    if (!receipt || busyRef.current) return;
    busyRef.current = true; setBusy("download"); setError("");
    try {
      const file = await call("explore_download", { id: receipt.id, kind }, z.object({ name: z.string(), base64: z.string(), mediaType: z.string() }));
      const bytes = Uint8Array.from(atob(file.base64), character => character.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], { type: file.mediaType }));
      const link = document.createElement("a"); link.href = url; link.download = file.name; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) { setError(failure(err)); }
    finally { busyRef.current = false; setBusy(""); }
  };
  const openSaved = (item: Saved) => {
    update({ explore: { query: item.query, name: item.name, savedId: item.id, revision: item.revision } });
    setShowSaved(false); setReceipt(undefined); setMessage(""); setError(""); setResult(undefined); setReopen(reopen + 1);
  };
  const columns = sameSource ? source.columns : [];
  const numeric = columns.filter(col => col.type === "number");
  const resultColumns = result?.columns ?? [];
  const selectedColumns = query?.select.length ? query.select : columns.map(col => col.name);
  const availableTables = query && !tables.some(item => datasetKey(item.dataset) === sourceKey)
    ? [...tables, { dataset: query.dataset, label: `${datasetLabel(query.dataset)} · saved query snapshot` }] : tables;
  const toggleField = (field: string) => {
    const next = selectedColumns.includes(field) ? selectedColumns.filter(col => col !== field) : [...selectedColumns, field];
    if (next.length) change({ select: next, sort: { column: "", direction: "asc" } });
  };
  return <section className="data-explorer" aria-label="Explore data">
    <header className="section-heading"><div><h2>Explore data</h2><p>Shape a table, pivot the full snapshot, and keep the query behind every export.</p></div>
      <button aria-expanded={showSaved} onClick={() => { setShowSaved(!showSaved); void loadSaved().catch(err => setError(failure(err))); }}>Saved queries · {saved.length}</button></header>
    {showSaved && <section className="explore-saved" aria-label="Saved exploration queries">
      <label>Find a saved query<input type="search" value={savedSearch} onChange={event => setSavedSearch(event.target.value)}/></label>
      {!saved.length && <p className="empty-inline">Name and save your first query below. Reopening restores its exact snapshot, filters and pivot.</p>}
      <div>{saved.filter(item => `${item.name} ${item.source.label}`.toLowerCase().includes(savedSearch.toLowerCase())).map(item =>
        <button key={item.id} disabled={!!busy} onClick={() => openSaved(item)}><strong>{item.name}</strong><small>{item.source.label} · {item.query.mode === "pivot" ? aggregateLabels[item.query.aggregate] : "Rows"} · {item.updatedAt.slice(0, 10)}</small></button>)}</div>
      {!!saved.length && !saved.some(item => `${item.name} ${item.source.label}`.toLowerCase().includes(savedSearch.toLowerCase())) && <p className="empty-inline">No saved query matches this search.</p>}
    </section>}
    {error && <><ErrorNotice error={error}/><button onClick={() => setRetry(retry + 1)}>Reload exploration data</button></>}
    {loading && <p role="status" className="loading">Reading registered tables and saved queries...</p>}
    {!loading && !query && <p className="empty-inline">Open a saved forecast or source snapshot from the Library to explore its tables. You can also reopen a Saved query.</p>}
    {query && draft && <>
      <fieldset className="explore-builder" disabled={!!busy}>
        <legend className="visually-hidden">Query builder</legend>
        <div className="explore-table-bar">
          <label>Saved table<select aria-label="Exploration table" value={sourceKey} onChange={event => {
            const item = availableTables.find(item => datasetKey(item.dataset) === event.target.value);
            if (item) choose(item.dataset);
          }}>{availableTables.map(item => <option key={datasetKey(item.dataset)} value={datasetKey(item.dataset)}>{item.label}</option>)}</select></label>
          <div className="model-mode" role="group" aria-label="Exploration layout">
            <button aria-pressed={query.mode === "rows"} onClick={() => change({ mode: "rows", sort: { column: "", direction: "asc" } })}>Rows</button>
            <button aria-pressed={query.mode === "pivot"} onClick={() => change({ mode: "pivot", value: query.value || numeric[0]?.name || "",
              aggregate: numeric.length ? query.aggregate : "count_rows", sort: { column: "", direction: "asc" } })}>Pivot</button>
          </div>
        </div>
        {sameSource ? <p className="explore-source-line">{source.rows.toLocaleString()} previewable rows · {source.columns.length} fields · verified snapshot {source.sha256.slice(0, 12)}
          {query.dataset.kind === "artifact" && <> · run {query.dataset.runId.slice(0, 8)}</>}</p> : <p role="status" className="loading">Reading full-table schema...</p>}
        {sameSource && source.withheldRows > 0 && <p className="notice">{source.withheldRows.toLocaleString()} protected observations remain withheld. Pivots, filters and exports use only this snapshot's saved tuning allocation.</p>}
        {sameSource && <div className="explore-layout">
          <aside className="explore-fields" aria-label="Available fields">
            <label>Find a field<input type="search" value={fieldSearch} onChange={event => setFieldSearch(event.target.value)} placeholder="Search fields"/></label>
            <p className="footnote">{query.mode === "rows" ? `${selectedColumns.length} selected · choose columns to export` : "Click + Row to group; choose a numeric value in the pivot."}</p>
            <div className="explore-field-list">{columns.filter(col => col.name.toLowerCase().includes(fieldSearch.toLowerCase())).map(col =>
              <div key={col.name}>{query.mode === "rows" ? <label className="inline"><input type="checkbox" aria-label={`Include ${col.name}`} checked={selectedColumns.includes(col.name)}
                disabled={selectedColumns.length === 1 && selectedColumns.includes(col.name)} onChange={() => toggleField(col.name)}/><span>{col.name}<small>{col.type} · {col.nulls.toLocaleString()} empty</small></span></label> :
                <><span>{col.name}<small>{col.type} · {col.nulls.toLocaleString()} empty</small></span><button aria-label={`Group rows by ${col.name}`}
                  disabled={query.rows.includes(col.name) || col.name === query.column || query.rows.length >= 3}
                  onClick={() => change({ rows: [...query.rows, col.name], sort: { column: "", direction: "asc" } })}>+ Row</button></>}</div>)}
              {!columns.some(col => col.name.toLowerCase().includes(fieldSearch.toLowerCase())) && <p className="empty-inline">No matching fields.</p>}
            </div>
          </aside>
          <div className="explore-configuration">
            {query.mode === "pivot" && <section aria-label="Pivot configuration">
              <h3>Pivot layout</h3>
              <div className="explore-row-fields"><span>Rows <small>up to 3</small></span>{query.rows.map((field, index) => <div key={field}><span>{index + 1}. {field}</span>
                <button className="icon" aria-label={`Remove row field ${field}`} onClick={() => change({ rows: query.rows.filter(col => col !== field), sort: { column: "", direction: "asc" } })}><X size={14}/></button></div>)}
                {!query.rows.length && <small>No row groups: one aggregate across all matching rows.</small>}</div>
              <div className="form-grid">
                <label>Pivot columns<select aria-label="Pivot column" value={query.column} onChange={event => change({ column: event.target.value, sort: { column: "", direction: "asc" } })}>
                  <option value="">No column split</option>{columns.filter(col => !query.rows.includes(col.name)).map(col => <option key={col.name}>{col.name}</option>)}</select></label>
                <label>Calculation<select aria-label="Pivot calculation" value={query.aggregate} onChange={event => change({ aggregate: exploreQuerySchema.shape.aggregate.parse(event.target.value) })}>
                  {Object.entries(aggregateLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
                <label>Values<select aria-label="Pivot value" disabled={query.aggregate === "count_rows"} value={query.value} onChange={event => change({ value: event.target.value })}>
                  <option value="">Choose a field</option>{columns.map(col => <option key={col.name}>{col.name}</option>)}</select></label>
              </div>
              <p className="footnote">Missing combinations stay empty, not zero. Averages and distinct counts are computed from matching source rows, not summed subtotals.</p>
            </section>}
            <section className="explore-filters" aria-label="Query filters"><div className="section-heading"><h3>Filters <span className="muted">all must match</span></h3>
              <button disabled={query.filters.length >= 12 || !columns.length} onClick={() => change({ filters: [...query.filters, { column: columns[0].name, operator: "eq", value: "" }] })}><Plus size={14}/>Add filter</button></div>
              {!query.filters.length && <p className="footnote">All previewable rows are included. Add a date, model, series or category filter.</p>}
              {query.filters.map((filter, index) => {
                const edit = (patch: Partial<typeof filter>) => change({ filters: query.filters.map((item, i) => i === index ? { ...item, ...patch } : item) });
                return <div className="explore-filter" key={index}>
                  <label>Field<select aria-label={`Filter ${index + 1} field`} value={filter.column} onChange={event => edit({ column: event.target.value })}>{columns.map(col => <option key={col.name}>{col.name}</option>)}</select></label>
                  <label>Condition<select aria-label={`Filter ${index + 1} condition`} value={filter.operator} onChange={event => edit({ operator: exploreQuerySchema.shape.filters.element.shape.operator.parse(event.target.value) })}>
                    {Object.entries(operatorLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
                  <label>Value<input aria-label={`Filter ${index + 1} value`} maxLength={1000} value={filter.value} disabled={["is_null", "not_null"].includes(filter.operator)} onChange={event => edit({ value: event.target.value })}/></label>
                  <button className="icon" aria-label={`Remove filter ${index + 1}`} onClick={() => change({ filters: query.filters.filter((_, i) => i !== index) })}><X size={15}/></button>
                </div>;
              })}
            </section>
            <div className="explore-sort"><label>Sort result by<select aria-label="Sort result by" value={query.sort.column} onChange={event => change({ sort: { ...query.sort, column: event.target.value } })}>
              <option value="">Saved row / group order</option>{resultColumns.map(col => <option key={col} value={col}>{result?.labels[col] ?? col}</option>)}</select></label>
              <label>Direction<select aria-label="Sort direction" value={query.sort.direction} onChange={event => change({ sort: { ...query.sort, direction: event.target.value === "desc" ? "desc" : "asc" } })}>
                <option value="asc">Ascending</option><option value="desc">Descending</option></select></label></div>
          </div>
        </div>}
      </fieldset>
      <div className="explore-run-bar"><button className="primary" disabled={!!busy || !sameSource} onClick={() => void run(query)}><Play size={15}/>{busy === "query" ? "Running query..." : "Run query"}</button>
        <button disabled={!!busy || !result || stale} onClick={() => void exportResult()}><Download size={15}/>{busy === "export" ? "Exporting..." : "Export full result"}</button>
        <span role="status">{stale ? "Query changed. Run it to update the result and enable export." : result ? `${result.matchedRows.toLocaleString()} matching source rows → ${result.total.toLocaleString()} result rows` : "Runs only on this saved local snapshot."}</span>
      </div>
      <div className="explore-save-bar">
        <label>Query name<input aria-label="Query name" maxLength={120} value={draft.name} disabled={!!busy} placeholder="e.g. Hosted minutes by month and model"
          onChange={event => update({ explore: { ...draft, name: event.target.value } })}/></label>
        <button disabled={!!busy || !sameSource || !draft.name.trim()} onClick={() => void save(false)}><Save size={15}/>{draft.savedId ? "Update saved query" : "Save query"}</button>
        {draft.savedId && <button disabled={!!busy || !draft.name.trim()} onClick={() => void save(true)}>Save a copy</button>}
      </div>
      {message && <p className="save-receipt" role="status">{message}</p>}
      {receipt && <section className="explore-export" aria-label="Export receipt"><h3>Export saved · {receipt.rows.toLocaleString()} result rows</h3>
        <p>Full result, not just the preview page. CSV and its reproducible query definition are saved together.</p>
        <div className="toolbar"><button disabled={!!busy || receipt.bytes > 4 * 1024 * 1024} onClick={() => void download("csv")}>Download CSV</button>
          <button disabled={!!busy} onClick={() => void download("query")}>Download query JSON</button></div>
        {receipt.bytes > 4 * 1024 * 1024 && <p className="footnote">This CSV exceeds the 4 MiB browser-transfer limit. The complete export is available at the local path below.</p>}
        {!!receipt.formulaEscapedCells && <p className="footnote">{receipt.formulaEscapedCells} formula-like cells escaped for spreadsheet safety.</p>}
        <details><summary>Local export paths</summary><p>{receipt.path}</p><p>{receipt.queryPath}</p></details>
      </section>}
      {result && <section className={stale ? "explore-stale" : ""} aria-label="Exploration output">
        {stale && <p className="notice">Previous query result — not the edited query above.</p>}
        <Table rows={result.rows} columns={result.columns} columnLabels={result.labels} raw title={result.query.mode === "pivot" ? "Pivot result" : "Data result"} inspect={inspect}/>
        <div className="table-footer"><span>{result.total ? result.offset + 1 : 0}–{result.offset + result.rows.length} of {result.total.toLocaleString()} result rows · unrounded values</span><div className="toolbar">
          <button disabled={!!busy || stale || !pages.length} onClick={() => void run(result.query, pages.at(-1) ?? 0, "previous")}>Previous result page</button>
          <button disabled={!!busy || stale || result.nextOffset === null} onClick={() => void run(result.query, result.nextOffset ?? 0, "next")}>Next result page</button>
        </div></div>
        {!result.matchedRows && <p className="empty-inline">No source rows matched. Remove or broaden a filter, then run the query again.</p>}
      </section>}
      <details className="explore-definition"><summary>Query definition and semantics</summary>
        <p className="footnote">Portable JSON query, not warehouse SQL. This pins an immutable snapshot and records selected fields, AND filters, pivot calculation and sort. All matching rows are evaluated before preview paging. Empty CSV cells are null; equality filters do not match null. Contains is case-insensitive. Sum/average/min/max require numeric fields; IDs and leading-zero codes remain text.</p>
        <pre className="file-preview" tabIndex={0}>{JSON.stringify(query, null, 2)}</pre>
      </details>
    </>}
  </section>;
}
