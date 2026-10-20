import { useEffect, useMemo, useRef, useState } from "react";
import { z } from "zod";
import { call, failure, rowSchema } from "./api";
import type { AnalysisProps } from "./Evidence";
import { ScaleControl } from "./Evidence";
import { AnalysisPlot } from "./AnalysisPlot";
import { ErrorNotice, Modal, Table } from "./components";
import { formatted, scales } from "./format";
import { calculateJudgment, calendarReceiptSchema, judgmentDraftSchema, judgmentEditSchema, neutralJudgment } from "./judgment";
import type { JudgmentDraft, JudgmentEdit } from "./judgment";
import { CalendarLayerControls } from "./CalendarLayerControls";

const sourceSchema = z.object({
  baseRunId: z.string(), baseName: z.string(), series: z.string(), model: z.string(), selectedModel: z.string(), units: z.string(),
  rows: z.array(z.object({ ds: z.string(), model_yhat: z.string(), selected_yhat: z.string(),
    calendar_yhat: z.string().optional(), calendar_delta: z.string().optional() }).passthrough()),
  calendarLayer: calendarReceiptSchema.nullable(), historyPeriods: z.array(z.string()),
  history: z.array(rowSchema), models: z.array(z.object({ id: z.string(), label: z.string(), complete: z.boolean(), endpoint: z.string().nullable() })),
  savedSettings: judgmentDraftSchema.nullable(), sourceHash: z.string(), method: z.string(), parentLabel: z.string(),
  versions: z.array(z.object({ id: z.string(), label: z.string(), createdAt: z.string(), parentRunId: z.string(), model: z.string() })),
});
const exportSchema = z.object({ id: z.string(), path: z.string(), queryPath: z.string(), rows: z.number() });

export function JudgmentWorksheet({ document: doc, view, update, dark, inspect, reload }: AnalysisProps & { reload: () => Promise<void> }) {
  const [data, setData] = useState<z.infer<typeof sourceSchema>>(), [error, setError] = useState("");
  const [loaded, setLoaded] = useState(""), [busy, setBusy] = useState(""), [search, setSearch] = useState("");
  const [paste, setPaste] = useState(""), [version, setVersion] = useState("");
  const [pendingVersion, setPendingVersion] = useState("");
  const [receipt, setReceipt] = useState<z.infer<typeof exportSchema>>();
  const pending = useRef(false);
  const [validCalendar, setValidCalendar] = useState(true);
  const [sourceAttempt, setSourceAttempt] = useState(0), [loadingSource, setLoadingSource] = useState(false);
  const requestedCalendar = useRef("");
  const origin = JSON.stringify([doc.runId, doc.series]);
  const cached = useRef({ origin: "", model: "", calendar: "" });
  const calendarKey = JSON.stringify(doc.judgment?.calendar ?? data?.savedSettings?.calendar ?? null);
  const requestKey = JSON.stringify([doc.runId, doc.series, doc.judgment?.model, doc.judgment?.calendar]);
  const current = useRef(requestKey); current.current = requestKey;
  useEffect(() => {
    if (data && cached.current.origin === origin && doc.judgment?.model === data.model && cached.current.calendar === calendarKey) { setLoaded(requestKey); return; }
    const controller = new AbortController();
    setError(""); setReceipt(undefined);
    if (data && cached.current.origin === origin && doc.judgment?.model === data.model && doc.judgment.calendar?.enabled && requestedCalendar.current !== requestKey) {
      setLoadingSource(false);
      return () => controller.abort();
    }
    setLoadingSource(true);
    const timer = setTimeout(() => {
      void call("judgment_source", { runId: doc.runId, series: doc.series, model: doc.judgment?.model, calendar: doc.judgment?.calendar }, sourceSchema, controller.signal)
        .then(next => { if (!controller.signal.aborted) {
          cached.current = { origin, model: next.model, calendar: JSON.stringify(doc.judgment?.calendar ?? next.savedSettings?.calendar ?? null) };
          setData(next); setLoaded(requestKey); setLoadingSource(false);
        } })
        .catch(err => { if (!controller.signal.aborted) { setError(failure(err)); setLoadingSource(false); } });
    }, data ? 400 : 0);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [requestKey, sourceAttempt]);
  const settings = doc.judgment ?? data?.savedSettings ?? neutralJudgment();
  const ready = !!data && (loaded === requestKey || (cached.current.origin === origin && doc.judgment?.model === data.model && cached.current.calendar === calendarKey));
  const calculation = useMemo(() => {
    if (!ready || !data) return { error: "", result: undefined };
    try { return { error: "", result: calculateJudgment(data.rows, settings) }; }
    catch (err) { return { error: failure(err), result: undefined }; }
  }, [data, ready, settings]);
  const change = (patch: Partial<JudgmentDraft>) => { update({ judgment: { ...settings, ...patch } }); setReceipt(undefined); };
  const loadSource = () => { requestedCalendar.current = requestKey; setSourceAttempt(value => value + 1); };
  const edit = (ds: string, field: keyof Omit<JudgmentEdit, "ds">, value: string) => {
    const next = { ds, percent: "", amount: "", override: "", note: "", ...settings.edits.find(row => row.ds === ds), [field]: value };
    change({ edits: [...settings.edits.filter(row => row.ds !== ds), next].filter(row => row.percent || row.amount || row.override || row.note) });
  };
  const perform = async (kind: string, action: () => Promise<void>) => {
    if (pending.current) return;
    pending.current = true; setBusy(kind); setError("");
    try { await action(); } catch (err) { setError(failure(err)); }
    finally { pending.current = false; setBusy(""); }
  };
  const importText = async (text: string, delimiter: "csv" | "tab") => {
    if (!ready || !data) return;
    const key = requestKey;
    const input = { runId: doc.runId, series: doc.series, model: data.model, calendar: settings.calendar, text, delimiter };
    const encoder = new TextEncoder();
    if (encoder.encode(text).length > 128 * 1024) throw new Error("Import is limited to 128 KiB; shorten notes or import fewer periods.");
    if (encoder.encode(JSON.stringify({ operation: "judgment_import", input })).length > 256 * 1024) throw new Error("Encoded import exceeds the local request limit; shorten notes or import fewer periods.");
    const imported = await call("judgment_import", input,
      z.object({ edits: z.array(judgmentEditSchema), rows: z.number(), sha256: z.string(), sourceHash: z.string() }));
    if (key !== current.current || imported.sourceHash !== data.sourceHash) throw new Error("The worksheet changed during import. Reload and import into the intended model.");
    change({ percent: "0", amount: "0", start: "", end: "", edits: imported.edits });
    setPaste("");
  };
  const save = () => perform("save", async () => {
    if (!data || !ready || !calculation.result || !validCalendar) return;
    const saved = await call("judgment_save", { runId: doc.runId, series: doc.series, settings, expectedSourceHash: data.sourceHash },
      z.object({ id: z.string(), label: z.string(), baseRunId: z.string() }));
    await reload();
    update({ runId: saved.id, title: saved.label, judgment: undefined, referenceRunId: saved.baseRunId, analysis: { previewModel: "", modelMode: "focus", models: ["*"] }, intervalLevel: 0 });
  });
  const download = (kind: "csv" | "query") => perform("download", async () => {
    if (!receipt) return;
    const file = await call("explore_download", { id: receipt.id, kind }, z.object({ name: z.string(), base64: z.string(), mediaType: z.string() }));
    const url = URL.createObjectURL(new Blob([Uint8Array.from(atob(file.base64), c => c.charCodeAt(0))], { type: file.mediaType }));
    const link = document.createElement("a"); link.href = url; link.download = file.name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  const scale = scales[doc.analysis.scale];
  const rows = calculation.result?.rows ?? [];
  const unchanged = data?.savedSettings && JSON.stringify(settings) === JSON.stringify(data.savedSettings);
  const modifiedDraft = JSON.stringify(settings) !== JSON.stringify(data?.savedSettings ?? neutralJudgment());
  const openVersion = (id: string) => {
    const selected = data?.versions.find(item => item.id === id);
    if (selected) update({ runId: selected.id, title: selected.label, judgment: undefined, analysis: { previewModel: "", models: ["*"] }, intervalLevel: 0 });
    setPendingVersion("");
  };
  const comparisonLabels = {
    selected_yhat: `Saved selection${data?.selectedModel ? ` · ${data.selectedModel}` : ""}`,
    model_yhat: `Chosen model · ${data?.model || data?.selectedModel || "saved selection"}`,
    calendar_yhat: "After calendar", calendar_delta: "Calendar change", judgment_delta: "Final tweaks",
    adjusted_yhat: "Analyst judgment",
  };
  const numberField = (label: string, field: "percent" | "amount") => <label>{label}<input aria-label={label} inputMode="decimal" value={settings[field]} maxLength={80} onChange={event => change({ [field]: event.target.value })}/></label>;
  return <section className="judgment-workspace" aria-label="Forecast adjustment worksheet">
    <header className="section-heading"><div><h2>Forecast adjustments</h2><p>Chosen model → optional calendar layer → final tweaks → saved version.</p></div><ScaleControl document={doc} update={update}/></header>
    <p className="boundary-note">The model winner is a recommendation, not a lock. These edits create a separate analyst scenario, not a better-scoring model. Inputs use original {data?.units || "source units"}; display scaling never changes their meaning.</p>
    {error && <ErrorNotice error={error}/>}
    {!ready && loadingSource && !error && <p role="status">Loading the full model path and saved versions...</p>}
    {!ready && error && <button disabled={loadingSource} onClick={loadSource}>Retry loading worksheet</button>}
    <h3>1. Choose the model</h3>
    <fieldset disabled={!!busy || !ready} className="judgment-controls"><legend className="visually-hidden">Model baseline</legend>
      <label>Find a model<input aria-label="Find adjustment model" value={search} onChange={event => setSearch(event.target.value)} placeholder="ARIMA, MSTL, ensemble..."/></label>
      <label className="judgment-model">Model baseline<select aria-label="Adjustment model" value={data?.model ?? settings.model} onChange={event => change({ model: event.target.value })}>
        {data?.models.filter(model => !model.id || model.id === settings.model || model.label.toLowerCase().includes(search.toLowerCase())).map(model =>
          <option key={model.id} value={model.id} disabled={!model.complete}>{model.label} · end {formatted(model.endpoint, "yhat", scale)}{!model.complete ? " · incomplete horizon" : ""}</option>)}
      </select><small>Baseline values are fixed; adjustments below are reapplied, never compounded on a previous version.</small></label>
    </fieldset>
    <CalendarLayerControls key={origin} settings={settings.calendar} change={calendar => change({ calendar })}
      periods={data?.rows.map(row => row.ds) ?? []} history={data?.historyPeriods ?? []} receipt={data?.calendarLayer}
      busy={!!busy || !data} stale={!ready} loading={loadingSource} calculate={loadSource} valid={setValidCalendar}/>
    <h3>3. Apply final tweaks</h3>
    <fieldset disabled={!!busy || !ready} className="judgment-controls"><legend className="visually-hidden">Final bulk adjustments</legend>
      {numberField("Bulk change (%)", "percent")}{numberField(`Absolute change per period (${data?.units || "original units"})`, "amount")}
      {(["start", "end"] as const).map(field => <label key={field}>{field === "start" ? "Apply from" : "Apply through"}<select aria-label={field === "start" ? "Apply from" : "Apply through"} value={settings[field]} onChange={event => change({ [field]: event.target.value })}>
        <option value="">Entire horizon</option>{data?.rows.map(row => <option key={row.ds}>{row.ds}</option>)}</select></label>)}
    </fieldset>
    <p className="footnote">Enter −10 for a 10% haircut, +10 for an uplift. The percentage applies after the calendar layer when enabled, then the absolute change is added once per period. Row inputs replace bulk inputs; a final override wins. Nothing is silently floored at zero.</p>
    <div className="toolbar"><button disabled={!!busy || !ready} onClick={() => change({ percent: "0", amount: "0", start: "", end: "", edits: [] })}>Clear adjustments</button>
      <button disabled={!ready || !!busy} onClick={() => update({ view: "Models", runId: data?.baseRunId, analysis: { previewModel: settings.model, modelMode: "focus" } })}>Review baseline evidence</button></div>
    {calculation.error && <ErrorNotice error={calculation.error}/>}
    {ready && <section className="judgment-grid-section" aria-label="Editable adjustment grid">
      <header className="section-heading"><h3>Period worksheet</h3><span className="muted">{data.rows.length} periods · {doc.series} · display: {doc.analysis.scale}</span></header>
      <dl className="judgment-totals" aria-label="Horizon adjustment comparison">
        <div><dt>Model horizon sum</dt><dd>{formatted(calculation.result?.totals.model_yhat, "yhat", scale) || "—"}</dd></div>
        {data.calendarLayer && <div><dt>After calendar</dt><dd>{formatted(calculation.result?.totals.calendar_yhat, "yhat", scale) || "—"}</dd></div>}
        <div><dt>Adjusted horizon sum</dt><dd>{formatted(calculation.result?.totals.adjusted_yhat, "yhat", scale) || "—"}</dd></div>
        <div><dt>Change vs model</dt><dd>{formatted(calculation.result?.totals.delta, "yhat", scale) || "—"} <small>({formatted(calculation.result?.totals.delta_pct, "ratio") || "—"}%)</small></dd></div>
      </dl>
      <div className="table-scroll judgment-grid" tabIndex={0}><table><thead><tr>
        {["Period", "Model baseline", ...(data.calendarLayer ? ["After calendar"] : []), "Change %", "Absolute change", "Final override", "Adjusted", "Δ vs model", "Running Δ", "Note"].map(label => <th key={label}>{label}</th>)}
      </tr></thead><tbody>{data.rows.map((row, i) => {
        const rowEdit = settings.edits.find(item => item.ds === row.ds);
        return <tr key={row.ds}><th scope="row">{row.ds}</th><td className="numeric">{formatted(row.model_yhat, "yhat", scale)}</td>
          {data.calendarLayer && <td className="numeric">{formatted(row.calendar_yhat, "yhat", scale)}</td>}
          {(["percent", "amount", "override"] as const).map(field => <td key={field}><input disabled={!!busy} aria-label={`${row.ds} ${field}`} inputMode="decimal" maxLength={80}
            value={rowEdit?.[field] ?? ""} placeholder={field === "override" ? "Calculated" : field === "percent" ? rows[i]?.percent ?? "Inherit" : rows[i]?.amount ?? "Inherit"}
            onChange={event => edit(row.ds, field, event.target.value)}/></td>)}
          <td className="numeric judgment-output">{formatted(rows[i]?.adjusted_yhat, "yhat", scale) || "—"}</td>
          <td className="numeric">{formatted(rows[i]?.delta, "yhat", scale) || "—"}</td><td className="numeric">{formatted(rows[i]?.cumulative_delta, "yhat", scale) || "—"}</td>
          <td><input disabled={!!busy} aria-label={`${row.ds} note`} maxLength={400} value={rowEdit?.note ?? ""} placeholder="Reason for exception" onChange={event => edit(row.ds, "note", event.target.value)}/></td></tr>;
      })}</tbody><tfoot><tr><th>Horizon sum</th><td className="numeric">{formatted(calculation.result?.totals.model_yhat, "yhat", scale) || "—"}</td>
        {data.calendarLayer && <td className="numeric">{formatted(calculation.result?.totals.calendar_yhat, "yhat", scale) || "—"}</td>}
        <td colSpan={3}>Selected series only</td><td className="numeric">{formatted(calculation.result?.totals.adjusted_yhat, "yhat", scale) || "—"}</td>
        <td className="numeric">{formatted(calculation.result?.totals.delta, "yhat", scale) || "—"}</td><td colSpan={2}>{formatted(calculation.result?.totals.delta_pct, "ratio")} % vs model</td></tr></tfoot></table></div>
      <p className="footnote">Horizon sums are for flow measures. For balances, ARR or rates, use the period value—not a sum across months. Blank overrides inherit; entering 0 explicitly sets the forecast to zero.</p>
    </section>}
    {!!rows.length && <><AnalysisPlot title="Model vs adjusted forecast" rows={rows} x="ds" fields={["selected_yhat", "model_yhat", ...(data?.calendarLayer ? ["calendar_yhat"] : []), "adjusted_yhat"]} fieldLabels={comparisonLabels} dashedFields={["adjusted_yhat"]} dark={dark} scale={scale}/>
      <p className="footnote">Saved selection and chosen model are unadjusted comparators. Dashed line is analyst judgment. No prediction intervals are implied.</p>
      <details><summary>Full comparison & calculation method</summary><p>{data?.method}</p><Table rows={rows} title="Adjustment comparison" columnLabels={comparisonLabels} inspect={inspect} scale={scale}/></details></>}
    <section className="judgment-save"><h3>{data?.parentLabel ? `Branch from ${data.parentLabel}` : "Save a named version"}</h3>
      <p>A save creates an immutable, selected-series version in the Library. Forecast charts, Data, Explore data, version comparisons and exports all read its adjusted values. Original model evidence stays on the baseline version.</p>
      <fieldset disabled={!!busy || !ready}><label>Version name<input aria-label="Adjustment version name" maxLength={120} value={settings.label} placeholder="Leadership base, downside, capacity haircut..." onChange={event => change({ label: event.target.value })}/></label>
        <label>Decision rationale<textarea aria-label="Adjustment rationale" maxLength={2000} rows={2} value={settings.reason} placeholder="Why this model and these assumptions?" onChange={event => change({ reason: event.target.value })}/></label></fieldset>
      <div className="toolbar"><button className="primary" disabled={!!busy || !ready || !validCalendar || !calculation.result || !settings.label.trim() || !settings.reason.trim() || !!unchanged} onClick={() => void save()}>{busy === "save" ? "Saving version..." : "Save new judgment version"}</button>
        <button disabled={!!busy || !ready || !validCalendar || !calculation.result} onClick={() => void perform("export", async () => {
          if (data) setReceipt(await call("judgment_export", { runId: doc.runId, series: doc.series, settings, expectedSourceHash: data.sourceHash }, exportSchema));
        })}>Export worksheet & recipe</button></div>
      {view.run.metadata.purpose === "judgment_scenario" && <button disabled={!!busy} onClick={() => update({ view: "Pipeline" })}>Inspect saved finalization pipeline</button>}
      {receipt && <div role="status"><p>Export saved · {receipt.rows} rows. CSV includes model, adjusted, difference and editable inputs.</p><div className="toolbar">
        <button disabled={!!busy} onClick={() => void download("csv")}>Download worksheet CSV</button><button disabled={!!busy} onClick={() => void download("query")}>Download judgment JSON</button></div></div>}
    </section>
    <details className="judgment-import"><summary>Upload or paste adjustments from Excel</summary>
      <p>Import CSV, or paste tab-separated cells with headers: <code>ds, percent, amount, override, note</code>. Only ds and one adjustment column are required; up to 500 periods and 128 KiB. Import replaces the worksheet's row edits and clears bulk changes to prevent double application. Save the result as another named version.</p>
      <label>Adjustment CSV<input aria-label="Upload adjustment CSV" type="file" accept=".csv,.tsv" disabled={!!busy || !ready} onChange={event => {
        const file = event.target.files?.[0]; event.target.value = "";
        if (file) void perform("import", async () => { if (file.size > 128 * 1024) throw new Error("Upload a CSV/TSV no larger than 128 KiB."); await importText(await file.text(), file.name.toLowerCase().endsWith(".tsv") ? "tab" : "csv"); });
      }}/></label>
      <label>Paste Excel cells<textarea aria-label="Paste adjustment cells" value={paste} disabled={!!busy} onChange={event => setPaste(event.target.value)} rows={4} placeholder={"ds\toverride\tnote\n2026-10-01\t100\tCapacity assumption"}/></label>
      <button disabled={!!busy || !ready || !paste.trim()} onClick={() => void perform("import", () => importText(paste, paste.split(/\r?\n/)[0].includes("\t") ? "tab" : "csv"))}>Import pasted rows</button>
    </details>
    <section className="judgment-versions"><h3>Saved judgment versions</h3><p>Branches retain their parent, original model and rationale. Opening a version replaces this worksheet's current draft.</p>
      <div className="toolbar"><label>Version<select aria-label="Judgment version" value={version} onChange={event => setVersion(event.target.value)}><option value="">Choose a version</option>
        {data?.versions.map(item => <option key={item.id} value={item.id}>{item.label} · {item.model || "saved selection"} · {new Date(item.createdAt).toLocaleString()}</option>)}</select></label>
        <button disabled={!version || !!busy || !ready} onClick={() => modifiedDraft ? setPendingVersion(version) : openVersion(version)}>Open judgment version</button>
        <button disabled={!version || !!busy} onClick={() => update({ view: "Compare", comparison: "versions", referenceRunId: version })}>Compare with this version</button>
      </div>{view.run.metadata.purpose === "judgment_scenario" && <p className="footnote">This version is a judgment scenario, not an official planning lock or a refitted model.</p>}
    </section>
    {pendingVersion && <Modal title="Replace unsaved judgment edits?" close={() => setPendingVersion("")}>
      <p>Your worksheet has changes that are not saved as a named judgment version. Opening another version replaces these edits. Keep editing to save them first.</p>
      <div className="toolbar"><button autoFocus onClick={() => setPendingVersion("")}>Keep editing</button>
        <button onClick={() => openVersion(pendingVersion)}>Discard edits and open version</button></div>
    </Modal>}
  </section>;
}
