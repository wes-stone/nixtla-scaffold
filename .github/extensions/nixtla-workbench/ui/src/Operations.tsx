import { useEffect, useState } from "react";
import { z } from "zod";
import { call, failure, profileSchema, rowSchema, text, viewSchema } from "./api";
import type { Catalog, Document, DocumentPatch, Row, View } from "./api";
import { Modal, Table } from "./components";

const started = z.object({ jobId: z.string() });
const ledgerSchema = z.object({
  rows: z.array(rowSchema), columns: z.array(z.string()), total: z.number(), nextOffset: z.number().nullable(),
});

export function Operations({ document: doc, view, catalog, update, reload, inspect, onError }: {
  document: Document; view: View; catalog: Catalog; update: (patch: DocumentPatch) => void;
  reload: () => Promise<void>; inspect: (row: Row) => void; onError: (error: string) => void;
}) {
  const judgment = view.run.metadata.purpose === "judgment_scenario";
  const draft = judgment && doc.maintenance.kind === "refresh" ? { ...doc.maintenance, kind: "actuals" as const } : doc.maintenance;
  const change = (patch: Partial<Document["maintenance"]>) => update({ maintenance: patch });
  const [busy, setBusy] = useState(false), [approval, setApproval] = useState(false);
  const [ledgerKind, setLedgerKind] = useState("actuals"), [offset, setOffset] = useState(0);
  const [result, setResult] = useState<z.infer<typeof ledgerSchema>>();
  const [question, setQuestion] = useState(""), [columns, setColumns] = useState<string[]>([]);
  const safe = async (action: () => Promise<void>) => {
    setBusy(true); onError("");
    try { await action(); } catch (error) { onError(failure(error)); } finally { setBusy(false); }
  };
  useEffect(() => {
    const controller = new AbortController();
    void call("ledger_view", { runId: doc.runId, kind: ledgerKind, series: doc.series || undefined, offset, limit: 100 }, ledgerSchema, controller.signal)
      .then(setResult).catch(error => { if (!controller.signal.aborted) onError(failure(error)); });
    return () => controller.abort();
  }, [doc.runId, doc.series, ledgerKind, offset, catalog.jobs.map(job => `${job.id}:${job.status}`).join("|")]);
  return <section>
    <h3>Maintain this forecast</h3>
    {judgment ? <><p>Record actuals against this saved judgment without rerunning it. To refresh models, open the original baseline; the judgment recipe is never discarded or rolled forward automatically.</p>
      <button disabled={busy || !text(view.run.metadata.judgmentBaseRunId)} onClick={() => update({
        runId: text(view.run.metadata.judgmentBaseRunId), title: view.run.name, view: "Pipeline", maintenance: { kind: "refresh", sourceId: "" },
      })}>Open original baseline to refresh</button></> :
      <p>Refresh creates a new version with the saved spec. An adopted ensemble retains its recipe; a new snapshot does not inherit old confirmation. Actuals are a separate ledger vintage, not a rerun.</p>}
    <div className="form-grid">
      <label>Operation<select aria-label="Maintenance operation" value={draft.kind} onChange={event => change({ kind: z.enum(["refresh", "actuals", "lock"]).parse(event.target.value) })}>
        {!judgment && <option value="refresh">Refresh saved forecast</option>}<option value="actuals">Record landed actuals</option><option value="lock">Lock a planning-ready version</option>
      </select></label>
      {draft.kind !== "lock" && <label>Saved snapshot<select aria-label="Maintenance snapshot" value={draft.sourceId} onChange={event => change({ sourceId: event.target.value })}>
        <option value="">Choose a snapshot...</option>{catalog.sources.map(source => <option key={source.id} value={source.id}>{source.name}</option>)}
      </select></label>}
      {draft.kind === "actuals" && <label>Known as of (with timezone)<input aria-label="Actuals known as of" value={draft.knownAsOf} onChange={event => change({ knownAsOf: event.target.value })} placeholder="2026-09-24T08:00:00-07:00"/></label>}
      {draft.kind !== "refresh" && <label>Revision or lock label<input aria-label="Revision or lock label" value={draft.label} onChange={event => change({ label: event.target.value })}/></label>}
    </div>
    {draft.kind !== "lock" && <details><summary>Inspect a new local source</summary>
      <label>New source file<input aria-label="Maintenance source file" value={draft.path} onChange={event => change({ path: event.target.value, sourceId: "", mapping: {} })}/></label>
      {question && <div className="notice">{question}</div>}
      {question && <div className="form-grid">{["time", "target", "id"].map(key => <label key={key}>{key} column<select value={draft.mapping[key] ?? ""} onChange={event => change({ mapping: { ...draft.mapping, [key]: event.target.value } })}>
        <option value="">Choose...</option>{columns.map(col => <option key={col}>{col}</option>)}</select></label>)}</div>}
      <button disabled={busy || !draft.path} onClick={() => void safe(async () => {
        const profile = await call("profile", { path: draft.path, horizon: view.spec.horizon, mapping: { ...draft.mapping, freq: view.spec.freq } }, profileSchema);
        change({ sourceId: profile.sourceId ?? "", mapping: profile.mapping }); setQuestion(profile.questions[0] ?? ""); setColumns(profile.columns);
        inspect(profile); await reload();
      })}>Inspect new snapshot</button>
    </details>}
    <button className="primary" disabled={busy || (draft.kind === "lock" ? !draft.label || !view.gate.planning_ready_claim_allowed : !draft.sourceId)}
      onClick={() => setApproval(true)}>Review {draft.kind === "refresh" ? "refresh" : draft.kind === "actuals" ? "actuals import" : "official lock"}</button>
    {draft.kind === "lock" && !view.gate.planning_ready_claim_allowed && <p className="notice">This version's saved gate does not permit a planning lock. No control can override that gate.</p>}
    <div className="section-heading"><h3>Operating ledger</h3><select aria-label="Ledger evidence" value={ledgerKind} onChange={event => { setLedgerKind(event.target.value); setOffset(0); }}>
      <option value="actuals">Actuals vintages</option><option value="performance">Forecast versus actuals</option><option value="locks">Official locks</option></select></div>
    {result && <><Table rows={result.rows} columns={result.columns} inspect={inspect} title="Ledger values"/>
      <div className="toolbar"><span>{result.total} recorded rows</span><button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 100))}>Earlier page</button>
        <button disabled={result.nextOffset === null} onClick={() => setOffset(result.nextOffset ?? 0)}>More ledger rows</button></div></>}
    {approval && <Modal title={`Approve ${draft.kind}`} close={() => setApproval(false)}>
      <p>{draft.kind === "refresh" ? "Run the saved full spec against this immutable snapshot. The prior version stays unchanged. Frozen blends refit their original policy; this does not approve publication or a planning lock." :
        draft.kind === "actuals" ? "Append this snapshot as an actuals vintage with the supplied known-as-of timestamp. The forecast stays unchanged; ledger comparisons will update." :
        "Record an official lock for this exact saved version. This is a deliberate ledger decision, not external publication."}</p>
      <p>Version: {view.run.name} · {view.run.id.slice(0, 12)}<br/>Snapshot: {catalog.sources.find(source => source.id === draft.sourceId)?.name ?? "Not applicable"}</p>
      <button className="primary" disabled={busy} onClick={() => void safe(async () => {
        await call("start_job", { kind: draft.kind, requestKey: crypto.randomUUID(), runId: doc.runId, approved: true,
          ...(draft.kind !== "lock" ? { sourceId: draft.sourceId } : { lockLabel: draft.label }),
          ...(draft.kind === "actuals" ? { knownAsOf: draft.knownAsOf, revisionLabel: draft.label } : {}) }, started);
        setApproval(false); await reload();
      })}>Approve {draft.kind}</button>
    </Modal>}
  </section>;
}

export function Confirmation({ runId, blend, horizon, freq, catalog, close, reload, onError }: {
  runId: string; blend: View["blends"][number]; horizon: unknown; freq: unknown; catalog: Catalog;
  close: () => void; reload: () => Promise<void>; onError: (error: string) => void;
}) {
  const [path, setPath] = useState(""), [sourceId, setSourceId] = useState("");
  const members = Array.isArray(blend.recipe.members) ? blend.recipe.members.filter((m): m is string => typeof m === "string") : [];
  const [reference, setReference] = useState(members.includes("Naive") ? "Naive" : members[0] ?? "");
  const [busy, setBusy] = useState(false), [finding, setFinding] = useState("");
  const safe = async (action: () => Promise<void>) => {
    setBusy(true); onError(""); try { await action(); } catch (error) { onError(failure(error)); } finally { setBusy(false); }
  };
  return <Modal title="Confirm a frozen ensemble" close={close}>
    <p>Reserve 20 non-overlapping calibration windows and one later confirmation window before target profiling. The recipe cannot change. Up to 22 local fits count against the 15-minute ceiling; insufficient history or previously exposed periods block execution.</p>
    <p>At horizon {text(horizon)}, this needs at least {Number(horizon) * 21 + Math.max(8, Number(horizon) * 2 + 1)} observations per series, with the reserved tail unexposed in this workspace.</p>
    <label>New protected source<input aria-label="Protected source file" value={path} onChange={event => { setPath(event.target.value); setSourceId(""); }}/></label>
    <button disabled={!path || busy} onClick={() => void safe(async () => {
      const profile = await call("profile", { path, horizon, holdoutWindows: 21, mapping: { freq } }, profileSchema);
      setSourceId(profile.sourceId ?? ""); setFinding(profile.questions[0] ?? profile.findings?.join(" ") ?? "Snapshot prepared."); await reload();
    })}>Prepare protected snapshot</button>
    {finding && <p role="status">{finding}</p>}
    <label>Protected snapshot<select aria-label="Protected snapshot" value={sourceId} onChange={event => setSourceId(event.target.value)}>
      <option value="">Choose...</option>{catalog.sources.filter(source => source.metadata.reservedWindows === 21).map(source => <option value={source.id} key={source.id}>{source.name}</option>)}</select></label>
    <label>Frozen reference model<select aria-label="Frozen reference model" value={reference} onChange={event => setReference(event.target.value)}>{members.map(member => <option key={member}>{member}</option>)}</select></label>
    <button className="primary" disabled={!sourceId || busy || !members.includes(reference)} onClick={() => void safe(async () => {
      await call("start_job", { kind: "confirm_blend", requestKey: crypto.randomUUID(), runId, blendId: blend.id,
        expectedReceipt: blend.receiptHash, sourceId, referenceModel: reference }, started);
      await reload(); close();
    })}>Run frozen confirmation</button>
    <p className="footnote">Calibration uses blend residuals, never averaged component bounds. Confirmation is consumed on submission, including failed or interrupted work. This does not automatically adopt, promote or publish.</p>
  </Modal>;
}

const proposalSchema = z.object({ proposalId: z.string(), expectedReceipt: z.string(), receipt: rowSchema });
export function ExternalComparison({ document: doc, view, update, reload, inspect, onError }: {
  document: Document; view: View; update: (patch: DocumentPatch) => void; reload: () => Promise<void>;
  inspect: (row: Row) => void; onError: (error: string) => void;
}) {
  const [proposal, setProposal] = useState<z.infer<typeof proposalSchema>>();
  const [busy, setBusy] = useState(false), [offset, setOffset] = useState(0);
  const [kind, setKind] = useState("external");
  const [result, setResult] = useState<View>();
  const present = view.artifacts.includes("external/comparison.csv");
  const safe = async (action: () => Promise<void>) => {
    setBusy(true); onError(""); try { await action(); } catch (error) { onError(failure(error)); } finally { setBusy(false); }
  };
  useEffect(() => {
    if (!present) return;
    const controller = new AbortController();
    setResult(undefined);
    void call("view", { runId: doc.runId, series: doc.series, kind, offset, limit: 100 }, viewSchema, controller.signal)
      .then(setResult).catch(error => { if (!controller.signal.aborted) onError(failure(error)); });
    return () => controller.abort();
  }, [present, doc.runId, doc.series, kind, offset]);
  return <section><h2>External & BYO evidence</h2>
    <p>Import canonical CSV outputs from FINN or your own model. Future-only forecasts are scenarios, not backtests. Historical cutoff labels allow advisory scoring, not a claim that their training was leakage-free.</p>
    <label>External forecast CSV<input aria-label="External forecast CSV" value={doc.externalPath} onChange={event => update({ externalPath: event.target.value })}/></label>
    <p className="footnote">Required: unique_id, ds, yhat. Optional: model and historical cutoff. Confirm the same {text(view.spec.freq)} calendar and {text(view.spec.unit_label) || "unrecorded"} target units; no unit conversion is inferred.</p>
    <button disabled={busy || !doc.externalPath} onClick={() => void safe(async () => {
      setProposal(await call("prepare_external", { runId: doc.runId, path: doc.externalPath, freq: text(view.spec.freq), units: text(view.spec.unit_label) }, proposalSchema));
    })}>Review external comparison</button>
    {present && <label>External evidence<select aria-label="External evidence" value={kind} onChange={event => { setKind(event.target.value); setOffset(0); }}>
      <option value="external">Forecast comparison</option><option value="external_forecasts">Imported forecast rows</option>
      {view.artifacts.includes("external/metrics.csv") && <option value="external_metrics">Historical advisory scores</option>}
    </select></label>}
    {result && <><Table rows={result.rows} columns={result.columns} title="External comparison values" inspect={inspect}/>
      <div className="table-footer"><span>{result.total} scoped rows</span><div className="toolbar">
        <button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous</button>
        <button disabled={result.nextOffset === null} onClick={() => setOffset(result.nextOffset ?? 0)}>Next</button>
        <button onClick={() => void safe(async () => inspect(await call("export_values", { runId: doc.runId, series: doc.series, kind }, rowSchema)))}>Export external CSV</button>
      </div></div></>}
    {proposal && <Modal title="Confirm external comparison scope" close={() => setProposal(undefined)}>
      <p>{text(proposal.receipt.authorization)}</p><p>Calendar: {text(proposal.receipt.freq)} · Units: {text(proposal.receipt.units) || "Not recorded; analyst must verify equivalence."}</p>
      <button disabled={busy} className="primary" onClick={() => void safe(async () => {
        await call("start_job", { kind: "pipeline", requestKey: crypto.randomUUID(), proposalId: proposal.proposalId, expectedReceipt: proposal.expectedReceipt, approved: true }, started);
        setProposal(undefined); await reload();
      })}>Confirm scope & import comparison</button>
    </Modal>}
  </section>;
}

export function PipelineAuthor({ document: doc, update, reload, inspect, onError }: {
  document: Document; update: (patch: DocumentPatch) => void; reload: () => Promise<void>;
  inspect: (row: Row) => void; onError: (error: string) => void;
}) {
  const draft = doc.pipeline;
  const change = (patch: Partial<Document["pipeline"]>) => update({ pipeline: { ...patch, proposalId: "", expectedReceipt: "" } });
  const [proposal, setProposal] = useState<z.infer<typeof proposalSchema>>();
  const [busy, setBusy] = useState(false);
  const safe = async (action: () => Promise<void>) => {
    setBusy(true); onError(""); try { await action(); } catch (error) { onError(failure(error)); } finally { setBusy(false); }
  };
  return <details className="saved-source"><summary>Author or refresh a source pipeline</summary>
    <p>Choose a Python extract, optional transform and declared query file. Review the exact copied scripts before running them. Existing multi-stage pipeline YAML can also be imported. Credentials remain with the configured provider.</p>
    <div className="form-grid">
      <label>Extract script<input aria-label="Pipeline extract script" value={draft.script} onChange={event => change({ script: event.target.value })}/></label>
      <label>Query file (optional)<input aria-label="Pipeline query file" value={draft.query} onChange={event => change({ query: event.target.value })}/></label>
      <label>Transform script (optional)<input aria-label="Pipeline transform script" value={draft.transform} onChange={event => change({ transform: event.target.value })}/></label>
    </div>
    <label>Extract arguments (JSON array)<input aria-label="Pipeline arguments" value={draft.args} onChange={event => change({ args: event.target.value })}/></label>
    {!doc.runId && <label>Forecast calendar (not required for extraction only)<input aria-label="Pipeline calendar" placeholder="MS, ME, D..." value={draft.frequency} onChange={event => change({ frequency: event.target.value })}/></label>}
    <p className="footnote">Extract output is declared as {"{output:extract}"}. The optional transform receives extract and prepared-output paths. Canonical output must contain unique_id, ds and y; other columns are retained.</p>
    <label>Or use an existing pipeline YAML<input aria-label="Pipeline configuration" value={draft.configPath} onChange={event => change({ configPath: event.target.value })}/></label>
    <label className="inline"><input type="checkbox" checked={draft.dataOnly} onChange={event => change({ dataOnly: event.target.checked })}/>Extract only; inspect the new source before modeling</label>
    <button disabled={busy || (!draft.configPath && !draft.script)} onClick={() => void safe(async () => {
      const input = draft.configPath ? { path: draft.configPath } : { definition: {
        name: doc.title, extracts: [{ name: "extract", script: draft.script, args: z.array(z.string()).parse(JSON.parse(draft.args)),
          outputs: { extract: "raw/source.csv" }, query_files: draft.query ? [draft.query] : [], timeout_seconds: 120 }],
        ...(draft.transform ? { transform: { name: "transform", script: draft.transform, args: ["{output:extract}", "{output:forecast_input}"], output: "prepared/transformed.csv" } } : {}),
        canonical: { source_output: "extract" },
        forecast: { horizon: doc.intake.horizon, freq: draft.frequency || null, model_policy: doc.intake.policy, paired_validation: true },
      } };
      const next = await call("prepare_pipeline", { ...input, runId: doc.runId, dataOnly: draft.dataOnly }, proposalSchema);
      setProposal(next); update({ pipeline: { proposalId: next.proposalId, expectedReceipt: next.expectedReceipt } });
    })}>Review pipeline execution</button>
    {proposal && draft.proposalId === proposal.proposalId && <Modal title="Review script-backed execution" close={() => setProposal(undefined)}>
      <p>{text(proposal.receipt.authorization)}</p><pre>{JSON.stringify(proposal.receipt.steps, null, 2)}</pre>
      <button onClick={() => inspect(proposal.receipt)}>Inspect complete receipt</button>
      <button className="primary" disabled={busy} onClick={() => void safe(async () => {
        await call("start_job", { kind: "pipeline", requestKey: crypto.randomUUID(), proposalId: proposal.proposalId,
          expectedReceipt: proposal.expectedReceipt, approved: true }, started);
        setProposal(undefined); await reload();
      })}>Approve reviewed pipeline</button>
    </Modal>}
  </details>;
}
