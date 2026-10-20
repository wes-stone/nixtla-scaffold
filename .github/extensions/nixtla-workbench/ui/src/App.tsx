import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { Activity, ArrowRight, ChevronRight, Database, FileText, Layers, MessageSquare, PanelLeft, Plus, RefreshCw, Square, X } from "lucide-react";
import { activeJob, call, catalogSchema, documentSchema, failure, profileSchema, rowSchema, stateSchema, text, viewSchema } from "./api";
import type { Catalog, Document, DocumentPatch, LibraryItem, Profile, Row, View, WorkspaceState } from "./api";
import { closeDocument, openDocument, reopenDocument, updateDocument } from "./domain";
import { ErrorNotice, Modal, Table, Timeline } from "./components";
import { Operations, Confirmation, PipelineAuthor, ExternalComparison } from "./Operations";
import { Explorer, libraryDocument } from "./Explorer";
import { SavedFile, SavedSource } from "./LibraryDocuments";
import { Analytics } from "./Analytics";
import { ForecastReview } from "./ForecastReview";
import { EnsembleReview } from "./EnsembleReview";
import { ExploreData } from "./ExploreData";
import { JudgmentWorksheet } from "./JudgmentWorksheet";
import { FinalizationPipeline } from "./FinalizationPipeline";
import "./styles.css";

const views: Document["view"][] = ["Data", "Explore data", "Forecast", "Adjust", "Models", "Backtests", "Diagnostics", "Compare", "Research", "Pipeline"];
const objectResult = z.record(z.unknown());
const jobResult = z.object({ jobId: z.string(), created: z.boolean(), status: z.string() });
const executionProposal = z.object({ proposalId: z.string(), expectedReceipt: z.string(), receipt: rowSchema });

export default function App() {
  const [catalog, setCatalog] = useState<Catalog>();
  const [state, setState] = useState<WorkspaceState>(stateSchema.parse({}));
  const [error, setError] = useState("");
  const [saveStatus, setSaveStatus] = useState("Saved locally");
  const [library, setLibrary] = useState(true);
  const [inspector, setInspector] = useState<Row>();
  const [dialog, setDialog] = useState<"import" | "ask" | "recover" | null>(null);
  const [path, setPath] = useState("");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const revision = useRef(0), initialized = useRef(false), locked = useRef(false);
  const stateRef = useRef(state), queue = useRef(Promise.resolve());
  const savedBody = useRef(""), pendingSaves = useRef(0);
  const active = state.documents.find(doc => doc.id === state.activeId);
  const reload = useCallback(async (reset = false) => {
    if (reset) await queue.current;
    const next = await call("catalog", {}, catalogSchema);
    if (initialized.current && next.revision < revision.current) return;
    setCatalog(next);
    const clean = !locked.current && pendingSaves.current === 0 && JSON.stringify(stateRef.current) === savedBody.current;
    if (!initialized.current || reset || (clean && next.revision > revision.current)) {
      const parsed = stateSchema.safeParse(next.state);
      if (!parsed.success) throw new Error("Saved workspace has an incompatible document shape; no edits were applied.");
      stateRef.current = parsed.data;
      savedBody.current = JSON.stringify(parsed.data);
      setState(parsed.data); revision.current = next.revision; initialized.current = true;
      setSaveStatus("Saved locally");
      if (reset) { locked.current = false; setError(""); }
    }
  }, []);
  useEffect(() => { void reload().catch(err => setError(failure(err))); }, [reload]);
  const hasActive = catalog?.jobs.some(job => activeJob(job.status)) ?? false;
  const connected = !!catalog;
  useEffect(() => {
    if (!connected) return;
    let pending = false, stopped = false;
    const synchronize = async () => {
      if (stopped || pending || document.visibilityState === "hidden") return;
      pending = true;
      try { await reload(); }
      catch (err) { if (!stopped) setError(failure(err)); }
      finally { pending = false; }
    };
    const timer = setInterval(() => void synchronize(), hasActive ? 2500 : 10000);
    window.addEventListener("focus", synchronize);
    document.addEventListener("visibilitychange", synchronize);
    return () => {
      stopped = true;
      clearInterval(timer);
      window.removeEventListener("focus", synchronize);
      document.removeEventListener("visibilitychange", synchronize);
    };
  }, [connected, hasActive, reload]);
  useEffect(() => { document.documentElement.dataset.theme = state.theme; }, [state.theme]);

  const commit = (change: (old: WorkspaceState) => WorkspaceState) => {
    if (locked.current) { setError("Reload the saved workspace to resolve the revision conflict before editing."); return; }
    try {
      const next = change(stateRef.current);
      stateRef.current = next; setState(next); setSaveStatus("Saving...");
      pendingSaves.current++;
      queue.current = queue.current.then(async () => {
        try {
          if (locked.current) return;
          const latest = stateRef.current;
          const body = JSON.stringify(latest);
          if (body === savedBody.current) { setSaveStatus("Saved locally"); return; }
          const saved = await call("save_state", { expectedRevision: revision.current, state: latest }, z.object({ revision: z.number() }));
          savedBody.current = body;
          revision.current = saved.revision; setSaveStatus(JSON.stringify(stateRef.current) === body ? "Saved locally" : "Saving...");
        } catch (err) {
          locked.current = true; setSaveStatus("Unsaved changes"); setError(failure(err));
        } finally {
          pendingSaves.current--;
        }
      });
    } catch (err) { setError(failure(err)); }
  };
  const open = (runId?: string, newTab = false) => {
    const run = catalog?.runs.find(item => item.id === runId);
    const doc = documentSchema.parse({ id: crypto.randomUUID(), runId, title: run?.name ?? "New analysis", view: runId ? "Forecast" : "Data" });
    commit(old => openDocument(old, doc, newTab));
    if (matchMedia("(max-width:760px)").matches) setLibrary(false);
  };
  const perform = async (action: () => Promise<void>) => {
    setBusy(true);
    if (!locked.current) setError("");
    try { await action(); }
    catch (err) { setError(failure(err)); }
    finally { setBusy(false); }
  };
  const openSource = (sourceId: string) => {
    const source = catalog?.sources.find(item => item.id === sourceId);
    const doc = documentSchema.parse({ id: crypto.randomUUID(), kind: "source", title: source?.name ?? "Saved source", view: "Data",
      intake: { sourceId, horizon: source?.metadata.confirmationHorizon ?? 6, mapping: source?.metadata.mapping ?? {} } });
    commit(old => openDocument(old, doc));
    if (matchMedia("(max-width:760px)").matches) setLibrary(false);
  };
  const selectLibraryItem = (item: LibraryItem) => {
    if (item.kind === "job") {
      void perform(async () => setInspector(await call("job", { jobId: item.id }, objectResult)));
      return;
    }
    commit(old => openDocument(old, libraryDocument(item, crypto.randomUUID())));
    setInspector(undefined);
    if (matchMedia("(max-width:760px)").matches) setLibrary(false);
  };
  const importRun = () => perform(async () => {
    const result = await call("import_run", { path }, z.object({ id: z.string(), reused: z.boolean() }));
    await reload(); setDialog(null);
    commit(old => openDocument(old, documentSchema.parse({ id: crypto.randomUUID(), runId: result.id, title: path.split(/[\\/]/).at(-1) ?? "Forecast" })));
  });
  const send = () => perform(async () => {
    await call("handoff", { runId: active?.runId, series: active?.series || undefined, document: active?.view ?? "Data", question }, z.object({ sent: z.boolean() }));
    setDialog(null); setQuestion("");
  });
  const jobRevision = catalog?.jobs.map(job => `${job.id}:${job.status}`).join("|") ?? "";

  return <div className="workbench">
    <a className="skip-link" href="#document">Skip to analysis</a>
    <header className="appbar">
      <div className="brand"><button className="icon" aria-label="Toggle library" onClick={() => setLibrary(!library)}><PanelLeft size={19}/></button>
        <Activity size={20} className="brand-icon"/><strong>Forecast <span>Workbench</span></strong><span className="local-label">Nixtla · local</span></div>
      <div className="toolbar"><span className="save-state" role="status">{saveStatus}</span>
        <select aria-label="Appearance" value={state.theme} disabled={!catalog?.initialized} onChange={event => {
          const theme = stateSchema.shape.theme.parse(event.target.value); commit(old => ({ ...old, theme }));
        }}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select>
        <button className="icon" aria-label="Refresh saved workspace" onClick={() => void perform(() => reload())}><RefreshCw size={16}/></button></div>
    </header>
    <div className={`workspace ${library ? "" : "library-hidden"} ${inspector ? "inspector-open" : ""}`}>
      <aside className="library" aria-label="Forecast library">
        <header><h2>Library</h2><button className="icon" aria-label="Create analysis tab" disabled={!catalog?.initialized} onClick={() => open(undefined, true)}><Plus size={17}/></button></header>
        {catalog?.initialized && <Explorer catalog={catalog} active={active} expanded={state.explorerExpanded}
          setExpanded={ids => commit(old => ({ ...old, explorerExpanded: ids }))} select={selectLibraryItem}
          closed={state.closedDocuments} reopen={id => commit(old => reopenDocument(old, id))}/>}
        <div className="library-bottom">
          <button disabled={!catalog?.initialized} onClick={() => open(undefined, true)}><Plus size={16}/> New analysis</button>
          <button className="quiet" disabled={!catalog?.initialized} onClick={() => { setPath(""); setDialog("import"); }}><Layers size={16}/> Import an existing run</button>
          <p>Original artifacts stay portable.<br/>A chart is never an approval.</p>
        </div>
      </aside>
      <main id="document" tabIndex={-1}>
        {hasActive && <button className="quiet" onClick={() => setDialog("recover")}>Recover interrupted session work</button>}
        <nav className="document-tabs" aria-label="Open analysis documents">
          {state.documents.map(doc => <div className={`document-tab ${doc.id === active?.id ? "active" : ""}`} key={doc.id}>
            <button aria-current={doc.id === active?.id ? "page" : undefined} onClick={() => commit(old => ({ ...old, activeId: doc.id }))}><FileText size={13}/>{doc.title}</button>
            <button className="tab-close" aria-label={`Close ${doc.title}`} onClick={() => commit(old => closeDocument(old, doc.id))}><X size={13}/></button>
          </div>)}
          <button className="new-tab" disabled={!catalog?.initialized} onClick={() => active
            ? commit(old => openDocument(old, { ...active, id: crypto.randomUUID() }, true)) : open(undefined, true)}><Plus size={14}/> New tab</button>
        </nav>
        <div className="document-body">
          {error && <><ErrorNotice error={error}/>{locked.current && <>
            <p>Your unsaved edits are still in this panel. Copy anything you need before reloading the saved workspace.</p>
            <button onClick={() => void perform(() => reload(true))}>Reload saved workspace</button>
          </>}</>}
          {!catalog && !error && <p className="loading" role="status">Opening local forecast evidence...</p>}
          {catalog && !catalog.initialized && <section className="welcome">
            <div className="welcome-mark"><Activity size={30}/></div><h1>Start with the data.<br/>Find the forecast worth trusting.</h1>
            <p>One place to inspect your history, compare Nixtla models and test ensembles. Copilot does the analysis; you keep the evidence and the decisions.</p>
            <button className="primary" disabled={busy} onClick={() => void perform(async () => {
              await call("initialize", {}, z.object({ initialized: z.boolean() })); await reload();
              open(undefined, true);
            })}>Create local workspace <ArrowRight size={16}/></button>
            <p className="footnote">Creates an owned catalog in this folder. No data is uploaded, no source is queried, and no forecast is started.</p>
          </section>}
          {catalog?.initialized && !active && <section className="welcome"><h1>Your forecasting workspace</h1>
            <p>Select an existing forecast, or bring in a source and let the analysis start there.</p>
            <div className="toolbar"><button className="primary" onClick={() => open(undefined, true)}>New analysis <ArrowRight size={16}/></button><button onClick={() => setDialog("import")}>Import a run</button></div></section>}
          {catalog?.initialized && active && <>
            <header className="document-heading"><div><h1>{active.title}</h1><p>{active.kind === "artifact" ? "Saved file · read-only" : active.kind === "source" ? "Saved data · read-only" : active.runId ? "Saved forecast evidence" : "Inspect first. Configure only what matters."}</p></div>
              <button onClick={() => { setQuestion(""); setDialog("ask"); }}><MessageSquare size={15}/> Ask Copilot</button></header>
            {(active.kind === "analysis" || active.kind === "source" || active.artifactName.toLowerCase().endsWith(".csv")) && <nav className="view-nav" aria-label="Analysis views">{(active.kind === "analysis" ? views : ["Data", "Explore data"] as const).map(view => <button key={view} className={view === active.view ? "selected" : ""}
              disabled={!active.runId && view !== "Data" && view !== "Explore data"} onClick={() => commit(old => updateDocument(old, active.id, { view }))}>{view}</button>)}</nav>}
            {active.view === "Explore data" ? <ExploreData key={active.id} document={active} update={patch => commit(old => updateDocument(old, active.id, patch))} inspect={setInspector}/> :
              active.kind === "artifact" ? <SavedFile key={active.id} document={active} inspect={setInspector}/> :
              active.kind === "source" ? <SavedSource key={active.id} document={active} inspect={setInspector} useSource={source => commit(old => openDocument(old, documentSchema.parse({
                id: crypto.randomUUID(), title: active.title, view: "Data",
                intake: { sourceId: source.sourceId, horizon: source.horizon, mapping: source.mapping },
              })))}/> :
              !active.runId ? <SourceDocument key={active.id} document={active} update={patch => commit(old => updateDocument(old, active.id, patch))} catalog={catalog} onChange={reload} onInspect={setInspector} onError={setError}/> :
              <EvidenceDocument key={active.id} document={active} catalog={catalog} jobRevision={jobRevision} theme={state.theme}
                update={patch => commit(old => updateDocument(old, active.id, patch))} inspect={setInspector}
                reload={reload} onError={setError} ask={value => { setQuestion(value); setDialog("ask"); }}/>}
          </>}
        </div>
        {!!catalog?.jobs.length && <section className="activity-strip" aria-label="Recent analysis jobs">
          {catalog.jobs.filter((job, index) => activeJob(job.status) || index < 3).map(job => <div key={job.id}>
            <Activity size={14}/><span><strong>{job.kind}</strong> · {job.status.replaceAll("_", " ")}{job.error && <small>{job.error}</small>}</span>
            {activeJob(job.status) && <button className="quiet" disabled={job.status === "cancel_requested"} onClick={() => void perform(async () => {
              const result = await call("cancel_job", { jobId: job.id }, objectResult);
              if (typeof result.warning === "string") setError(result.warning);
              await reload();
            })}><Square size={12}/> Stop</button>}
            {job.resumeAvailable && <button className="quiet" disabled={busy} title="Reuse the frozen request and saved baseline; no fresh time or research budget."
              onClick={() => void perform(async () => {
                await call("resume_job", { jobId: job.id, requestKey: crypto.randomUUID() }, jobResult); await reload();
              })}>Resume · {Math.floor(job.remainingSeconds ?? 0)}s left</button>}
            {typeof job.result?.runId === "string" && <button className="link" onClick={() => open(String(job.result?.runId))}>Open forecast <ArrowRight size={13}/></button>}
            {typeof job.result?.sourceId === "string" && <button className="link" onClick={() => openSource(String(job.result?.sourceId))}>Inspect source <ArrowRight size={13}/></button>}
            <button className="link" onClick={() => void perform(async () => {
              const detail = await call("job", { jobId: job.id }, objectResult);
              const evidence = await call("job_evidence", { jobId: job.id }, objectResult);
              setInspector({ ...detail, researchEvidence: evidence });
            })}>Details</button>
          </div>)}
        </section>}
      </main>
      {inspector && <aside className="inspector" aria-label="Evidence inspector"><header><h2>Evidence</h2><button className="icon" aria-label="Close inspector" onClick={() => setInspector(undefined)}><X size={17}/></button></header>
        <p>Exact values and provenance for the selected scope.</p><dl>{Object.entries(inspector).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{text(value) || "Not available"}</dd></div>)}</dl></aside>}
    </div>
    {dialog === "import" && <Modal title="Import an existing forecast" close={() => setDialog(null)}>
      <p>Choose a completed Nixtla run folder. Its evidence is copied into an immutable local snapshot; the original is not changed.</p>
      <label>Run folder<input autoFocus value={path} onChange={event => setPath(event.target.value)} placeholder="C:\...\runs\forecast"/></label>
      <button className="primary" disabled={busy || !path.trim()} onClick={() => void importRun()}>Import saved evidence</button>
    </Modal>}
    {dialog === "ask" && <Modal title="Continue in Copilot" close={() => setDialog(null)}>
      <p>This sends your question with the current forecast and series scope. It does not authorize a model run, adoption, external access or publishing.</p>
      <label>What should we investigate?<textarea autoFocus rows={5} maxLength={4000} value={question} onChange={event => setQuestion(event.target.value)}/></label>
      <button className="primary" disabled={busy || !question.trim()} onClick={() => void send()}>Send question to Copilot <ArrowRight size={16}/></button>
    </Modal>}
    {dialog === "recover" && <Modal title="Reconcile interrupted work" close={() => setDialog(null)}>
      <p>Verify recorded process identities. Jobs belonging to a live session stay active. If an owner has died, stop only its verified orphan process tree and preserve an interrupted receipt. Partial artifacts stay on disk; nothing is replayed.</p>
      <button disabled={busy} className="primary" onClick={() => void perform(async () => {
        setInspector(await call("recover_jobs", { approved: true }, objectResult)); await reload(); setDialog(null);
      })}>Approve orphan recovery</button>
    </Modal>}
  </div>;
}

function SourceDocument({ document: doc, update, catalog, onChange, onInspect, onError }: {
  document: Document; update: (patch: DocumentPatch) => void;
  catalog: Catalog; onChange: () => Promise<void>; onInspect: (row: Row) => void; onError: (error: string) => void;
}) {
  const [profile, setProfile] = useState<Profile>();
  const [external, setExternal] = useState<z.infer<typeof executionProposal>>();
  const { path, mapping, sourceId, horizon, policy, repair } = doc.intake;
  const change = (patch: Partial<Document["intake"]>) => update({ intake: patch });
  const setMapping = (mapping: Record<string, string>) => change({ mapping });
  const setSourceId = (sourceId: string) => change({ sourceId });
  const [busy, setBusy] = useState(false);
  const run = async (id: string, kind = "forecast") => {
    await call("start_job", {
      kind, requestKey: crypto.randomUUID(), sourceId: id,
      name: path.split(/[\\/]/).at(-1)?.replace(/\.[^.]+$/, "") || catalog.sources.find(source => source.id === id)?.name || doc.title,
      spec: { ...z.record(z.unknown()).parse(doc.intake.advanced.trim() ? JSON.parse(doc.intake.advanced) : {}),
        horizon, model_policy: policy, ...(repair ? { fill_method: repair } : {}) },
    }, jobResult);
    await onChange();
  };
  const inspect = async (forecast: boolean) => {
    setBusy(true); onError("");
    try {
      const result = await call("profile", { path, mapping, horizon, holdoutWindows: doc.intake.holdoutWindows }, profileSchema);
      setProfile(result); change({ mapping: result.mapping, sourceId: result.sourceId ?? "" });
      await onChange();
      if (forecast && result.sourceId && !result.questions.length) await run(result.sourceId);
    } catch (error) { onError(failure(error)); }
    finally { setBusy(false); }
  };
  const start = async (kind: string) => {
    setBusy(true); onError("");
    try { await run(sourceId, kind); }
    catch (error) { onError(failure(error)); }
    finally { setBusy(false); }
  };
  return <section className="source-document">
    <div className="source-intro"><h2>What are we forecasting?</h2><p>Bring a historical table. We’ll inspect its shape and calendar before choosing the feasible models.</p></div>
    <label className="source-path">Local CSV or saved-value workbook<input value={path} onChange={event => { change({ path: event.target.value, sourceId: "", mapping: {} }); setProfile(undefined); }}
      placeholder="C:\...\monthly_revenue.csv"/></label>
    <div className="source-controls"><label>Forecast horizon<input aria-label="Forecast horizon" type="number" min={1} max={120} value={horizon} onChange={event => change({ horizon: Number(event.target.value) })}/><small>Periods at the source’s frequency</small></label>
      <div className="toolbar"><button disabled={busy || !path} onClick={() => void inspect(false)}>Inspect data</button>
        <button className="primary" disabled={busy || !path || horizon < 1} onClick={() => void inspect(true)}>{busy ? "Analyzing..." : "Analyze & forecast"}<ArrowRight size={15}/></button></div></div>
    <p className="boundary-note">Local native models, rolling-origin evidence and up to two advisory blend comparisons within 15 minutes. No cloud calls or package installation. Stop remains available while models run.</p>
    <details><summary>Advanced native settings and protected research</summary><p>Optional full ForecastSpec settings retain hierarchy, transformations, drivers, feature recipes and installed model families. No arbitrary executable is authorized here.</p>
      <label>Advanced ForecastSpec JSON<textarea aria-label="Advanced ForecastSpec JSON" rows={5} value={doc.intake.advanced} onChange={event => change({ advanced: event.target.value })} placeholder={'{"hierarchy": ["region"], "hierarchy_reconciliation": "bottom_up"}'}/></label>
      <label>Protected windows<select aria-label="Protected windows" value={doc.intake.holdoutWindows} onChange={event => change({ holdoutWindows: Number(event.target.value) })}><option value={1}>One native-research confirmation window</option><option value={21}>20 blend calibration windows + 1 confirmation</option></select></label>
      <p className="footnote">Inspect only to preserve the protected tail. A full-history forecast consumes it.</p>
      <button disabled={!sourceId || busy} onClick={() => void (async () => {
        setBusy(true); onError("");
        try {
          const spec = { ...z.record(z.unknown()).parse(doc.intake.advanced.trim() ? JSON.parse(doc.intake.advanced) : {}),
            horizon, model_policy: policy, ...(repair ? { fill_method: repair } : {}) };
          setExternal(await call("prepare_forecast", { sourceId, spec }, executionProposal));
        } catch (error) { onError(failure(error)); } finally { setBusy(false); }
      })()}>Review custom / external execution</button>
    </details>
    {external && <Modal title="Approve custom or external execution" close={() => setExternal(undefined)}>
      <p>{text(external.receipt.authorization)}</p><pre>{JSON.stringify(external.receipt.spec, null, 2)}</pre>
      <button className="primary" disabled={busy} onClick={() => void (async () => {
        setBusy(true); onError("");
        try {
          await call("start_job", { kind: "pipeline", requestKey: crypto.randomUUID(), proposalId: external.proposalId,
            expectedReceipt: external.expectedReceipt, approved: true }, jobResult);
          setExternal(undefined); await onChange();
        } catch (error) { onError(failure(error)); } finally { setBusy(false); }
      })()}>Approve exact execution</button>
    </Modal>}
    {profile && <section className="findings"><h2>What the data says</h2>
      {profile.findings?.map(finding => <p key={finding}>{finding}</p>)}
      {!!profile.questions.length && <div className="notice">{profile.questions[0]}</div>}
      {profile.questions.length > 0 && <div className="form-grid">
        {(["time", "target", "id"] as const).map(key => <label key={key}>{key === "id" ? "Series key (optional)" : `${key} column`}<select value={mapping[key] ?? ""} onChange={event => setMapping({ ...mapping, [key]: event.target.value })}>
          <option value="">Choose...</option>{profile.columns.map(col => <option key={col}>{col}</option>)}</select></label>)}
        <label>Frequency<input value={mapping.freq ?? ""} onChange={event => setMapping({ ...mapping, freq: event.target.value })} placeholder="MS, ME, D, W-SUN..."/></label>
        <button disabled={busy} onClick={() => void inspect(false)}>Apply mapping & inspect</button>
      </div>}
      <Table rows={profile.preview} title="Source preview" inspect={onInspect}/>
      {profile.assumptions?.map(assumption => <p className="footnote" key={assumption}>{assumption}</p>)}
    </section>}
    <section className="saved-source"><h3>Use a saved source snapshot</h3><div className="form-grid">
      <label>Snapshot<select aria-label="Snapshot" value={sourceId} onChange={event => setSourceId(event.target.value)}><option value="">Select a snapshot</option>
        {sourceId && !catalog.sources.some(source => source.id === sourceId) && <option value={sourceId}>{doc.title}</option>}
        {catalog.sources.map(source => <option key={source.id} value={source.id}>{source.name}</option>)}</select></label>
      <label>Model policy<select aria-label="Model policy" value={policy} onChange={event => change({ policy: event.target.value })}><option value="standard">Standard · eligible native models</option><option value="baseline">Simple baselines</option><option value="statsforecast">StatsForecast</option><option value="mlforecast">MLForecast (requires installed runtime)</option><option value="all">All installed eligible families</option></select></label>
      <label>Missing-observation handling<select aria-label="Missing-observation handling" value={repair} onChange={event => change({ repair: event.target.value })}><option value="">Require a decision when missing</option><option value="ffill">Carry last observation forward</option><option value="zero">Explicit zero fill</option><option value="interpolate">Interpolate</option><option value="drop">Drop missing observations</option></select></label>
    </div><div className="toolbar"><button disabled={!sourceId || busy} onClick={() => void start("forecast")}>Run forecast</button><button disabled={!sourceId || busy} onClick={() => void start("research")}>Run bounded research</button></div></section>
    <PipelineAuthor document={doc} update={update} reload={onChange} inspect={onInspect} onError={onError}/>
  </section>;
}

function EvidenceDocument({ document: doc, catalog, jobRevision, theme, update, inspect, reload, onError, ask }: {
  document: Document; catalog: Catalog; jobRevision: string; theme: string;
  update: (patch: DocumentPatch) => void; inspect: (row: Row) => void;
  reload: () => Promise<void>; onError: (error: string) => void; ask: (question: string) => void;
}) {
  const [view, setView] = useState<View>(), [historyRows, setHistoryRows] = useState<Row[]>([]);
  const [backtest, setBacktest] = useState<View>(), [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false), [cutoff, setCutoff] = useState("");
  const { policy: blendPolicy, members, weights } = doc.recipe;
  const note = doc.note;
  const setBlendPolicy = (policy: string) => update({ recipe: { ...doc.recipe, policy } });
  const setMembers = (members: string[]) => update({ recipe: { ...doc.recipe, members } });
  const setWeights = (weights: Record<string, string>) => update({ recipe: { ...doc.recipe, weights } });
  const setNote = (note: string) => update({ note });
  const [receipt, setReceipt] = useState(""), [ensembleOpen, setEnsembleOpen] = useState(false);
  const [adoption, setAdoption] = useState(false), [refresh, setRefresh] = useState(0);
  const [confirm, setConfirm] = useState(false);
  const [submitting, setSubmitting] = useState(false), submittingRef = useRef(false);
  const [loadedScope, setLoadedScope] = useState("");
  const runId = doc.runId ?? "";
  const kind = doc.view === "Data" ? "history" : doc.view === "Compare" && doc.comparison === "models" ? (doc.blendId ? "ensemble_metrics" : "metrics") : "forecast";
  const scope = JSON.stringify([runId, doc.series, doc.view, kind, doc.blendId, doc.comparison, offset, cutoff]);
  useEffect(() => { setOffset(0); setCutoff(""); setView(undefined); setBacktest(undefined); }, [runId, doc.series, doc.view]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    const load = async () => {
      const next = await call("view", { runId, series: doc.series || undefined, kind, blendId: doc.blendId || undefined, offset, limit: 500 }, viewSchema, controller.signal);
      if (controller.signal.aborted) return;
      if (!doc.series && next.series.length) { update({ series: next.series[0] }); return; }
      let history: View | undefined, bt: View | undefined;
      if (doc.view === "Forecast") {
        history = await call("view", { runId, series: doc.series, kind: "history", limit: 500, tail: true }, viewSchema, controller.signal);
      }
      if (doc.view === "Compare" && doc.comparison === "models") {
        bt = await call("view", { runId, series: doc.series, blendId: doc.blendId || undefined, kind: doc.blendId ? "ensemble_backtest" : "backtest", cutoff: cutoff || undefined, tail: true, limit: 500 }, viewSchema, controller.signal);
      }
      if (controller.signal.aborted) return;
      setView(next);
      if (history) setHistoryRows(history.rows);
      if (bt) setBacktest(bt);
      setLoadedScope(scope);
    };
    void load().catch(error => { if (!controller.signal.aborted) onError(failure(error)); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [runId, doc.series, doc.view, doc.blendId, doc.comparison, offset, cutoff, jobRevision, refresh]);
  const dark = theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme:dark)").matches);
  const safe = async (action: () => Promise<void>) => {
    if (submittingRef.current) return;
    submittingRef.current = true; setSubmitting(true); onError("");
    try { await action(); } catch (error) { onError(failure(error)); }
    finally { submittingRef.current = false; setSubmitting(false); }
  };
  if (!view || loadedScope !== scope) return <p className="loading" role="status">Reading scoped evidence...</p>;
  const periods = backtest?.cutoffs ?? [];
  const activeCutoff = cutoff || periods.at(-1) || "";
  const replay = backtest?.rows.filter(row => text(row.cutoff) === activeCutoff) ?? [];
  const selectedBlend = view.blends.find(blend => blend.id === doc.blendId);
  const chartFields = ["y", ...(doc.blendId ? ["Blend"] : []), ...(selectedBlend && Array.isArray(selectedBlend.recipe.members)
    ? selectedBlend.recipe.members.filter((member): member is string => typeof member === "string") : view.models)];
  const sourceId = text(view.run.metadata.sourceId);
  const seriesGates: Row[] = Array.isArray(view.gate.series) ? view.gate.series.filter((item): item is Row => typeof item === "object" && item !== null && !Array.isArray(item)) : [];
  const scopedGate = seriesGates.find(item => text(item.unique_id) === doc.series) ?? view.gate;
  const failedGates = Array.isArray(scopedGate.failed_gates) ? scopedGate.failed_gates.map(item => text(item).replaceAll("_", " ")).join(", ") : "";
  return <>
    {loading && <p className="footnote" role="status">Refreshing saved evidence; the current scope remains visible.</p>}
    <div className="scope-bar"><label>Series<select aria-label="Series" value={doc.series} onChange={event => update({ series: event.target.value })}>{view.series.map(series => <option key={series}>{series}</option>)}</select></label>
      <span>{text(view.spec.freq)} · {text(view.spec.horizon)} forecast periods · {text(view.spec.unit_label) || "Units not recorded"}</span>
      <button className="quiet" onClick={() => inspect({ ...view.gate, spec: view.spec })}>Evidence status <ChevronRight size={14}/></button></div>
    <div className="readiness"><span className="status">{text(view.gate.status).replaceAll("_", " ")}</span>
      <p>{scopedGate.planning_ready_claim_allowed === false ? `Planning not approved.${failedGates ? ` Failed checks: ${failedGates}.` : " Inspect the saved evidence before relying on this forecast."}` : text(view.gate.reason) || "Readiness belongs to the saved evidence, not the display settings."}</p></div>
    {view.run.metadata.purpose === "judgment_scenario" && <p className="notice"><strong>{text(view.run.metadata.judgmentLabel)} · analyst judgment.</strong> Model: {text(view.run.metadata.judgmentModel) || "saved selection"}. No backtest or prediction intervals transfer to these adjusted values. <button className="link" onClick={() => update({ runId: text(view.run.metadata.judgmentBaseRunId), judgment: undefined, view: "Forecast", analysis: { previewModel: "", models: ["*"] } })}>Open original baseline</button></p>}
    {doc.view === "Adjust" && <JudgmentWorksheet key={`${doc.runId}:${doc.series}`} document={doc} view={view} update={update} dark={dark} inspect={inspect} reload={reload}/>}
    {doc.view === "Forecast" && <>
      <div className="section-heading"><div><h2>Actuals and forecast</h2><p>One saved version. History and future remain distinct.</p></div><button className="quiet" onClick={() => ask("Explain this series' forecast and its main limitations using the saved evidence.")}>Why this forecast?</button></div>
      <ForecastReview document={doc} view={view} update={update} dark={dark} inspect={inspect} historyRows={historyRows}/>
      <p className="footnote">Showing up to 500 most recent history observations and the current forecast page. All observations remain in Data and scoped exports. Saved intervals are not synthesized; inspect calibration before relying on them.</p>
    </>}
    {["Models", "Backtests", "Diagnostics"].includes(doc.view) && <Analytics document={doc} view={view} update={update} dark={dark} inspect={inspect}/>}
    {doc.view === "Data" && <><h2>Historical evidence</h2><p>Original observations at the forecast grain. Missingness, freshness and transformations remain inspectable.</p>
      <Timeline rows={view.rows} fields={["y"]} dark={dark} inspect={inspect}/><button className="quiet" onClick={() => inspect(view.profile)}>Inspect data profile</button></>}
    {doc.view === "Compare" && <nav className="comparison-nav" aria-label="Comparison type">
      <button aria-pressed={doc.comparison === "models"} onClick={() => update({ comparison: "models" })}>Models & ensembles</button>
      <button aria-pressed={doc.comparison === "versions"} onClick={() => update({ comparison: "versions" })}>Version changes</button>
      <button aria-pressed={doc.comparison === "external"} onClick={() => update({ comparison: "external" })}>External & BYO</button></nav>}
    {doc.view === "Compare" && doc.comparison === "external" && <ExternalComparison document={doc} view={view} update={update} reload={reload} inspect={inspect} onError={onError}/>}
    {doc.view === "Compare" && doc.comparison === "versions" && <VersionComparison document={doc} catalog={catalog} update={update} inspect={inspect} dark={dark} onError={onError}/>}
    {doc.view === "Compare" && doc.comparison === "models" && <>
      <div className="section-heading"><div><h2>Models and ensembles</h2><p>Review the existing tournament, then test a frozen blend on paired historical windows.</p></div>
        <button onClick={() => setEnsembleOpen(!ensembleOpen)}><Layers size={15}/>{ensembleOpen ? "Close recipe" : "Build ensemble"}</button></div>
      <p className="footnote">{view.spec.paired_validation ? "This tournament uses shared finite-row support within each series." : "Legacy model scores may use different row support."} New blend comparisons pair their own cohort; no result receives automatic promotion.</p>
      {!doc.blendId && <EnsembleReview document={doc} view={view} update={update} dark={dark} inspect={inspect}/>}
      <label>Comparison evidence<select aria-label="Comparison evidence" value={doc.blendId} onChange={event => { update({ blendId: event.target.value }); setOffset(0); setCutoff(""); }}>
        <option value="">Original model tournament</option>{view.blends.map(blend => <option key={blend.id} value={blend.id}>{text(blend.recipe.policy)} · {blend.id.slice(0, 8)}</option>)}</select></label>
      {ensembleOpen && <section className="ensemble-editor"><h3>Ensemble recipe</h3><div className="form-grid">
        <label>Weighting policy<select aria-label="Weighting policy" value={blendPolicy} onChange={event => setBlendPolicy(event.target.value)}><option value="equal">Equal weights</option><option value="fixed">Your fixed weights</option><option value="inverse_error">Prior-error weights</option><option value="top_k">Top 3 on prior evidence</option><option value="family_diverse">Family-diverse average</option><option value="greedy">Objective-aware greedy (experimental)</option><option value="horizon_greedy" disabled>Horizon groups (not qualified)</option></select></label>
        <p>{blendPolicy === "horizon_greedy" ? "Horizon grouping did not improve the global blend in development. Use a supported policy instead." : blendPolicy === "greedy" ? "Learns only from at least three fully observed prior origins, using this run’s declared objective. This is an experimental challenger, not an automatic winner." : "Membership and weight-learning rules are frozen into a new experiment. Missing components exclude a row, rather than silently changing the blend."}</p></div>
        <div className="member-list">{view.models.map(model => <div key={model}><label className="inline"><input type="checkbox" checked={members.includes(model)} onChange={event => setMembers(event.target.checked ? [...members, model] : members.filter(m => m !== model))}/>{model}</label>
          {blendPolicy === "fixed" && members.includes(model) && <input aria-label={`${model} weight`} type="number" min={0} step="any" value={weights[model] ?? "1"} onChange={event => setWeights({ ...weights, [model]: event.target.value })}/>}</div>)}</div>
        <button className="primary" disabled={submitting || !members.length || members.length > 12 || blendPolicy === "horizon_greedy"} onClick={() => void safe(async () => {
          const settings = z.object({ horizon: z.number().int().positive(), selection_policy: z.object({ objective: z.enum(["daily_rmse", "monthly_landing"]) }).optional() }).parse(view.spec);
          await call("start_job", { kind: "blend", requestKey: crypto.randomUUID(), runId, recipe: {
            policy: blendPolicy, members, max_models: 3, ...(blendPolicy === "fixed" ? { weights: members.map(m => Number(weights[m] ?? "1")) } : {}),
            ...(blendPolicy === "greedy" ? { version: 2, objective: settings.selection_policy?.objective ?? "daily_rmse" } : {}),
          } }, jobResult); await reload(); setEnsembleOpen(false);
        })}>Backtest this recipe <ArrowRight size={15}/></button>
      </section>}
      {!!view.blends.length && <section className="blend-results">
        <p className="footnote">{selectedBlend ? `${text(selectedBlend.receipt.paired_rows)} paired rows · ${text(selectedBlend.receipt.excluded_rows)} excluded · ${selectedBlend.receipt.confirmation_status === "independently_reserved_within_workspace" ? "Reserved confirmation and blend-residual intervals; inspect small-sample coverage." : "Point-only retrospective evidence."}` : `${view.blends.length} saved ensemble experiments are available in the comparison selector.`} Adoption does not create a planning lock.</p>
        {selectedBlend && <div className="toolbar"><button onClick={() => inspect({ recipe: selectedBlend.recipe, ...selectedBlend.receipt })}>Inspect selected recipe</button>
          <button onClick={() => setConfirm(true)}>Calibrate & confirm</button>
          <button disabled={!selectedBlend.receipt.paired_rows || ensembleOpen} onClick={() => setAdoption(true)}>{selectedBlend.receipt.confirmation_status === "independently_reserved_within_workspace" ? "Adopt reviewed version" : "Adopt directional version"}</button></div>}</section>}
      {replay.length > 0 && <><div className="section-heading"><h3>Replay a forecast origin</h3><label>Historical origin<select aria-label="Historical origin" value={activeCutoff} onChange={event => setCutoff(event.target.value)}>{periods.map(period => <option key={period}>{period}</option>)}</select></label></div>
        <Timeline rows={replay} fields={chartFields} origin={activeCutoff} dark={dark} inspect={inspect}/>
        <Table rows={replay} title="Origin evidence" inspect={inspect}/></>}
      {backtest && backtest.total > backtest.rows.length && <p className="footnote">Replay is bounded to 500 rows. Choose an origin to load its scope; export retains every backtest row.</p>}
    </>}
    {doc.view === "Research" && <>
      <div className="section-heading"><div><h2>Research with a reason</h2><p>Falsifiable hypotheses, paired controls and an explicit stopping point.</p></div>
        <button onClick={() => ask("Prepare an unexposed research source for this horizon. Profile only its tuning partition, preserve confirmation, then propose a bounded native research run.")}>Prepare protected research <ArrowRight size={15}/></button></div>
      {!sourceId && <p className="notice">This imported version has no workbench source binding. Inspect its source as a new snapshot before running research.</p>}
      <p>Native research uses the existing optimizer with at most two iterations and one variant per iteration. Its original chronological confirmation and claim rules remain in force.</p>
      <p className="footnote">This saved forecast has already seen its history. To start confirmatory research, use New analysis → Inspect data on an unexposed source, then Run bounded research. A renamed file does not restore untouched evidence.</p>
      {!!view.hypotheses.length && <Table rows={view.hypotheses} title="Why these experiments" inspect={inspect}/>}
      <button className="quiet" onClick={() => ask("Inspect this forecast's diagnostics and identify the most valuable next falsifiable improvement. Do not start external execution.")}>Investigate the next hypothesis in Copilot</button>
      <section className="research-notes"><h3>Notes tied to this version</h3><label>Finding or decision<textarea rows={4} value={note} maxLength={16000} onChange={event => setNote(event.target.value)}/></label>
        <button disabled={!note.trim()} onClick={() => void safe(async () => { await call("save_note", { runId, body: note }, objectResult); setNote(""); await reload(); setRefresh(n => n + 1); setReceipt("Note saved to this version."); })}>Save note</button>
        {view.notes.map((saved, i) => <article key={i}><small>{text(saved.created_at)}</small><p>{text(saved.body)}</p></article>)}
      </section>
    </>}
    {doc.view === "Pipeline" && <>
      {view.run.metadata.purpose === "judgment_scenario" ? <FinalizationPipeline document={doc} view={view} update={update} dark={dark} inspect={inspect}/> : <>
      <div className="section-heading"><div><h2>From source to evidence</h2><p>Inspect, refresh and maintain this forecast without changing its original version.</p></div></div>
      <div className="pipeline">{[
        { label: "Source snapshot", detail: sourceId ? "Bound to this run" : "Imported evidence", value: view.run.metadata, icon: Database },
        { label: "Model policy", detail: text(view.spec.model_policy), value: view.spec, icon: Layers },
        { label: "Evaluation", detail: "Rolling-origin artifacts", value: view.gate, icon: Activity },
        { label: "Saved version", detail: "Original files retained", value: { runId, artifacts: view.artifacts }, icon: FileText },
      ].map(stage => <button key={stage.label} onClick={() => inspect(stage.value)}><stage.icon size={19}/><strong>{stage.label}</strong><small>{stage.detail}</small></button>)}</div>
      <p>Finalize a chosen model with an optional calendar layer and final tweaks. Save a named version to retain the ordered recipe and exact stage comparison.</p>
      <button onClick={() => update({ view: "Adjust" })}>Open finalization worksheet</button></>}
      <Operations document={doc} view={view} catalog={catalog} update={update} reload={reload} inspect={inspect} onError={onError}/>
      <PipelineAuthor document={doc} update={update} reload={reload} inspect={inspect} onError={onError}/>
    </>}
    {(["Data", "Forecast"].includes(doc.view) || (doc.view === "Compare" && doc.comparison === "models")) && <>
      {doc.view !== "Forecast" && <Table rows={view.rows} columns={view.columns} inspect={inspect} title={doc.view === "Compare" ? "Saved model scores" : "Exact evidence values"}/>}
      <div className="table-footer"><span>{offset + 1}–{Math.min(offset + view.rows.length, view.total)} of {view.total} rows</span><div className="toolbar">
        <button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 500))}>Previous</button>
        <button disabled={view.nextOffset === null} onClick={() => setOffset(view.nextOffset ?? 0)}>Next</button>
        <button onClick={() => void safe(async () => {
          const output = await call("export_values", { runId, kind, series: doc.series, blendId: doc.blendId || undefined }, z.object({ path: z.string(), rows: z.number(), sha256: z.string() }));
          setReceipt(`Saved ${output.rows} rows to ${output.path}`);
        })}>Export scoped CSV</button></div></div>
    </>}
    {receipt && <p className="save-receipt" role="status">{receipt}</p>}
    {adoption && selectedBlend && <Modal title="Adopt the reviewed ensemble version" close={() => setAdoption(false)}>
      <p>This creates a new version for every series using this exact recipe. The previous forecast stays unchanged. This is not a promotion, publication or official lock. Readiness is regenerated from the candidate's own evidence.</p>
      <p>{selectedBlend.receipt.confirmation_status === "independently_reserved_within_workspace" ? "Reserved confirmation and blend-residual interval calibration are recorded. Inspect coverage and policy results before relying on the bounds." : "This candidate has point-only retrospective evidence, not planning-ready approval."}</p>
      <p>Recipe: {text(selectedBlend.recipe.policy)} · {text(selectedBlend.receipt.paired_rows)} paired observations.</p>
      <button className="primary" onClick={() => { setAdoption(false); void safe(async () => {
        await call("start_job", { kind: "adopt", requestKey: crypto.randomUUID(), runId, blendId: selectedBlend.id, expectedReceipt: selectedBlend.receiptHash, approved: true }, jobResult);
        await reload();
      }); }}>{selectedBlend.receipt.confirmation_status === "independently_reserved_within_workspace" ? "Approve new reviewed version" : "Approve new directional version"}</button>
    </Modal>}
    {confirm && selectedBlend && <Confirmation runId={runId} blend={selectedBlend} horizon={view.spec.horizon} freq={view.spec.freq} catalog={catalog}
      close={() => setConfirm(false)} reload={reload} onError={onError}/>}
    <details className="evidence-files"><summary>Source artifacts · {view.artifacts.length} original files</summary><div className="artifact-list">
      {view.artifacts.filter(name => /\.(json|csv|md|jsonl)$/.test(name)).map(name => <button key={name} onClick={() => void safe(async () => {
        const artifact = await call("artifact", { runId, name }, z.object({ name: z.string(), text: z.string(), sha256: z.string() }));
        inspect(artifact);
      })}><FileText size={14}/>{name}<ChevronRight size={14}/></button>)}</div>
    </details>
  </>;
}

const comparisonSchema = z.object({
  rows: z.array(rowSchema), columns: z.array(z.string()), total: z.number(),
  nextOffset: z.number().nullable(), limitations: z.string(),
});
function VersionComparison({ document: doc, catalog, update, inspect, dark, onError }: {
  document: Document; catalog: Catalog; update: (patch: DocumentPatch) => void;
  inspect: (row: Row) => void; dark: boolean; onError: (message: string) => void;
}) {
  const [result, setResult] = useState<z.infer<typeof comparisonSchema>>();
  const [offset, setOffset] = useState(0);
  useEffect(() => {
    setResult(undefined);
    if (!doc.referenceRunId) return;
    const controller = new AbortController();
    void call("compare_versions", { runId: doc.runId, referenceRunId: doc.referenceRunId, series: doc.series, offset, limit: 500 }, comparisonSchema, controller.signal)
      .then(setResult).catch(error => { if (!controller.signal.aborted) onError(failure(error)); });
    return () => controller.abort();
  }, [doc.runId, doc.referenceRunId, doc.series, offset]);
  return <section>
    <div className="section-heading"><div><h2>What changed between versions?</h2><p>Compare the saved forecasts at matching series and periods, without rewriting either version.</p></div></div>
    <label>Reference version<select aria-label="Reference version" value={doc.referenceRunId} onChange={event => { setOffset(0); update({ referenceRunId: event.target.value }); }}>
      <option value="">Choose a saved version...</option>{catalog.runs.filter(run => run.id !== doc.runId).map(run => <option key={run.id} value={run.id}>{text(run.metadata.judgmentLabel) || run.name} · {new Date(run.created_at).toLocaleString()}</option>)}</select></label>
    {result && <><p className="boundary-note">{result.limitations}</p><Timeline rows={result.rows} fields={["current", "previous"]} dark={dark} inspect={inspect}/>
      <Table rows={result.rows} columns={result.columns} title="Exact version deltas" inspect={inspect}/>
      <div className="table-footer"><span>{result.total} scoped rows</span><div className="toolbar"><button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 500))}>Previous</button>
        <button disabled={result.nextOffset === null} onClick={() => setOffset(result.nextOffset ?? 0)}>Next</button></div></div></>}
  </section>;
}
