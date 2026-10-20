import { useEffect, useRef, useState } from "react";
import type { ReactNode, KeyboardEvent } from "react";
import { Activity, ChevronRight, ChevronsDownUp, Database, FileText, Folder, FolderOpen, History, Search, X } from "lucide-react";
import { call, documentSchema, failure, libraryPageSchema } from "./api";
import type { Catalog, Document, LibraryItem } from "./api";

export function libraryDocument(item: LibraryItem, id: string): Document {
  return documentSchema.parse({
    id, runId: item.runId ?? undefined,
    title: item.kind === "artifact" ? item.artifactName?.split("/").at(-1) ?? item.name : item.family ?? item.name,
    kind: item.kind === "source" ? "source" : item.kind === "artifact" ? "artifact" : "analysis",
    artifactName: item.artifactName ?? "", intake: { sourceId: item.sourceId ?? "" },
    view: item.kind === "source" ? "Data" : item.kind === "note" ? "Research" : "Forecast",
  });
}

function useEntries(input: Record<string, unknown>, stamp: string) {
  const [items, setItems] = useState<LibraryItem[]>([]), [total, setTotal] = useState(0);
  const [next, setNext] = useState<number | null>(null), [busy, setBusy] = useState(true), [error, setError] = useState("");
  const controller = useRef<AbortController | undefined>(undefined), requestedOffset = useRef(0);
  const signature = JSON.stringify(input);
  const previousSignature = useRef(signature);
  const load = async (offset: number) => {
    controller.current?.abort();
    const current = new AbortController(); controller.current = current; requestedOffset.current = offset;
    setBusy(true); setError("");
    try {
      const page = await call("library", { ...input, offset, limit: 50 }, libraryPageSchema, current.signal);
      if (current.signal.aborted) return;
      setItems(previous => offset ? [...previous, ...page.items] : page.items);
      setTotal(page.total); setNext(page.nextOffset);
    } catch (err) { if (!current.signal.aborted) setError(failure(err)); }
    finally { if (!current.signal.aborted) setBusy(false); }
  };
  useEffect(() => {
    if (signature !== previousSignature.current) { setItems([]); setTotal(0); setNext(null); }
    previousSignature.current = signature;
    void load(0);
    return () => controller.current?.abort();
  }, [signature, stamp]);
  return { items, total, next, busy, error, load, retry: () => load(requestedOffset.current) };
}

type Shared = {
  expanded: string[]; toggle: (id: string) => void; stamp: string;
  select: (item: LibraryItem) => void; active?: Document;
};
function Branch({ id, name, count, children, shared }: {
  id: string; name: string; count?: number; children: ReactNode; shared: Shared;
}) {
  const open = shared.expanded.includes(id);
  return <div className="explorer-branch">
    <button className="explorer-toggle" aria-expanded={open} onClick={() => shared.toggle(id)} title={name}>
      <ChevronRight size={13} className={open ? "rotated" : ""}/>{open ? <FolderOpen size={15}/> : <Folder size={15}/>}
      <span>{name}</span>{count !== undefined && <small>{count.toLocaleString()}</small>}
    </button>
    {open && <div className="explorer-children">{children}</div>}
  </div>;
}
function Leaf({ item, shared, label, detail }: { item: LibraryItem; shared: Shared; label?: string; detail?: string }) {
  const selected = item.kind === "source" ? shared.active?.kind === "source" && shared.active.intake.sourceId === item.sourceId :
    item.kind === "artifact" ? shared.active?.kind === "artifact" && shared.active.runId === item.runId && shared.active.artifactName === item.artifactName :
    item.kind === "version" ? shared.active?.kind === "analysis" && shared.active.runId === item.runId : false;
  return <button className={`explorer-leaf ${selected ? "selected" : ""}`} title={[label ?? item.name, detail].filter(Boolean).join("\n")}
    aria-current={selected ? "page" : undefined} onClick={() => shared.select(item)}>
    {item.kind === "source" ? <Database size={14}/> : item.kind === "job" ? <Activity size={14}/> : <FileText size={14}/>}
    <span>{label ?? item.name}{detail && <small>{detail}</small>}</span>
  </button>;
}
function Entries({ input, shared, children }: {
  input: Record<string, unknown>; shared: Shared; children?: (items: LibraryItem[]) => ReactNode;
}) {
  const result = useEntries(input, shared.stamp);
  return <>
    {children ? children(result.items) : result.items.map(item => <Leaf key={`${item.kind}:${item.id}`} item={item} shared={shared}
      detail={item.kind === "source" ? `${item.seriesCount ?? "?"} series · ${item.frequency ?? "Unrecorded calendar"}` :
        item.kind === "job" ? `${item.status ?? ""} · ${date(item.createdAt)}` : item.family ?? undefined}/>)}
    {result.busy && <p className="explorer-message" role="status">Reading library…</p>}
    {result.error && <div className="explorer-message" role="alert">{result.error}<button onClick={() => void result.retry()}>Retry</button></div>}
    {!result.busy && !result.error && !result.items.length && <p className="explorer-message">Nothing saved here yet.</p>}
    {!!result.items.length && <div className="explorer-page"><small>{result.items.length.toLocaleString()} of {result.total.toLocaleString()}</small>
      {result.next !== null && <button disabled={result.busy} onClick={() => void result.load(result.next ?? 0)}>Load more</button>}</div>}
  </>;
}
const date = (value?: string) => value ? new Date(value).toLocaleString() : "";

export function fileGroups(items: LibraryItem[], prefix = "") {
  const directories = new Map<string, LibraryItem[]>(), files: LibraryItem[] = [];
  for (const item of items) {
    const relative = (item.artifactName ?? item.name).slice(prefix.length);
    const separator = relative.indexOf("/");
    if (separator < 0) files.push(item);
    else {
      const name = relative.slice(0, separator);
      directories.set(name, [...(directories.get(name) ?? []), item]);
    }
  }
  return { directories: [...directories].sort(([a], [b]) => a.localeCompare(b)), files };
}
function Files({ items, runId, shared, prefix = "" }: { items: LibraryItem[]; runId: string; shared: Shared; prefix?: string }) {
  const groups = fileGroups(items, prefix);
  return <>{groups.directories.map(([name, nested]) => <Branch key={name} id={`files:${runId}:${prefix}${name}`} name={name} shared={shared}>
    <Files items={nested} runId={runId} shared={shared} prefix={`${prefix}${name}/`}/>
  </Branch>)}{groups.files.map(item => <Leaf key={item.id} item={item} shared={shared} label={(item.artifactName ?? item.name).slice(prefix.length)}/>)}</>;
}
function Family({ item, shared }: { item: LibraryItem; shared: Shared }) {
  return <Branch id={`family:${item.id}`} name={item.name} count={item.versionCount} shared={shared}>
    <Branch id={`versions:${item.id}`} name="Versions" count={item.versionCount} shared={shared}>
      <Entries input={{ section: "versions", family: item.name }} shared={shared}>{items => items.map(version => (
        <Branch key={version.id} id={`version:${version.id}`} name={`${version.versionLabel ? `${version.versionLabel} · ` : ""}${date(version.createdAt)} · ${version.id.slice(0, 6)}`} shared={shared}>
          <Leaf item={version} shared={shared} label="Open forecast" detail={version.purpose === "judgment_scenario" ? "Analyst judgment · not backtested" : version.adopted ? "Adopted ensemble" : version.purpose === "component_refit" ? "Component refit" : "Saved model run"}/>
          <Branch id={`artifacts:${version.id}`} name="Files" shared={shared}>
            <Entries input={{ section: "files", runId: version.id }} shared={shared}>{files => <Files items={files} runId={version.id} shared={shared}/>}</Entries>
          </Branch>
        </Branch>
      ))}</Entries>
    </Branch>
    <Branch id={`notes:${item.id}`} name="Saved notes" shared={shared}><Entries input={{ section: "notes", family: item.name }} shared={shared}/></Branch>
  </Branch>;
}

function navigate(event: KeyboardEvent<HTMLElement>) {
  const current = event.target;
  if (!(current instanceof HTMLButtonElement)) return;
  const nodes = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button")].filter(node => node.offsetParent !== null && !node.disabled);
  const index = nodes.indexOf(current);
  if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
    event.preventDefault();
    nodes[event.key === "Home" ? 0 : event.key === "End" ? nodes.length - 1 : Math.max(0, Math.min(nodes.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)))]?.focus();
  } else if (event.key === "ArrowRight" && current.getAttribute("aria-expanded") === "false") {
    event.preventDefault(); current.click();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    if (current.getAttribute("aria-expanded") === "true") current.click();
    else current.closest(".explorer-children")?.parentElement?.querySelector<HTMLButtonElement>(":scope > button")?.focus();
  }
}

export function Explorer({ catalog, expanded, setExpanded, active, select, closed, reopen }: {
  catalog: Catalog; expanded: string[]; setExpanded: (ids: string[]) => void;
  active?: Document; select: (item: LibraryItem) => void; closed: Document[]; reopen: (id: string) => void;
}) {
  const [search, setSearch] = useState(""), [query, setQuery] = useState("");
  useEffect(() => { const timer = setTimeout(() => setQuery(search.trim()), 250); return () => clearTimeout(timer); }, [search]);
  const shared: Shared = { expanded, active, select,
    toggle: id => setExpanded(expanded.includes(id) ? expanded.filter(value => value !== id) : [...expanded, id]),
    stamp: `${catalog.totalRuns}:${catalog.totalSources}:${catalog.totalNotes}:${catalog.totalJobs}`,
  };
  const live = { ...shared, stamp: `${shared.stamp}:${catalog.jobs.map(job => `${job.id}:${job.status}`).join("|")}` };
  return <>
    <div className="explorer-tools"><label className="search"><Search size={15}/><input aria-label="Search library" placeholder="Find forecasts, data, files…" value={search} maxLength={200} onChange={event => setSearch(event.target.value)}/></label>
      <button className="icon" aria-label={search ? "Clear library search" : "Collapse explorer folders"} onClick={() => search ? setSearch("") : setExpanded([])}>{search ? <X size={15}/> : <ChevronsDownUp size={15}/>}</button>
    </div>
    <nav className="library-tree" aria-label="Forecasts, data and files" onKeyDown={navigate}>
      {query ? <><p className="explorer-message">Results across this entire library</p><Entries input={{ section: "search", query }} shared={live}/></> : <>
        <Branch id="forecasts" name="Forecasts" count={catalog.totalForecasts} shared={shared}>
          <Entries input={{ section: "forecasts" }} shared={shared}>{items => items.map(item => <Family key={item.id} item={item} shared={shared}/>)}</Entries>
        </Branch>
        <Branch id="sources" name="Data" count={catalog.totalSources} shared={shared}><Entries input={{ section: "sources" }} shared={shared}/></Branch>
        <Branch id="analyses" name="Saved notes" count={catalog.totalNotes} shared={shared}><Entries input={{ section: "notes" }} shared={shared}/></Branch>
        <Branch id="activity" name="Activity" count={catalog.totalJobs} shared={shared}><Entries input={{ section: "jobs" }} shared={live}/></Branch>
        {!!closed.length && <Branch id="closed" name="Recently closed" count={closed.length} shared={shared}>
          {closed.slice().reverse().map(doc => <button key={doc.id} className="explorer-leaf" aria-label={`Reopen draft: ${doc.title}`} onClick={() => reopen(doc.id)}>
            <History size={14}/><span>{doc.title}<small>{doc.view} · recoverable draft</small></span>
          </button>)}
        </Branch>}
      </>}
    </nav>
    <p className="library-location" title={catalog.workspace}><Database size={12}/> SQLite library · this workspace</p>
  </>;
}
