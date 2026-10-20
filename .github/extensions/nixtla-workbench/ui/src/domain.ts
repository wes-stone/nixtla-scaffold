import type { Document, DocumentPatch, WorkspaceState } from "./api";

export function openDocument(state: WorkspaceState, document: Document, newTab = false): WorkspaceState {
  if (newTab && state.documents.length >= 30) throw new Error("Close a document before opening more than 30 tabs.");
  const active = state.documents.find(item => item.id === state.activeId);
  if (!newTab && active) {
    if (active.kind === document.kind && active.runId === document.runId && active.runId && active.artifactName === document.artifactName) {
      return updateDocument(state, active.id, { view: document.view });
    }
    if (state.closedDocuments.length >= 200) throw new Error("The recoverable document limit is reached; reopen an existing document.");
    const closedDocuments = [...state.closedDocuments.filter(doc => doc.id !== active.id), active];
    return { ...state, closedDocuments, activeId: document.id, documents: state.documents.map(item => item.id === active.id ? document : item) };
  }
  return { ...state, documents: [...state.documents, document], activeId: document.id };
}

export function closeDocument(state: WorkspaceState, id: string): WorkspaceState {
  const closed = state.documents.find(doc => doc.id === id);
  if (!closed) return state;
  if (state.closedDocuments.length >= 200) throw new Error("The recoverable document limit is reached; reopen an existing document.");
  const documents = state.documents.filter(item => item.id !== id);
  return { ...state, documents, closedDocuments: [...state.closedDocuments, closed], activeId: state.activeId === id ? documents.at(-1)?.id ?? "" : state.activeId };
}

export function reopenDocument(state: WorkspaceState, id: string): WorkspaceState {
  const doc = state.closedDocuments.find(item => item.id === id);
  if (!doc) throw new Error("This closed document no longer exists.");
  return openDocument({ ...state, closedDocuments: state.closedDocuments.filter(item => item.id !== id) }, doc, true);
}

export function updateDocument(state: WorkspaceState, id: string, patch: DocumentPatch): WorkspaceState {
  return { ...state, documents: state.documents.map(item => item.id === id ? {
    ...item, ...patch, intake: { ...item.intake, ...patch.intake },
    maintenance: { ...item.maintenance, ...patch.maintenance }, pipeline: { ...item.pipeline, ...patch.pipeline },
    recipe: { ...item.recipe, ...patch.recipe },
    analysis: { ...item.analysis, ...patch.analysis },
  } : item) };
}
