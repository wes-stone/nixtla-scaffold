import { describe, expect, it } from "vitest";
import { fileGroups, libraryDocument } from "./Explorer";
import { documentSchema, stateSchema } from "./api";
import type { LibraryItem } from "./api";
import { openDocument } from "./domain";

describe("persistent library navigation", () => {
  it("opens saved data read-only, not as a model execution form", () => {
    const doc = libraryDocument({ kind: "source", id: "source", sourceId: "source", name: "Monthly data" }, "tab");
    expect(doc).toMatchObject({ kind: "source", view: "Data", intake: { sourceId: "source" } });
    expect(doc.runId).toBeUndefined();
  });
  it("keeps file documents separate from a forecast in the same version", () => {
    let state = openDocument(stateSchema.parse({}), documentSchema.parse({ id: "forecast", runId: "run" }), true);
    const file = libraryDocument({ kind: "artifact", id: "file", name: "forecast.csv", artifactName: "forecast.csv", runId: "run" }, "file");
    state = openDocument(state, file);
    expect(state.documents[0].kind).toBe("artifact");
    expect(state.closedDocuments[0].id).toBe("forecast");
    state = openDocument(state, documentSchema.parse({ id: "back", runId: "run" }));
    expect(state.documents[0].kind).toBe("analysis");
  });
  it("groups immutable artifact paths without flattening duplicate names", () => {
    const items: LibraryItem[] = ["forecast.csv", "audit/metrics.csv", "output/metrics.csv"].map(name => ({ kind: "artifact", id: name, name, artifactName: name }));
    const tree = fileGroups(items);
    expect(tree.files.map(item => item.name)).toEqual(["forecast.csv"]);
    expect(tree.directories.map(([name]) => name)).toEqual(["audit", "output"]);
    expect(fileGroups(tree.directories[0][1], "audit/").files[0].name).toBe("audit/metrics.csv");
  });
  it("restores folder expansion through the saved workspace schema", () => {
    const expanded = ["forecasts", "family:Revenue", "versions:Revenue", "version:one", "artifacts:one"];
    expect(stateSchema.parse(JSON.parse(JSON.stringify({ explorerExpanded: expanded }))).explorerExpanded).toEqual(expanded);
  });
});
