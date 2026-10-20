import { test, expect } from "@playwright/test";
import { startServer } from "../../server.mjs";

const draft = (id, title) => ({ id, title, view: "Data", intake: { path: "" } });
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

let server, saved, revision, writes, operations, catalogDelay, saveDelay;
test.beforeEach(async () => {
  revision = 1;
  writes = [];
  operations = [];
  catalogDelay = undefined;
  saveDelay = undefined;
  saved = { documents: [draft("initial", "Initial analysis")], closedDocuments: [], activeId: "initial", theme: "light", explorerExpanded: [] };
  server = await startServer({
    workspace: process.cwd(),
    handoff: async () => {},
    service: {
      async call(operation, input = {}) {
        operations.push(operation);
        if (operation === "catalog") {
          const result = { initialized: true, runs: [], sources: [], jobs: [], state: structuredClone(saved), revision };
          const delay = catalogDelay;
          catalogDelay = undefined;
          if (delay) await delay.promise;
          return result;
        }
        if (operation === "library") return { items: [], total: 0, offset: 0, nextOffset: null };
        if (operation === "save_state") {
          const delay = saveDelay;
          saveDelay = undefined;
          if (delay) await delay.promise;
          if (input.expectedRevision !== revision) {
            const error = new Error("Workspace changed in another view. Reload saved state before editing.");
            error.code = "revision_conflict";
            throw error;
          }
          saved = structuredClone(input.state);
          writes.push(structuredClone(input));
          return { revision: ++revision };
        }
        throw new Error(`Unexpected operation: ${operation}`);
      },
    },
  });
});
test.afterEach(async () => {
  catalogDelay?.resolve();
  saveDelay?.resolve();
  await server?.close();
  expect(operations).not.toContain("start_job");
});

async function open(page) {
  await page.goto(server.url);
  await expect(page.getByRole("heading", { name: "Initial analysis", exact: true })).toBeVisible();
}

function agentUpdate() {
  saved.documents.push(draft("agent-june", "Agent forecast through June"));
  saved.activeId = "agent-june";
  revision++;
}

test("refresh adopts agent-saved documents and their revision instead of freezing the next edit", async ({ page }) => {
  await open(page);
  agentUpdate();
  await page.getByRole("button", { name: /Refresh saved/ }).click();
  await expect(page.getByRole("heading", { name: "Agent forecast through June", exact: true })).toBeVisible({ timeout: 3000 });
  await page.getByLabel("Appearance").selectOption("dark");
  await expect(page.locator(".save-state")).toHaveText("Saved locally");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(saved.documents).toHaveLength(2);
  expect(writes.at(-1).expectedRevision).toBe(2);
});

test("an idle panel receives agent updates on focus and visibility restoration", async ({ page }) => {
  await open(page);
  agentUpdate();
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(page.getByRole("heading", { name: "Agent forecast through June", exact: true })).toBeVisible({ timeout: 3000 });
  saved.documents[1].title = "Updated from another panel";
  revision++;
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
  await expect(page.getByRole("heading", { name: "Updated from another panel", exact: true })).toBeVisible({ timeout: 3000 });
  expect(writes).toHaveLength(0);
});

test("visible idle polling synchronizes without starting models or saving a clean workspace", async ({ page }) => {
  await page.clock.install();
  await open(page);
  agentUpdate();
  await page.clock.runFor(10050);
  await expect(page.getByRole("heading", { name: "Agent forecast through June", exact: true })).toBeVisible({ timeout: 3000 });
  expect(writes).toHaveLength(0);
});

test("refresh never replaces an unsaved draft during an in-flight conflicting save", async ({ page }) => {
  await open(page);
  const pending = deferred();
  saveDelay = pending;
  await page.getByLabel("Local CSV or saved-value workbook").fill("C:\\draft\\unsaved.csv");
  await expect.poll(() => operations.filter(operation => operation === "save_state").length).toBe(1);
  agentUpdate();
  await page.getByRole("button", { name: /Refresh saved/ }).click();
  await expect(page.getByLabel("Local CSV or saved-value workbook")).toHaveValue("C:\\draft\\unsaved.csv");
  await expect(page.getByRole("heading", { name: "Initial analysis", exact: true })).toBeVisible();
  pending.resolve();
  await expect(page.locator(".save-state")).toHaveText("Unsaved changes");
  await expect(page.getByRole("alert")).toContainText("Workspace changed");
  await page.getByRole("button", { name: /Refresh saved/ }).click();
  await expect(page.getByLabel("Local CSV or saved-value workbook")).toHaveValue("C:\\draft\\unsaved.csv");
  await expect(page.getByRole("alert")).toContainText("Workspace changed");
  await expect(page.getByRole("button", { name: "Reload saved workspace", exact: true })).toBeVisible();
  expect(saved.documents).toHaveLength(2);
  expect(saved.documents[0].intake.path).toBe("");
  expect(writes).toHaveLength(0);
});

test("a delayed catalog response cannot roll back an acknowledged save", async ({ page }) => {
  await open(page);
  const pending = deferred();
  catalogDelay = pending;
  const before = operations.filter(operation => operation === "catalog").length;
  await page.getByRole("button", { name: /Refresh saved/ }).click();
  await expect.poll(() => operations.filter(operation => operation === "catalog").length).toBeGreaterThan(before);
  await page.getByLabel("Local CSV or saved-value workbook").fill("C:\\draft\\saved.csv");
  await expect(page.locator(".save-state")).toHaveText("Saved locally");
  expect(revision).toBe(2);
  pending.resolve();
  await expect(page.getByLabel("Local CSV or saved-value workbook")).toHaveValue("C:\\draft\\saved.csv");
  await page.getByLabel("Appearance").selectOption("dark");
  await expect(page.locator(".save-state")).toHaveText("Saved locally");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(writes.at(-1).expectedRevision).toBe(2);
});

test("explicit conflict recovery reloads the saved workspace and restores editing", async ({ page }) => {
  await open(page);
  agentUpdate();
  await page.getByLabel("Local CSV or saved-value workbook").fill("C:\\draft\\local.csv");
  await expect(page.locator(".save-state")).toHaveText("Unsaved changes");
  await page.getByRole("button", { name: "Reload saved workspace", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Agent forecast through June", exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.getByLabel("Appearance").selectOption("dark");
  await expect(page.locator(".save-state")).toHaveText("Saved locally");
  expect(saved.documents).toHaveLength(2);
});
