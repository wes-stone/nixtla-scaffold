import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_ANOMALY_RUN;
const digest = data => createHash("sha256").update(data).digest("hex");
let service, server, workspace;

test.beforeEach(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set a disposable NIXTLA_E2E_OUTPUT directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "anomalies-"));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  const run = await service.call("import_run", { path: resolve(source), name: "Anomaly acceptance" });
  const view = await service.call("view", { runId: run.id });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "anomalies", runId: run.id, title: "Anomaly acceptance", series: view.series[0],
      view: "Diagnostics", analysis: { section: "Configuration" } }], activeId: "anomalies", theme: "light",
  } });
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterEach(async () => { await server?.close(); await service?.close(); });

test("saved anomaly decisions are readable, recoverable and never start a fit", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_ANOMALY_RUN to a saved anomaly run.");
  const original = digest(await readFile(join(source, "manifest.json")));
  const errors = [], jobs = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api") && request.postDataJSON()?.operation === "start_job") jobs.push(request.url());
  });
  await page.goto(server.url);
  const panel = page.getByRole("region", { name: "Calendar and anomaly review", exact: true });
  await expect(panel.getByRole("region", { name: "Training review at the forecast cutoff", exact: true })).toBeVisible();
  await expect(panel).toContainText("Raw actuals are preserved");
  await expect(panel.locator("canvas").first()).toBeVisible();
  await panel.getByRole("heading", { name: "Calendar and anomaly review" }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "anomalies-desktop.png") });
  const decisions = panel.getByRole("region", { name: "Flagged observations and decisions", exact: true });
  await decisions.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "anomalies-decisions-desktop.png") });
  const disclosure = panel.getByText("Complete anomaly evidence and exports", { exact: true });
  await disclosure.focus();
  await page.keyboard.press("Enter");
  const receipts = panel.getByRole("region", { name: "Origin-local anomaly receipts", exact: true });
  await expect(receipts).toBeVisible();
  const before = await receipts.locator("tbody").textContent();
  await page.getByLabel("Display units", { exact: true }).selectOption("millions");
  await expect(receipts.locator("tbody")).toHaveText(before);
  await disclosure.click();
  await page.route("**/api", async route => {
    const body = route.request().postDataJSON();
    if (body?.operation === "evidence" && body?.input?.name?.includes("anomaly_")) {
      await route.fulfill({ status: 500, contentType: "application/json",
        body: JSON.stringify({ error: { code: "fixture", message: "Anomaly evidence unavailable" } }) });
    } else await route.continue();
  });
  await page.getByRole("button", { name: "Reload configuration evidence" }).click();
  await expect(panel.getByRole("alert")).toContainText("Anomaly evidence unavailable");
  await page.unroute("**/api");
  await page.getByRole("button", { name: "Reload configuration evidence" }).click();
  await expect(panel.getByRole("alert")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await expect(panel.getByRole("region", { name: "Training review at the forecast cutoff", exact: true })).toBeVisible();
  await expect(panel.locator("canvas").first()).toBeVisible();
  await expect(panel.getByText("Reading saved analytical evidence...", { exact: true })).toHaveCount(0);
  await panel.getByRole("heading", { name: "Calendar and anomaly review" }).scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: join(workspace, "anomalies-mobile-dark.png") });
  await decisions.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "anomalies-decisions-mobile-dark.png") });
  expect(jobs).toEqual([]);
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(digest(await readFile(join(source, "manifest.json")))).toBe(original);
  console.log(`Anomaly evidence acceptance workspace: ${workspace}`);
});
