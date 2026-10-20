import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_NATIVE_FITTED_RUN;
const legacy = process.env.NIXTLA_E2E_NATIVE_FITTED_LEGACY;
const hash = data => createHash("sha256").update(data).digest("hex");
let service, server, workspace, saved;

async function selectRun(path, title) {
  const run = await service.call("import_run", { path: resolve(path), name: title });
  const view = await service.call("view", { runId: run.id });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "native-fitted", runId: run.id, title, series: view.series[0],
      view: "Diagnostics", analysis: { section: "Configuration", previewModel: "AutoMFLES", scale: "millions" } }],
    activeId: "native-fitted", theme: "light",
  } });
  return run;
}

test.beforeEach(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set a disposable NIXTLA_E2E_OUTPUT directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "native-fitted-"));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  saved = await selectRun(source, "Native fitted acceptance");
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterEach(async () => { await server?.close(); await service?.close(); });

test("native fitted flags, exclusions and exports remain inspectable without model jobs", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_NATIVE_FITTED_RUN to a saved fixed-recipe acceptance run.");
  const original = hash(await readFile(join(source, "manifest.json")));
  const errors = [], jobs = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api") && request.postDataJSON()?.operation === "start_job") jobs.push(request.url());
  });
  await page.goto(server.url);
  const configurationPanel = page.getByRole("region", { name: "Forecast configuration", exact: true });
  const panel = page.getByRole("region", { name: "Native fitted-model anomalies", exact: true });
  const scope = panel.getByRole("region", { name: "Native fitted diagnostic scope", exact: true });
  await expect(scope).toBeVisible();
  await expect(configurationPanel).toContainText("Excluded from selection under the saved policy");
  await expect(panel).toContainText("not calibrated future intervals");
  await expect(panel.locator("canvas").first()).toBeVisible();
  await expect(panel.getByRole("region", { name: "Native fitted anomaly flags", exact: true })).toBeVisible();
  const receipt = await service.call("evidence", { runId: saved.id, name: "audit/native_fitted_points.csv",
    series: "series_1", models: ["AutoMFLES"], limit: 2000 });
  expect(receipt.rows).toHaveLength(182);
  expect(receipt.rows.filter(row => row.flagged)).toHaveLength(25);
  await panel.getByRole("heading", { name: "Native fitted-model anomalies", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "native-fitted-desktop.png") });
  const before = await scope.locator("tbody").textContent();
  await page.getByLabel("Display units", { exact: true }).selectOption("raw");
  await expect(scope.locator("tbody")).toHaveText(before);
  await page.getByLabel("Display units", { exact: true }).selectOption("millions");
  const disclosure = panel.getByText("Native fitted evidence and exports", { exact: true });
  await disclosure.focus();
  await page.keyboard.press("Enter");
  await expect(panel.getByRole("region", { name: "Native fitted values", exact: true })).toBeVisible();
  await panel.getByRole("button", { name: "Export full scope", exact: true }).first().click();
  await expect(panel).toContainText("Saved 182 full-precision rows");
  await disclosure.click();
  await page.getByLabel("Configuration model").selectOption("Ridge");
  await expect(panel).toContainText("This model adapter did not record native fitted intervals");
  await page.getByLabel("Configuration model").selectOption("SeasonalWindowAverage");
  await expect(panel).toContainText("does not accept interval levels");
  await page.getByLabel("Configuration model").selectOption("AutoMFLES");
  await expect(scope).toBeVisible();
  await page.route("**/api", async route => {
    const body = route.request().postDataJSON();
    if (body?.operation === "evidence" && body?.input?.name?.includes("native_fitted_")) {
      await route.fulfill({ status: 500, contentType: "application/json",
        body: JSON.stringify({ error: { code: "fixture", message: "Native fitted evidence unavailable" } }) });
    } else await route.continue();
  });
  await page.getByRole("button", { name: "Reload configuration evidence" }).click();
  await expect(panel.getByRole("alert")).toContainText("Native fitted evidence unavailable");
  await page.unroute("**/api");
  await page.getByRole("button", { name: "Reload configuration evidence" }).click();
  await expect(scope).toBeVisible();
  await expect(panel.getByRole("alert")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await expect(panel.locator("canvas").first()).toBeVisible();
  await panel.getByRole("heading", { name: "Native fitted-model anomalies", exact: true }).scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: join(workspace, "native-fitted-mobile-dark.png") });
  await panel.locator("canvas").first().screenshot({ path: join(workspace, "native-fitted-mobile-plot.png") });
  if (legacy) {
    await page.goto("about:blank");
    await selectRun(legacy, "Legacy native evidence");
    await page.goto(server.url);
    await expect(panel).toContainText("Native fitted evidence was not saved for this version");
  }
  expect(jobs).toEqual([]);
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "manifest.json")))).toBe(original);
  console.log(`Native fitted evidence acceptance workspace: ${workspace}`);
});
