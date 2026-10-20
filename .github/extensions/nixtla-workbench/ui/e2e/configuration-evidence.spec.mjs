import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_CONFIGURATION_RUN;
const legacy = process.env.NIXTLA_E2E_LEGACY_RUN;
const hash = data => createHash("sha256").update(data).digest("hex");
let service, server, workspace;

test.beforeEach(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set a disposable NIXTLA_E2E_OUTPUT directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "configuration-"));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  await selectRun(source, "Configuration study control");
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterEach(async () => { await server?.close(); await service?.close(); });

async function selectRun(path, title) {
  const run = await service.call("import_run", { path: resolve(path), name: title });
  const view = await service.call("view", { runId: run.id });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "configuration", runId: run.id, title, series: view.series[0],
      view: path === legacy ? "Diagnostics" : "Models", analysis: { section: "Configuration" } }],
    activeId: "configuration", theme: "light",
  } });
}

test("configuration evidence is transparent, recoverable, unscaled and read-only", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_CONFIGURATION_RUN to a saved configuration study run.");
  const original = hash(await readFile(join(source, "manifest.json")));
  const errors = [], jobs = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api") && request.postDataJSON()?.operation === "start_job") jobs.push(request.url());
  });
  await page.goto(server.url);
  await page.getByRole("button", { name: "Review configuration", exact: true }).click();
  const panel = page.getByRole("region", { name: "Forecast configuration", exact: true });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("What the data says");
  await expect(panel).toContainText("Calendar priors and complete cycles alone do not prove seasonality");
  await panel.getByLabel("Configuration model").selectOption("AutoMFLES");
  await expect(panel).toContainText("reconstructed from native grid and observed scores");
  await expect(panel.getByRole("region", { name: "Selected native search configuration", exact: true })).toBeVisible();
  await panel.getByRole("heading", { name: "Forecast configuration", exact: true }).scrollIntoViewIfNeeded();
  if (process.env.NIXTLA_E2E_CAPTURE !== "0") await page.screenshot({ path: join(workspace, "configuration-desktop.png") });
  await panel.getByLabel("Configuration model").scrollIntoViewIfNeeded();
  if (process.env.NIXTLA_E2E_CAPTURE !== "0") await page.screenshot({ path: join(workspace, "configuration-model-desktop.png") });
  await panel.getByLabel("Configuration model").selectOption("Ridge");
  await expect(panel).toContainText("Not reported");
  await expect(panel.getByRole("region", { name: "Selected native search configuration", exact: true })).toHaveCount(0);
  await panel.getByLabel("Configuration model").selectOption("AutoMFLES");
  const details = panel.getByText("Complete configuration evidence and exports", { exact: true });
  await details.focus();
  await page.keyboard.press("Enter");
  const diagnostic = panel.getByRole("region", { name: "Series configuration diagnostics", exact: true });
  await expect(diagnostic).toBeVisible();
  const before = await diagnostic.locator("tbody").textContent();
  await page.getByLabel("Display units", { exact: true }).selectOption("millions");
  await expect(diagnostic.locator("tbody")).toHaveText(before);
  await details.click();
  await page.route("**/api", async route => {
    const body = route.request().postDataJSON();
    if (body?.operation === "evidence" && body?.input?.name?.includes("configuration_")) {
      await route.fulfill({ status: 500, contentType: "application/json",
        body: JSON.stringify({ error: { code: "fixture", message: "Configuration evidence unavailable" } }) });
    } else await route.continue();
  });
  await panel.getByRole("button", { name: "Reload configuration evidence" }).click();
  await expect(panel.getByRole("alert")).toContainText("Configuration evidence unavailable");
  await page.unroute("**/api");
  await panel.getByRole("button", { name: "Reload configuration evidence" }).click();
  await expect(panel.getByRole("alert")).toHaveCount(0);
  await expect(panel).toContainText("Inside the model");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await panel.getByRole("heading", { name: "Forecast configuration", exact: true }).scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  if (process.env.NIXTLA_E2E_CAPTURE !== "0") await page.screenshot({ path: join(workspace, "configuration-mobile-dark.png") });
  await panel.getByLabel("Configuration model").scrollIntoViewIfNeeded();
  if (process.env.NIXTLA_E2E_CAPTURE !== "0") await page.screenshot({ path: join(workspace, "configuration-model-mobile-dark.png") });
  if (legacy) {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto("about:blank");
    await selectRun(legacy, "Legacy saved forecast");
    await page.goto(server.url);
    await expect(panel).toContainText("This saved version predates configuration evidence");
    await expect(panel).toContainText("Opening this page does not reconstruct evidence");
  }
  expect(jobs).toEqual([]);
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "manifest.json")))).toBe(original);
  console.log(`Configuration evidence acceptance workspace: ${workspace}`);
});
