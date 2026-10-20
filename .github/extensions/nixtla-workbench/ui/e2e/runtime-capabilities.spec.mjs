import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_MODEL_RUN;
let service, server, workspace, original;
const hash = value => createHash("sha256").update(value).digest("hex");
test.beforeAll(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to a disposable artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "runtime-"));
  original = hash(await readFile(join(source, "forecast.csv")));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  const run = await service.call("import_run", { path: resolve(source), name: "Runtime inspection fixture" });
  const view = await service.call("view", { runId: run.id });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "runtime", runId: run.id, title: "Runtime inspection", series: view.series[0], view: "Forecast" }],
    activeId: "runtime", theme: "light",
  } });
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterAll(async () => { await server?.close(); await service?.close(); });

test("runtime inspection stays separate, handles errors and never starts a fit", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_MODEL_RUN to a saved run. No fits are performed.");
  const errors = [], jobs = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    const body = request.url().endsWith("/api") ? request.postDataJSON() : null;
    if (body?.operation === "start_job") jobs.push(body);
  });
  await page.goto(server.url);
  const browser = page.getByRole("region", { name: "Model browser", exact: true });
  await expect(browser).toBeVisible();
  await expect(browser.getByRole("button", { name: "Show weighted ensemble", exact: true })).toBeVisible();
  await expect(browser.locator(".model-roster")).not.toContainText("Loading evidence");
  const savedModels = await browser.locator(".model-roster-row").count();
  const panel = browser.locator(".runtime-capabilities");
  await panel.locator(":scope > summary").focus();
  await page.keyboard.press("Enter");
  await expect(panel).toContainText("53 registered recipes");
  await expect(panel).toContainText("Not a count of successful fits");
  await expect(panel.getByRole("button", { name: "Refresh runtime status" })).toBeEnabled();
  expect(await browser.locator(".model-roster-row").count()).toBe(savedModels);
  await panel.screenshot({ path: join(workspace, "runtime-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await panel.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.screenshot({ path: join(workspace, "runtime-mobile-dark.png") });
  await page.route("**/api", async route => {
    if (route.request().postDataJSON()?.operation === "runtime_capabilities") {
      await route.fulfill({ status: 500, contentType: "application/json",
        body: JSON.stringify({ error: { code: "probe", message: "Runtime temporarily unavailable" } }) });
    } else await route.continue();
  });
  await panel.getByRole("button", { name: "Refresh runtime status" }).click();
  await expect(panel.getByRole("alert")).toContainText("previous result is stale");
  await expect(panel.getByRole("button", { name: "Refresh runtime status" })).toBeEnabled();
  await page.unroute("**/api");
  await panel.getByRole("button", { name: "Refresh runtime status" }).click();
  await expect(panel.getByRole("alert")).toHaveCount(0);
  await expect(panel.getByRole("button", { name: "Refresh runtime status" })).toBeEnabled();
  expect(jobs).toEqual([]);
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "forecast.csv")))).toBe(original);
  console.log(`Runtime screenshots: ${workspace}`);
});
