import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_INTERVAL_RUN;
const legacy = process.env.NIXTLA_E2E_LEGACY_RUN;
const legacyScores = process.env.NIXTLA_E2E_LEGACY_SCORE_RUN;
const hash = data => createHash("sha256").update(data).digest("hex");
let service, server, workspace, original;

test.beforeEach(async ({}, testInfo) => {
  service = server = undefined;
  const selectedSource = testInfo.title.startsWith("legacy interval") ? legacyScores : source;
  if (!selectedSource) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set a disposable NIXTLA_E2E_OUTPUT directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "interval-"));
  original = hash(await readFile(join(selectedSource, "forecast.csv")));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  await selectRun(selectedSource, "Interval fixture");
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterEach(async () => { await server?.close(); await service?.close(); });

async function selectRun(path, title) {
  const run = await service.call("import_run", { path: resolve(path), name: title });
  const view = await service.call("view", { runId: run.id });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "interval", runId: run.id, title, series: view.series[0], view: "Forecast" }],
    activeId: "interval", theme: "light",
  } });
}

test("saved intervals are readable, recoverable and never start a forecast", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_INTERVAL_RUN to a saved experimental run.");
  const errors = [], jobs = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api") && request.postDataJSON()?.operation === "start_job") jobs.push(request.url());
  });
  await page.goto(server.url);
  const panel = page.getByRole("region", { name: "Experimental interval evidence", exact: true });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("split_conformal");
  await expect(panel).toContainText("Current calibration support");
  await expect(panel).toContainText("Matched interval comparison");
  await expect(panel).toContainText("do not certify month totals");
  if (legacyScores) await expect(panel).not.toContainText("Legacy overall and horizon Winkler scores are hidden");
  await panel.screenshot({ path: join(workspace, "interval-desktop.png") });
  const details = panel.getByText("Coverage and scores by horizon and origin", { exact: true });
  await details.focus();
  await page.keyboard.press("Enter");
  await expect(panel).toContainText("Saved interval evaluation");
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await expect(panel).toBeVisible();
  await expect(panel.getByRole("button", { name: "Reload interval evidence" })).toBeEnabled();
  await page.route("**/api", async route => {
    const body = route.request().postDataJSON();
    if (body?.operation === "evidence" && body?.input?.name?.includes("interval_")) {
      await route.fulfill({ status: 500, contentType: "application/json",
        body: JSON.stringify({ error: { code: "fixture", message: "Interval evidence unavailable" } }) });
    } else await route.continue();
  });
  await panel.getByRole("button", { name: "Reload interval evidence" }).click();
  await expect(panel.getByRole("alert")).toContainText("Interval evidence unavailable");
  await page.unroute("**/api");
  await panel.getByRole("button", { name: "Reload interval evidence" }).click();
  await expect(panel.getByRole("alert")).toHaveCount(0);
  await expect(panel).toContainText("Current calibration support");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await panel.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.screenshot({ path: join(workspace, "interval-mobile-dark.png") });
  if (legacy) {
    await selectRun(legacy, "Native interval fixture");
    await page.reload();
    await expect(page.getByRole("region", { name: "Model browser", exact: true })).toBeVisible();
    await expect(panel).toHaveCount(0);
  }
  expect(jobs).toEqual([]);
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "forecast.csv")))).toBe(original);
  console.log(`Interval evidence screenshots: ${workspace}`);
});

test("legacy interval aggregates are masked without changing archived evidence", async ({ page }) => {
  test.skip(!legacyScores, "Set NIXTLA_E2E_LEGACY_SCORE_RUN to an unversioned interval run.");
  await page.goto(server.url);
  const panel = page.getByRole("region", { name: "Experimental interval evidence", exact: true });
  await expect(panel).toContainText("Legacy overall and horizon Winkler scores are hidden");
  const summary = panel.getByRole("region", { name: "Matched interval comparison (bounded preview)", exact: true });
  await expect(summary).toBeVisible();
  const header = await summary.locator("thead th").allTextContents();
  const scoreColumn = header.indexOf("mean interval score");
  expect(scoreColumn).toBeGreaterThanOrEqual(0);
  const values = await summary.locator("tbody tr").evaluateAll((rows, column) => rows.map(row => row.children[column].textContent), scoreColumn);
  expect(values.length).toBeGreaterThan(0);
  expect(values.every(value => value === "\u2014")).toBe(true);
  await panel.getByText("Coverage and scores by horizon and origin", { exact: true }).click();
  await expect(panel.getByRole("region", { name: "Saved interval evaluation", exact: true })).toBeVisible();
  await expect(panel.getByText("Legacy overall and horizon Winkler scores are hidden", { exact: false })).toHaveCount(2);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(legacyScores, "forecast.csv")))).toBe(original);
});
