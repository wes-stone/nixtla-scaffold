import { test, expect as baseExpect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_RANKING_RUN;
const expect = baseExpect.configure({ timeout: 120000 });
let service, server, workspace, run, audit, original;
const hash = value => createHash("sha256").update(value).digest("hex");
test.beforeAll(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to a disposable artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "ranking-"));
  original = hash(await readFile(join(source, "forecast.csv")));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  run = await service.call("import_run", { path: resolve(source), name: "Q2 saved Huber tournament" });
  audit = await service.call("evidence", { runId: run.id, name: "appendix/model_audit.csv", series: "Total", limit: 2000 });
  expect(audit.rows.find(row => row.model === "Huber")).toMatchObject({ selected_model: "Huber", selection_eligible: true });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "ranking", runId: run.id, title: "Q2 tournament", series: "Total", view: "Forecast",
      analysis: { previewModel: "AutoARIMA_MSTLFeatures", models: ["AutoARIMA_MSTLFeatures", "Huber"], modelMode: "compare" } }],
    activeId: "ranking", theme: "light",
  } });
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterAll(async () => { await server?.close(); await service?.close(); });

test("Q2 winner, accuracy ordering and paired weighted ensemble remain visible without changing forecasts", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_RANKING_RUN to the saved Q2 all-model tournament. No fits are performed.");
  test.setTimeout(420000);
  const errors = [], mutations = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    const body = request.url().endsWith("/api") ? request.postDataJSON() : null;
    if (body && ["start_job", "adopt_blend", "import_run"].includes(body.operation)) mutations.push(body);
  });
  await page.goto(server.url);
  const browser = page.getByRole("region", { name: "Model browser", exact: true });
  const rows = browser.locator(".model-roster-row");
  await expect(browser.getByLabel("Saved tournament winner")).toContainText("Winner: Huber", { timeout: 120000 });
  await expect(browser.locator(".focused-model strong")).toHaveText("AutoARIMA_MSTLFeatures");
  await expect(rows.first()).toContainText("#1");
  await expect(rows.first().getByRole("button", { name: "Focus Huber", exact: true })).toBeVisible();
  await expect(rows.first()).toContainText("431.64K");
  const eligible = audit.rows.filter(row => row.selection_eligible === true && row.family !== "ensemble")
    .sort((a, b) => a.rmse - b.rmse || a.mae - b.mae || Math.abs(a.bias) - Math.abs(b.bias) || a.model.localeCompare(b.model));
  expect((await rows.locator(".model-focus").evaluateAll(nodes => nodes.map(node => node.getAttribute("aria-label").slice(6)))).slice(0, eligible.length))
    .toEqual(eligible.map(row => row.model));
  await browser.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "winner-desktop.png") });
  await browser.getByLabel("Accuracy metric", { exact: true }).selectOption("bias");
  await expect(rows.first()).toContainText("RandomWalkWithDrift");
  await expect(browser.getByLabel("Saved tournament winner")).toContainText("Huber");
  await browser.getByLabel("Accuracy metric", { exact: true }).selectOption("rmse");
  await browser.getByLabel("Sort models", { exact: true }).selectOption("name");
  await expect(rows.first()).toContainText("AutoARIMA");
  await browser.getByLabel("Sort models", { exact: true }).selectOption("accuracy");
  await browser.getByLabel("Find a model", { exact: true }).fill("weighted");
  await expect(rows).toHaveCount(1);
  await expect(rows.locator(".model-rank")).toHaveCount(0);
  await expect(rows).toContainText("Separate blend cohort");
  await browser.getByRole("button", { name: "Show weighted ensemble", exact: true }).click();
  await expect(browser.locator(".focused-model strong")).toHaveText("WeightedEnsemble");
  await expect(page.locator(".model-stage .chart-legend")).toContainText("WeightedEnsemble");
  await browser.getByLabel("Find a model", { exact: true }).fill("");
  await page.setViewportSize({ width: 620, height: 900 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await browser.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "winner-narrow-dark.png") });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.setViewportSize({ width: 390, height: 844 });
  await browser.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "winner-mobile.png") });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await browser.getByRole("button", { name: "Weights & paired backtest", exact: true }).click();
  const ensemble = page.getByRole("region", { name: "Weighted ensemble evidence", exact: true });
  await expect(ensemble).toContainText("46.0% higher RMSE than Huber");
  await expect(ensemble).toContainText("243 matched predictions");
  await expect(ensemble).toContainText("81 warm-up rows");
  await expect(ensemble).toContainText("444,816.50");
  await expect(ensemble).toContainText("649,227.93");
  await ensemble.getByText("Saved component weights", { exact: true }).click();
  await expect(ensemble.getByRole("region", { name: "Weighted ensemble components", exact: true })).toContainText("0.0395");
  await ensemble.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "weighted-ensemble-desktop.png") });
  await ensemble.getByRole("button", { name: "View saved blend forecast", exact: true }).click();
  await expect(browser.locator(".focused-model strong")).toHaveText("WeightedEnsemble");
  await browser.getByRole("button", { name: "Show winner", exact: true }).click();
  await expect(browser.locator(".focused-model strong")).toHaveText("Huber");
  await page.getByRole("button", { name: "Models", exact: true }).click();
  const tournament = page.getByRole("region", { name: "Component tournament", exact: true });
  await expect(tournament.locator("tbody tr").first()).toContainText("Huber");
  await expect(tournament.locator("tbody tr").first()).toContainText("Saved winner");
  await expect(tournament).not.toContainText("WeightedEnsemble");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(errors).toEqual([]);
  expect(mutations).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "forecast.csv")))).toBe(original);
  console.log(`Ranking screenshots: ${workspace}`);
});
