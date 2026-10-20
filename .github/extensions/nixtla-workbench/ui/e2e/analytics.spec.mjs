import { test, expect } from "@playwright/test";
import { cp, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

let server, service, workspace, run, original, series, firstForecast, lastActual, componentCount, diagnosticCount, ledgerCount;
const source = process.env.NIXTLA_E2E_RUN;
const hash = value => createHash("sha256").update(value).digest("hex");
const display = (value, scale = 1) => (Number(value) / scale).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
test.beforeAll(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to a disposable artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "analytics-"));
  original = hash(await readFile(join(source, "forecast.csv")));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  run = await service.call("import_run", { path: resolve(source), name: "Saved analytical review" });
  const view = await service.call("view", { runId: run.id });
  series = view.series[0];
  firstForecast = view.rows.find(row => String(row.unique_id) === series).yhat;
  const history = await service.call("view", { runId: run.id, kind: "history", series });
  lastActual = history.rows.at(-1).y;
  const audit = await service.call("evidence", { runId: run.id, name: "appendix/model_audit.csv", series });
  componentCount = audit.rows.filter(row => row.selection_eligible === true && row.family !== "ensemble").length;
  diagnosticCount = audit.rows.length - componentCount;
  ledgerCount = (await service.call("ledger_view", { runId: run.id, series, kind: "snapshot", limit: 500 })).rows.length;
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "review", runId: run.id, title: "Saved analytical review", series, view: "Forecast" }],
    activeId: "review", theme: "light",
  } });
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterAll(async () => { await server?.close(); await service?.close(); });

test("saved evidence: two decimals, full arena, origins, diagnostics and responsive themes", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_RUN to a synthetic monthly fixture with 38 observations from July 2023, two six-period backtests and AutoETS/Holt paths. No fits are performed.");
  test.setTimeout(240000);
  const errors = [], executions = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api") && request.postDataJSON()?.operation === "start_job") executions.push(request.postDataJSON());
  });
  await page.goto(server.url);
  const values = page.getByRole("region", { name: "Forecast values", exact: true });
  await expect(values.getByRole("cell", { name: display(firstForecast), exact: true }).first()).toBeVisible();
  await page.getByLabel("Display units").selectOption("billions");
  await expect(values.getByRole("cell", { name: display(firstForecast, 1e9), exact: true }).first()).toBeVisible();
  await page.getByLabel("Prediction interval", { exact: true }).selectOption("80");
  await expect(values.getByRole("columnheader", { name: "80% lower" })).toBeVisible();
  await page.getByRole("button", { name: "Wide review table" }).click();
  await expect(page.getByRole("heading", { name: "Wide copy/paste summary" })).toBeVisible();
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await expect(page.getByRole("region", { name: "Component tournament", exact: true }).locator("tbody tr")).toHaveCount(componentCount);
  await page.getByText("Diagnostic candidates and separate ensemble cohorts", { exact: true }).click();
  await expect(page.getByRole("region", { name: "Separate-cohort diagnostics", exact: true }).locator("tbody tr")).toHaveCount(diagnosticCount);
  await page.getByRole("button", { name: "Pareto frontier", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Pareto frontier · RMSE versus MAE" })).toBeInViewport();
  await page.screenshot({ path: join(workspace, "pareto-desktop.png") });
  await page.getByRole("button", { name: "Forecast race", exact: true }).click();
  await page.getByLabel("Find a model", { exact: true }).fill("AutoETS");
  await page.getByRole("button", { name: "Focus AutoETS", exact: true }).click();
  await page.getByLabel("Find a model", { exact: true }).fill("");
  const race = page.locator("#model-race");
  await expect(race.getByRole("option", { name: "2023-07-01", exact: true })).toHaveCount(1);
  await expect(race).toContainText("38 saved actual observations through 2026-08-01");
  await race.getByLabel("Inspect a period").selectOption("2026-08-01");
  await expect(page.locator(".inspector dl > div").filter({ has: page.locator("dt", { hasText: /^y$/ }) }).locator("dd")).toHaveText(String(lastActual));
  await race.getByLabel("Inspect a period").selectOption("2026-09-01");
  await expect(page.locator(".inspector")).toContainText("AutoETS");
  await expect(page.locator(".inspector dt").filter({ hasText: /^y$/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Close inspector", exact: true }).click();
  await page.getByLabel("Compare Holt", { exact: true }).check();
  await expect(race.locator(".chart-legend li")).toHaveCount(3);
  await expect(page.getByRole("region", { name: "Component tournament", exact: true }).locator("tbody tr")).toHaveCount(componentCount);
  await expect(race.getByRole("option", { name: "2023-07-01", exact: true })).toHaveCount(1);
  await page.getByRole("button", { name: "Focus AutoETS", exact: true }).click();
  await expect(page.getByRole("region", { name: "Component tournament", exact: true }).locator("tbody tr")).toHaveCount(componentCount);
  await race.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "model-race-actuals-desktop.png") });
  await page.setViewportSize({ width: 620, height: 900 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await race.scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "model-race-actuals-narrow.png") });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("light");
  await page.getByRole("button", { name: "Backtests", exact: true }).click();
  const origin = page.getByLabel("Backtest origin", { exact: true });
  await expect(origin.locator("option")).toHaveCount(2);
  await expect(page.locator("#backtest-replay")).toHaveAttribute("aria-busy", "false");
  const replayCanvas = await page.locator("#backtest-replay canvas").first().elementHandle();
  const replayChart = page.locator("#backtest-replay .chart");
  const fullAxis = /^Forecast timeline, 2023-07-01 to 2026-08-01; /;
  await expect(replayChart).toHaveAttribute("aria-label", fullAxis);
  const originRequests = [];
  const trackOrigins = request => {
    if (request.url().endsWith("/api") && request.postDataJSON()?.operation === "evidence" && request.postDataJSON()?.input?.cutoff) originRequests.push(request.postDataJSON());
  };
  page.on("request", trackOrigins);
  const latest = await origin.inputValue();
  await page.getByRole("button", { name: "Previous window", exact: true }).click();
  await expect(origin).not.toHaveValue(latest);
  await expect(page.locator("#backtest-replay")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator(".window-summary")).toContainText("2025-09-01 – 2026-02-01");
  expect(await replayCanvas.evaluate(node => node.isConnected)).toBe(true);
  await expect(page.getByRole("button", { name: "Focus forecast", exact: true })).toHaveCount(0);
  await expect(replayChart).toHaveAttribute("aria-label", fullAxis);
  await page.getByLabel("Keep value scale fixed", { exact: true }).uncheck();
  await expect(replayChart).toHaveAttribute("aria-label", fullAxis);
  await page.getByLabel("Keep value scale fixed", { exact: true }).check();
  await page.locator("#backtest-replay").getByLabel("Inspect a period").selectOption("2026-08-01");
  await expect(page.locator(".inspector dl > div").filter({ has: page.locator("dt", { hasText: /^y$/ }) }).locator("dd")).toHaveText(String(lastActual));
  await expect(page.locator(".inspector dt").filter({ hasText: /^AutoETS$/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Close inspector", exact: true }).click();
  await expect(page.getByLabel("Origin timeline")).toHaveCount(0);
  await page.getByRole("button", { name: "Next window", exact: true }).click();
  await expect(origin).toHaveValue(latest);
  await expect(page.locator("#backtest-replay")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator(".window-summary")).toContainText("2026-03-01 – 2026-08-01");
  await expect(replayChart).toHaveAttribute("aria-label", fullAxis);
  expect(originRequests).toEqual([]);
  page.off("request", trackOrigins);
  await page.getByText("Detailed backtest evidence and exports", { exact: true }).click();
  await expect(page.getByRole("region", { name: "Selected-origin predictions and saved intervals", exact: true }).locator("tbody tr").first()).toBeVisible();
  await page.getByLabel("Backtest interval", { exact: true }).selectOption("80");
  await expect(page.locator(".focused-model strong")).toHaveText("AutoETS");
  await page.locator("#backtest-replay").scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "backtests-desktop.png") });
  await page.setViewportSize({ width: 620, height: 900 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await page.locator("#backtest-replay").scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "backtests-narrow-dark.png") });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("light");
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Nominal versus empirical coverage by model and horizon" })).toBeVisible();
  await page.getByRole("button", { name: "Residuals", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Saved bias, outlier, white-noise and structural-break tests" })).toBeVisible();
  await page.getByRole("button", { name: "Seasonality", exact: true }).click();
  await page.getByLabel("Year starts in").selectOption("7");
  await expect(page.getByRole("heading", { name: "Seasonality credibility and complete cycles" })).toBeVisible();
  await page.getByRole("button", { name: "Hierarchy", exact: true }).click();
  await expect(page.getByText(/No hierarchy tables were produced/)).toBeVisible();
  await page.getByRole("button", { name: "Drivers", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Drivers evidence explorer" })).toBeVisible();
  await page.getByRole("button", { name: "Ledger", exact: true }).click();
  await expect(page.getByRole("region", { name: "Ledger evidence · bounded to 500 rows" })).toContainText(`${ledgerCount} rows shown`);
  await page.getByRole("button", { name: "Challengers", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Challengers evidence explorer" })).toBeVisible();
  await page.getByRole("button", { name: "Feeders", exact: true }).click();
  await page.getByLabel("Feeders evidence table").selectOption("appendix/model_pareto_frontier.csv");
  await expect(page.getByRole("button", { name: "Export full scope" })).toBeVisible();
  await page.getByRole("button", { name: "Forecast", exact: true }).click();
  await page.getByRole("button", { name: "Return to saved selection", exact: true }).click();
  await expect(values.getByRole("cell", { name: display(firstForecast, 1e9), exact: true }).first()).toBeVisible();
  await page.getByLabel("Appearance").selectOption("dark");
  await page.setViewportSize({ width: 620, height: 900 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByRole("heading", { name: "Actuals and forecast" }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(workspace, "forecast-narrow-dark.png") });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.reload();
  await expect(page.getByLabel("Display units")).toHaveValue("billions");
  expect(errors).toEqual([]);
  expect(executions).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "forecast.csv")))).toBe(original);
  console.log(`Analytical acceptance screenshots: ${workspace}`);
});

test("paged replay retains the old window and canvas until the selected window is ready", async ({ page }) => {
  test.skip(!source, "Requires a saved evidence fixture.");
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "slow-replay", runId: run.id, series, view: "Backtests", analysis: { cutoff: "2025-08-01" } }],
    activeId: "slow-replay", theme: "light",
  } });
  let release, selectedRequest = false;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route("**/api", async route => {
    const request = route.request().postDataJSON();
    if (request.operation !== "evidence" || request.input.name !== "appendix/backtest_long.csv") return route.continue();
    const response = await route.fetch(), body = await response.json();
    if (!request.input.cutoff) {
      body.result.rows = body.result.rows.slice(0, 5);
      body.result.nextOffset = 5;
    } else if (request.input.cutoff === "2026-02-01") {
      selectedRequest = true;
      await gate;
    }
    await route.fulfill({ response, json: body });
  });
  await page.goto(server.url);
  await expect(page.locator("#backtest-replay")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator(".window-summary")).toContainText("2025-09-01 – 2026-02-01");
  const canvas = await page.locator("#backtest-replay canvas").first().elementHandle();
  await page.getByRole("button", { name: "Next window", exact: true }).click();
  await expect.poll(() => selectedRequest).toBe(true);
  await expect(page.locator("#backtest-replay")).toHaveAttribute("aria-busy", "true");
  await expect(page.getByRole("status").filter({ hasText: "previous window remains visible" })).toBeVisible();
  await expect(page.locator(".window-summary")).toContainText("2025-09-01 – 2026-02-01");
  expect(await canvas.evaluate(node => node.isConnected)).toBe(true);
  release();
  await expect(page.locator("#backtest-replay")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator(".window-summary")).toContainText("2026-03-01 – 2026-08-01");
  expect(await canvas.evaluate(node => node.isConnected)).toBe(true);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
});

test("absent backtest windows show an explicit empty state, not a player or endless loading", async ({ page }) => {
  test.skip(!source, "Requires a saved evidence fixture.");
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "empty-replay", runId: run.id, series, view: "Backtests" }],
    activeId: "empty-replay", theme: "light",
  } });
  await page.route("**/api", async route => {
    const request = route.request().postDataJSON();
    if (request.operation !== "evidence" || request.input.name !== "appendix/backtest_long.csv") return route.continue();
    const response = await route.fetch(), body = await response.json();
    Object.assign(body.result, { available: false, rows: [], cutoffs: [], total: 0, nextOffset: null, message: "No saved backtest artifact." });
    await route.fulfill({ response, json: body });
  });
  await page.goto(server.url);
  await expect(page.getByText("No saved backtest windows are available for this series and model selection.", { exact: true })).toBeVisible();
  await expect(page.getByRole("group", { name: "Backtest window navigation" })).toHaveCount(0);
  await expect(page.getByText("Loading saved backtest windows and actuals...", { exact: true })).toHaveCount(0);
});

test("optional hierarchy, scenario and external evidence remain scoped and chartable", async ({ page }) => {
  test.skip(!source, "Requires a saved evidence fixture.");
  const fixture = join(workspace, "optional-fixture");
  await cp(source, fixture, { recursive: true });
  await writeFile(join(fixture, "appendix", "hierarchy_rollup.csv"),
    `parent_unique_id,ds,parent_yhat,child_sum_yhat\n${series},2026-09-01,1234567890.125,1234567890.125\n`);
  await writeFile(join(fixture, "appendix", "scenario_forecast.csv"),
    `unique_id,ds,scenario_name,yhat,yhat_scenario\n${series},2026-09-01,Synthetic scenario,1234567890.125,1350000000.25\n`);
  await mkdir(join(fixture, "byo_model"), { recursive: true });
  await writeFile(join(fixture, "byo_model", "forecast_comparison.csv"),
    `unique_id,ds,scenario_name,external_model,scaffold_yhat,external_yhat,comparison_status\n${series},2026-09-01,Test only,Synthetic fixture,1234567890.125,1500000000.5,aligned\n`);
  const optional = await service.call("import_run", { path: fixture, name: "Optional evidence test fixture" });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "optional", runId: optional.id, title: "Optional evidence test fixture", series, view: "Diagnostics", analysis: { section: "Hierarchy" } }],
    activeId: "optional", theme: "light",
  } });
  await page.goto(server.url);
  await page.getByLabel("Hierarchy evidence table").selectOption("appendix/hierarchy_rollup.csv");
  await expect(page.getByRole("region", { name: "hierarchy rollup", exact: true })).toContainText("1,234,567,890.13");
  await page.getByRole("button", { name: "Drivers", exact: true }).click();
  await page.getByLabel("Drivers evidence table").selectOption("appendix/scenario_forecast.csv");
  await expect(page.getByRole("region", { name: "scenario forecast", exact: true })).toContainText("1,350,000,000.25");
  await page.getByRole("button", { name: "Challengers", exact: true }).click();
  await page.getByLabel("Challengers evidence table").selectOption("byo_model/forecast_comparison.csv");
  await expect(page.getByRole("region", { name: "forecast comparison", exact: true })).toContainText("1,500,000,000.50");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
});
