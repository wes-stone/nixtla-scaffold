import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

let service, server, workspace, dailyRun, monthlyRun;
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const iso = date => date.toISOString().slice(0, 10);

async function fixture(name, freq, history, forecast) {
  const input = join(workspace, name);
  await mkdir(join(input, "appendix"), { recursive: true });
  await writeFile(join(input, "manifest.json"), JSON.stringify({ spec: { freq, horizon: forecast.length, unit_label: "synthetic units" } }));
  await writeFile(join(input, "appendix", "observed_history.csv"), "unique_id,ds,y\n" + history.map(([date, value]) => `Synthetic,${date},${value ?? ""}`).join("\n"));
  await writeFile(join(input, "appendix", "history.csv"), "unique_id,ds,y\n" + history.map(([date, value]) => `Synthetic,${date},${value === null ? 999999 : value}`).join("\n"));
  const selected = "unique_id,ds,yhat,model,yhat_lo_80,yhat_hi_80\n" +
    forecast.map(([date, value]) => `Synthetic,${date},${value ?? ""},AutoETS,${value === null ? "" : value - 10},${value === null ? "" : value + 10}`).join("\n");
  await writeFile(join(input, "forecast.csv"), selected);
  await writeFile(join(input, "appendix", "forecast_long.csv"), selected + "\n" +
    forecast.map(([date, value]) => `Synthetic,${date},${value === null ? "" : value * 2},Ridge,,`).join("\n"));
  return service.call("import_run", { path: input, name });
}

async function activate(run, series = "Synthetic") {
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "seasonal", runId: run.id, title: run.name, series, view: "Forecast" }],
    activeId: "seasonal", theme: "light",
  } });
}

test.beforeAll(async () => {
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to a disposable artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "seasonal-"));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  const history = Array.from({ length: 77 }, (_, i) => [iso(new Date(Date.UTC(2026, 7, i + 1))), i === 0 ? 0 : i === 76 ? null : 100 + (i % 7 < 5 ? 80 : 20)]);
  dailyRun = await fixture("Synthetic daily seasonality", "D", history.filter(([date]) => date !== "2026-09-02"),
    Array.from({ length: 31 }, (_, i) => [iso(new Date(Date.UTC(2026, 9, i + 17))), i === 2 ? null : 230 + (i % 7 < 5 ? 60 : 0)]));
  monthlyRun = await fixture("Synthetic monthly seasonality", "MS",
    Array.from({ length: 32 }, (_, i) => [iso(new Date(Date.UTC(2024, i, 1))), 100 + i]).filter(([date]) => date !== "2026-02-01"),
    Array.from({ length: 6 }, (_, i) => [iso(new Date(Date.UTC(2026, i + 8, 1))), 160 + i]));
  server = await startServer({ workspace, service });
});
test.afterAll(async () => { await server?.close(); await service?.close(); });

test("daily overlays preserve gaps, model choice, state, errors and responsive chart access", async ({ page }) => {
  const errors = [], actions = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api")) actions.push(request.postDataJSON()?.operation);
  });
  await activate(dailyRun);
  await page.goto(server.url);
  await page.getByRole("group", { name: "Forecast chart view", exact: true }).getByRole("button", { name: "Seasonality", exact: true }).click();
  const panel = page.getByRole("region", { name: "Seasonal forecast", exact: true });
  await expect(panel.getByLabel("Seasonal focus period")).toHaveValue("2026-10");
  await expect(panel.getByRole("list", { name: "Seasonal chart series" })).toContainText("2026-08 · actual");
  await expect(panel).toContainText("2026-10 · AutoETS forecast");
  await expect(page.getByLabel("Prediction interval", { exact: true })).toBeDisabled();
  await panel.getByText("Exact seasonal values", { exact: true }).click();
  const values = panel.getByRole("region", { name: "Seasonal dates and values" });
  await expect(values).toContainText("2026-10-17");
  await expect(values).not.toContainText("999,999");
  await expect(values).not.toContainText("2026-11-01");
  await expect(values.locator("tbody tr").filter({ hasText: "2026-08-01" })).toContainText("0.00");
  await expect(values.locator("tbody tr").filter({ hasText: "2026-10-16" }).locator(".missing")).toHaveCount(3);
  await panel.getByText("Exact seasonal values", { exact: true }).click();
  await page.getByRole("button", { name: "Focus Ridge", exact: true }).click();
  await expect(panel).toContainText("2026-10 · Ridge forecast");
  await page.getByRole("button", { name: "Return to saved selection", exact: true }).click();
  await expect(panel).toContainText("2026-10 · AutoETS forecast");
  await panel.getByLabel("Seasonal focus period").selectOption("2026-11");
  await expect(panel).toContainText("2026-11 · AutoETS forecast");
  await panel.getByLabel("Seasonal focus period").selectOption("2026-10");
  await panel.getByLabel("Seasonal alignment").selectOption("date");
  await expect(panel).toContainText("Day-of-month comparison");
  await expect.poll(async () => (await service.call("catalog")).state.documents[0].analysis.seasonalAlignment).toBe("date");
  await page.reload();
  await expect(panel.getByLabel("Seasonal alignment")).toHaveValue("date");
  await panel.getByLabel("Seasonal alignment").selectOption("weekday");
  await panel.getByRole("heading").scrollIntoViewIfNeeded();
  await panel.screenshot({ path: join(workspace, "daily-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await panel.getByRole("heading").scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.screenshot({ path: join(workspace, "daily-mobile-dark.png") });
  await page.route("**/api", async route => {
    const body = route.request().postDataJSON();
    if (body?.operation === "evidence" && body.input?.name === "appendix/observed_history.csv")
      await route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ error: { code: "fixture", message: "Seasonal history unavailable" } }) });
    else await route.continue();
  });
  await panel.getByRole("button", { name: "Reload seasonal evidence" }).click();
  await expect(panel.getByRole("alert")).toContainText("Seasonal history unavailable");
  await expect(panel.locator(".chart")).toHaveCount(0);
  await page.unroute("**/api");
  await panel.getByRole("button", { name: "Reload seasonal evidence" }).click();
  await expect(panel.locator(".chart")).toBeVisible();
  await page.getByRole("group", { name: "Forecast chart view", exact: true }).getByRole("button", { name: "Timeline", exact: true }).click();
  await expect(page.getByRole("region", { name: "Forecast timeline", exact: true })).toBeVisible();
  await expect(page.getByLabel("Prediction interval", { exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
  expect(actions).not.toContain("start_job");
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  console.log(`Seasonal visual evidence: ${workspace}`);
});

test("monthly and diagnostic views share the fiscal-year overlay without changing forecasts", async ({ page }) => {
  await activate(monthlyRun);
  await page.goto(server.url);
  await page.getByRole("group", { name: "Forecast chart view" }).getByRole("button", { name: "Seasonality", exact: true }).click();
  const panel = page.getByRole("region", { name: "Seasonal forecast", exact: true });
  await expect(panel).toContainText("Month-of-year comparison");
  await expect(panel.getByLabel("Seasonal focus period")).toHaveValue("2026");
  await panel.getByLabel("Year starts in").selectOption("7");
  await expect(panel).toContainText("2026/27 · AutoETS forecast");
  await panel.getByLabel("Seasonal comparison periods").selectOption("1");
  await expect(panel.getByRole("list")).not.toContainText("2025/26");
  await panel.getByRole("heading").scrollIntoViewIfNeeded();
  await panel.screenshot({ path: join(workspace, "monthly-desktop.png") });
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByRole("navigation", { name: "Diagnostic section" }).getByRole("button", { name: "Seasonality", exact: true }).click();
  await expect(panel.getByLabel("Year starts in")).toHaveValue("7");
  await expect(panel).toContainText("2026/27 · AutoETS forecast");
  expect((await service.call("catalog")).jobs).toHaveLength(0);
});

test("a saved daily forecast renders unchanged in the seasonality view", async ({ page }) => {
  const path = process.env.NIXTLA_E2E_SEASONAL_RUN ?? join(workspace, "Synthetic daily seasonality");
  const original = await readFile(join(path, "forecast.csv"));
  const run = await service.call("import_run", { path: resolve(path), name: "Saved seasonal review" });
  const view = await service.call("view", { runId: run.id });
  expect(["D", "B", "1D", "1B"]).toContain(view.spec.freq);
  const series = view.series[0];
  const first = view.rows.filter(row => String(row.unique_id) === series)
    .sort((a, b) => String(a.ds).localeCompare(String(b.ds)))[0];
  expect(first).toBeDefined();
  const focus = String(first.ds).slice(0, 7);
  await activate(run, series);
  await page.goto(server.url);
  await page.getByRole("group", { name: "Forecast chart view" }).getByRole("button", { name: "Seasonality", exact: true }).click();
  const panel = page.getByRole("region", { name: "Seasonal forecast", exact: true });
  await expect(panel.getByLabel("Seasonal focus period")).toHaveValue(focus);
  await expect(panel).toContainText(`${focus} · ${first.model} forecast`);
  await expect(panel).not.toContainText("observed-only history artifact");
  await expect(panel.getByRole("alert")).toHaveCount(0);
  await panel.getByRole("heading").scrollIntoViewIfNeeded();
  await panel.screenshot({ path: join(workspace, "saved-seasonal-desktop.png") });
  expect(hash(await readFile(join(path, "forecast.csv")))).toBe(hash(original));
  expect((await service.call("catalog")).jobs).toHaveLength(0);
});
