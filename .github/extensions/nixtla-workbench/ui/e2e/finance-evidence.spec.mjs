import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

const source = process.env.NIXTLA_E2E_FINANCE_RUN;
let service, server, workspace, original;
const hash = data => createHash("sha256").update(data).digest("hex");
test.beforeAll(async () => {
  if (!source) return;
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to an artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "finance-"));
  original = hash(await readFile(join(source, "forecast.csv")));
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  const run = await service.call("import_run", { path: resolve(source), name: "Finance engineering fixture" });
  const view = await service.call("view", { runId: run.id });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "finance", runId: run.id, title: "Finance evidence", series: view.series[0], view: "Forecast" }],
    activeId: "finance", theme: "light",
  } });
  server = await startServer({ workspace, service, handoff: async () => {} });
});
test.afterAll(async () => { await server?.close(); await service?.close(); });

test("finance scores and aggregate limitations persist without fitting", async ({ page }) => {
  test.skip(!source, "Set NIXTLA_E2E_FINANCE_RUN to a saved finance fixture.");
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(server.url);
  const panel = page.getByRole("region", { name: "Finance totals evidence", exact: true });
  await expect(panel).toContainText("monthly Landing MAPE");
  await expect(panel).toContainText("insufficient_calibration");
  await expect(panel).toContainText("never summed daily bounds");
  const details = panel.getByText("Inspect origin and closed-month windows", { exact: true });
  await details.focus();
  await page.keyboard.press("Enter");
  await expect(panel).toContainText("Finance evaluation windows");
  await panel.screenshot({ path: join(workspace, "finance-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle library", exact: true }).click();
  await page.getByLabel("Appearance").selectOption("dark");
  await panel.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.screenshot({ path: join(workspace, "finance-mobile-dark.png") });
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toHaveLength(0);
  expect(hash(await readFile(join(source, "forecast.csv")))).toBe(original);
  console.log(`Finance screenshots: ${workspace}`);
});
