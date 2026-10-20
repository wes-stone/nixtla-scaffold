import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

let service, server, workspace;
test.beforeAll(async () => {
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to a disposable artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "forecast-controls-"));
  const input = join(workspace, "synthetic-run");
  await mkdir(join(input, "appendix"), { recursive: true });
  await writeFile(join(input, "manifest.json"), JSON.stringify({ spec: { horizon: 2, freq: "MS", unit_label: "minutes" } }));
  const forecast = "unique_id,ds,yhat,model,yhat_lo_80,yhat_hi_80\nSynthetic minutes,2026-10-01,100,AutoETS,90,110\nSynthetic minutes,2026-11-01,120,AutoETS,110,130\n";
  await writeFile(join(input, "forecast.csv"), forecast);
  await writeFile(join(input, "appendix", "forecast_long.csv"), forecast);
  await writeFile(join(input, "appendix", "history.csv"), "unique_id,ds,y\nSynthetic minutes,2026-08-01,90\nSynthetic minutes,2026-09-01,100\n");
  const { runtime, contract } = await configuration();
  service = createService({ workspace, python: runtime.python, contract });
  await service.call("initialize");
  const run = await service.call("import_run", { path: input, name: "Synthetic control layout" });
  const catalog = await service.call("catalog");
  await service.call("save_state", { expectedRevision: catalog.revision, state: {
    documents: [{ id: "controls", runId: run.id, title: "Synthetic control layout", series: "Synthetic minutes", view: "Forecast" }],
    activeId: "controls", theme: "light",
  } });
  server = await startServer({ workspace, service });
});
test.afterAll(async () => { await server?.close(); await service?.close(); });

test("aligned forecast controls and keyboard navigation to the wide review", async ({ page }) => {
  test.setTimeout(240000);
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(server.url);
  const controls = page.locator(".model-stage > .analysis-controls");
  const units = controls.getByLabel("Display units", { exact: true });
  const interval = controls.getByLabel("Prediction interval", { exact: true });
  const open = controls.getByRole("button", { name: "Wide review table", exact: true });
  await expect(units).toBeVisible();
  await expect(page.getByRole("region", { name: "Forecast values", exact: true })).toContainText("100.00");

  for (const width of [1440, 1228, 1000, 620, 390]) {
    await page.setViewportSize({ width, height: 900 });
    if (width === 620) await page.getByRole("button", { name: "Toggle library", exact: true }).click();
    await controls.scrollIntoViewIfNeeded();
    const boxes = await controls.evaluate(element => {
      const rect = node => {
        const box = node.getBoundingClientRect();
        return { top: box.top, bottom: box.bottom, left: box.left, right: box.right };
      };
      return { selects: [...element.querySelectorAll("select")].map(rect), button: rect(element.querySelector("button")), hint: rect(element.querySelector("small")) };
    });
    expect(Math.abs(boxes.selects[0].top - boxes.selects[1].top)).toBeLessThanOrEqual(1);
    expect(Math.abs(boxes.selects[0].bottom - boxes.selects[1].bottom)).toBeLessThanOrEqual(1);
    if (width >= 620) expect(Math.abs(boxes.button.bottom - boxes.selects[0].bottom)).toBeLessThanOrEqual(1);
    expect(boxes.hint.top).toBeGreaterThanOrEqual(Math.max(boxes.button.bottom, ...boxes.selects.map(box => box.bottom)));
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (width === 1440 || width === 390) await controls.screenshot({ path: join(workspace, `controls-${width}.png`) });
  }

  await units.focus();
  await page.keyboard.press("Tab");
  await expect(interval).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(open).toBeFocused();
  await page.keyboard.press("Enter");
  const review = page.getByRole("region", { name: "Wide forecast review", exact: true });
  await expect(review).toBeFocused();
  await expect(review.getByRole("heading", { name: "Wide forecast review", exact: true })).toBeInViewport();
  await expect(review.getByRole("region", { name: "Wide copy/paste summary", exact: true })).toContainText("100.00");
  await page.screenshot({ path: join(workspace, "wide-review-mobile.png") });
  await open.click();
  await expect(review).toBeFocused();
  await expect(review.getByRole("heading", { name: "Wide forecast review", exact: true })).toBeInViewport();
  await review.getByRole("button", { name: "Back to forecast", exact: true }).click();
  await expect(review).toHaveCount(0);
  await expect(open).toBeFocused();
  await expect(open).toBeInViewport();
  await expect(open).toHaveAttribute("aria-expanded", "false");
  await page.setViewportSize({ width: 1440, height: 900 });
  await open.click();
  await expect(review).toBeFocused();
  await expect(review.getByRole("heading", { name: "Wide forecast review", exact: true })).toBeInViewport();
  await expect(review.getByRole("region", { name: "Wide copy/paste summary", exact: true })).toContainText("100.00");
  await page.screenshot({ path: join(workspace, "wide-review-desktop.png") });
  expect(errors).toEqual([]);
  expect((await service.call("catalog")).jobs).toEqual([]);
  console.log(`Forecast control artifacts: ${workspace}`);
});
