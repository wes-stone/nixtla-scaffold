import { test, expect } from "@playwright/test";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { configuration, createService, startServer } from "../../server.mjs";

test("experimental policies preserve objective, paired replay and non-adoption", async ({ page }) => {
  if (!process.env.NIXTLA_E2E_OUTPUT) throw new Error("Set NIXTLA_E2E_OUTPUT to a disposable artifact directory.");
  await mkdir(process.env.NIXTLA_E2E_OUTPUT, { recursive: true });
  const workspace = await mkdtemp(join(resolve(process.env.NIXTLA_E2E_OUTPUT), "ensembles-"));
  const source = join(workspace, "series.csv");
  await writeFile(source, ["unique_id,ds,y", ...Array.from({ length: 60 }, (_, i) =>
    `001,${2020 + Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, "0")}-01,${100 + i}`)].join("\n"));
  const { runtime, contract } = await configuration();
  const service = createService({ workspace, python: runtime.python, contract });
  const server = await startServer({ workspace, service, handoff: async () => {} });
  try {
    await service.call("initialize");
    const profile = await service.call("profile", { path: source });
    const started = await service.call("start_job", { kind: "forecast", requestKey: "fixture", sourceId: profile.sourceId,
      spec: { horizon: 6, run_mode: "quick", model_policy: "baseline", levels: [],
        evaluation: { max_windows: 8, step_size: 6 } } });
    await expect.poll(async () => (await service.call("job", { jobId: started.jobId })).status, { timeout: 120000 }).toBe("completed");
    const job = await service.call("job", { jobId: started.jobId });
    await service.call("save_state", { expectedRevision: 0, state: {
      documents: [{ id: "ensemble", runId: job.result.runId, title: "Ensemble policies", series: "001", view: "Compare", comparison: "models" }],
      activeId: "ensemble",
    } });
    await page.goto(server.url);
    for (const policy of ["greedy"]) {
      await expect(page.getByRole("button", { name: "Build ensemble", exact: true })).toBeVisible({ timeout: 90000 });
      await page.getByRole("button", { name: "Build ensemble", exact: true }).click();
      await page.getByLabel("Weighting policy").selectOption(policy);
      await expect(page.getByRole("option", { name: "Horizon groups (not qualified)", exact: true })).toHaveJSProperty("disabled", true);
      await page.getByRole("checkbox", { name: "Naive", exact: true }).check();
      await page.getByRole("checkbox", { name: "RandomWalkWithDrift", exact: true }).check();
      await expect(page.getByText(/Learns only from at least three/)).toBeVisible();
      await page.locator(".ensemble-editor").screenshot({ path: join(workspace, `${policy}-desktop.png`) });
      await page.setViewportSize({ width: 620, height: 900 });
      await page.getByRole("button", { name: "Toggle library", exact: true }).click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.locator(".ensemble-editor").screenshot({ path: join(workspace, `${policy}-narrow.png`) });
      await page.getByRole("button", { name: "Toggle library", exact: true }).click();
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.getByRole("button", { name: /Backtest this recipe/ }).click();
      await expect.poll(async () => (await service.call("view", { runId: job.result.runId })).blends
        .some(blend => blend.recipe.policy === policy), { timeout: 90000 }).toBe(true);
      const view = await service.call("view", { runId: job.result.runId });
      const blend = view.blends.find(item => item.recipe.policy === policy);
      expect(blend.recipe.objective).toBe("daily_rmse");
      expect(blend.receipt.promotion_recommended).toBe(false);
      expect(blend.receipt.paired_rows).toBeGreaterThan(0);
    }
    await expect(page.getByRole("alert")).toHaveCount(0);
    expect((await service.call("catalog")).runs.every(run => !run.metadata.adopted)).toBe(true);
  } finally {
    await service.close();
    await server.close();
  }
});
