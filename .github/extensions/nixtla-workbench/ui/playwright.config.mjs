import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 180000,
  expect: { timeout: 45000 },
  use: {
    actionTimeout: 20000,
    channel: process.env.NIXTLA_BROWSER_CHANNEL ?? "msedge",
    headless: true,
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
});
