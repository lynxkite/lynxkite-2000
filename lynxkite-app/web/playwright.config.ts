import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: process.env.CI ? 30000 : 10000,
  fullyParallel: false,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  retries: 0,
  maxFailures: 3,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["html"]] : "html",
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: "http://127.0.0.1:8000",
    trace: "on",
    testIdAttribute: "data-nodeid", // Useful for easily selecting nodes using getByTestId
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "cd ../../examples && LYNXKITE_SUPPRESS_OP_ERRORS=1 lynxkite",
    port: 8000,
    reuseExistingServer: false,
  },
});
