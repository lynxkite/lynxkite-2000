import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "../..");
const examplesDir = path.join(repoRoot, "examples");
const lynxkiteExecutable =
  process.platform === "win32"
    ? path.join(repoRoot, ".venv", "Scripts", "lynxkite.exe")
    : path.join(repoRoot, ".venv", "bin", "lynxkite");

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
    permissions: ["clipboard-read", "clipboard-write"], // Needed for testing copy-paste functionality
  },
  projects: [
    {
      name: "setup",
      testMatch: /global\.setup\.ts/,
    },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
      dependencies: ["setup"],
    },
  ],
  webServer: {
    command: `"${lynxkiteExecutable}"`,
    cwd: examplesDir,
    env: {
      ...process.env,
      LYNXKITE_SUPPRESS_OP_ERRORS: "1",
    },
    port: 8000,
    reuseExistingServer: true,
  },
});
