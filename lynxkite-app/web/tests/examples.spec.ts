// Test the execution of the example workspaces
import type { Page } from "@playwright/test";
import { test } from "@playwright/test";
import { Splash } from "./lynxkite";

const WORKSPACES = ["Airlines demo", "Image processing", "NetworkX demo"];

async function openWorkspace(page: Page, name: string) {
  const splash = await Splash.openRoot(page);
  await splash.openFolder("Basic examples");
  const ws = await splash.openWorkspace(name);
  await ws.waitForNodesToLoad();
  await ws.expectCurrentWorkspaceIs(name);
  return ws;
}

for (const name of WORKSPACES) {
  test(name, async ({ page }) => {
    const ws = await openWorkspace(page, name);
    await ws.execute();
    await ws.expectErrorFree();
  });
}

test("Model use", async ({ page }) => {
  const ws = await openWorkspace(page, "Model use");
  await ws.execute({ timeout: 30000 }); // Actually trains the model.
  await ws.expectErrorFree();
  let b = ws.boxByTitle("Train/test split");
  await b.expectParameterOptions("table name", ["", "records"]);
  b = ws.boxByTitle("Train model");
  await b.expectParameterOptions("model name", ["", "model"]);
  b = ws.boxByTitle("View vectors");
  await b.locator.locator(".params-expander").click();
  await b.expectParameterOptions("table name", [
    "",
    "records",
    "records_test",
    "records_train",
    "training",
  ]);
  await b.expectParameterOptions("vector column", ["", "index", "prediction", "x", "y"]);
  await b.expectParameterOptions("label column", ["", "index", "prediction", "x", "y"]);
});
