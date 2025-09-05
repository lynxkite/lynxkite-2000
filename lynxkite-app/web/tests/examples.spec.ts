// Test the execution of the example workspaces
import { test } from "@playwright/test";
import { Workspace } from "./lynxkite";

const WORKSPACES = [
  "Airlines demo",
  // "Graph RAG",
  "Image processing",
  "NetworkX demo",
];

for (const name of WORKSPACES) {
  test(name, async ({ page }) => {
    const ws = await Workspace.open(page, name);
    await ws.execute();
    await ws.expectErrorFree();
  });
}

test("Model use", async ({ page }) => {
  const ws = await Workspace.open(page, "Model use");
  await ws.execute({ timeout: 30000 }); // Actually trains the model.
  await ws.expectErrorFree();
  let b = ws.boxByTitle("Train/test split");
  await b.expectParameterOptions("table name", ["", "df"]);
  b = ws.boxByTitle("Train model");
  await b.expectParameterOptions("model name", ["", "model"]);
  b = ws.boxByTitle("View vectors");
  await b.locator.locator(".params-expander").click();
  await b.expectParameterOptions("table name", ["", "df", "df_test", "df_train", "training"]);
  await b.expectParameterOptions("vector column", ["", "index", "pred", "x", "y"]);
  await b.expectParameterOptions("label column", ["", "index", "pred", "x", "y"]);
});
