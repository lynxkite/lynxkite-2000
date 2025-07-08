// Tests error reporting.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "error_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("error_spec_test");
});

test("missing parameter", async () => {
  // Test the correct error message is displayed when a required parameter is missing,
  // and that the error message is removed when the parameter is filled.
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox = workspace.getBox("Scale-free graph 1");
  await expect(graphBox.locator(".error")).toHaveText("n is unset.");
  await graphBox.getByLabel("n", { exact: true }).fill("10");
  await expect(graphBox.locator(".error")).not.toBeVisible();
});

test("unknown operation", async () => {
  // Test that the correct error is displayed when the operation does not belong to
  // the current environment.
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox = workspace.getBox("Scale-free graph 1");
  await graphBox.getByLabel("n", { exact: true }).fill("10");
  await workspace.setEnv("Pillow");
  const csvBox = workspace.getBox("Scale-free graph 1");
  await expect(csvBox.locator(".error")).toHaveText("Unknown operation.");
  await workspace.setEnv("LynxKite Graph Analytics");
  await expect(csvBox.locator(".error")).not.toBeVisible();
});
