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
  await workspace.addBox("Create scale-free graph");
  const graphBox = workspace.getBox("Create scale-free graph 1");
  await graphBox.locator("input").fill("");
  expect(await graphBox.locator(".error").innerText()).toBe(
    "invalid literal for int() with base 10: ''",
  );
  await graphBox.locator("input").fill("10");
  await expect(graphBox.locator(".error")).not.toBeVisible();
});

test("unknown operation", async () => {
  // Test that the correct error is displayed when the operation does not belong to
  // the current environment.
  await workspace.addBox("Create scale-free graph");
  await workspace.setEnv("LynxScribe");
  const csvBox = workspace.getBox("Create scale-free graph 1");
  const errorText = await csvBox.locator(".error").innerText();
  expect(errorText).toBe('Operation "Create scale-free graph" not found.');
  await workspace.setEnv("LynxKite Graph Analytics");
  await expect(csvBox.locator(".error")).not.toBeVisible();
});
