// Tests copy-pasting functionality, and tests that normal copy-paste operations also work as expected.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "box_copy_pasting_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("box_copy_pasting_spec_test");
});

test("Copy-paste box", async () => {
  await workspace.addBox("Import Parquet");
  await workspace.selectBox("Import Parquet 1");
  await workspace.copySelection();
  await workspace.pasteSelection();
  await expect(workspace.getBox("Import Parquet 2")).toBeVisible();
});

test("Copy-paste connected boxes", async () => {
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  await workspace.addBox("View tables");
  await workspace.connectBoxes("Import PyKEEN dataset 1", "View tables 1");
  await workspace.selectBoxes(["Import PyKEEN dataset 1", "View tables 1"]);
  await workspace.copySelection();
  await workspace.pasteSelection();
  await expect(workspace.getBox("Import PyKEEN dataset 2")).toBeVisible();
  const tableBox = workspace.getBox("View tables 2");
  await expect(tableBox).toBeVisible();
  await expect(tableBox.locator(".error")).not.toBeVisible();
});

test("Cut-paste box", async () => {
  await workspace.addBox("Import Parquet");
  await workspace.selectBox("Import Parquet 1");
  await workspace.cutSelection();
  await expect(workspace.getBox("Import Parquet 1")).not.toBeVisible();
  await workspace.pasteSelection();
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
});

test("Copy-paste normal text", async () => {
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox = workspace.getBox("Scale-free graph 1");
  await graphBox.getByLabel("n", { exact: true }).fill("10");
  await graphBox.getByLabel("n", { exact: true }).selectText();
  await workspace.copySelection();
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox2 = workspace.getBox("Scale-free graph 2");
  await graphBox2.getByLabel("n", { exact: true }).click();
  await workspace.pasteSelection();
  await expect(graphBox2.getByLabel("n", { exact: true })).toHaveValue("10");
});

test("Copy boxes and paste into text field", async () => {
  await workspace.addBox("Import Parquet");
  await workspace.selectBox("Import Parquet 1");
  await workspace.copySelection();
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox = workspace.getBox("Scale-free graph 1");
  await graphBox.getByLabel("n", { exact: true }).click();
  await workspace.pasteSelection();
  const expectedValueRegex =
    /^(?=.*"edges":)(?=.*"nodes")(?=.*"type":\s*"basic")(?=.*"name":\s*"Import Parquet")/s;
  await expect(graphBox.getByLabel("n", { exact: true })).toHaveValue(expectedValueRegex);
});
