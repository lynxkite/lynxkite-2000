// Tests undo/redo functionality
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "undo_redo_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("undo_redo_spec_test");
});

test("undo/redo add_node transaction", async () => {
  await workspace.addBox("Import Parquet");
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
  await workspace.page.keyboard.press("Control+z");
  await expect(workspace.getBox("Import Parquet 1")).not.toBeVisible();
  await workspace.page.keyboard.press("Control+y");
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
});

test("undo/redo add_edge transaction", async () => {
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  await workspace.addBox("View tables");
  await new Promise((resolve) => setTimeout(resolve, 600));
  await workspace.connectBoxes("Import PyKEEN dataset 1", "View tables 1");
  const tableBox = workspace.getBox("View tables 1");
  await expect(tableBox.locator(".error")).not.toBeVisible();
  await workspace.page.keyboard.press("Control+z");
  await expect(tableBox.locator(".error")).toBeVisible();
  await workspace.page.keyboard.press("Control+y");
  await expect(tableBox.locator(".error")).not.toBeVisible();
});

test("undo/redo box dragging", async () => {
  await workspace.addBox("Import Parquet");
  const originalPos = await workspace.getBox("Import Parquet 1").boundingBox();
  await new Promise((resolve) => setTimeout(resolve, 600));
  await workspace.moveBox("Import Parquet 1", { offsetX: 100, offsetY: 100 });
  const newPos = await workspace.getBox("Import Parquet 1").boundingBox();
  expect(newPos?.x).toBeGreaterThan(originalPos?.x);
  expect(newPos?.y).toBeGreaterThan(originalPos?.y);
  await workspace.page.keyboard.press("Control+z");
  const undonePos = await workspace.getBox("Import Parquet 1").boundingBox();
  expect(undonePos?.x).toBeCloseTo(originalPos?.x, 1);
  expect(undonePos?.y).toBeCloseTo(originalPos?.y, 1);
  await workspace.page.keyboard.press("Control+y");
  const redonePos = await workspace.getBox("Import Parquet 1").boundingBox();
  expect(redonePos?.x).toBeGreaterThan(originalPos?.x);
  expect(redonePos?.y).toBeGreaterThan(originalPos?.y);
});

test("undo/redo grouping boxes", async () => {
  const consoleMessages: { type: string; text: string }[] = [];
  workspace.page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      consoleMessages.push({ type: msg.type(), text: msg.text() });
    }
  });
  await workspace.addBox("Import Parquet");
  await workspace.addBox("View tables");
  await workspace.connectBoxes("Import Parquet 1", "View tables 1");
  await workspace.selectBoxes(["Import Parquet 1", "View tables 1"]);
  await new Promise((resolve) => setTimeout(resolve, 600));
  await workspace.groupSelection();
  await workspace.page.keyboard.press("Control+z");
  await expect(workspace.getBox("Group 1")).not.toBeVisible();
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
  await expect(workspace.getBox("View tables 1")).toBeVisible();
  expect(consoleMessages).toEqual([]);

  await workspace.page.keyboard.press("Control+y");
  await expect(workspace.getBox("Group 1")).toBeVisible();
  await expect(workspace.getBox("Group 1").locator('[data-id="Import Parquet 1"]')).toBeVisible();
  await expect(workspace.getBox("Group 1").locator('[data-id="View tables 1"]')).toBeVisible();
  expect(consoleMessages).toEqual([]);
});

test("undo/redo normal text input", async () => {
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox = workspace.getBox("Scale-free graph 1");
  await new Promise((resolve) => setTimeout(resolve, 600));
  await graphBox.getByLabel("n", { exact: true }).fill("10");
  await workspace.page.keyboard.press("Control+z");
  await expect(graphBox.getByLabel("n", { exact: true })).toHaveValue("");
  await workspace.page.keyboard.press("Control+y");
  await expect(graphBox.getByLabel("n", { exact: true })).toHaveValue("10");
});
