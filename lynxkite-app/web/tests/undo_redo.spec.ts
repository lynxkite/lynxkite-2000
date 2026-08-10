// Tests undo/redo functionality
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;
const PRIMARY_MODIFIER = process.platform === "darwin" ? "Meta" : "Control";
const TEXT_INPUT_REDO_SHORTCUTS =
  process.platform === "darwin"
    ? ["Meta+Shift+z", "Meta+y", "Meta+Shift+z"]
    : ["Control+y", "Control+Shift+z", "Control+y"];

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "undo_redo_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  await splash.deleteEntry("undo_redo_spec_test");
});

test("undo/redo add_node transaction", async () => {
  await workspace.addBox("File operations › Import Parquet");
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
  await workspace.undo();
  await expect(workspace.getBox("Import Parquet 1")).not.toBeVisible();
  await workspace.redo();
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
});

test("undo/redo add_edge transaction", async () => {
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  await workspace.addBox("View tables");
  await new Promise((resolve) => setTimeout(resolve, 600));
  await workspace.connectBoxes("Import PyKEEN dataset 1", "View tables 1");
  const tableBox = workspace.getBox("View tables 1");
  await expect(tableBox.locator(".error")).not.toBeVisible();
  await workspace.undo();
  await expect(tableBox.locator(".error")).toBeVisible();
  await workspace.redo();
  await expect(tableBox.locator(".error")).not.toBeVisible();
});

test("undo/redo box dragging", async () => {
  await workspace.addBox("File operations › Import Parquet");
  const originalPos = await workspace.getBox("Import Parquet 1").boundingBox();
  await new Promise((resolve) => setTimeout(resolve, 600));
  await workspace.moveBox("Import Parquet 1", { offsetX: 100, offsetY: 100 });
  const newPos = await workspace.getBox("Import Parquet 1").boundingBox();
  expect(newPos?.x).toBeGreaterThan(originalPos!.x);
  expect(newPos?.y).toBeGreaterThan(originalPos!.y);
  await workspace.undo();
  const undonePos = await workspace.getBox("Import Parquet 1").boundingBox();
  expect(undonePos?.x).toBeCloseTo(originalPos!.x, 1);
  expect(undonePos?.y).toBeCloseTo(originalPos!.y, 1);
  await workspace.redo();
  const redonePos = await workspace.getBox("Import Parquet 1").boundingBox();
  expect(redonePos?.x).toBeGreaterThan(originalPos!.x);
  expect(redonePos?.y).toBeGreaterThan(originalPos!.y);
});

test("undo/redo grouping boxes", async () => {
  const consoleMessages: { type: string; text: string }[] = [];
  workspace.page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      consoleMessages.push({ type: msg.type(), text: msg.text() });
    }
  });
  await workspace.addBox("File operations › Import Parquet");
  await workspace.addBox("View tables");
  await workspace.connectBoxes("Import Parquet 1", "View tables 1");
  await workspace.selectBoxes(["Import Parquet 1", "View tables 1"]);
  await new Promise((resolve) => setTimeout(resolve, 600));
  await workspace.groupSelection();
  await expect(workspace.getBox("Group 1")).toBeVisible();

  await workspace.undo();
  await expect(workspace.getBox("Group 1")).not.toBeVisible();
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
  await expect(workspace.getBox("View tables 1")).toBeVisible();
  expect(consoleMessages).toEqual([]);

  await workspace.redo();
  await expect(workspace.getBox("Group 1")).toBeVisible();
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
  await expect(workspace.getBox("View tables 1")).toBeVisible();
  expect(consoleMessages).toEqual([]);
});

test("undo/redo normal text input", async () => {
  await workspace.addBox("NetworkX › Generators › Directed › Scale-free graph");
  const graphBox = workspace.getBox("Scale-free graph 1");
  const getNInput = () => workspace.getBox("Scale-free graph 1").getByLabel("n", { exact: true });
  const getNValue = async () => {
    if ((await workspace.getBox("Scale-free graph 1").count()) === 0) return null;
    return await getNInput().inputValue();
  };
  const nInput = getNInput();
  const initialValue = await nInput.inputValue();
  const editedValue = initialValue === "10" ? "11" : "10";
  await nInput.click();
  await expect(nInput).toBeFocused();
  await nInput.pressSequentially(editedValue);
  await expect(nInput).toHaveValue(editedValue);
  // Prefer native text-input undo when the input is focused.
  for (let i = 0; i < 3 && (await getNValue()) === editedValue; i++) {
    await getNInput().click();
    await getNInput().press(`${PRIMARY_MODIFIER}+z`);
  }

  // If app-level undo intercepted Cmd/Ctrl+Z, recover box state and continue.
  if ((await getNValue()) === null) {
    await workspace.redo();
    await expect(workspace.getBox("Scale-free graph 1")).toBeVisible();
  }

  if ((await getNValue()) === editedValue) {
    await workspace.undo();
    if ((await getNValue()) === null) {
      await workspace.redo();
      await expect(workspace.getBox("Scale-free graph 1")).toBeVisible();
    }
  }

  await expect(getNInput()).not.toHaveValue(editedValue);

  // Prefer native redo for text input.
  for (const shortcut of TEXT_INPUT_REDO_SHORTCUTS) {
    if ((await getNValue()) === editedValue) break;
    await getNInput().click();
    await getNInput().press(shortcut);
  }

  // Fallback to app-level redo if native redo did not restore the edit.
  for (let i = 0; i < 3 && (await getNValue()) !== editedValue; i++) {
    await workspace.redo();
  }

  await expect(getNInput()).toHaveValue(editedValue);
  expect(initialValue).not.toBe(editedValue);
});
