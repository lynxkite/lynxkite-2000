// Tests the basic directory operations, such as creating and deleting folders and workspaces.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

test.describe("Directory operations", () => {
  let splash: Splash;

  test.beforeAll(async ({ browser }) => {
    const page = await browser.newPage();
    splash = await Splash.open(page);
  });

  test("Create & delete workspace", async () => {
    const workspaceName = `TestWorkspace-${Date.now()}`;
    const workspace = await Workspace.empty(splash.page, workspaceName);
    await workspace.expectCurrentWorkspaceIs(workspaceName);
    // Add a box so the workspace is saved
    await workspace.addBox("Import Parquet");
    await workspace.close();
    await splash.deleteEntry(workspaceName);
    await expect(splash.getEntry(workspaceName)).not.toBeVisible();
  });

  test("Create & delete folder", async () => {
    const folderName = `TestFolder-${Date.now()}`;
    await splash.createFolder(folderName);
    await expect(splash.currentFolder()).toHaveText(`testing sandbox/${folderName}`);
    await splash.goHome();
    await splash.deleteEntry(folderName);
    await expect(splash.getEntry(folderName)).not.toBeVisible();
  });
});

test("Nested folders & workspaces operations", async ({ page }) => {
  const parentFolderName = "ParentFolder";
  const childFolderName = "ChildFolder";
  const workspaceName = "NestedWorkspace";
  const splash = await Splash.open(page);

  await splash.deleteEntryIfExists(parentFolderName);

  try {
    await test.step("Create parent folder", async () => {
      await splash.createFolder(parentFolderName);
      await expect(splash.currentFolder()).toHaveText(`testing sandbox/${parentFolderName}`);
    });

    await test.step("Create nested folder", async () => {
      await splash.createFolder(childFolderName);
      await expect(splash.currentFolder()).toHaveText(
        `testing sandbox/${parentFolderName}/${childFolderName}`,
      );
      await splash.toParent();
      await expect(splash.currentFolder()).toHaveText(`testing sandbox/${parentFolderName}`);
    });

    await test.step("Delete nested folder", async () => {
      await splash.deleteEntry(childFolderName);
      await expect(splash.getEntry(childFolderName)).not.toBeVisible();
    });

    await test.step("Create nested workspace", async () => {
      const workspace = await splash.createWorkspace(workspaceName);
      await workspace.expectCurrentWorkspaceIs(workspaceName);
      await workspace.close();
      await expect(splash.getEntry(workspaceName)).toBeVisible();
    });

    await test.step("Delete nested workspace", async () => {
      await splash.deleteEntry(workspaceName);
      await expect(splash.getEntry(workspaceName)).not.toBeVisible();
    });

    await test.step("Rename folder", async () => {
      await splash.createFolder("RenameTest");
      await splash.toParent();
      const renamedName = "RenameTest-Renamed";
      await splash.renameEntry("RenameTest", renamedName);
      await expect(splash.getEntry(renamedName)).toBeVisible();
      await splash.deleteEntry(renamedName);
    });
  } finally {
    await test.step("Delete parent folder", async () => {
      await splash.goHome();
      await splash.deleteEntryIfExists(parentFolderName);
    });
  }
});
