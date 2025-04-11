// Tests the basic directory operations, such as creating and deleting folders and workspaces.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

test.describe("Directory operations", () => {
  let splash: Splash;

  test.beforeAll(async ({ browser }) => {
    const page = await browser.newPage();
    // To make deletion confirmation dialog to be automatically accepted
    page.on("dialog", async (dialog) => {
      await dialog.accept();
    });
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
    await expect(splash.currentFolder()).toHaveText(folderName);
    await splash.goHome();
    await splash.deleteEntry(folderName);
    await expect(splash.getEntry(folderName)).not.toBeVisible();
  });
});

test.describe
  .serial("Nested folders & workspaces operations", () => {
    let splash: Splash;

    test.beforeEach(() => {
      // Nested navigation doesn't work yet
      test.skip();
    });

    test.beforeAll(async ({ browser }) => {
      const page = await browser.newPage();
      // To make deletion confirmation dialog to be automatically accepted
      page.on("dialog", async (dialog) => {
        await dialog.accept();
      });
      splash = await Splash.open(page);
      await splash.createFolder("TestFolder");
    });

    test.afterAll(async () => {
      //cleanup
      test.skip();
      await splash.goHome();
      await splash.deleteEntry("TestFolder");
    });

    test("Create nested folder", async () => {
      await splash.createFolder("TestFolder2");
      await expect(splash.currentFolder()).toHaveText("TestFolder2");
      await splash.toParent();
    });

    test("Delete nested folder", async () => {
      await splash.deleteEntry("TestFolder2");
      await expect(splash.getEntry("TestFolder2")).not.toBeVisible();
    });

    test("Create nested workspace", async () => {
      const workspace = splash.createWorkspace("TestWorkspace");
      await workspace.expectCurrentWorkspaceIs("TestWorkspace");
      await workspace.close();
    });

    test("Delete nested workspace", async () => {
      await splash.deleteEntry("TestWorkspace");
      await expect(splash.getEntry("TestWorkspace")).not.toBeVisible();
    });
  });
