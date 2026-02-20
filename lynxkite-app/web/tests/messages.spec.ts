// Tests message functionality
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "message_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("message_spec_test");
});

test("message shows up", async () => {
  // Test that the message is displayed when a message is printed.
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  const messageBox = workspace.getBox("Import PyKEEN dataset 1");
  await expect(messageBox.locator(".node-message")).toHaveText(
    "Dataset contains 14 nodes, 55 relations and 1992 edges in total.",
  );
});

test("message does not propagate to downstream boxes", async () => {
  // Test that the message is not displayed in downstream boxes.
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  const graphBox = workspace.getBox("Import PyKEEN dataset 1");
  await expect(graphBox.locator(".node-message")).toHaveText(
    "Dataset contains 14 nodes, 55 relations and 1992 edges in total.",
  );

  await workspace.addBox("View tables");
  await workspace.connectBoxes("Import PyKEEN dataset 1", "View tables 1");
  const inputBox = workspace.getBox("View tables 1");
  await expect(inputBox.locator(".node-message")).not.toBeVisible();
});

test("message caches correctly", async () => {
  // Test that the message is still displayed after execution, when the box is cached.
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  const graphBox = workspace.getBox("Import PyKEEN dataset 1");
  await expect(graphBox.locator(".node-message")).toHaveText(
    "Dataset contains 14 nodes, 55 relations and 1992 edges in total.",
  );

  await workspace.execute();
  await expect(graphBox.locator(".node-message")).toHaveText(
    "Dataset contains 14 nodes, 55 relations and 1992 edges in total.",
  );
});
