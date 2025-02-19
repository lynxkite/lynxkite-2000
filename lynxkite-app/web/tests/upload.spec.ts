import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
// Test uploading a file in an import box.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(
    await browser.newPage(),
    "upload_spec_test",
  );
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("upload_spec_test");
});

test("can upload and import a simple CSV", async () => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const csvPath = join(__dirname, "data", "upload_test.csv");

  await workspace.addBox("Import CSV");
  const csvBox = workspace.getBox("Import CSV 1");
  const filenameInput = csvBox.locator("input.input-bordered").nth(0);
  await filenameInput.click();
  await filenameInput.fill(csvPath);
  await filenameInput.press("Enter");

  await workspace.addBox("View tables");
  const tableBox = workspace.getBox("View tables 1");
  await workspace.connectBoxes("Import CSV 1", "View tables 1");

  const tableRows = tableBox.locator("table tbody tr");
  await expect(tableRows).toHaveCount(4);
});
