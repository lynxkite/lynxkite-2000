import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
// Test uploading a file in an import box.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "import_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("import_spec_test");
});

async function validateImport(workspace: Workspace, fileName: string, fileFormat: string) {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const filePath = join(__dirname, "data", fileName);

  await workspace.addBox("Import file");
  const importBox = workspace.getBox("Import file 1");
  const fileFormatSelect = await importBox
    .locator("label.param", { hasText: "file format" })
    .locator("select");
  await fileFormatSelect.selectOption(fileFormat);
  const filePathInput = await importBox
    .locator("label.param", { hasText: "file path" })
    .locator("input");
  await filePathInput.click();
  await filePathInput.fill(filePath);
  await filePathInput.press("Enter");
  const tableNameInput = await importBox
    .locator("label.param", { hasText: "table name" })
    .locator("input");
  await tableNameInput.click();
  await tableNameInput.fill("table");
  await tableNameInput.press("Enter");

  await workspace.addBox("View tables");
  const tableBox = workspace.getBox("View tables 1");
  await workspace.connectBoxes("Import file 1", "View tables 1");

  const tableRows = tableBox.locator("table tbody tr");
  await expect(tableRows).toHaveCount(4);
}

test("Can import a CSV file", async () => {
  await validateImport(workspace, "import_test.csv", "csv");
});

test("Can import a parquet file", async () => {
  await validateImport(workspace, "import_test.parquet", "parquet");
});

test("Can import a JSON file", async () => {
  await validateImport(workspace, "import_test.json", "json");
});

// Needs openpyxl. It's the same code as the other formats, so not worth installing it in CI.
test.skip("Can import an Excel file", async () => {
  await validateImport(workspace, "import_test.xlsx", "excel");
});
