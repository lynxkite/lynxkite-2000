// Test uploading a file in an import box.
import { test, expect } from '@playwright/test';
import { Workspace } from './lynxkite';
import { join, dirname  } from 'path';
import { fileURLToPath } from 'url';


let workspace: Workspace;


test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage());
  await workspace.setEnv('PyTorch model'); // Workaround until we fix the default environment
  await workspace.setEnv('LynxKite Graph Analytics');
});


test('can upload and import a simple CSV', async () => {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = dirname(__filename);
    const csvPath = join(__dirname, 'data', 'upload_test.csv');

    await workspace.addBox('Import CSV');
    const csvBox =  workspace.getBox('Import CSV 1');
    const filenameInput = csvBox.locator('input.input-bordered').nth(0);
    await filenameInput.click();
    await filenameInput.fill(csvPath);
    await filenameInput.press('Enter');
    
    await workspace.addBox('View tables');
    const tableBox = workspace.getBox('View tables 1');
    await workspace.connectBoxes('Import CSV 1', 'View tables 1');

    const tableRows = tableBox.locator('table tbody tr');
    await expect(tableRows).toHaveCount(4);
});
