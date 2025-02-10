// Tests some basic operations like box creation, deletion, and dragging.
import { test, expect } from '@playwright/test';
import { Workspace } from './lynxkite';


let workspace: Workspace;


test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage());
  await workspace.setEnv('PyTorch model'); // Workaround until we fix the default environment
  await workspace.setEnv('LynxKite Graph Analytics');
});


test('Box creation & deletion per env', async () => {
  const envs = await workspace.getEnvs();
  for(const env of envs) {
    await workspace.setEnv(env);
    const catalog = await workspace.getCatalog();
    expect(catalog).not.toHaveLength(0);
    const op = catalog[0];
    await workspace.addBox(op);
    await expect(workspace.getBox(`${op} 1`)).toBeVisible();
    await workspace.deleteBoxes([`${op} 1`]);
    await expect(workspace.getBox(`${op} 1`)).not.toBeVisible();
  }
});


test('Delete multi-handle boxes', async () => {
  await workspace.addBox('Compute PageRank');
  await workspace.deleteBoxes(['Compute PageRank 1']);
  await expect(workspace.getBox('Compute PageRank 1')).not.toBeVisible();
});


test ('Drag box', async () => {
  await workspace.addBox('Import Parquet');
  const originalPos = await workspace.getBox('Import Parquet 1').boundingBox();
  await workspace.moveBox('Import Parquet 1', {offsetX: 100, offsetY: 100});
  const newPos = await workspace.getBox('Import Parquet 1').boundingBox();
  // Exact position is not guaranteed, but it should have moved
  expect(newPos.x).toBeGreaterThan(originalPos.x);
  expect(newPos.y).toBeGreaterThan(originalPos.y);
});



