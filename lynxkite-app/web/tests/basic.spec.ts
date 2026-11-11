// Tests some basic operations like box creation, deletion, and dragging.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "basic_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  await splash.deleteEntry("basic_spec_test");
});

for (const delay of [0, 350]) {
  test(`Background click dismisses node search without reopening (delay ${delay} ms)`, async () => {
    await workspace.getCatalog();
    await workspace.page.keyboard.press("/");
    const search = workspace.page.locator(".node-search");
    await expect(search).toBeVisible();

    const background = workspace.page.locator(".react-flow__pane");
    await background.click({ position: { x: 20, y: 20 }, delay });
    await expect(search).not.toBeVisible();

    await background.click({ position: { x: 100, y: 100 } });
    await expect(search).toBeVisible();
  });
}

test("Box creation & deletion per env", async () => {
  const envs = await workspace.getEnvs();
  for (const env of envs) {
    await workspace.setEnv(env);
    const catalog = (await workspace.getCatalog()).filter((box) => box !== "Comment");
    expect(catalog).not.toHaveLength(0);
    const op = catalog[0];
    await workspace.addBox(op);
    await expect(workspace.getBox(`${op} 1`)).toBeVisible();
    await workspace.deleteBoxes([`${op} 1`]);
    await expect(workspace.getBox(`${op} 1`)).not.toBeVisible();
  }
});

test("Delete multi-handle boxes", async () => {
  await workspace.addBox("NetworkX › Algorithms › Link analysis › PageRank alg › PageRank");
  await workspace.deleteBoxes(["PageRank 1"]);
  await expect(workspace.getBox("PageRank 1")).not.toBeVisible();
});

test("Drag box", async () => {
  await workspace.addBox("File operations › Import Parquet");
  const originalPos = await workspace.getBox("Import Parquet 1").boundingBox();
  await workspace.moveBox("Import Parquet 1", { offsetX: 100, offsetY: 100 });
  const newPos = await workspace.getBox("Import Parquet 1").boundingBox();
  // Exact position is not guaranteed, but it should have moved
  expect(newPos.x).toBeGreaterThan(originalPos.x);
  expect(newPos.y).toBeGreaterThan(originalPos.y);
});

test("Delete selection", async () => {
  await workspace.addBox("Table operations › Enter table data");
  await workspace.addBox("View tables");
  await workspace.connectBoxes("Enter table data 1", "View tables 1");
  await workspace.moveBox("Enter table data 1", { offsetX: 100, offsetY: 100 });
  await workspace.addBox("Table operations › Enter table data");
  await workspace.addBox("View tables");
  await workspace.connectBoxes("Enter table data 2", "View tables 2");
  await workspace.moveBox("Enter table data 2", { offsetX: 100, offsetY: 100 });
  await workspace.selectBoxes(["Enter table data 1", "View tables 1"]);
  await workspace.page.keyboard.press("Backspace");
  await expect(workspace.getBox("Enter table data 1")).not.toBeVisible();
  await expect(workspace.getBox("View tables 1")).not.toBeVisible();
  await expect(workspace.getBox("Enter table data 2")).toBeVisible({ timeout: 500 });
  await expect(workspace.getBox("View tables 2")).toBeVisible({ timeout: 500 });
});
