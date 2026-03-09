// Tests some basic operations like box creation, deletion, dragging, and copy-pasting.
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "basic_spec_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("basic_spec_test");
});

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
  await workspace.addBox("Import Parquet");
  const originalPos = await workspace.getBox("Import Parquet 1").boundingBox();
  await workspace.moveBox("Import Parquet 1", { offsetX: 100, offsetY: 100 });
  const newPos = await workspace.getBox("Import Parquet 1").boundingBox();
  // Exact position is not guaranteed, but it should have moved
  expect(newPos.x).toBeGreaterThan(originalPos.x);
  expect(newPos.y).toBeGreaterThan(originalPos.y);
});

test("Copy-paste box", async () => {
  await workspace.addBox("Import Parquet");
  await workspace.selectBox("Import Parquet 1");
  await workspace.copySelection();
  await workspace.pasteSelection();
  await expect(workspace.getBox("Import Parquet 2")).toBeVisible();
});

test("Copy-paste connected boxes", async () => {
  await workspace.addBox("Graph embedding and link prediction › Import PyKEEN dataset");
  await workspace.addBox("View tables");
  await workspace.connectBoxes("Import PyKEEN dataset 1", "View tables 1");
  await workspace.selectBoxes(["Import PyKEEN dataset 1", "View tables 1"]);
  await workspace.copySelection();
  await workspace.pasteSelection();
  await expect(workspace.getBox("Import PyKEEN dataset 2")).toBeVisible();
  const tableBox = workspace.getBox("View tables 2");
  await expect(tableBox).toBeVisible();
  await expect(tableBox.locator(".error")).not.toBeVisible();
});

test("Cut-paste box", async () => {
  await workspace.addBox("Import Parquet");
  await workspace.selectBox("Import Parquet 1");
  await workspace.cutSelection();
  await expect(workspace.getBox("Import Parquet 1")).not.toBeVisible();
  await workspace.pasteSelection();
  await expect(workspace.getBox("Import Parquet 1")).toBeVisible();
});
