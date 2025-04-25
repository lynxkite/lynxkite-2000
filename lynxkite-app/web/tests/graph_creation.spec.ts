// Test the graph creation box in LynxKite
import { expect, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "graph_creation_spec_test");
  await workspace.addBox("NX › Scale-Free Graph");
  await workspace.getBox("NX › Scale-Free Graph 1").getByLabel("n", { exact: true }).fill("10");
  await workspace.addBox("Create graph");
  await workspace.connectBoxes("NX › Scale-Free Graph 1", "Create graph 1");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  splash.page.on("dialog", async (dialog) => {
    await dialog.accept();
  });
  await splash.deleteEntry("graph_creation_spec_test");
});

test("Tables are displayed in the Graph creation box", async () => {
  const graphBox = await workspace.getBox("Create graph 1");
  const nodesTableHeader = graphBox.locator(".graph-tables .df-head", {
    hasText: "nodes",
  });
  const edgesTableHeader = graphBox.locator(".graph-tables .df-head", {
    hasText: "edges",
  });
  const nodesTable = nodesTableHeader.locator("xpath=//following-sibling::table[1]");
  const edgesTable = edgesTableHeader.locator("xpath=//following-sibling::table[1]");
  await expect(nodesTableHeader).toBeVisible();
  await expect(edgesTableHeader).toBeVisible();
  await expect(nodesTable).not.toBeVisible();
  await expect(edgesTable).not.toBeVisible();
  await nodesTableHeader.click();
  await expect(nodesTable).toBeVisible();
  await edgesTableHeader.click();
  await expect(edgesTable).toBeVisible();
});

test("Adding and removing relationships", async () => {
  const graphBox = await workspace.getBox("Create graph 1");
  const addRelationshipButton = await graphBox.locator(".add-relationship-button");
  await addRelationshipButton.click();
  const formData: Record<string, string> = {
    name: "relation_1",
    df: "edges",
    source_column: "source_id",
    target_column: "target_id",
    source_table: "nodes",
    target_table: "nodes",
    source_key: "node_id",
    target_key: "node_id",
  };
  for (const [fieldName, fieldValue] of Object.entries(formData)) {
    const inputField = await graphBox.locator(
      `.graph-relation-attributes input[name="${fieldName}"]`,
    );
    await inputField.fill(fieldValue);
  }
  await graphBox.locator(".submit-relationship-button").click();
  // check that the relationship has been saved in the backend
  await workspace.page.reload();
  const graphBoxAfterReload = await workspace.getBox("Create graph 1");
  const relationHeader = await graphBoxAfterReload.locator(".graph-relations .df-head", {
    hasText: "relation_1",
  });
  await expect(relationHeader).toBeVisible();
  await relationHeader.locator("button").click(); // Delete the relationship
  await expect(relationHeader).not.toBeVisible();
});

test("Output of the box is a bundle", async () => {
  await workspace.addBox("View tables");
  const tableView = await workspace.getBox("View tables 1");
  await workspace.connectBoxes("Create graph 1", "View tables 1");
  const nodesTableHeader = await tableView.locator(".df-head", {
    hasText: "nodes",
  });
  const edgesTableHeader = await tableView.locator(".df-head", {
    hasText: "edges",
  });
  await expect(nodesTableHeader).toBeVisible();
  await expect(edgesTableHeader).toBeVisible();
});
