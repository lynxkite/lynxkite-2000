// Test the execution of the example workspaces
import { expect, test } from "@playwright/test";
import { Workspace } from "./lynxkite";

const WORKSPACES = [
  "Airlines demo",
  // "Graph RAG",
  "Image processing",
  "NetworkX demo",
  "Model use",
];

for (const name of WORKSPACES) {
  test(name, async ({ page }) => {
    const ws = await Workspace.open(page, name);
    await ws.execute();
    await ws.expectErrorFree();
  });
}
