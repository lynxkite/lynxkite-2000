// Test the execution of the example workspaces
import { expect, test } from "@playwright/test";
import { Workspace } from "./lynxkite";

const WORKSPACES = [
  // "AIMO",
  "Airlines demo",
  "Bio Cypher demo",
  // "Graph RAG",
  "Image processing",
  // "LynxScribe demo",
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
