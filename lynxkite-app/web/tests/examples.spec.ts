// Test the execution of the example workspaces
import { expect, test } from "@playwright/test";
import { Workspace } from "./lynxkite";

test("LynxKite Graph Analytics example", async ({ page }) => {
  const ws = await Workspace.open(page, "NetworkX demo");
  await ws.expectErrorFree(process.env.CI ? 2000 : 1000);
});

test("Pytorch example", async ({ page }) => {
  const ws = await Workspace.open(page, "PyTorch demo");
  await ws.expectErrorFree();
});

test.fail("AIMO example", async ({ page }) => {
  // Fails because of missing OPENAI_API_KEY
  const ws = await Workspace.open(page, "AIMO");
  await ws.expectErrorFree();
});

test.fail("LynxScribe example", async ({ page }) => {
  // Fails because of missing OPENAI_API_KEY
  const ws = await Workspace.open(page, "LynxScribe demo");
  await ws.expectErrorFree();
});

test.fail("Graph RAG", async ({ page }) => {
  // Fails due to some issue with ChromaDB
  const ws = await Workspace.open(page, "Graph RAG");
  await ws.expectErrorFree(process.env.CI ? 2000 : 500);
});

test.fail("RAG chatbot app", async ({ page }) => {
  // Fail due to all operation being unknown
  const ws = await Workspace.open(page, "RAG chatbot app");
  await ws.expectErrorFree();
});

test("Airlines demo", async ({ page }) => {
  const ws = await Workspace.open(page, "Airlines demo");
  await ws.expectErrorFree(process.env.CI ? 10000 : 500);
});

test("Pillow example", async ({ page }) => {
  const ws = await Workspace.open(page, "Image processing");
  await ws.expectErrorFree();
});
