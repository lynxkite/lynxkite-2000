// Test the execution of the example workspaces
import { test, expect } from '@playwright/test';
import { Workspace } from './lynxkite';


test('LynxKite Graph Analytics example', async ({ page }) => {
    const ws = await Workspace.open(page, "NetworkX demo");
    expect(await ws.isErrorFree(process.env.CI? 2000: 1000)).toBeTruthy();
});


test('Pytorch example', async ({ page }) => {
    const ws = await Workspace.open(page, "PyTorch demo");
    expect(await ws.isErrorFree()).toBeTruthy();
});


test.fail('AIMO example', async ({ page }) => {
    // Fails because of missing OPENAI_API_KEY
    const ws = await Workspace.open(page, "AIMO");
    expect(await ws.isErrorFree()).toBeTruthy();
});

test.fail('LynxScribe example', async ({ page }) => {
    // Fails because of missing OPENAI_API_KEY
    const ws = await Workspace.open(page, "LynxScribe demo");
    expect(await ws.isErrorFree()).toBeTruthy();
});


test.fail('Graph RAG', async ({ page }) => {
    // Fails due to some issue with ChromaDB
    const ws = await Workspace.open(page, "Graph RAG");
    expect(await ws.isErrorFree(process.env.CI? 2000: 500)).toBeTruthy();
});


test.fail('RAG chatbot app', async ({ page }) => {
    // Fail due to all operation being unknown
    const ws = await Workspace.open(page, "RAG chatbot app");
    expect(await ws.isErrorFree()).toBeTruthy();
});


test.fail('night demo', async ({ page }) => {
    // airlines.graphml file not found
    // requires cugraph
    const ws = await Workspace.open(page, "night demo");
    expect(await ws.isErrorFree(process.env.CI? 10000: 500)).toBeTruthy();
});


test('Pillow example', async ({ page }) => {
    const ws = await Workspace.open(page, "Image processing");
    expect(await ws.isErrorFree()).toBeTruthy();
});

