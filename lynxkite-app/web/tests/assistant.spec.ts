// Tests for the assistant panel. The assistant API is mocked.
import { expect, type Page, test } from "@playwright/test";
import { Splash, Workspace } from "./lynxkite";

let workspace: Workspace;

test.beforeEach(async ({ browser }) => {
  workspace = await Workspace.empty(await browser.newPage(), "assistant_path_test");
});

test.afterEach(async () => {
  await workspace.close();
  const splash = await new Splash(workspace.page);
  await splash.deleteEntry("assistant_path_test");
});

async function getConfig(page: Page) {
  return await (await page.request.get("/api/config")).json();
}

// Returns the bodies of the requests sent to the assistant.
async function mockAssistant(page: Page) {
  const requests: any[] = [];
  await page.route("**/api/assistant/stream", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: "text/plain; charset=utf-8",
      body: "ok",
    });
  });
  return requests;
}

async function openAssistant(page: Page) {
  await page.locator('[data-tooltip-html*="Toggle assistant"] button').click();
  await expect(page.locator(".assistant-editor")).toBeVisible();
}

async function sendPrompt(page: Page, prompt: string) {
  await page.locator(".assistant-editor").click();
  await page.keyboard.type(prompt);
  await page.getByRole("button", { name: "Send" }).click();
}

// A new workspace has no path in the CRDT. The assistant must still send the file path.
test("sends the workspace path for a new workspace", async () => {
  const page = workspace.page;
  test.skip(!(await getConfig(page)).assistant_available, "assistant plugin not installed");
  const requests = await mockAssistant(page);
  await openAssistant(page);
  await sendPrompt(page, "hello");

  const expectedPath = decodeURIComponent(new URL(page.url()).pathname)
    .replace(/^\/edit\//, "")
    .replace(/\/$/, "");
  await expect.poll(() => requests.at(-1)?.workspace).toBe(expectedPath);
  expect(expectedPath.endsWith("assistant_path_test.lynxkite")).toBe(true);
});

test("sends the selected mode", async () => {
  const page = workspace.page;
  test.skip(!(await getConfig(page)).assistant_available, "assistant plugin not installed");
  const requests = await mockAssistant(page);
  await openAssistant(page);
  const ask = page.getByRole("radio", { name: "Ask" });
  const agent = page.getByRole("radio", { name: "Agent" });
  await expect(agent).toBeChecked();
  await expect(agent).toBeEnabled();
  await sendPrompt(page, "add a box");
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].mode).toBe("agent");

  await ask.click();
  await expect(ask).toBeChecked();
  await expect(page.locator(".assistant-editor")).toHaveAttribute(
    "data-placeholder",
    /cannot make changes/,
  );
  await sendPrompt(page, "what does this do?");
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1].mode).toBe("ask");
});

test("users without write access can only use Ask mode", async () => {
  const page = workspace.page;
  const config = await getConfig(page);
  test.skip(!config.assistant_available, "assistant plugin not installed");
  // Pretend that authentication is on and the user can only read this workspace.
  await page.route("**/api/config", (route) =>
    route.fulfill({
      json: {
        ...config,
        authentication_issuer: "https://auth.invalid",
        authentication_audience: "lynxkite",
      },
    }),
  );
  await page.route("**/api/permissions?*", (route) =>
    route.fulfill({ json: { read: true, write: false } }),
  );
  const requests = await mockAssistant(page);
  try {
    await page.reload();
    await expect(page.getByText("Read-only")).toBeVisible();
    await openAssistant(page);
    await expect(page.getByRole("radio", { name: "Ask" })).toBeChecked();
    await expect(page.getByRole("radio", { name: "Agent" })).toBeDisabled();
    await sendPrompt(page, "what does this do?");
    await expect.poll(() => requests.length).toBe(1);
    expect(requests[0].mode).toBe("ask");
  } finally {
    // Restore the real config, so that the workspace can be deleted.
    await page.unrouteAll();
    await page.reload();
  }
});
