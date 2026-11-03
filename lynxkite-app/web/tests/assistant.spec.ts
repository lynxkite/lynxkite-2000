// A new workspace has no path in the CRDT. The assistant must still send the file path.
import { expect, test } from "@playwright/test";
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

test("sends the workspace path for a new workspace", async () => {
  const page = workspace.page;
  const config = await (await page.request.get("/api/config")).json();
  test.skip(!config.assistant_available, "assistant plugin not installed");

  let requestedWorkspace: string | undefined;
  await page.route("**/api/assistant/stream", async (route) => {
    requestedWorkspace = route.request().postDataJSON().workspace;
    await route.fulfill({
      status: 200,
      contentType: "text/plain; charset=utf-8",
      body: "ok",
    });
  });

  await page.locator('[data-tooltip-html*="Toggle assistant"] button').click();
  const editor = page.locator(".assistant-editor");
  await expect(editor).toBeVisible();
  await editor.click();
  await page.keyboard.type("hello");
  await page.getByRole("button", { name: "Send" }).click();

  const expectedPath = decodeURIComponent(new URL(page.url()).pathname)
    .replace(/^\/edit\//, "")
    .replace(/\/$/, "");
  await expect.poll(() => requestedWorkspace).toBe(expectedPath);
  expect(expectedPath.endsWith("assistant_path_test.lynxkite.json")).toBe(true);
});
