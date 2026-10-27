// Tests the sidebar search, favorites and recently opened lists.
import { expect, type Page, test } from "@playwright/test";
import { Splash } from "./lynxkite";

const SANDBOX = "testing sandbox";

function sidebarSection(page: Page, title: string) {
  return page.locator(".sidebar-section").filter({ hasText: title });
}

// Sidebar items carry the full path as their title, so this matches exactly one entry.
function sidebarItem(page: Page, sectionTitle: string, path: string) {
  return sidebarSection(page, sectionTitle).locator(`.sidebar-list-item[title="${path}"]`);
}

test("Search finds a folder and opens it", async ({ page }) => {
  const folderName = `SidebarSearch-${Date.now()}`;
  const splash = await Splash.open(page);
  await splash.createFolder(folderName);
  try {
    await page.getByPlaceholder("Search files...").fill(folderName);
    const result = page.locator(`.sidebar-search-results a[title="${SANDBOX}/${folderName}"]`);
    await expect(result).toBeVisible();
    await result.click();
    await expect(splash.currentFolder()).toHaveText(`${SANDBOX}/${folderName}`);
    // Picking a result clears the search.
    await expect(page.getByPlaceholder("Search files...")).toHaveValue("");
  } finally {
    await splash.toParent();
    await expect(splash.currentFolder()).toHaveText(SANDBOX);
    await splash.deleteEntry(folderName);
  }
});

test("Search shows a message when nothing matches", async ({ page }) => {
  await Splash.open(page);
  await page.getByPlaceholder("Search files...").fill(`no-such-file-${Date.now()}`);
  await expect(page.locator(".sidebar-search-results")).toContainText("No matches");
});

test("Favorites follow renames and deletes", async ({ page }) => {
  const folderName = `SidebarFavorite-${Date.now()}`;
  const renamedName = `${folderName}-Renamed`;
  const splash = await Splash.open(page);
  await splash.createFolder(folderName);
  try {
    await page.getByRole("button", { name: `Toggle favorite for ${folderName}` }).click();
    await expect(sidebarItem(page, "Favorites", `${SANDBOX}/${folderName}`)).toBeVisible();

    await splash.renameEntry(folderName, renamedName);
    await expect(sidebarItem(page, "Favorites", `${SANDBOX}/${folderName}`)).toHaveCount(0);
    await expect(sidebarItem(page, "Favorites", `${SANDBOX}/${renamedName}`)).toBeVisible();

    await splash.deleteEntry(renamedName);
    await expect(sidebarItem(page, "Favorites", `${SANDBOX}/${renamedName}`)).toHaveCount(0);
  } finally {
    await splash.deleteEntryIfExists(folderName);
    await splash.deleteEntryIfExists(renamedName);
  }
});

test("Recently opened workspaces are listed and removed on delete", async ({ page }) => {
  const workspaceName = `SidebarRecent-${Date.now()}`;
  const path = `${SANDBOX}/${workspaceName}.lynxkite.json`;
  const splash = await Splash.open(page);
  try {
    const workspace = await splash.createWorkspace(workspaceName);
    await workspace.close();
    await expect(sidebarItem(page, "Recent", path)).toBeVisible();

    await splash.deleteEntry(workspaceName);
    await expect(sidebarItem(page, "Recent", path)).toHaveCount(0);
  } finally {
    await splash.deleteEntryIfExists(workspaceName);
  }
});
