// Shared testing utilities.
import { type Locator, type Page, expect } from "@playwright/test";

// Mirrors the "id" filter.
export function toId(x) {
  return x.toLowerCase().replace(/[ !?,./]/g, "-");
}

export const ROOT = "automated-tests";

export class Workspace {
  readonly page: Page;
  name: string;

  constructor(page: Page, workspaceName: string) {
    this.page = page;
    this.name = workspaceName;
  }

  // Starts with a brand new workspace.
  static async empty(page: Page, workspaceName: string): Promise<Workspace> {
    const splash = await Splash.open(page);
    return await splash.createWorkspace(workspaceName);
  }

  static async open(page: Page, workspaceName: string): Promise<Workspace> {
    const splash = await Splash.open(page);
    const ws = await splash.openWorkspace(workspaceName);
    await ws.waitForNodesToLoad();
    await ws.expectCurrentWorkspaceIs(workspaceName);
    return ws;
  }

  async getEnvs() {
    // Return all available workspace environments
    const envs = this.page.locator('select[name="workspace-env"] option');
    await expect(envs).not.toHaveCount(0);
    return await envs.allInnerTexts();
  }

  async setEnv(env: string) {
    await this.page.locator('select[name="workspace-env"]').selectOption(env);
  }

  async expectCurrentWorkspaceIs(name) {
    await expect(this.page.locator(".ws-name")).toHaveText(name);
  }

  async waitForNodesToLoad() {
    // This method should be used only on non empty workspaces
    await this.page.locator(".react-flow__nodes").waitFor();
    await this.page.locator(".react-flow__node").first().waitFor();
  }

  async addBox(boxName) {
    // TODO: Support passing box parameters.
    const allBoxes = await this.getBoxes().all();
    await this.page.locator(".ws-name").click();
    await this.page.keyboard.press("/");
    await this.page.locator(".node-search").getByText(boxName, { exact: true }).click();
    await expect(this.getBoxes()).toHaveCount(allBoxes.length + 1);
  }

  async getCatalog() {
    await this.page.locator(".ws-name").click();
    await this.page.keyboard.press("/");
    const results = this.page.locator(".node-search .matches .search-result");
    await expect(results.first()).toBeVisible();
    const catalog = await results.allInnerTexts();
    // Dismiss the catalog menu
    await this.page.keyboard.press("Escape");
    await expect(this.page.locator(".node-search")).not.toBeVisible();
    return catalog;
  }

  async selectBox(boxId: string) {
    const box = this.getBox(boxId);
    // Click on the resizer, so we don't click on any parameters by accident.
    await box.locator(".react-flow__resize-control").click();
    await expect(box).toHaveClass(/selected/);
  }

  async deleteBoxes(boxIds: string[]) {
    for (const boxId of boxIds) {
      await this.selectBox(boxId);
      await this.page.keyboard.press("Backspace");
      await expect(this.getBox(boxId)).not.toBeVisible();
    }
  }

  getBox(boxId: string) {
    return this.page.locator(`[data-id="${boxId}"]`);
  }

  getBoxes() {
    return this.page.locator(".react-flow__node");
  }

  getBoxHandle(boxId: string, pos: string) {
    return this.page.locator(`.connectable[data-nodeid="${boxId}"][data-handlepos="${pos}"]`);
  }

  async moveBox(
    boxId: string,
    offset?: { offsetX: number; offsetY: number },
    targetPosition?: { x: number; y: number },
  ) {
    // Move a box around, it is a best effort operation, the exact target position may not be reached
    const box = await this.getBox(boxId).locator(".title").boundingBox();
    if (!box) {
      return;
    }
    const boxCenterX = box.x + box.width / 2;
    const boxCenterY = box.y + box.height / 2;
    await this.page.mouse.move(boxCenterX, boxCenterY);
    await this.page.mouse.down();
    if (targetPosition) {
      await this.page.mouse.move(targetPosition.x, targetPosition.y);
    } else if (offset) {
      // Without steps the movement is too fast and the box is not dragged. The more steps,
      // the better the movement is captured
      await this.page.mouse.move(boxCenterX + offset.offsetX, boxCenterY + offset.offsetY, {
        steps: 5,
      });
    }
    await this.page.mouse.up();
  }

  async connectBoxes(sourceId: string, targetId: string) {
    const sourceHandle = this.getBoxHandle(sourceId, "right");
    const targetHandle = this.getBoxHandle(targetId, "left");
    await expect(sourceHandle).toBeVisible();
    await expect(targetHandle).toBeVisible();
    await sourceHandle.hover();
    await this.page.mouse.down();
    await expect(this.page.locator(".react-flow__connectionline")).toBeVisible();
    await targetHandle.hover();
    await this.page.mouse.up();
    await expect(
      this.page.locator(`.react-flow__edge[aria-label="Edge from ${sourceId} to ${targetId}"]`),
    ).toBeVisible();
  }

  async execute() {
    const request = this.page.waitForResponse(/api[/]execute_workspace/);
    await this.page.keyboard.press("r");
    await request;
  }

  async expectErrorFree(executionWaitTime?) {
    await expect(this.getBoxes().locator("text=⚠️").first()).not.toBeVisible();
  }

  async close() {
    await this.page.getByRole("link", { name: "close" }).click();
  }
}

export class Splash {
  page: Page;
  root: Locator;

  constructor(page) {
    this.page = page;
    this.root = page.locator("#splash");
  }

  // Opens the LynxKite directory browser in the root.
  static async open(page: Page): Promise<Splash> {
    await page.goto("/");
    await page.evaluate(() => {
      window.sessionStorage.clear();
      window.localStorage.clear();
    });
    await page.reload();
    const splash = new Splash(page);
    return splash;
  }

  workspace(name: string) {
    return this.page.getByRole("link", { name: name, exact: true });
  }

  getEntry(name: string) {
    return this.page.locator(".entry").filter({ hasText: name }).first();
  }

  async createWorkspace(name: string) {
    await this.page.getByRole("button", { name: "New workspace" }).click();
    const nameBox = this.page.locator('input[name="entryName"]');
    await nameBox.fill(name);
    await nameBox.press("Enter");
    const ws = new Workspace(this.page, name);
    await ws.setEnv("LynxKite Graph Analytics");
    return ws;
  }

  async openWorkspace(name: string) {
    await this.workspace(name).click();
    return new Workspace(this.page, name);
  }

  async createFolder(folderName: string) {
    await this.page.getByRole("button", { name: "New folder" }).click();
    const nameBox = this.page.locator('input[name="entryName"]');
    await nameBox.fill(folderName);
    await nameBox.press("Enter");
  }

  async deleteEntry(entryName: string) {
    await this.getEntry(entryName).locator("button").click();
    await this.page.reload();
  }

  currentFolder() {
    return this.page.locator(".current-folder");
  }

  async goHome() {
    await this.page.getByRole("link", { name: "home" }).click();
  }
}
