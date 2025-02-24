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
  static async empty(page: Page, workspaceName?: string): Promise<Workspace> {
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
    //TODO: Support passing box parameters (id, position, etc.)
    const allBoxes = await this.getBoxes();
    if (allBoxes) {
      // Avoid overlapping with existing nodes
      const numNodes = allBoxes.length;
      await this.page.mouse.wheel(0, numNodes * 500);
    }

    // Some x,y offset, otherwise the box handle may fall outside the viewport.
    await this.page
      .locator(".react-flow__pane")
      .click({ position: { x: 20, y: 20 } });
    await this.page.locator(".node-search").getByText(boxName).click();
    await this.page.keyboard.press("Escape");
    // Workaround to wait for the deselection animation after choosing a box. Otherwise, the next box will not be added.
    await new Promise((resolve) => setTimeout(resolve, 200));
  }

  async getCatalog() {
    await this.page.locator(".react-flow__pane").click();
    const catalog = await this.page
      .locator(".node-search .matches .search-result")
      .allInnerTexts();
    // Dismiss the catalog menu
    await this.page.keyboard.press("Escape");
    await new Promise((resolve) => setTimeout(resolve, 200));
    return catalog;
  }

  async deleteBoxes(boxIds: string[]) {
    for (const boxId of boxIds) {
      await this.getBoxHandle(boxId).first().click();
      await this.page.keyboard.press("Backspace");
    }
  }

  getBox(boxId: string) {
    return this.page.locator(`[data-id="${boxId}"]`);
  }

  getBoxes() {
    return this.page.locator(".react-flow__node").all();
  }

  getBoxHandle(boxId: string) {
    return this.page.getByTestId(boxId);
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
      await this.page.mouse.move(
        boxCenterX + offset.offsetX,
        boxCenterY + offset.offsetY,
        { steps: 5 },
      );
    }
    await this.page.mouse.up();
  }

  async connectBoxes(sourceId: string, targetId: string) {
    const sourceHandle = this.getBoxHandle(sourceId);
    const targetHandle = this.getBoxHandle(targetId);
    await sourceHandle.hover();
    await this.page.mouse.down();
    await targetHandle.hover();
    await this.page.mouse.up();
  }

  async isErrorFree(executionWaitTime?): Promise<boolean> {
    // TODO: Workaround, to account for workspace execution. Once
    // we have a load indicator we can use that instead.
    await new Promise((resolve) =>
      setTimeout(resolve, executionWaitTime ? executionWaitTime : 500),
    );
    const boxes = await this.getBoxes();
    for (const box of boxes) {
      if (await box.locator(".error").isVisible()) {
        return false;
      }
    }
    return true;
  }

  async close() {
    await this.page.locator('a[href="/dir/"]').click();
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
    return this.page.getByRole("link", { name: name });
  }

  getEntry(name: string) {
    return this.page.locator(".entry").filter({ hasText: name }).first();
  }

  async createWorkspace(name?: string) {
    await this.page.getByRole("button", { name: "New workspace" }).click();
    await this.page.locator('input[name="workspaceName"]').click();
    let workspaceName: string;
    if (name) {
      workspaceName = name;
      await this.page.locator('input[name="workspaceName"]').fill(name);
    } else {
      workspaceName = await this.page
        .locator('input[name="workspaceName"]')
        .inputValue();
    }
    await this.page.locator('input[name="workspaceName"]').press("Enter");
    const ws = new Workspace(this.page, workspaceName);
    await ws.setEnv("LynxKite Graph Analytics");
    return ws;
  }

  async openWorkspace(name: string) {
    await this.workspace(name).click();
    return new Workspace(this.page, name);
  }

  async createFolder(folderName?: string) {
    await this.page.getByRole("button", { name: "New folder" }).click();
    await this.page.locator('input[name="folderName"]').click();
    if (folderName) {
      await this.page.locator('input[name="folderName"]').fill(folderName);
    }
    await this.page.locator('input[name="folderName"]').press("Enter");
  }

  async deleteEntry(entryName: string) {
    await this.getEntry(entryName).locator("button").click();
    await this.page.reload();
  }

  currentFolder() {
    return this.page.locator(".current-folder");
  }

  async goHome() {
    await this.page.locator('a[href="/dir/"]').click();
  }
}
