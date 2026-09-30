// Tests file upload in the directory browser.
import { expect, test } from "@playwright/test";
import { Splash } from "./lynxkite";

test("Can upload a file via the Upload file button", async ({ page }) => {
  const splash = await Splash.open(page);
  const fileName = `upload-test-${Date.now()}.txt`;
  const contents = Buffer.from("hello from upload test");
  let finishUpload!: () => void;
  const holdUploadResponse = new Promise<void>((resolve) => {
    finishUpload = resolve;
  });
  await page.route(/api\/upload/, async (route) => {
    const response = await route.fetch();
    await holdUploadResponse;
    await route.fulfill({ response });
  });

  const fileInput = page.locator('input[type="file"]');
  const upload = page.waitForResponse(/api\/upload/);
  await fileInput.setInputFiles({
    name: fileName,
    mimeType: "text/plain",
    buffer: contents,
  });
  const uploadStatus = page.getByRole("status");
  await expect(uploadStatus).toContainText("Uploading files (0/1)");
  const progress = uploadStatus.getByRole("progressbar");
  await expect(progress).toHaveAttribute("max", contents.length.toString());
  await expect(progress).toHaveClass(/progress-info w-full/);

  finishUpload();
  await upload;
  await expect(uploadStatus).not.toBeVisible();

  await expect(splash.getEntry(fileName)).toBeVisible({ timeout: 10000 });

  await splash.deleteEntry(fileName);
  await expect(splash.getEntry(fileName)).not.toBeVisible();
});

test("Can upload multiple files at once", async ({ page }) => {
  const splash = await Splash.open(page);
  const file1 = `upload-test-multi-1-${Date.now()}.txt`;
  const file2 = `upload-test-multi-2-${Date.now()}.txt`;

  const fileInput = page.locator('input[type="file"]');
  const uploads = Promise.all([
    page.waitForResponse(/api\/upload/),
    page.waitForResponse(/api\/upload/),
  ]);
  await fileInput.setInputFiles([
    { name: file1, mimeType: "text/plain", buffer: Buffer.from("file one") },
    { name: file2, mimeType: "text/plain", buffer: Buffer.from("file two") },
  ]);
  await uploads;

  await expect(splash.getEntry(file1)).toBeVisible({ timeout: 10000 });
  await expect(splash.getEntry(file2)).toBeVisible({ timeout: 10000 });

  await splash.deleteEntry(file1);
  await splash.deleteEntry(file2);
  await expect(splash.getEntry(file1)).not.toBeVisible();
  await expect(splash.getEntry(file2)).not.toBeVisible();
});

test("Can drop a folder with nested files and empty subfolders", async ({ page }) => {
  const splash = await Splash.open(page);
  const folderName = `upload-folder-${Date.now()}`;

  await page.evaluate((name) => {
    const file = new File(["nested upload"], "hello.txt", { type: "text/plain" });
    const fileEntry = {
      name: file.name,
      isFile: true,
      file: (callback: (file: File) => void) => {
        callback(file);
      },
    } as FileSystemFileEntry;
    const folderEntry = (entryName: string, children: FileSystemEntry[]) =>
      ({
        name: entryName,
        isDirectory: true,
        createReader: () => ({
          readEntries: (callback: (entries: FileSystemEntry[]) => void) => {
            callback(children.splice(0, 1));
          },
        }),
      }) as FileSystemDirectoryEntry;
    const root = folderEntry(name, [folderEntry("nested", [fileEntry]), folderEntry("empty", [])]);
    const event = new DragEvent("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", {
      value: { items: [{ kind: "file", webkitGetAsEntry: () => root }], files: [] },
    });
    document.querySelector(".entry-drop-zone")!.dispatchEvent(event);
  }, folderName);

  await expect(splash.getEntry(folderName)).toBeVisible({ timeout: 10000 });
  await splash.getEntry(folderName).click();
  await expect(splash.getEntry("nested")).toBeVisible();
  await expect(splash.getEntry("empty")).toBeVisible();
  await splash.getEntry("nested").click();
  await expect(splash.getEntry("hello.txt")).toBeVisible();
});
