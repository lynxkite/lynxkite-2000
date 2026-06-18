// Tests file upload in the directory browser.
import { expect, test } from "@playwright/test";
import { Splash } from "./lynxkite";

test("Can upload a file via the Upload file button", async ({ page }) => {
  const splash = await Splash.open(page);
  const fileName = `upload-test-${Date.now()}.txt`;

  const fileInput = page.locator('input[type="file"]');
  const upload = page.waitForResponse(/api\/upload/);
  await fileInput.setInputFiles({
    name: fileName,
    mimeType: "text/plain",
    buffer: Buffer.from("hello from upload test"),
  });
  await upload;

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
