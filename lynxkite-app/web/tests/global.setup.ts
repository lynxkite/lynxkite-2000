import { test as setup } from "@playwright/test";
import { Splash } from "./lynxkite";

setup("create testing sandbox", async ({ page }) => {
  const splash = await Splash.openRoot(page);
  await splash.deleteEntryIfExists("testing sandbox");
  await splash.createFolder("testing sandbox");
});
