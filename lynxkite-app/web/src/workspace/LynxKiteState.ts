import { createContext } from "react";
import type { Workspace } from "../apiTypes.ts";

export const LynxKiteState = createContext({
  workspace: {} as Workspace,
  canWrite: true,
});

export const LynxKiteNodeState = createContext({
  iconized: false,
});
