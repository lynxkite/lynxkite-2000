import { createContext } from "react";
import { Workspace } from "../apiTypes.ts";

export const LynxKiteState = createContext({ workspace: {} as Workspace });
