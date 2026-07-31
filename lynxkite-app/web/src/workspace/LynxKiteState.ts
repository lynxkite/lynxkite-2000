import { createContext } from "react";
import type { Workspace } from "../apiTypes.ts";

// The full workspace. Consumed only by hooks that read stable slices of it
// (e.g. `path`, `env`). Kept separate from the per-frame node render state so
// that reading those slices does not subscribe a component to the frequently
// re-created `workspace` object.
export const LynxKiteState = createContext({
  workspace: {} as Workspace,
  canWrite: true,
});

// Node render state that changes very rarely (only when the zoom crosses the
// iconize threshold). This is intentionally separate from `workspace`: the
// workspace object gets a new reference on every CRDT update (i.e. every drag
// frame), so if `iconized` lived on the same context every node would re-render
// on every frame, bypassing the node-level `memo`.
export const LynxKiteNodeState = createContext({
  iconized: false,
});
