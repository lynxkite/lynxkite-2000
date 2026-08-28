// Defines a basic placeholder workspace. Used for static exports, which run without a backend.
import { applyEdgeChanges, applyNodeChanges, type Edge, type Node } from "@xyflow/react";
import { useEffect, useState } from "react";
import type { Workspace as WorkspaceType } from "../apiTypes.ts";
import { getStaticWorkspaceConfig } from "../common.ts";
import { type CRDTWorkspace, EMPTY_WORKSPACE } from "./crdt.ts";

function workspaceToFlowState(ws: WorkspaceType, oldNodes: Node[] = []) {
  const oldNodesById = Object.fromEntries(oldNodes.map((n) => [n.id, n]));
  const feNodes = (ws.nodes || []).map((node) => {
    if (node.type !== "node_group") {
      node.dragHandle = ".drag-handle";
    }
    const mergedNode = { ...oldNodesById[node.id], ...node } as Node;
    if (node.parentId === undefined) {
      delete mergedNode.parentId;
    }
    if (node.extent === undefined) {
      delete mergedNode.extent;
    }
    if (node.width != null && node.height != null) {
      mergedNode.measured = { width: node.width, height: node.height };
    }
    return mergedNode;
  });
  return { feNodes, feEdges: (ws.edges || []) as Edge[] };
}

export function useStaticWorkspace() {
  const config = getStaticWorkspaceConfig();
  const [state, setState] = useState<CRDTWorkspace>(() => ({ ...EMPTY_WORKSPACE }));

  useEffect(() => {
    if (!config) return;
    fetch(config.workspace)
      .then((res) => res.json())
      .then((ws: WorkspaceType) => {
        const withPath = { ...ws, path: config.workspace };
        setState((oldState) => ({
          ...EMPTY_WORKSPACE,
          ws: withPath,
          ...workspaceToFlowState(withPath, oldState.feNodes),
        }));
      });
  }, [config]);

  return {
    ...state,
    onFENodesChange: (changes: any[]) => {
      setState((oldState) => ({
        ...oldState,
        feNodes: applyNodeChanges(
          changes.filter((ch) => ch.type === "select"),
          oldState.feNodes,
        ),
      }));
    },
    onFEEdgesChange: (changes: any[]) => {
      setState((oldState) => ({
        ...oldState,
        feEdges: applyEdgeChanges(
          changes.filter((ch) => ch.type === "select"),
          oldState.feEdges,
        ),
      }));
    },
  };
}
