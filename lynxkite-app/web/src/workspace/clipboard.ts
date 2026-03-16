import type { Edge, Node, XYPosition } from "@xyflow/react";
import { Map as YMap } from "yjs";
import type { Workspace } from "../apiTypes.ts";
import { nodeToYMap } from "./crdt.ts";

let localClipboard = "";

export function writeClipboard(text: string) {
  localClipboard = text;
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
  }
}

export async function readClipboard() {
  if (navigator.clipboard?.readText) {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        return text;
      }
    } catch {
      // Fall back to in-memory clipboard.
    }
  }
  return localClipboard;
}

export function copySelection(nodes: Node[], edges: Edge[], workspaceData: Workspace) {
  const selectedNodes = nodes.filter((n) => n.selected);
  const selectedNodeIds = new Set(selectedNodes.map((node) => node.id));
  const selectedEdges = edges.filter(
    (edge) => selectedNodeIds.has(edge.source) && selectedNodeIds.has(edge.target),
  );
  const data = {
    edges: selectedEdges,
    env: workspaceData?.env ?? "LynxKite Graph Analytics",
    execution_options: workspaceData?.execution_options ?? {},
    nodes: selectedNodes,
    paused: workspaceData?.paused ?? false,
  };
  writeClipboard(JSON.stringify(data));
}

export async function pasteSelection(
  crdt: any,
  cursorScreenPos: { current: XYPosition | null },
  reactFlow: any,
  setMessage: (msg: string | null) => void,
) {
  const text = await readClipboard();
  if (!text) {
    setMessage("Clipboard is empty.");
    return;
  }
  try {
    const data = JSON.parse(text);
    if (!Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
      setMessage("Clipboard does not contain valid node data.");
      return;
    }
    const copiedNodes: Node[] = data.nodes;
    const copiedEdges: Edge[] = data.edges;
    if (copiedNodes.length === 0) {
      setMessage("Clipboard does not contain any nodes.");
      return;
    }
    const anchor = cursorScreenPos.current
      ? reactFlow.screenToFlowPosition(cursorScreenPos.current)
      : undefined;
    const bounds = copiedNodes.reduce(
      (acc: { minX: number; minY: number; maxX: number; maxY: number }, node: any) => {
        const position = node.position ?? { x: 0, y: 0 };
        return {
          minX: Math.min(acc.minX, position.x),
          minY: Math.min(acc.minY, position.y),
          maxX: Math.max(acc.maxX, position.x),
          maxY: Math.max(acc.maxY, position.y),
        };
      },
      {
        minX: Number.POSITIVE_INFINITY,
        minY: Number.POSITIVE_INFINITY,
        maxX: Number.NEGATIVE_INFINITY,
        maxY: Number.NEGATIVE_INFINITY,
      },
    );
    const copiedCenter = {
      x: (bounds.minX + bounds.maxX) / 2,
      y: (bounds.minY + bounds.maxY) / 2,
    };
    const offset =
      anchor && Number.isFinite(copiedCenter.x) && Number.isFinite(copiedCenter.y)
        ? { x: anchor.x - copiedCenter.x, y: anchor.y - copiedCenter.y }
        : { x: 20, y: 20 };
    const usedIds = new Set((crdt?.ws?.nodes || []).map((node: any) => node.id));
    const findFreeIdInBatch = (prefix: string) => {
      let i = 1;
      let id = `${prefix} ${i}`;
      while (usedIds.has(id)) {
        i += 1;
        id = `${prefix} ${i}`;
      }
      usedIds.add(id);
      return id;
    };
    const idMap = new Map<string, string>();
    for (const node of copiedNodes) {
      const newId = findFreeIdInBatch((node.data?.title as string) || "Node");
      idMap.set(node.id, newId);
    }
    const pastedNodes = copiedNodes.map((node) => {
      const position = node.position ?? { x: 0, y: 0 };
      const parentId = idMap.get(node.parentId as string);
      const isTopLevel = !parentId;
      return {
        ...node,
        id: idMap.get(node.id)!,
        parentId,
        selected: false,
        position: {
          x: isTopLevel ? position.x + offset.x : position.x,
          y: isTopLevel ? position.y + offset.y : position.y,
        },
      };
    });
    let skippedEdges = 0;
    const pastedEdges: Edge[] = [];
    for (const edge of copiedEdges) {
      const source = idMap.get(edge.source);
      const target = idMap.get(edge.target);
      const sourceHandle =
        typeof edge.sourceHandle === "string" && edge.sourceHandle.length > 0
          ? edge.sourceHandle
          : undefined;
      const targetHandle =
        typeof edge.targetHandle === "string" && edge.targetHandle.length > 0
          ? edge.targetHandle
          : undefined;
      if (!source || !target || !sourceHandle || !targetHandle) {
        skippedEdges += 1;
        continue;
      }
      pastedEdges.push({
        ...edge,
        selected: false,
        id: `${source} ${sourceHandle} ${target} ${targetHandle}`,
        source,
        sourceHandle,
        target,
        targetHandle,
      });
    }
    crdt?.applyChange((conn: any) => {
      const wnodes = conn.ws.get("nodes");
      const wedges = conn.ws.get("edges");
      for (const node of pastedNodes) {
        wnodes.push([nodeToYMap(node)]);
      }
      for (const edge of pastedEdges) {
        const edgeMap = new YMap<any>();
        for (const [key, value] of Object.entries(edge)) {
          edgeMap.set(key, value);
        }
        wedges.push([edgeMap]);
      }
    });
    if (skippedEdges > 0) {
      setMessage(
        `Pasted nodes, skipped ${skippedEdges} dangling edge${skippedEdges > 1 ? "s" : ""}.`,
      );
    }
  } catch (error) {
    setMessage("Failed to paste nodes from clipboard.");
    console.error("Failed to paste nodes from clipboard.", error);
  }
}

export async function cutSelection(
  nodes: Node[],
  edges: Edge[],
  crdt: Workspace,
  deleteSelection: () => void,
) {
  copySelection(nodes, edges, crdt);
  deleteSelection();
}
