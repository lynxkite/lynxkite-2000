// Auto-connect logic extracted from Workspace.tsx for readability.

import type { Edge, Node, XYPosition } from "@xyflow/react";
import { getHandles } from "./nodes/LynxKiteNode.tsx";

export const MIN_AUTO_CONNECT_DISTANCE = 220;

export function allowsMultipleConnections(_inputType: unknown): boolean {
  // Assume all inputs allow multiple connections for now.
  // If it's annoying, we can switch to assuming nothing allows multiple connections.
  return true;
}

export function getNodeHandles(node: Node, edges: Edge[]) {
  const inputs = (node.data as any)?.meta?.inputs ?? [];
  const outputs = (node.data as any)?.meta?.outputs ?? [];
  return getHandles({ edges }, node.id, inputs, outputs).map((h) => ({
    ...h,
    acceptsMultipleConnections: allowsMultipleConnections(undefined),
  }));
}

export function getHandlePosition(
  absPos: XYPosition,
  node: Node,
  handle: { position: string; offsetPercentage?: number },
) {
  const width = node.width ?? 200;
  const height = node.height ?? 200;
  const offset = (handle.offsetPercentage ?? 50) / 100;
  if (handle.position === "left") return { x: absPos.x, y: absPos.y + height * offset };
  if (handle.position === "right") return { x: absPos.x + width, y: absPos.y + height * offset };
  if (handle.position === "top") return { x: absPos.x + width * offset, y: absPos.y };
  return { x: absPos.x + width * offset, y: absPos.y + height };
}

export function findClosestHandlePair(
  sourceNode: Node,
  targetNode: Node,
  edges: Edge[],
  getAbsPos: (node: Node) => XYPosition,
) {
  const sourceHandles = getNodeHandles(sourceNode, edges).filter((h) => h.type === "source");
  const targetHandles = getNodeHandles(targetNode, edges).filter((h) => h.type === "target");
  if (!sourceHandles.length || !targetHandles.length) {
    return null;
  }
  let bestPair: { sourceHandle: string; targetHandle: string; distance: number } | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const sh of sourceHandles) {
    const sp = getHandlePosition(getAbsPos(sourceNode), sourceNode, sh);
    for (const th of targetHandles) {
      const isOccupied = edges.some(
        (e) => e.target === targetNode.id && e.targetHandle === th.name,
      );
      if (isOccupied && !th.acceptsMultipleConnections) {
        continue;
      }
      const tp = getHandlePosition(getAbsPos(targetNode), targetNode, th);
      const distance = Math.hypot(sp.x - tp.x, sp.y - tp.y);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestPair = { sourceHandle: sh.name, targetHandle: th.name, distance };
      }
    }
  }
  return bestPair;
}

export function edgeExists(
  source: string,
  sourceHandle: string,
  target: string,
  targetHandle: string,
  edges: Edge[],
) {
  return edges.some(
    (e) =>
      e.source === source &&
      e.sourceHandle === sourceHandle &&
      e.target === target &&
      e.targetHandle === targetHandle,
  );
}
