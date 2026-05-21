// Auto-connect logic extracted from Workspace.tsx for readability.

import { type Edge, type Node, useReactFlow, type XYPosition } from "@xyflow/react";
import { type MouseEvent, useCallback, useMemo, useState } from "react";
import type { Workspace, WorkspaceEdge } from "../apiTypes.ts";
import { getHandles } from "./nodes/LynxKiteNode.tsx";

const MIN_AUTO_CONNECT_DISTANCE = 220;

function getNodeHandles(node: Node, edges: Edge[]) {
  const inputs = (node.data as any)?.meta?.inputs ?? [];
  const outputs = (node.data as any)?.meta?.outputs ?? [];
  return getHandles({ edges } as Workspace, node.id, inputs, outputs);
}

function getHandlePosition(
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

function findClosestHandlePair(
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

function edgeExists(
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

export function useAutoConnect(edges: Edge[], crdt: any) {
  const reactFlow = useReactFlow();
  // A preview of the auto-connect edge while dragging a node.
  const [previewEdge, setPreviewEdge] = useState<Edge | null>(null);
  const renderedEdges = useMemo(() => {
    return previewEdge ? [...edges.filter((e) => e.id !== previewEdge.id), previewEdge] : edges;
  }, [edges, previewEdge]);

  function getAbsPos(node: Node) {
    const internal = reactFlow.getInternalNode(node.id);
    return internal?.internals.positionAbsolute ?? node.position;
  }

  const getClosestEdge = useCallback(
    (draggedNode: Node) => {
      const draggedPos = getAbsPos(draggedNode);
      if (!draggedPos || draggedNode.type === "node_group") {
        return null;
      }

      let bestEdge: WorkspaceEdge | null = null;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const n of reactFlow.getNodes()) {
        if (n.id === draggedNode.id || n.type === "node_group") {
          continue;
        }
        const pos = getAbsPos(n);
        const closeNodeIsSource = pos.x < draggedPos.x;
        const sourceNode = closeNodeIsSource ? n : draggedNode;
        const targetNode = closeNodeIsSource ? draggedNode : n;
        const bestPair = findClosestHandlePair(sourceNode, targetNode, edges, getAbsPos);
        if (!bestPair) continue;
        if (bestPair.distance >= MIN_AUTO_CONNECT_DISTANCE || bestPair.distance >= bestDistance) {
          continue;
        }
        bestDistance = bestPair.distance;
        bestEdge = {
          id: `${sourceNode.id} ${bestPair.sourceHandle} ${targetNode.id} ${bestPair.targetHandle}`,
          source: sourceNode.id,
          sourceHandle: bestPair.sourceHandle,
          target: targetNode.id,
          targetHandle: bestPair.targetHandle,
        };
      }

      return bestEdge;
    },
    [reactFlow, edges],
  );

  const onNodeDrag = useCallback(
    (_event: MouseEvent | TouchEvent, draggedNode: Node) => {
      const closeEdge = getClosestEdge(draggedNode);
      if (
        !closeEdge ||
        edgeExists(
          closeEdge.source,
          closeEdge.sourceHandle!,
          closeEdge.target,
          closeEdge.targetHandle!,
          edges,
        )
      ) {
        if (previewEdge) setPreviewEdge(null);
        return;
      }
      const previewId = `preview:${closeEdge.id}`;
      if (previewEdge?.id === previewId) return;
      setPreviewEdge({
        ...closeEdge,
        id: previewId,
        className: "temp-preview-edge",
        style: {
          strokeDasharray: "8 6",
          strokeLinecap: "round",
        },
        selectable: false,
        deletable: false,
        focusable: false,
      });
    },
    [getClosestEdge, edges, previewEdge],
  );

  const onNodeDragStop = useCallback(
    (_event: MouseEvent | TouchEvent, draggedNode: Node) => {
      const closeEdge = getClosestEdge(draggedNode);
      setPreviewEdge(null);
      if (!closeEdge) return;
      if (
        edgeExists(
          closeEdge.source,
          closeEdge.sourceHandle!,
          closeEdge.target,
          closeEdge.targetHandle!,
          edges,
        )
      ) {
        return;
      }
      crdt?.addEdge(closeEdge);
    },
    [getClosestEdge, crdt, edges],
  );
  return { onNodeDrag, onNodeDragStop, renderedEdges };
}
