// Auto-connect boxes when they get close, with a preview edge.
// Based on https://reactflow.dev/examples/nodes/proximity-connect.

import { type Edge, type Node, type ReactFlowInstance, useReactFlow } from "@xyflow/react";
import type { InternalNodeBase } from "@xyflow/system";
import { type MouseEvent, useCallback, useMemo, useState } from "react";
import type { WorkspaceEdge } from "../apiTypes.ts";

const MIN_AUTO_CONNECT_DISTANCE = 100;

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

function allHandles(node: InternalNodeBase) {
  return [
    ...(node.internals.handleBounds?.source ?? []),
    ...(node.internals.handleBounds?.target ?? []),
  ];
}

// Finds the shortest edge between the handles of the dragged node and the handles of other nodes,
// if it's within the threshold distance.
function closestEdge(reactFlow: ReactFlowInstance, draggedNode: Node): WorkspaceEdge | null {
  if (draggedNode.type === "node_group") return null;
  const internalNode = reactFlow.getInternalNode(draggedNode.id);
  if (!internalNode) return null;
  const draggedHandles = allHandles(internalNode);
  if (!draggedHandles.length) return null;
  const draggedPos = internalNode.internals.positionAbsolute;
  let bestEdge: WorkspaceEdge | null = null;
  let bestDistance = MIN_AUTO_CONNECT_DISTANCE;
  for (const n of reactFlow.getNodes()) {
    if (n.id === draggedNode.id || n.type === "node_group") continue;
    const i = reactFlow.getInternalNode(n.id);
    if (!i) continue;
    const handles = allHandles(i);
    const pos = i.internals.positionAbsolute;
    for (const h1 of draggedHandles) {
      if (!h1.id) continue;
      for (const h2 of handles) {
        if (!h2.id || h1.type === h2.type) continue;
        const hp1 = {
          x: draggedPos.x + h1.x,
          y: draggedPos.y + h1.y,
        };
        const hp2 = {
          x: pos.x + h2.x,
          y: pos.y + h2.y,
        };
        const distance = Math.hypot(hp1.x - hp2.x, hp1.y - hp2.y);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestEdge = {
            id: `${draggedNode.id} ${h1.id} ${n.id} ${h2.id}`,
            source: h1.type === "source" ? draggedNode.id : n.id,
            sourceHandle: h1.type === "source" ? h1.id : h2.id,
            target: h1.type === "target" ? draggedNode.id : n.id,
            targetHandle: h1.type === "target" ? h1.id : h2.id,
          };
        }
      }
    }
  }
  return bestEdge;
}

export function useAutoConnect(edges: Edge[], crdt: any) {
  const reactFlow = useReactFlow();
  const [previewEdge, setPreviewEdge] = useState<Edge | null>(null);
  const renderedEdges = useMemo(() => {
    return previewEdge ? [...edges.filter((e) => e.id !== previewEdge.id), previewEdge] : edges;
  }, [edges, previewEdge]);
  const getClosestEdge = useCallback((n: Node) => closestEdge(reactFlow, n), [reactFlow]);

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
  return {
    // Handlers to register with React Flow.
    onNodeDrag,
    onNodeDragStop,
    // Includes a preview edge while dragging.
    renderedEdges,
  };
}
