// The LynxKite workspace editor.

import {
  Background,
  BackgroundVariant,
  type Connection,
  type Edge,
  MarkerType,
  type Node,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type XYPosition,
} from "@xyflow/react";
import axios from "axios";
import { type MouseEvent, memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import useSWR, { type Fetcher } from "swr";
import Arrow from "~icons/tabler/arrow-wave-right-up.jsx";
import Backspace from "~icons/tabler/backspace.jsx";
import GridDots from "~icons/tabler/grid-dots.jsx";
import LibraryMinus from "~icons/tabler/library-minus.jsx";
import LibraryPlus from "~icons/tabler/library-plus.jsx";
import Pause from "~icons/tabler/player-pause.jsx";
import Play from "~icons/tabler/player-play.jsx";
import RotateClockwise from "~icons/tabler/rotate-clockwise.jsx";
import Transfer from "~icons/tabler/transfer.jsx";
import Close from "~icons/tabler/x.jsx";
import type { Op as OpsOp, WorkspaceNode } from "../apiTypes.ts";
import favicon from "../assets/favicon.ico";
import { usePath } from "../common.ts";
import Tooltip from "../Tooltip.tsx";
import { nodeToYMap, useCRDTWorkspace } from "./crdt.ts";
import EnvironmentSelector from "./EnvironmentSelector";
import ExecutionOptions from "./ExecutionOptions.tsx";
import { snapChangesToGrid } from "./grid.ts";
import LynxKiteEdge from "./LynxKiteEdge.tsx";
import { LynxKiteState } from "./LynxKiteState";
import NodeSearch, { buildCategoryHierarchy, type Catalogs } from "./NodeSearch.tsx";
import NodeWithGraphCreationView from "./nodes/GraphCreationNode.tsx";
import Group from "./nodes/Group.tsx";
import NodeWithComment from "./nodes/NodeWithComment.tsx";
import NodeWithGradio from "./nodes/NodeWithGradio.tsx";
import NodeWithImage from "./nodes/NodeWithImage.tsx";
import NodeWithMolecule from "./nodes/NodeWithMolecule.tsx";
import NodeWithParams from "./nodes/NodeWithParams";
import NodeWithTableView from "./nodes/NodeWithTableView.tsx";
import NodeWithVisualization from "./nodes/NodeWithVisualization.tsx";

// The workspace gets re-rendered on every frame when a node is moved.
// Surprisingly, re-rendering the icons is very expensive in dev mode.
// Memoizing them fixes it.
const DeleteIcon = memo(Backspace);
const GridIcon = memo(GridDots);
const GridOffIcon = memo(Arrow);
const GroupIcon = memo(LibraryPlus);
const UngroupIcon = memo(LibraryMinus);
const RestartIcon = memo(RotateClockwise);
const PlayIcon = memo(Play);
const PauseIcon = memo(Pause);
const CloseIcon = memo(Close);
const ChangeTypeIcon = memo(Transfer);
const MIN_AUTO_CONNECT_DISTANCE = 220;

function allowsMultipleConnections(inputType: unknown): boolean {
  const LIST_TYPE_RE = /(^|[^a-z])list\b|typing\.list\[|list\[/i;
  if (!inputType) return false;
  if (typeof inputType === "string") {
    return LIST_TYPE_RE.test(inputType);
  }
  if (typeof inputType !== "object") {
    return false;
  }
  const t = inputType as Record<string, unknown>;
  if (typeof t.type === "string") {
    const lower = t.type.toLowerCase();
    if (lower === "array") return true;
    if (LIST_TYPE_RE.test(t.type)) return true;
  }
  if (typeof t.origin === "string" && t.origin.toLowerCase() === "list") return true;
  if ("items" in t) return true;
  for (const key of ["anyOf", "oneOf", "allOf"] as const) {
    const variants = t[key];
    if (!Array.isArray(variants)) continue;
    if (variants.some((v) => allowsMultipleConnections(v))) {
      return true;
    }
  }
  // Last-resort heuristic for serialized typing metadata variants.
  try {
    const serialized = JSON.stringify(t).toLowerCase();
    if (serialized.includes('"type":"array"')) return true;
    if (serialized.includes('"origin":"list"')) return true;
    if (LIST_TYPE_RE.test(serialized)) return true;
  } catch {
    // Ignore non-serializable metadata shapes.
  }
  return false;
}

export default function Workspace(props: any) {
  return (
    <ReactFlowProvider>
      <LynxKiteFlow {...props} />
    </ReactFlowProvider>
  );
}

function LynxKiteFlow() {
  const reactFlow = useReactFlow();
  const reactFlowContainer = useRef<HTMLDivElement>(null);
  const [isShiftPressed, setIsShiftPressed] = useState(false);
  const [gridSnapEnabled, setGridSnapEnabled] = useState(
    () => localStorage.getItem("gridSnapEnabled") === "true",
  );
  const path = usePath().replace(/^[/]edit[/]/, "");
  const [message, setMessage] = useState(null as string | null);
  const [previewEdge, setPreviewEdge] = useState<Edge | null>(null);
  const shortPath = path!
    .split("/")
    .pop()!
    .replace(/[.]lynxkite[.]json$/, "");
  const crdt = useCRDTWorkspace(path);
  const nodes = crdt.feNodes;
  const edges = crdt.feEdges;
  const renderedEdges = useMemo(
    () => (previewEdge ? [...edges.filter((e) => e.id !== previewEdge.id), previewEdge] : edges),
    [edges, previewEdge],
  );

  // Track Shift key state
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Shift") {
        setIsShiftPressed(true);
      }
    }

    function handleKeyUp(event: KeyboardEvent): void {
      if (event.key === "Shift") {
        setIsShiftPressed(false);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useEffect(() => {
    localStorage.setItem("gridSnapEnabled", String(gridSnapEnabled));
  }, [gridSnapEnabled]);

  const fetcher: Fetcher<Catalogs> = (resource: string, init?: RequestInit) =>
    fetch(resource, init).then((res) => res.json());
  const encodedPathForAPI = path!
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const catalog = useSWR(`/api/catalog?workspace=${encodedPathForAPI}`, fetcher);
  const categoryHierarchy = useMemo(() => {
    if (!catalog.data || !crdt?.ws?.env) return undefined;
    return buildCategoryHierarchy(catalog.data[crdt.ws.env]);
  }, [catalog, crdt]);
  const [suppressSearchUntil, setSuppressSearchUntil] = useState(0);
  const [nodeSearchSettings, setNodeSearchSettings] = useState(
    undefined as
      | {
          pos: XYPosition;
        }
      | undefined,
  );
  const nodeTypes = useMemo(
    () => ({
      basic: NodeWithParams,
      visualization: NodeWithVisualization,
      image: NodeWithImage,
      table_view: NodeWithTableView,
      service: NodeWithTableView,
      gradio: NodeWithGradio,
      graph_creation_view: NodeWithGraphCreationView,
      molecule: NodeWithMolecule,
      comment: NodeWithComment,
      node_group: Group,
    }),
    [],
  );
  const edgeTypes = useMemo(
    () => ({
      default: LynxKiteEdge,
    }),
    [],
  );

  // Global keyboard shortcuts.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Show the node search dialog on "/".
      if (nodeSearchSettings || isTypingInFormElement()) return;
      if (event.key === "/" && categoryHierarchy) {
        event.preventDefault();
        setNodeSearchSettings({
          pos: getBestPosition(),
        });
      } else if (event.key === "r") {
        event.preventDefault();
        executeWorkspace();
      }
    };
    // TODO: Switch to keydown once https://github.com/xyflow/xyflow/pull/5055 is merged.
    document.addEventListener("keyup", handleKeyDown);
    return () => {
      document.removeEventListener("keyup", handleKeyDown);
    };
  }, [categoryHierarchy, nodeSearchSettings]);

  function getBestPosition() {
    const W = reactFlowContainer.current!.clientWidth;
    const H = reactFlowContainer.current!.clientHeight;
    const w = 200;
    const h = 200;
    const SPEED = 20;
    const GAP = 50;
    const pos = { x: 100, y: 100 };
    while (pos.y < H) {
      // Find a position that is not occupied by a node.
      const fpos = reactFlow.screenToFlowPosition(pos);
      const occupied = crdt?.ws?.nodes?.some((n) => {
        const np = n.position;
        return (
          np.x < fpos.x + w + GAP &&
          np.x + (n.width ?? 0) + GAP > fpos.x &&
          np.y < fpos.y + h + GAP &&
          np.y + (n.height ?? 0) + GAP > fpos.y
        );
      });
      if (!occupied) {
        return pos;
      }
      // Move the position to the right and down until we find a free spot.
      pos.x += SPEED;
      if (pos.x + w > W) {
        pos.x = 100;
        pos.y += SPEED;
      }
    }
    return { x: 100, y: 100 };
  }

  function isTypingInFormElement() {
    const activeElement = document.activeElement;
    return (
      activeElement &&
      (activeElement.tagName === "INPUT" ||
        activeElement.tagName === "TEXTAREA" ||
        (activeElement as HTMLElement).isContentEditable)
    );
  }

  const closeNodeSearch = useCallback(() => {
    setNodeSearchSettings(undefined);
    setSuppressSearchUntil(Date.now() + 200);
  }, []);
  const toggleNodeSearch = useCallback(
    (event: MouseEvent) => {
      if (!categoryHierarchy) return;
      if (suppressSearchUntil > Date.now()) return;
      if (nodeSearchSettings) {
        closeNodeSearch();
        return;
      }
      event.preventDefault();
      setNodeSearchSettings({
        pos: { x: event.clientX, y: event.clientY },
      });
    },
    [categoryHierarchy, crdt.ws, nodeSearchSettings, suppressSearchUntil, closeNodeSearch],
  );
  function findFreeId(prefix: string) {
    let i = 1;
    let id = `${prefix} ${i}`;
    const used = new Set(crdt?.ws?.nodes?.map((n) => n.id));
    while (used.has(id)) {
      i += 1;
      id = `${prefix} ${i}`;
    }
    return id;
  }
  function addNode(node: Partial<WorkspaceNode>) {
    crdt?.addNode(node);
  }
  function nodeFromMeta(meta: OpsOp): Partial<WorkspaceNode> {
    const node: Partial<WorkspaceNode> = {
      type: meta.type,
      height: 200,
      data: {
        meta: meta,
        title: meta.name,
        op_id: meta.id || meta.name,
        params: Object.fromEntries(meta.params.map((p) => [p.name, p.default])),
      },
    };
    return node;
  }
  const addNodeFromSearch = useCallback(
    (meta: OpsOp) => {
      const node = nodeFromMeta(meta);
      const nss = nodeSearchSettings!;
      node.position = reactFlow.screenToFlowPosition({
        x: nss.pos.x,
        y: nss.pos.y,
      });
      node.id = findFreeId(node.data!.title);
      addNode(node);
      closeNodeSearch();
    },
    [nodeSearchSettings, reactFlow, closeNodeSearch],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      setSuppressSearchUntil(Date.now() + 200);
      const edge = {
        id: `${connection.source} ${connection.sourceHandle} ${connection.target} ${connection.targetHandle}`,
        source: connection.source,
        sourceHandle: connection.sourceHandle!,
        target: connection.target,
        targetHandle: connection.targetHandle!,
      };
      crdt?.addEdge(edge);
    },
    [crdt],
  );

  function getAbsPos(node: Node) {
    const internal = reactFlow.getInternalNode(node.id);
    return internal?.internals.positionAbsolute ?? node.position;
  }

  function getNodeHandles(node: Node) {
    const nodeMetaInputs = (node.data as any)?.meta?.inputs ?? [];
    const nodeData = (node.data as any) ?? {};
    const opId = nodeData.op_id;
    const opMeta = nodeData.meta ?? {};
    const env = crdt?.ws?.env;
    const envCatalog = (env && catalog.data?.[env]) || {};
    const catalogKeyCandidates = [opId, opMeta.id, opMeta.name, nodeData.title].filter(
      (k): k is string => typeof k === "string" && k.length > 0,
    );
    const matchedCatalogOp = catalogKeyCandidates
      .map((k) => (envCatalog as any)[k])
      .find((entry) => entry && Array.isArray(entry.inputs));
    const catalogInputs = matchedCatalogOp?.inputs || [];
    const catalogInputsByName = new Map((catalogInputs as any[]).map((h: any) => [h.name, h]));
    const inputs = nodeMetaInputs.map((h: any) => {
      const catalogInput = catalogInputsByName.get(h.name);
      const resolvedType = h.type ?? catalogInput?.type;
      return {
        ...h,
        type: "target",
        acceptsMultipleConnections: allowsMultipleConnections(resolvedType),
      };
    });
    const outputs = ((node.data as any)?.meta?.outputs ?? []).map((h: any) => ({
      ...h,
      type: "source",
    }));
    const handles = [...inputs, ...outputs] as Array<{
      position: "top" | "bottom" | "left" | "right";
      name: string;
      type: "source" | "target";
      index?: number;
      offsetPercentage?: number;
      acceptsMultipleConnections?: boolean;
    }>;
    const counts = { top: 0, bottom: 0, left: 0, right: 0 };
    for (const h of handles) {
      h.index = counts[h.position];
      counts[h.position] += 1;
    }
    for (const h of handles) {
      h.offsetPercentage = (100 * (h.index! + 1)) / (counts[h.position] + 1);
    }
    // Keep unknown/renamed handles available for auto-connect if they already exist on edges.
    for (const e of edges) {
      if (
        e.target === node.id &&
        !handles.find((h) => h.type === "target" && h.name === e.targetHandle)
      ) {
        handles.push({
          position: "left",
          name: e.targetHandle!,
          type: "target",
          offsetPercentage: 50,
          acceptsMultipleConnections: false,
        });
      }
      if (
        e.source === node.id &&
        !handles.find((h) => h.type === "source" && h.name === e.sourceHandle)
      ) {
        handles.push({
          position: "right",
          name: e.sourceHandle!,
          type: "source",
          offsetPercentage: 50,
        });
      }
    }
    return handles;
  }

  function getHandlePosition(node: Node, handle: { position: string; offsetPercentage?: number }) {
    const p = getAbsPos(node);
    const width = node.width ?? 200;
    const height = node.height ?? 200;
    const offset = (handle.offsetPercentage ?? 50) / 100;
    if (handle.position === "left") return { x: p.x, y: p.y + height * offset };
    if (handle.position === "right") return { x: p.x + width, y: p.y + height * offset };
    if (handle.position === "top") return { x: p.x + width * offset, y: p.y };
    return { x: p.x + width * offset, y: p.y + height };
  }

  function findClosestHandlePair(sourceNode: Node, targetNode: Node) {
    const sourceHandles = getNodeHandles(sourceNode).filter((h) => h.type === "source");
    const targetHandles = getNodeHandles(targetNode).filter((h) => h.type === "target");
    if (!sourceHandles.length || !targetHandles.length) {
      return null;
    }
    let bestPair: { sourceHandle: string; targetHandle: string; distance: number } | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const sh of sourceHandles) {
      const sp = getHandlePosition(sourceNode, sh);
      for (const th of targetHandles) {
        const isOccupied = edges.some(
          (e) => e.target === targetNode.id && e.targetHandle === th.name,
        );
        if (isOccupied && !th.acceptsMultipleConnections) {
          continue;
        }
        const tp = getHandlePosition(targetNode, th);
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
    edgeList: Edge[] = edges,
  ) {
    return edgeList.some(
      (e) =>
        e.source === source &&
        e.sourceHandle === sourceHandle &&
        e.target === target &&
        e.targetHandle === targetHandle,
    );
  }

  const getClosestEdge = useCallback(
    (draggedNode: Node) => {
      const draggedPos = getAbsPos(draggedNode);
      if (!draggedPos || draggedNode.type === "node_group") {
        return null;
      }

      let bestEdge: Edge | null = null;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const n of reactFlow.getNodes()) {
        if (n.id === draggedNode.id || n.type === "node_group") {
          continue;
        }
        const pos = getAbsPos(n);
        const closeNodeIsSource = pos.x < draggedPos.x;
        const sourceNode = closeNodeIsSource ? n : draggedNode;
        const targetNode = closeNodeIsSource ? draggedNode : n;
        const bestPair = findClosestHandlePair(sourceNode, targetNode);
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
        )
      ) {
        setPreviewEdge(null);
        return;
      }
      setPreviewEdge({
        ...closeEdge,
        id: `preview:${closeEdge.id}`,
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
    [getClosestEdge, edges],
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
        )
      ) {
        return;
      }
      crdt?.addEdge(closeEdge);
    },
    [getClosestEdge, crdt, edges],
  );
  const parentDir = path!.split("/").slice(0, -1).join("/");
  function onDragOver(e: React.DragEvent<HTMLDivElement>) {
    e.stopPropagation();
    e.preventDefault();
  }
  async function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.stopPropagation();
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    const formData = new FormData();
    formData.append("file", file);
    if (!catalog.data || !crdt?.ws?.env) {
      return;
    }
    try {
      await axios.post("/api/upload", formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((100 * progressEvent.loaded) / progressEvent.total!);
          if (percentCompleted === 100) setMessage("Processing file...");
          else setMessage(`Uploading ${percentCompleted}%`);
        },
      });
      setMessage(null);
      const cat = catalog.data[crdt.ws.env];
      const node = nodeFromMeta(cat["Import file"]);
      node.id = findFreeId(node.data!.title);
      node.position = reactFlow.screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });
      node.data!.params.file_path = `uploads/${file.name}`;
      if (file.name.includes(".csv")) {
        node.data!.params.file_format = "csv";
      } else if (file.name.includes(".parquet")) {
        node.data!.params.file_format = "parquet";
      } else if (file.name.includes(".json")) {
        node.data!.params.file_format = "json";
      } else if (file.name.includes(".xls")) {
        node.data!.params.file_format = "excel";
      }
      addNode(node);
    } catch (error) {
      setMessage("File upload failed.");
      console.error("File upload failed.", error);
    }
  }
  async function executeWorkspace() {
    const response = await axios.post(`/api/execute_workspace?name=${encodeURIComponent(path)}`);
    if (response.status !== 200) {
      setMessage("Workspace execution failed.");
    }
  }
  function deleteSelection() {
    const selectedNodes = nodes.filter((n) => n.selected);
    const selectedEdges = edges.filter((e) => e.selected);
    reactFlow.deleteElements({ nodes: selectedNodes, edges: selectedEdges });
  }
  function changeBox() {
    const [selectedNode] = nodes.filter((n) => n.selected);
    reactFlow.updateNodeData(selectedNode.id, { op_id: "" });
  }
  function groupSelection() {
    const selectedNodes = nodes.filter((n) => n.selected && !n.parentId);
    const groupNode = {
      id: findFreeId("Group"),
      type: "node_group",
      position: { x: 0, y: 0 },
      width: 0,
      height: 0,
      data: { title: "Group", params: {} },
      selected: true,
    };
    let top = Number.POSITIVE_INFINITY;
    let left = Number.POSITIVE_INFINITY;
    let bottom = Number.NEGATIVE_INFINITY;
    let right = Number.NEGATIVE_INFINITY;
    const PAD = 50;
    for (const node of selectedNodes) {
      if (node.position.y - PAD < top) top = node.position.y - PAD;
      if (node.position.x - PAD < left) left = node.position.x - PAD;
      if (node.position.y + PAD + node.height! > bottom)
        bottom = node.position.y + PAD + node.height!;
      if (node.position.x + PAD + node.width! > right) right = node.position.x + PAD + node.width!;
      node.selected = false;
    }
    groupNode.position = {
      x: left,
      y: top,
    };
    groupNode.width = right - left;
    groupNode.height = bottom - top;
    crdt.applyChange((conn) => {
      const wnodes = conn.ws.get("nodes");
      wnodes.unshift([nodeToYMap(groupNode)]);
      const selectedNodesById = new Map(selectedNodes.map((n) => [n.id, n]));
      for (const node of wnodes) {
        const feNode = selectedNodesById.get(node.get("id"));
        if (feNode) {
          const pos = feNode.position;
          node.set("position", {
            x: pos.x - left,
            y: pos.y - top,
          });
          node.set("parentId", groupNode.id);
          node.set("extent", "parent");
          node.set("selected", false);
        }
      }
    });
  }
  function ungroupSelection() {
    const groups = Object.fromEntries(
      nodes
        .filter((n) => n.selected && n.type === "node_group" && !n.parentId)
        .map((n) => [n.id, n]),
    );
    crdt.applyChange((conn) => {
      const wnodes = conn.ws.get("nodes");
      for (const node of wnodes) {
        const g = groups[node.get("parentId") as string];
        if (!g) continue;
        const pos = node.get("position") as XYPosition;
        node.set("position", {
          x: pos.x + g.position.x,
          y: pos.y + g.position.y,
        });
        node.set("parentId", undefined);
        node.set("extent", undefined);
        node.set("selected", true);
      }
      const groupIndices: number[] = wnodes
        .map((n: any, idx: number) => ({ id: n.get("id"), idx }))
        .filter(({ id }: { id: string }) => id in groups)
        .map(({ idx }: { idx: number }) => idx);
      groupIndices.sort((a, b) => b - a);
      for (const groupIdx of groupIndices) {
        wnodes.delete(groupIdx, 1);
      }
    });
  }
  const selected = nodes.filter((n) => n.selected);
  const isAnyGroupSelected = nodes.some((n) => n.selected && n.type === "node_group");
  return (
    <div className="workspace">
      <div className="top-bar bg-neutral">
        <Link className="logo" to="/">
          <img alt="" src={favicon} />
        </Link>
        <div className="ws-name">{shortPath}</div>
        <title>{shortPath}</title>
        {crdt?.ws && (
          <>
            <ExecutionOptions
              env={crdt.ws.env || ""}
              value={crdt.ws.execution_options}
              onChange={crdt.setExecutionOptions}
            />
            <EnvironmentSelector
              options={Object.keys(catalog.data || {})}
              value={crdt.ws.env || ""}
              onChange={crdt.setEnv}
            />
          </>
        )}
        <div className="tools text-secondary">
          {crdt?.ws && (
            <>
              <Tooltip doc="Group selected nodes">
                <button
                  className="btn btn-link"
                  disabled={selected.length < 2}
                  onClick={groupSelection}
                >
                  <GroupIcon />
                </button>
              </Tooltip>
              <Tooltip doc="Ungroup selected nodes">
                <button
                  className="btn btn-link"
                  disabled={!isAnyGroupSelected}
                  onClick={ungroupSelection}
                >
                  <UngroupIcon />
                </button>
              </Tooltip>
              <Tooltip doc="Delete selected nodes and edges">
                <button
                  className="btn btn-link"
                  disabled={selected.length === 0}
                  onClick={deleteSelection}
                >
                  <DeleteIcon />
                </button>
              </Tooltip>
              <Tooltip doc="Change selected box to a different box">
                <button
                  className="btn btn-link"
                  disabled={selected.length !== 1}
                  onClick={changeBox}
                >
                  <ChangeTypeIcon />
                </button>
              </Tooltip>
              <Tooltip doc={gridSnapEnabled ? "Disable grid snapping" : "Enable grid snapping"}>
                <button
                  className="btn btn-link"
                  onClick={() => setGridSnapEnabled(!gridSnapEnabled)}
                >
                  {gridSnapEnabled ? <GridIcon /> : <GridOffIcon />}
                </button>
              </Tooltip>
              <Tooltip
                doc={crdt.ws.paused ? "Resume automatic execution" : "Pause automatic execution"}
              >
                <button
                  className="btn btn-link"
                  onClick={() => crdt.setPausedState(!crdt.ws?.paused)}
                >
                  {crdt.ws.paused ? <PlayIcon /> : <PauseIcon />}
                </button>
              </Tooltip>
              <Tooltip doc="Re-run the workspace">
                <button className="btn btn-link" onClick={executeWorkspace}>
                  <RestartIcon />
                </button>
              </Tooltip>
            </>
          )}
          <Tooltip doc="Close workspace">
            <Link
              className="btn btn-link"
              to={`/dir/${parentDir
                .split("/")
                .map((segment) => encodeURIComponent(segment))
                .join("/")}`}
              aria-label="close"
            >
              <CloseIcon />
            </Link>
          </Tooltip>
        </div>
      </div>
      <div
        className="reactflow-container"
        onDragOver={onDragOver}
        onDrop={onDrop}
        ref={reactFlowContainer}
      >
        {crdt?.ws ? (
          <LynxKiteState.Provider value={{ workspace: crdt.ws }}>
            <ReactFlow
              nodes={nodes}
              edges={renderedEdges}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              fitView
              onNodesChange={(changes) => {
                changes = snapChangesToGrid(
                  changes,
                  isShiftPressed || gridSnapEnabled,
                  crdt?.ws?.nodes || [],
                );
                crdt?.onFENodesChange?.(changes);
              }}
              onEdgesChange={crdt?.onFEEdgesChange}
              onPaneClick={toggleNodeSearch}
              onConnect={onConnect}
              onNodeDrag={onNodeDrag}
              onNodeDragStop={onNodeDragStop}
              proOptions={{ hideAttribution: true }}
              maxZoom={10}
              minZoom={0.1}
              zoomOnScroll={true}
              panOnScroll={false}
              panOnDrag={[0]}
              selectionOnDrag={false}
              preventScrolling={true}
              defaultEdgeOptions={{
                markerEnd: {
                  type: MarkerType.ArrowClosed,
                  color: "#888",
                  width: 15,
                  height: 15,
                },
              }}
              fitViewOptions={{ maxZoom: 1 }}
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={35}
                size={6}
                color="#f0f0f0"
                bgColor="#fafafa"
                offset={3}
              />
              {nodeSearchSettings && categoryHierarchy && (
                <NodeSearch
                  pos={nodeSearchSettings.pos}
                  categoryHierarchy={categoryHierarchy}
                  onCancel={closeNodeSearch}
                  onClick={addNodeFromSearch}
                />
              )}
            </ReactFlow>
          </LynxKiteState.Provider>
        ) : (
          <div className="workspace-loading">Loading workspace...</div>
        )}
        {message && (
          <div className="workspace-message">
            <span className="close" onClick={() => setMessage(null)}>
              <Close />
            </span>
            {message}
          </div>
        )}
      </div>
    </div>
  );
}
