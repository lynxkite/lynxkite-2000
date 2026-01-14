// The LynxKite workspace editor.

import {
  Background,
  BackgroundVariant,
  type Connection,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type XYPosition,
} from "@xyflow/react";
import axios from "axios";
import { type MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import useSWR, { type Fetcher } from "swr";
// @ts-expect-error
import Backspace from "~icons/tabler/backspace.jsx";
// @ts-expect-error
import UngroupIcon from "~icons/tabler/library-minus.jsx";
// @ts-expect-error
import GroupIcon from "~icons/tabler/library-plus.jsx";
// @ts-expect-error
import Pause from "~icons/tabler/player-pause.jsx";
// @ts-expect-error
import Play from "~icons/tabler/player-play.jsx";
// @ts-expect-error
import Restart from "~icons/tabler/rotate-clockwise.jsx";
// @ts-expect-error
import Transfer from "~icons/tabler/transfer.jsx";
// @ts-expect-error
import Close from "~icons/tabler/x.jsx";
import type { Op as OpsOp, WorkspaceNode } from "../apiTypes.ts";
import favicon from "../assets/favicon.ico";
import { usePath } from "../common.ts";
import Tooltip from "../Tooltip.tsx";
import { nodeToYMap, useCRDTWorkspace } from "./crdt.ts";
import EnvironmentSelector from "./EnvironmentSelector";
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
  const path = usePath().replace(/^[/]edit[/]/, "");
  const [message, setMessage] = useState(null as string | null);
  const shortPath = path!
    .split("/")
    .pop()!
    .replace(/[.]lynxkite[.]json$/, "");
  const crdt = useCRDTWorkspace(path);
  const nodes = crdt.feNodes;
  const edges = crdt.feEdges;

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
  if (!crdt?.ws) {
    return <div>Loading workspace...</div>;
  }
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
    const PAD = 10;
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
      const selectedNodeIds = new Set(selectedNodes.map((n) => n.id));
      for (const node of wnodes) {
        if (selectedNodeIds.has(node.get("id"))) {
          node.set("position", {
            x: node.get("position").x - left,
            y: node.get("position").y - top,
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
        <EnvironmentSelector
          options={Object.keys(catalog.data || {})}
          value={crdt.ws.env || ""}
          onChange={crdt.setEnv}
        />
        <div className="tools text-secondary">
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
              <Backspace />
            </button>
          </Tooltip>
          <Tooltip doc="Change selected box to a different box">
            <button className="btn btn-link" disabled={selected.length !== 1} onClick={changeBox}>
              <Transfer />
            </button>
          </Tooltip>
          <Tooltip
            doc={crdt.ws.paused ? "Resume automatic execution" : "Pause automatic execution"}
          >
            <button className="btn btn-link" onClick={() => crdt.setPausedState(!crdt.ws?.paused)}>
              {crdt.ws.paused ? <Play /> : <Pause />}
            </button>
          </Tooltip>
          <Tooltip doc="Re-run the workspace">
            <button className="btn btn-link" onClick={executeWorkspace}>
              <Restart />
            </button>
          </Tooltip>
          <Tooltip doc="Close workspace">
            <Link
              className="btn btn-link"
              to={`/dir/${parentDir
                .split("/")
                .map((segment) => encodeURIComponent(segment))
                .join("/")}`}
              aria-label="close"
            >
              <Close />
            </Link>
          </Tooltip>
        </div>
      </div>
      <div
        style={{ height: "100%", width: "100vw" }}
        onDragOver={onDragOver}
        onDrop={onDrop}
        ref={reactFlowContainer}
      >
        <LynxKiteState.Provider value={{ workspace: crdt.ws }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            onNodesChange={(changes) => {
              if (isShiftPressed) {
                changes = snapChangesToGrid(changes, isShiftPressed, crdt?.ws?.nodes || []);
              }
              crdt?.onFENodesChange?.(changes);
            }}
            onEdgesChange={crdt?.onFEEdgesChange}
            onPaneClick={toggleNodeSearch}
            onConnect={onConnect}
            proOptions={{ hideAttribution: true }}
            maxZoom={10}
            minZoom={0.2}
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
              gap={40}
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
