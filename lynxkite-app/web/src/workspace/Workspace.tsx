// The LynxKite workspace editor.

import { getYjsDoc, syncedStore } from "@syncedstore/core";
import {
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  Controls,
  type Edge,
  MarkerType,
  type Node,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useUpdateNodeInternals,
  type XYPosition,
} from "@xyflow/react";
import axios from "axios";
import { type MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import useSWR, { type Fetcher } from "swr";
import { WebsocketProvider } from "y-websocket";
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
import Close from "~icons/tabler/x.jsx";
import type { WorkspaceNode, Workspace as WorkspaceType } from "../apiTypes.ts";
import favicon from "../assets/favicon.ico";
import { usePath } from "../common.ts";
import Tooltip from "../Tooltip.tsx";
// import NodeWithTableView from './NodeWithTableView';
import EnvironmentSelector from "./EnvironmentSelector";
import LynxKiteEdge from "./LynxKiteEdge.tsx";
import { LynxKiteState } from "./LynxKiteState";
import NodeSearch, { buildCategoryHierarchy, type Catalogs, type OpsOp } from "./NodeSearch.tsx";
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
  const updateNodeInternals = useUpdateNodeInternals();
  const reactFlow = useReactFlow();
  const reactFlowContainer = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState([] as Node[]);
  const [edges, setEdges] = useState([] as Edge[]);
  const path = usePath().replace(/^[/]edit[/]/, "");
  const shortPath = path!
    .split("/")
    .pop()!
    .replace(/[.]lynxkite[.]json$/, "");
  const [state, setState] = useState({ workspace: {} as WorkspaceType });
  const [message, setMessage] = useState(null as string | null);
  const [pausedUIState, setPausedUIState] = useState(false);
  useEffect(() => {
    const state = syncedStore({ workspace: {} as WorkspaceType });
    setState(state);
    const doc = getYjsDoc(state);
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const encodedPath = path!
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    const wsProvider = new WebsocketProvider(
      `${proto}//${location.host}/ws/crdt`,
      encodedPath,
      doc,
    );
    if (state.workspace && typeof state.workspace.paused === "undefined") {
      state.workspace.paused = false;
    }
    const onChange = (_update: any, origin: any, _doc: any, _tr: any) => {
      if (origin === wsProvider) {
        // An update from the CRDT. Apply it to the local state.
        // This is only necessary because ReactFlow keeps secret internal copies of our stuff.
        if (!state.workspace) return;
        if (!state.workspace.nodes) return;
        if (!state.workspace.edges) return;
        for (const n of state.workspace.nodes) {
          if (n.type !== "node_group" && n.dragHandle !== ".drag-handle") {
            n.dragHandle = ".drag-handle";
          }
        }
        const nodes = reactFlow.getNodes();
        const selection = nodes.filter((n) => n.selected).map((n) => n.id);
        const newNodes = state.workspace.nodes.map((n) =>
          selection.includes(n.id) ? { ...n, selected: true } : n,
        );
        setNodes([...newNodes] as Node[]);
        setEdges([...state.workspace.edges] as Edge[]);
        for (const node of state.workspace.nodes) {
          // Make sure the internal copies are updated.
          updateNodeInternals(node.id);
        }
        setPausedUIState(state.workspace.paused || false);
      }
    };
    doc.on("update", onChange);
    return () => {
      doc.destroy();
      wsProvider.destroy();
    };
  }, [path, updateNodeInternals]);

  const onNodesChange = useCallback(
    (changes: any[]) => {
      // An update from the UI. Apply it to the local state...
      setNodes((nds) => applyNodeChanges(changes, nds));
      // ...and to the CRDT state. (Which could be the same, except for ReactFlow's internal copies.)
      const wnodes = state.workspace?.nodes;
      if (!wnodes) return;
      for (const ch of changes) {
        const nodeIndex = wnodes.findIndex((n) => n.id === ch.id);
        if (nodeIndex === -1) continue;
        const node = wnodes[nodeIndex];
        if (!node) continue;
        // Position events sometimes come with NaN values. Ignore them.
        if (
          ch.type === "position" &&
          !Number.isNaN(ch.position.x) &&
          !Number.isNaN(ch.position.y)
        ) {
          getYjsDoc(state).transact(() => {
            node.position.x = ch.position.x;
            node.position.y = ch.position.y;
          });
          // Update edge positions.
          updateNodeInternals(ch.id);
        } else if (ch.type === "select") {
        } else if (ch.type === "dimensions") {
          getYjsDoc(state).transact(() => {
            node.width = ch.dimensions.width;
            node.height = ch.dimensions.height;
          });
          // Update edge positions when node size changes.
          updateNodeInternals(ch.id);
        } else if (ch.type === "remove") {
          wnodes.splice(nodeIndex, 1);
        } else if (ch.type === "replace") {
          // Ideally we would only update the parameter that changed. But ReactFlow does not give us that detail.
          getYjsDoc(state).transact(() => {
            if (node.data.collapsed !== ch.item.data.collapsed) {
              node.data.collapsed = ch.item.data.collapsed;
              // Update edge positions when node collapses/expands.
              setTimeout(() => updateNodeInternals(ch.id), 0);
            }
            if (node.data.__execution_delay !== ch.item.data.__execution_delay) {
              node.data.__execution_delay = ch.item.data.__execution_delay;
            }
            for (const [key, value] of Object.entries(ch.item.data.params)) {
              if (node.data.params[key] !== value) {
                node.data.params[key] = value;
              }
            }
          });
        } else {
          console.log("Unknown node change", ch);
        }
      }
    },
    [state, updateNodeInternals],
  );
  const onEdgesChange = useCallback(
    (changes: any[]) => {
      setEdges((eds) => applyEdgeChanges(changes, eds));
      const wedges = state.workspace?.edges;
      if (!wedges) return;
      for (const ch of changes) {
        const edgeIndex = wedges.findIndex((e) => e.id === ch.id);
        if (ch.type === "remove") {
          wedges.splice(edgeIndex, 1);
        } else if (ch.type === "select") {
        } else {
          console.log("Unknown edge change", ch);
        }
      }
    },
    [state],
  );

  const fetcher: Fetcher<Catalogs> = (resource: string, init?: RequestInit) =>
    fetch(resource, init).then((res) => res.json());
  const encodedPathForAPI = path!
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const catalog = useSWR(`/api/catalog?workspace=${encodedPathForAPI}`, fetcher);
  const categoryHierarchy = useMemo(() => {
    if (!catalog.data || !state.workspace.env) return undefined;
    return buildCategoryHierarchy(catalog.data[state.workspace.env]);
  }, [catalog.data, state.workspace.env]);
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
      const occupied = state.workspace.nodes!.some((n) => {
        const np = n.position;
        return (
          np.x < fpos.x + w + GAP &&
          np.x + n.width + GAP > fpos.x &&
          np.y < fpos.y + h + GAP &&
          np.y + n.height + GAP > fpos.y
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
    [categoryHierarchy, state, nodeSearchSettings, suppressSearchUntil, closeNodeSearch],
  );
  function findFreeId(prefix: string) {
    let i = 1;
    let id = `${prefix} ${i}`;
    const used = new Set(state.workspace.nodes!.map((n) => n.id));
    while (used.has(id)) {
      i += 1;
      id = `${prefix} ${i}`;
    }
    return id;
  }
  function addNode(node: Partial<WorkspaceNode>) {
    state.workspace.nodes!.push(node as WorkspaceNode);
    setNodes([...nodes, node as WorkspaceNode]);
  }
  function nodeFromMeta(meta: OpsOp): Partial<WorkspaceNode> {
    const node: Partial<WorkspaceNode> = {
      type: meta.type,
      data: {
        meta: { value: meta },
        title: meta.name,
        op_id: meta.id,
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
    [nodeSearchSettings, state, reactFlow, nodes, closeNodeSearch],
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
      state.workspace.edges!.push(edge);
      setEdges((oldEdges) => [...oldEdges, edge]);
    },
    [state],
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
    try {
      await axios.post("/api/upload", formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((100 * progressEvent.loaded) / progressEvent.total!);
          if (percentCompleted === 100) setMessage("Processing file...");
          else setMessage(`Uploading ${percentCompleted}%`);
        },
      });
      setMessage(null);
      const cat = catalog.data![state.workspace.env!];
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
  function togglePause() {
    state.workspace.paused = !state.workspace.paused;
    setPausedUIState(state.workspace.paused);
  }
  function deleteSelection() {
    const selectedNodes = nodes.filter((n) => n.selected);
    const selectedEdges = edges.filter((e) => e.selected);
    reactFlow.deleteElements({ nodes: selectedNodes, edges: selectedEdges });
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
    }
    groupNode.position = {
      x: left,
      y: top,
    };
    groupNode.width = right - left;
    groupNode.height = bottom - top;
    setNodes([
      { ...(groupNode as WorkspaceNode), selected: true },
      ...nodes.map((n) =>
        n.selected
          ? {
              ...n,
              position: { x: n.position.x - left, y: n.position.y - top },
              parentId: groupNode.id,
              extent: "parent" as const,
              selected: false,
            }
          : n,
      ),
    ]);
    getYjsDoc(state).transact(() => {
      state.workspace.nodes!.unshift(groupNode as WorkspaceNode);
      const selectedNodeIds = new Set(selectedNodes.map((n) => n.id));
      for (const node of state.workspace.nodes!) {
        if (selectedNodeIds.has(node.id)) {
          node.position.x -= left;
          node.position.y -= top;
          node.parentId = groupNode.id;
          node.extent = "parent";
          node.selected = false;
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
    setNodes(
      nodes
        .filter((n) => !groups[n.id])
        .map((n) => {
          const g = groups[n.parentId!];
          if (!g) return n;
          return {
            ...n,
            position: {
              x: n.position.x + g.position.x,
              y: n.position.y + g.position.y,
            },
            parentId: undefined,
            extent: undefined,
            selected: true,
          };
        }),
    );
    getYjsDoc(state).transact(() => {
      const wnodes = state.workspace.nodes!;
      for (const node of state.workspace.nodes!) {
        const g = groups[node.parentId as string];
        if (!g) continue;
        node.position.x += g.position.x;
        node.position.y += g.position.y;
        node.parentId = undefined;
        node.extent = undefined;
      }
      for (const groupId in groups) {
        const groupIdx = wnodes.findIndex((n) => n.id === groupId);
        wnodes.splice(groupIdx, 1);
      }
    });
  }
  const areMultipleNodesSelected = nodes.filter((n) => n.selected).length > 1;
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
          value={state.workspace.env!}
          onChange={(env) => {
            state.workspace.env = env;
          }}
        />
        <div className="tools text-secondary">
          {areMultipleNodesSelected && (
            <Tooltip doc="Group selected nodes">
              <button className="btn btn-link" onClick={groupSelection}>
                <GroupIcon />
              </button>
            </Tooltip>
          )}
          {isAnyGroupSelected && (
            <Tooltip doc="Ungroup selected nodes">
              <button className="btn btn-link" onClick={ungroupSelection}>
                <UngroupIcon />
              </button>
            </Tooltip>
          )}
          <Tooltip doc="Delete selected nodes and edges">
            <button className="btn btn-link" onClick={deleteSelection}>
              <Backspace />
            </button>
          </Tooltip>
          <Tooltip doc={pausedUIState ? "Resume automatic execution" : "Pause automatic execution"}>
            <button className="btn btn-link" onClick={togglePause}>
              {pausedUIState ? <Play /> : <Pause />}
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
        <LynxKiteState.Provider value={state}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onPaneClick={toggleNodeSearch}
            onConnect={onConnect}
            proOptions={{ hideAttribution: true }}
            maxZoom={10}
            minZoom={0.2}
            zoomOnScroll={false}
            panOnScroll={true}
            panOnDrag={false}
            selectionOnDrag={true}
            panOnScrollSpeed={1}
            preventScrolling={false}
            defaultEdgeOptions={{
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: "black",
                width: 15,
                height: 15,
              },
              style: {
                strokeWidth: 2,
                stroke: "black",
              },
            }}
            fitViewOptions={{ maxZoom: 1 }}
          >
            <Controls />
            {nodeSearchSettings && categoryHierarchy && (
              <NodeSearch
                pos={nodeSearchSettings.pos}
                categoryHierarchy={categoryHierarchy}
                onCancel={closeNodeSearch}
                onAdd={addNodeFromSearch}
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
