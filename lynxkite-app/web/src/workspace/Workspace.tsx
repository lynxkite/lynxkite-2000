import { getYjsDoc, syncedStore } from "@syncedstore/core";
import {
  type Connection,
  Controls,
  type Edge,
  MarkerType,
  MiniMap,
  type Node,
  ReactFlow,
  ReactFlowProvider,
  type XYPosition,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import axios from "axios";
import { type MouseEvent, useCallback, useEffect, useMemo, useState } from "react";
// The LynxKite workspace editor.
import { useParams } from "react-router";
import useSWR, { type Fetcher } from "swr";
import { WebsocketProvider } from "y-websocket";
// @ts-ignore
import Atom from "~icons/tabler/atom.jsx";
// @ts-ignore
import Backspace from "~icons/tabler/backspace.jsx";
// @ts-ignore
import Close from "~icons/tabler/x.jsx";
import type { Workspace, WorkspaceNode } from "../apiTypes.ts";
import favicon from "../assets/favicon.ico";
// import NodeWithTableView from './NodeWithTableView';
import EnvironmentSelector from "./EnvironmentSelector";
import { LynxKiteState } from "./LynxKiteState";
import NodeSearch, { type OpsOp, type Catalog, type Catalogs } from "./NodeSearch.tsx";
import NodeWithGraphCreationView from "./nodes/GraphCreationNode.tsx";
import NodeWithImage from "./nodes/NodeWithImage.tsx";
import NodeWithParams from "./nodes/NodeWithParams";
import NodeWithTableView from "./nodes/NodeWithTableView.tsx";
import NodeWithVisualization from "./nodes/NodeWithVisualization.tsx";

export default function (props: any) {
  return (
    <ReactFlowProvider>
      <LynxKiteFlow {...props} />
    </ReactFlowProvider>
  );
}

function LynxKiteFlow() {
  const updateNodeInternals = useUpdateNodeInternals();
  const reactFlow = useReactFlow();
  const [nodes, setNodes] = useState([] as Node[]);
  const [edges, setEdges] = useState([] as Edge[]);
  const { path } = useParams();
  const [state, setState] = useState({ workspace: {} as Workspace });
  const [message, setMessage] = useState(null as string | null);
  useEffect(() => {
    const state = syncedStore({ workspace: {} as Workspace });
    setState(state);
    const doc = getYjsDoc(state);
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsProvider = new WebsocketProvider(`${proto}//${location.host}/ws/crdt`, path!, doc);
    const onChange = (_update: any, origin: any, _doc: any, _tr: any) => {
      if (origin === wsProvider) {
        // An update from the CRDT. Apply it to the local state.
        // This is only necessary because ReactFlow keeps secret internal copies of our stuff.
        if (!state.workspace) return;
        if (!state.workspace.nodes) return;
        if (!state.workspace.edges) return;
        for (const n of state.workspace.nodes) {
          if (n.dragHandle !== ".bg-primary") {
            n.dragHandle = ".bg-primary";
          }
        }
        setNodes([...state.workspace.nodes] as Node[]);
        setEdges([...state.workspace.edges] as Edge[]);
        for (const node of state.workspace.nodes) {
          // Make sure the internal copies are updated.
          updateNodeInternals(node.id);
        }
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
            Object.assign(node.position, ch.position);
          });
        } else if (ch.type === "select") {
        } else if (ch.type === "dimensions") {
          getYjsDoc(state).transact(() => Object.assign(node, ch.dimensions));
        } else if (ch.type === "remove") {
          wnodes.splice(nodeIndex, 1);
        } else if (ch.type === "replace") {
          // Ideally we would only update the parameter that changed. But ReactFlow does not give us that detail.
          const u = {
            collapsed: ch.item.data.collapsed,
            // The "..." expansion on a Y.map returns an empty object. Copying with fromEntries/entries instead.
            params: {
              ...Object.fromEntries(Object.entries(ch.item.data.params)),
            },
            __execution_delay: ch.item.data.__execution_delay,
          };
          getYjsDoc(state).transact(() => Object.assign(node.data, u));
        } else {
          console.log("Unknown node change", ch);
        }
      }
    },
    [state],
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
  const catalog = useSWR("/api/catalog", fetcher);
  const [suppressSearchUntil, setSuppressSearchUntil] = useState(0);
  const [nodeSearchSettings, setNodeSearchSettings] = useState(
    undefined as
      | {
          pos: XYPosition;
          boxes: Catalog;
        }
      | undefined,
  );
  const nodeTypes = useMemo(
    () => ({
      basic: NodeWithParams,
      visualization: NodeWithVisualization,
      image: NodeWithImage,
      table_view: NodeWithTableView,
      graph_creation_view: NodeWithGraphCreationView,
    }),
    [],
  );

  // Global keyboard shortcuts.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Show the node search dialog on "/".
      if (event.key === "/" && !nodeSearchSettings && !isTypingInFormElement()) {
        event.preventDefault();
        setNodeSearchSettings({
          pos: { x: 100, y: 100 },
          boxes: catalog.data![state.workspace.env!],
        });
      }
    };
    // TODO: Switch to keydown once https://github.com/xyflow/xyflow/pull/5055 is merged.
    document.addEventListener("keyup", handleKeyDown);
    return () => {
      document.removeEventListener("keyup", handleKeyDown);
    };
  }, [catalog.data, nodeSearchSettings, state.workspace.env]);

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
      if (suppressSearchUntil > Date.now()) return;
      if (nodeSearchSettings) {
        closeNodeSearch();
        return;
      }
      event.preventDefault();
      setNodeSearchSettings({
        pos: { x: event.clientX, y: event.clientY },
        boxes: catalog.data![state.workspace.env!],
      });
    },
    [catalog, state, nodeSearchSettings, suppressSearchUntil, closeNodeSearch],
  );
  function addNode(node: Partial<WorkspaceNode>, state: { workspace: Workspace }, nodes: Node[]) {
    const title = node.data?.title;
    let i = 1;
    node.id = `${title} ${i}`;
    const wnodes = state.workspace.nodes!;
    while (wnodes.find((x) => x.id === node.id)) {
      i += 1;
      node.id = `${title} ${i}`;
    }
    wnodes.push(node as WorkspaceNode);
    setNodes([...nodes, node as WorkspaceNode]);
  }
  function nodeFromMeta(meta: OpsOp): Partial<WorkspaceNode> {
    const node: Partial<WorkspaceNode> = {
      type: meta.type,
      data: {
        meta: meta,
        title: meta.name,
        params: Object.fromEntries(Object.values(meta.params).map((p) => [p.name, p.default])),
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
      addNode(node, state, nodes);
      closeNodeSearch();
    },
    [nodeSearchSettings, state, reactFlow, nodes, closeNodeSearch],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      setSuppressSearchUntil(Date.now() + 200);
      const edge = {
        id: `${connection.source} ${connection.target}`,
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
      addNode(node, state, nodes);
    } catch (error) {
      setMessage("File upload failed.");
    }
  }
  return (
    <div className="workspace">
      <div className="top-bar bg-neutral">
        <a className="logo" href="">
          <img alt="" src={favicon} />
        </a>
        <div className="ws-name">{path}</div>
        <EnvironmentSelector
          options={Object.keys(catalog.data || {})}
          value={state.workspace.env!}
          onChange={(env) => {
            state.workspace.env = env;
          }}
        />
        <div className="tools text-secondary">
          <a href="">
            <Atom />
          </a>
          <a href="">
            <Backspace />
          </a>
          <a href={`/dir/${parentDir}`}>
            <Close />
          </a>
        </div>
      </div>
      <div style={{ height: "100%", width: "100vw" }} onDragOver={onDragOver} onDrop={onDrop}>
        <LynxKiteState.Provider value={state}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onPaneClick={toggleNodeSearch}
            onConnect={onConnect}
            proOptions={{ hideAttribution: true }}
            maxZoom={1}
            minZoom={0.2}
            zoomOnScroll={false}
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
          >
            <Controls />
            <MiniMap />
            {nodeSearchSettings && (
              <NodeSearch
                pos={nodeSearchSettings.pos}
                boxes={nodeSearchSettings.boxes}
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
