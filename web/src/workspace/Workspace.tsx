// The LynxKite workspace editor.
import { useParams } from "react-router";
import useSWR from 'swr';
import { useMemo, useCallback, useState } from "react";
import favicon from '../assets/favicon.ico';
import {
  ReactFlow,
  Controls,
  MiniMap,
  MarkerType,
  useReactFlow,
  useUpdateNodeInternals,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  type XYPosition,
  type Node,
  type Edge,
  type Connection,
  type NodeTypes,
} from '@xyflow/react';
// @ts-ignore
import ArrowBack from '~icons/tabler/arrow-back.jsx';
// @ts-ignore
import Backspace from '~icons/tabler/backspace.jsx';
// @ts-ignore
import Atom from '~icons/tabler/atom.jsx';
import { syncedStore, getYjsDoc } from "@syncedstore/core";
import { useSyncedStore } from "@syncedstore/react";
import { WebsocketProvider } from "y-websocket";
import NodeWithParams from './nodes/NodeWithParams';
// import NodeWithVisualization from './NodeWithVisualization';
// import NodeWithImage from './NodeWithImage';
// import NodeWithTableView from './NodeWithTableView';
// import NodeWithSubFlow from './NodeWithSubFlow';
// import NodeWithArea from './NodeWithArea';
// import NodeSearch from './NodeSearch';
import EnvironmentSelector from './EnvironmentSelector';
import { LynxKiteState } from './LynxKiteState';
import '@xyflow/react/dist/style.css';
import { Workspace } from "../apiTypes.ts";

export default function (props: any) {
  return (
    <ReactFlowProvider>
      <LynxKiteFlow {...props} />
    </ReactFlowProvider>
  );
}


function LynxKiteFlow() {
  const updateNodeInternals = useUpdateNodeInternals();
  const { screenToFlowPosition } = useReactFlow();
  const [nodes, setNodes] = useState([] as Node[]);
  const [edges, setEdges] = useState([] as Edge[]);
  const { path } = useParams();

  const sstore = syncedStore({ workspace: {} });
  const doc = getYjsDoc(sstore);
  const wsProvider = useMemo(() => new WebsocketProvider("ws://localhost:8000/ws/crdt", path!, doc), [path]);
  wsProvider; // Just to disable the lint warning. The life cycle of this object is a mystery.
  const state: { workspace: Workspace } = useSyncedStore(sstore);
  const onNodesChange = useCallback(
    (changes: any[]) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
      for (const ch of changes) {
        if (ch.type === 'position') {
          const node = state.workspace?.nodes?.find((n) => n.id === ch.id);
          if (node) {
            node.position = ch.position;
          }
        }
      }
    },
    [],
  );
  const onEdgesChange = useCallback(
    (changes: any[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );
  if (state?.workspace?.nodes && JSON.stringify(nodes) !== JSON.stringify([...state.workspace.nodes as Node[]])) {
    const updated = Object.fromEntries(state.workspace.nodes.map((n) => [n.id, n]));
    const oldNodes = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const updatedNodes = nodes.filter(n => updated[n.id]).map((n) => ({ ...n, ...updated[n.id] })) as Node[];
    const newNodes = state.workspace.nodes.filter((n) => !oldNodes[n.id]);
    const allNodes = [...updatedNodes, ...newNodes];
    if (JSON.stringify(allNodes) !== JSON.stringify(nodes)) {
      setNodes(allNodes as Node[]);
    }
  }
  if (state?.workspace?.edges && JSON.stringify(edges) !== JSON.stringify([...state.workspace.edges as Edge[]])) {
    const updated = Object.fromEntries(state.workspace.edges.map((e) => [e.id, e]));
    const oldEdges = Object.fromEntries(edges.map((e) => [e.id, e]));
    const updatedEdges = edges.filter(e => updated[e.id]).map((e) => ({ ...e, ...updated[e.id] })) as Edge[];
    const newEdges = state.workspace.edges.filter((e) => !oldEdges[e.id]);
    const allEdges = [...updatedEdges, ...newEdges];
    if (JSON.stringify(allEdges) !== JSON.stringify(edges)) {
      setEdges(allEdges as Edge[]);
    }
  }

  const fetcher = (resource: string, init?: RequestInit) => fetch(resource, init).then(res => res.json());
  const catalog = useSWR('/api/catalog', fetcher);

  const nodeTypes = useMemo(() => ({
    basic: NodeWithParams,
    table_view: NodeWithParams,
  }), []);
  const parentDir = path!.split('/').slice(0, -1).join('/');
  return (
    <div className="workspace">
      <div className="top-bar bg-neutral">
        <a className="logo" href=""><img src={favicon} /></a>
        <div className="ws-name">
          {path}
        </div>
        <EnvironmentSelector
          options={Object.keys(catalog.data || {})}
          value={state.workspace.env!}
          onChange={(env) => { state.workspace.env = env; }}
        />
        <div className="tools text-secondary">
          <a href=""><Atom /></a>
          <a href=""><Backspace /></a>
          <a href={'/dir/' + parentDir}><ArrowBack /></a>
        </div>
      </div>
      <div style={{ height: "100%", width: '100vw' }}>
        <LynxKiteState.Provider value={state.workspace}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes} fitView
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            proOptions={{ hideAttribution: true }}
            maxZoom={3}
            minZoom={0.3}
            defaultEdgeOptions={{ markerEnd: { type: MarkerType.Arrow } }}
          >
            <Controls />
            {/* {#if nodeSearchSettings}
          <NodeSearch pos={nodeSearchSettings.pos} boxes={nodeSearchSettings.boxes} on:cancel={closeNodeSearch} on:add={addNode} />
          {/if} */}
          </ReactFlow>
        </LynxKiteState.Provider>
      </div>
    </div>

  );
}
