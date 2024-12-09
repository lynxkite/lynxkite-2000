// The LynxKite workspace editor.
import { useParams } from "react-router";
import useSWR from 'swr';
import { useMemo } from "react";
import favicon from '../assets/favicon.ico';
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  MarkerType,
  useReactFlow,
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

export default function () {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { path } = useParams();

  const sstore = syncedStore({ workspace: {} });
  const doc = getYjsDoc(sstore);
  const wsProvider = new WebsocketProvider("ws://localhost:8000/ws/crdt", path!, doc);
  wsProvider; // Just to disable the lint warning. The life cycle of this object is a mystery.
  const state = useSyncedStore(sstore);

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
          value={state.workspace?.env}
          onChange={(env) => state.workspace.env = env}
        />
        <div className="tools text-secondary">
          <a href=""><Atom /></a>
          <a href=""><Backspace /></a>
          <a href={'/dir/' + parentDir}><ArrowBack /></a>
        </div>
      </div>
      <div style={{ height: "100%", width: '100vw' }}>
        <LynxKiteState.Provider value={state}>
          <ReactFlow nodes={state.workspace?.nodes} edges={state.workspace?.edges} nodeTypes={nodeTypes} fitView
            proOptions={{ hideAttribution: true }}
            maxZoom={3}
            minZoom={0.3}
            defaultEdgeOptions={{ markerEnd: { type: MarkerType.Arrow } }}
          >
            <Controls />
            <MiniMap />
            {/* {#if nodeSearchSettings}
          <NodeSearch pos={nodeSearchSettings.pos} boxes={nodeSearchSettings.boxes} on:cancel={closeNodeSearch} on:add={addNode} />
          {/if} */}
          </ReactFlow>
        </LynxKiteState.Provider>
      </div>
    </div>

  );
}
