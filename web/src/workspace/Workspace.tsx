// The LynxKite workspace editor.
import { useParams } from "react-router";
import useSWR from 'swr';
import { useEffect, useMemo, useCallback, useState } from "react";
import favicon from '../assets/favicon.ico';
import {
  ReactFlow,
  Controls,
  MarkerType,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  useUpdateNodeInternals,
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
  const updateNodeInternals = useUpdateNodeInternals()
  const [nodes, setNodes] = useState([] as Node[]);
  const [edges, setEdges] = useState([] as Edge[]);
  const { path } = useParams();
  const [state, setState] = useState({ workspace: {} as Workspace });
  useEffect(() => {
    const state = syncedStore({ workspace: {} as Workspace });
    setState(state);
    const doc = getYjsDoc(state);
    const wsProvider = new WebsocketProvider("ws://localhost:8000/ws/crdt", path!, doc);
    const onChange = (update: any, origin: any, doc: any, tr: any) => {
      if (origin === wsProvider) {
        // An update from the CRDT. Apply it to the local state.
        // This is only necessary because ReactFlow keeps secret internal copies of our stuff.
        if (!state.workspace) return;
        if (!state.workspace.nodes) return;
        if (!state.workspace.edges) return;
        setNodes([...state.workspace.nodes] as Node[]);
        setEdges([...state.workspace.edges] as Edge[]);
        for (const node of state.workspace.nodes) {
          // Make sure the internal copies are updated.
          updateNodeInternals(node.id);
        }
      }
    };
    doc.on('update', onChange);
    return () => {
      doc.destroy();
      wsProvider.destroy();
    }
  }, [path]);

  const onNodesChange = (changes: any[]) => {
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
      if (ch.type === 'position' && !isNaN(ch.position.x) && !isNaN(ch.position.y)) {
        Object.assign(node.position, ch.position);
      } else if (ch.type === 'select') {
      } else if (ch.type === 'dimensions') {
      } else if (ch.type === 'replace') {
        node.data.collapsed = ch.item.data.collapsed;
        node.data.params = { ...ch.item.data.params };
      } else {
        console.log('Unknown node change', ch);
      }
    }
  };
  const onEdgesChange = (changes: any[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  };

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
        <LynxKiteState.Provider value={state}>
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
