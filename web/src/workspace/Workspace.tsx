// The LynxKite workspace editor.
import { useParams } from "react-router";
import useSWR, { Fetcher } from 'swr';
import { useEffect, useMemo, useCallback, useState, MouseEvent } from "react";
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
  useReactFlow,
  MiniMap,
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
import { Workspace, WorkspaceNode } from "../apiTypes.ts";
import NodeSearch, { OpsOp, Catalog, Catalogs } from "./NodeSearch.tsx";

export default function (props: any) {
  return (
    <ReactFlowProvider>
      <LynxKiteFlow {...props} />
    </ReactFlowProvider>
  );
}


function LynxKiteFlow() {
  const updateNodeInternals = useUpdateNodeInternals()
  const reactFlow = useReactFlow();
  const [nodes, setNodes] = useState([] as Node[]);
  const [edges, setEdges] = useState([] as Edge[]);
  const { path } = useParams();
  const [state, setState] = useState({ workspace: {} as Workspace });
  useEffect(() => {
    const state = syncedStore({ workspace: {} as Workspace });
    setState(state);
    const doc = getYjsDoc(state);
    const wsProvider = new WebsocketProvider("ws://localhost:8000/ws/crdt", path!, doc);
    const onChange = (_update: any, origin: any, _doc: any, _tr: any) => {
      if (origin === wsProvider) {
        // An update from the CRDT. Apply it to the local state.
        // This is only necessary because ReactFlow keeps secret internal copies of our stuff.
        if (!state.workspace) return;
        if (!state.workspace.nodes) return;
        if (!state.workspace.edges) return;
        console.log('update', JSON.parse(JSON.stringify(state.workspace)));
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
        getYjsDoc(state).transact(() => {
          Object.assign(node.position, ch.position);
        });
      } else if (ch.type === 'select') {
      } else if (ch.type === 'dimensions') {
        getYjsDoc(state).transact(() => Object.assign(node, ch.dimensions));
      } else if (ch.type === 'remove') {
        wnodes.splice(nodeIndex, 1);
      } else if (ch.type === 'replace') {
        // Ideally we would only update the parameter that changed. But ReactFlow does not give us that detail.
        const u = {
          collapsed: ch.item.data.collapsed,
          // The "..." expansion on a Y.map returns an empty object. Copying with fromEntries/entries instead.
          params: { ...Object.fromEntries(Object.entries(ch.item.data.params)) },
          __execution_delay: ch.item.data.__execution_delay,
        };
        getYjsDoc(state).transact(() => Object.assign(node.data, u));
      } else {
        console.log('Unknown node change', ch);
      }
    }
  };
  const onEdgesChange = (changes: any[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  };

  const fetcher: Fetcher<Catalogs> = (resource: string, init?: RequestInit) => fetch(resource, init).then(res => res.json());
  const catalog = useSWR('/api/catalog', fetcher);

  const nodeTypes = useMemo(() => ({
    basic: NodeWithParams,
    table_view: NodeWithParams,
  }), []);
  function closeNodeSearch() {
    setNodeSearchSettings(undefined);
  }
  function toggleNodeSearch(event: MouseEvent) {
    if (nodeSearchSettings) {
      closeNodeSearch();
      return;
    }
    event.preventDefault();
    setNodeSearchSettings({
      pos: { x: event.clientX, y: event.clientY },
      boxes: catalog.data![state.workspace.env!],
    });
  }
  function addNode(meta: OpsOp) {
    const node: Partial<WorkspaceNode> = {
      type: meta.type,
      data: {
        meta: meta,
        title: meta.name,
        params: Object.fromEntries(
          Object.values(meta.params).map((p) => [p.name, p.default])),
      },
    };
    const nss = nodeSearchSettings!;
    node.position = reactFlow.screenToFlowPosition({ x: nss.pos.x, y: nss.pos.y });
    const title = meta.name;
    let i = 1;
    node.id = `${title} ${i}`;
    const wnodes = state.workspace.nodes!;
    while (wnodes.find((x) => x.id === node.id)) {
      i += 1;
      node.id = `${title} ${i}`;
    }
    wnodes.push(node as WorkspaceNode);
    setNodes([...nodes, node as WorkspaceNode]);
    closeNodeSearch();
  }
  const [nodeSearchSettings, setNodeSearchSettings] = useState(undefined as {
    pos: XYPosition,
    boxes: Catalog,
  } | undefined);

  function nodeClick(e: any) {
    const node = e.detail.node;
    const meta = node.data.meta;
    if (!meta) return;
    const sub_nodes = meta.sub_nodes;
    if (!sub_nodes) return;
    const event = e.detail.event;
    if (event.target.classList.contains('title')) return;
    setNodeSearchSettings({
      pos: { x: event.clientX, y: event.clientY },
      boxes: sub_nodes,
    });
  }
  function onConnect(params: Connection) {
    // const edge = {
    //   id: `${params.source} ${params.target}`,
    //   source: params.source,
    //   sourceHandle: params.sourceHandle,
    //   target: params.target,
    //   targetHandle: params.targetHandle,
    // };
    // state.workspace.edges!.push(edge);
  }
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
            onPaneClick={toggleNodeSearch}
            onNodeClick={nodeClick}
            onConnect={onConnect}
            proOptions={{ hideAttribution: true }}
            maxZoom={3}
            minZoom={0.3}
            defaultEdgeOptions={{ markerEnd: { type: MarkerType.Arrow } }}
          >
            <Controls />
            <MiniMap />
            {nodeSearchSettings &&
              <NodeSearch pos={nodeSearchSettings.pos} boxes={nodeSearchSettings.boxes} onCancel={closeNodeSearch} onAdd={addNode} />
            }
          </ReactFlow>
        </LynxKiteState.Provider>
      </div>
    </div>

  );
}
