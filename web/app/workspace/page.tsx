'use client';
import useSWR from 'swr';
import { useMemo } from "react";
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
} from '@xyflow/React';
// @ts-ignore
import ArrowBack from '~icons/tabler/arrow-back.jsx';
// @ts-ignore
import Backspace from '~icons/tabler/backspace.jsx';
// @ts-ignore
import Atom from '~icons/tabler/atom.jsx';
// import NodeWithParams from './NodeWithParams';
// import NodeWithVisualization from './NodeWithVisualization';
// import NodeWithImage from './NodeWithImage';
// import NodeWithTableView from './NodeWithTableView';
// import NodeWithSubFlow from './NodeWithSubFlow';
// import NodeWithArea from './NodeWithArea';
// import NodeSearch from './NodeSearch';
import EnvironmentSelector from './EnvironmentSelector';
import '@xyflow/react/dist/style.css';

export default function Home() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  let path = '';

  // const { screenToFlowPosition } = useReactFlow();
  // const queryClient = useQueryClient();
  // const backendWorkspace = useQuery(['workspace', path], async () => {
  //   const res = await fetch(`/api/load?path=${path}`);
  //   return res.json();
  // }, { staleTime: 10000, retry: false });
  // const mutation = useMutation(async (update) => {
  //   const res = await fetch('/api/save', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //     body: JSON.stringify(update),
  //   });
  //   return await res.json();
  // }, {
  //   onSuccess: data => queryClient.setQueryData(['workspace', path], data),
  // });

  // const nodeTypes: NodeTypes = {
  //   basic: NodeWithParams,
  //   visualization: NodeWithVisualization,
  //   image: NodeWithImage,
  //   table_view: NodeWithTableView,
  //   sub_flow: NodeWithSubFlow,
  //   area: NodeWithArea,
  // };

  // const nodes = writable<Node[]>([]);
  // const edges = writable<Edge[]>([]);
  // let doNotSave = true;
  // $: if ($backendWorkspace.isSuccess) {
  //   doNotSave = true; // Change is coming from the backend.
  //   nodes.set(JSON.parse(JSON.stringify($backendWorkspace.data?.nodes || [])));
  //   edges.set(JSON.parse(JSON.stringify($backendWorkspace.data?.edges || [])));
  //   doNotSave = false;
  // }

  // function closeNodeSearch() {
  //   nodeSearchSettings = undefined;
  // }
  // function toggleNodeSearch({ detail: { event } }) {
  //   if (nodeSearchSettings) {
  //     closeNodeSearch();
  //     return;
  //   }
  //   event.preventDefault();
  //   nodeSearchSettings = {
  //     pos: { x: event.clientX, y: event.clientY },
  //     boxes: $catalog.data[$backendWorkspace.data?.env],
  //   };
  // }
  // function addNode(e) {
  //   const meta = { ...e.detail };
  //   nodes.update((n) => {
  //     const node = {
  //       type: meta.type,
  //       data: {
  //         meta: meta,
  //         title: meta.name,
  //         params: Object.fromEntries(
  //           Object.values(meta.params).map((p) => [p.name, p.default])),
  //       },
  //     };
  //     node.position = screenToFlowPosition({ x: nodeSearchSettings.pos.x, y: nodeSearchSettings.pos.y });
  //     const title = node.data.title;
  //     let i = 1;
  //     node.id = `${title} ${i}`;
  //     while (n.find((x) => x.id === node.id)) {
  //       i += 1;
  //       node.id = `${title} ${i}`;
  //     }
  //     node.parentId = nodeSearchSettings.parentId;
  //     if (node.parentId) {
  //       node.extent = 'parent';
  //       const parent = n.find((x) => x.id === node.parentId);
  //       node.position = { x: node.position.x - parent.position.x, y: node.position.y - parent.position.y };
  //     }
  //     return [...n, node]
  //   });
  //   closeNodeSearch();
  // }
  const fetcher = (resource: string, init?: RequestInit) => fetch(resource, init).then(res => res.json());
  const catalog = useSWR('/api/catalog', fetcher);

  // let nodeSearchSettings: {
  //   pos: XYPosition,
  //   boxes: any[],
  //   parentId: string,
  // };

  // const graph = derived([nodes, edges], ([nodes, edges]) => ({ nodes, edges }));
  // // Like JSON.stringify, but with keys sorted.
  // function orderedJSON(obj: any) {
  //   const allKeys = new Set();
  //   JSON.stringify(obj, (key, value) => (allKeys.add(key), value));
  //   return JSON.stringify(obj, Array.from(allKeys).sort());
  // }
  // graph.subscribe(async (g) => {
  //   if (doNotSave) return;
  //   const dragging = g.nodes.find((n) => n.dragging);
  //   if (dragging) return;
  //   const resizing = g.nodes.find((n) => n.data?.beingResized);
  //   if (resizing) return;
  //   scheduleSave(g);
  // });
  // let saveTimeout;
  // function scheduleSave(g) {
  //   // A slight delay, so we don't send a million requests when a node is resized, for example.
  //   clearTimeout(saveTimeout);
  //   saveTimeout = setTimeout(() => save(g), 500);
  // }
  // function save(g) {
  //   g = JSON.parse(JSON.stringify(g));
  //   for (const node of g.nodes) {
  //     delete node.measured;
  //     delete node.selected;
  //     delete node.dragging;
  //     delete node.beingResized;
  //   }
  //   for (const node of g.edges) {
  //     delete node.markerEnd;
  //     delete node.selected;
  //   }
  //   g.env = $backendWorkspace.data?.env;
  //   const ws = orderedJSON(g);
  //   const bd = orderedJSON($backendWorkspace.data);
  //   if (ws === bd) return;
  //   console.log('changed', JSON.stringify(diff(g, $backendWorkspace.data), null, 2));
  //   $mutation.mutate({ path, ws: g });
  // }
  // function nodeClick(e) {
  //   const node = e.detail.node;
  //   const meta = node.data.meta;
  //   if (!meta) return;
  //   const sub_nodes = meta.sub_nodes;
  //   if (!sub_nodes) return;
  //   const event = e.detail.event;
  //   if (event.target.classList.contains('title')) return;
  //   nodeSearchSettings = {
  //     pos: { x: event.clientX, y: event.clientY },
  //     boxes: sub_nodes,
  //     parentId: node.id,
  //   };
  // }
  // $: parentDir = path.split('/').slice(0, -1).join('/');

  const nodeTypes = useMemo(() => ({}), []);
  return (

    <div className="page">
      <div className="top-bar">
        <div className="ws-name">
          <a href=""><img src="/favicon.ico" /></a>
          {path}
        </div>
        <div className="tools">
          <EnvironmentSelector
            options={Object.keys(catalog.data || {})}
            value={'asd'}
            onChange={(env) => 1}
          />
          <a href=""><Atom /></a>
          <a href=""><Backspace /></a>
          <a href="#dir?path={parentDir}"><ArrowBack /></a>
        </div>
      </div>
      <div style={{ height: "100%", width: '100vw' }}>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView
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
      </div>
    </div>

  );
}
