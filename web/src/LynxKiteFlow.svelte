<script lang="ts">
  import { setContext } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import {
    SvelteFlow,
    Controls,
    Background,
    MiniMap,
    MarkerType,
    useSvelteFlow,
    type XYPosition,
    type Node,
    type Edge,
    type Connection,
    type NodeTypes,
  } from '@xyflow/svelte';
  import NodeWithParams from './NodeWithParams.svelte';
  import NodeWithParamsVertical from './NodeWithParamsVertical.svelte';
  import NodeWithGraphView from './NodeWithGraphView.svelte';
  import NodeWithTableView from './NodeWithTableView.svelte';
  import NodeWithSubFlow from './NodeWithSubFlow.svelte';
  import NodeSearch from './NodeSearch.svelte';
  import '@xyflow/svelte/dist/style.css';

  const { screenToFlowPosition } = useSvelteFlow();

  const nodeTypes: NodeTypes = {
    basic: NodeWithParams,
    vertical: NodeWithParamsVertical,
    graph_view: NodeWithGraphView,
    table_view: NodeWithTableView,
    sub_flow: NodeWithSubFlow,
  };

  export let path = '';
  const nodes = writable<Node[]>([]);
  const edges = writable<Edge[]>([]);
  let workspaceLoaded = false;
  async function fetchWorkspace(path) {
    if (!path) return;
    const res = await fetch(`/api/load?path=${path}`);
    const j = await res.json();
    nodes.set(j.nodes);
    edges.set(j.edges);
    backendWorkspace = orderedJSON(j);
    workspaceLoaded = true;
  }
  $: fetchWorkspace(path);

  function closeNodeSearch() {
    nodeSearchSettings = undefined;
  }
  function toggleNodeSearch({ detail: { event } }) {
    if (nodeSearchSettings) {
      closeNodeSearch();
      return;
    }
    event.preventDefault();
    nodeSearchSettings = {
      pos: { x: event.clientX, y: event.clientY },
      boxes: $boxes,
    };
  }
  function addNode(e) {
    const node = {...e.detail};
    nodes.update((n) => {
      node.position = screenToFlowPosition({x: nodeSearchSettings.pos.x, y: nodeSearchSettings.pos.y});
      node.data = { ...node.data };
      const title = node.data.title;
      node.data.params = Object.fromEntries(
        node.data.params.map((p) => [p.name, p.default]));
      let i = 1;
      node.id = `${title} ${i}`;
      while (n.find((x) => x.id === node.id)) {
        i += 1;
        node.id = `${title} ${i}`;
      }
      node.parentNode = nodeSearchSettings.parentNode;
      if (node.parentNode) {
        node.extent = 'parent';
        const parent = n.find((x) => x.id === node.parentNode);
        node.position = { x: node.position.x - parent.position.x, y: node.position.y - parent.position.y };
      }
      return [...n, node]
    });
    closeNodeSearch();
  }
  const boxes = writable([]);
  async function getBoxes() {
    const res = await fetch('/api/catalog');
    const j = await res.json();
    boxes.set(j);
  }
  getBoxes();

  let nodeSearchSettings: {
    pos: XYPosition,
    boxes: any[],
    parentNode: string,
  };

  const graph = derived([nodes, edges], ([nodes, edges]) => ({ nodes, edges }));
  let backendWorkspace: string;
  // Like JSON.stringify, but with keys sorted.
  function orderedJSON(obj: any) {
    const allKeys = new Set();
    JSON.stringify(obj, (key, value) => (allKeys.add(key), value));
    return JSON.stringify(obj, Array.from(allKeys).sort());
  }
  graph.subscribe(async (g) => {
    if (!workspaceLoaded) {
      return;
    }
    const dragging = g.nodes.find((n) => n.dragging);
    if (dragging) return;
    g = JSON.parse(JSON.stringify(g));
    for (const node of g.nodes) {
      delete node.computed;
    }
    const ws = orderedJSON(g);
    if (ws === backendWorkspace) return;
    // console.log('save', '\n' + ws, '\n' + backendWorkspace);
    backendWorkspace = ws;
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path, ws: g }),
    });
    const j = await res.json();
    backendWorkspace = orderedJSON(j);
    nodes.set(j.nodes);
  });
  function onconnect(connection: Connection) {
    edges.update((edges) => {
      // Only one source can connect to a given target.
      return edges.filter((e) =>
        e.source === connection.source
        || e.target !== connection.target
        || e.targetHandle !== connection.targetHandle);
    });
  }
  function getMeta(title) {
    return $boxes.find((m) => m.data.title === title);
  }
  setContext('LynxKiteFlow', { getMeta });
  function nodeClick(e) {
    const node = e.detail.node;
    const meta = getMeta(node.data.title);
    if (!meta) return;
    const sub_nodes = meta.sub_nodes;
    if (!sub_nodes) return;
    const event = e.detail.event;
    if (event.target.classList.contains('title')) return;
    nodeSearchSettings = {
      pos: { x: event.clientX, y: event.clientY },
      boxes: sub_nodes,
      parentNode: node.id,
    };
  }

</script>

<div style:height="100%">
  <SvelteFlow {nodes} {edges} {nodeTypes} fitView
    on:paneclick={toggleNodeSearch}
    on:nodeclick={nodeClick}
    proOptions={{ hideAttribution: true }}
    maxZoom={1.5}
    minZoom={0.3}
    onconnect={onconnect}
    >
    <Controls />
    <MiniMap />
    {#if nodeSearchSettings}
      <NodeSearch pos={nodeSearchSettings.pos} boxes={nodeSearchSettings.boxes} on:cancel={closeNodeSearch} on:add={addNode} />
    {/if}
  </SvelteFlow>
</div>
