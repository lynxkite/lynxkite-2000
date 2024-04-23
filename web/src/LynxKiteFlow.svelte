<script lang="ts">
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
  import NodeWithGraphView from './NodeWithGraphView.svelte';
  import NodeWithTableView from './NodeWithTableView.svelte';
  import NodeSearch from './NodeSearch.svelte';
  import '@xyflow/svelte/dist/style.css';

  const { screenToFlowPosition } = useSvelteFlow();

  const nodeTypes: NodeTypes = {
    basic: NodeWithParams,
    graph_view: NodeWithGraphView,
    table_view: NodeWithTableView,
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
    nodeSearchPos = undefined;
  }
  function toggleNodeSearch({ detail: { event } }) {
    if (nodeSearchPos) {
      closeNodeSearch();
      return;
    }
    event.preventDefault();
    nodeSearchPos = {
      top: event.offsetY,
      left: event.offsetX - 155,
    };
  }
  function addNode(e) {
    const node = {...e.detail};
    nodes.update((n) => {
      node.position = screenToFlowPosition({x: nodeSearchPos.left, y: nodeSearchPos.top});
      const title = node.data.title;
      let i = 1;
      node.id = `${title} ${i}`;
      while (n.find((x) => x.id === node.id)) {
        i += 1;
        node.id = `${title} ${i}`;
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

  let nodeSearchPos: XYPosition | undefined = undefined;

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
      return edges.filter((e) => e.source === connection.source || e.target !== connection.target);
    });
  }

</script>

<div style:height="100%">
  <SvelteFlow {nodes} {edges} {nodeTypes} fitView
    on:paneclick={toggleNodeSearch}
    proOptions={{ hideAttribution: true }}
    maxZoom={1.5}
    minZoom={0.3}
    onconnect={onconnect}
    >
    <Background patternColor="#39bcf3" />
    <Controls />
    <MiniMap />
    {#if nodeSearchPos}<NodeSearch boxes={$boxes} on:cancel={closeNodeSearch} on:add={addNode} pos={nodeSearchPos} />{/if}
  </SvelteFlow>
</div>
