<script lang="ts">
  import { writable, derived } from 'svelte/store';
  import {
    SvelteFlow,
    Controls,
    Background,
    MiniMap,
    MarkerType,
    Position,
    useSvelteFlow,
    type XYPosition,
    type Node,
    type Edge,
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

  const nodes = writable<Node[]>([]);

  const edges = writable<Edge[]>([
    {
      id: '3-1',
      source: '3',
      target: '1',
      // markerEnd: { type: MarkerType.ArrowClosed },
    },
    {
      id: '3-4',
      source: '1',
      target: '4',
      // markerEnd: { type: MarkerType.ArrowClosed },
    },
  ]);

  function closeNodeSearch() {
    nodeSearchPos = undefined;
  }
  function toggleNodeSearch({ detail: { event } }) {
    if (nodeSearchPos) {
      closeNodeSearch();
      return;
    }
    event.preventDefault();
    const width = 500;
    const height = 200;
    nodeSearchPos = {
      top: event.clientY < height - 200 ? event.clientY : undefined,
      left: event.clientX < width - 200 ? event.clientX : undefined,
      right: event.clientX >= width - 200 ? width - event.clientX : undefined,
      bottom: event.clientY >= height - 200 ? height - event.clientY : undefined
    };
    nodeSearchPos = {
      top: event.clientY,
      left: event.clientX - 150,
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
  let backendWorkspace;
  // Like JSON.stringify, but with keys sorted.
  function orderedJSON(obj: any) {
    const allKeys = new Set();
    JSON.stringify(obj, (key, value) => (allKeys.add(key), value));
    return JSON.stringify(obj, Array.from(allKeys).sort());
  }
  graph.subscribe(async (g) => {
    const dragging = g.nodes.find((n) => n.dragging);
    if (dragging) return;
    g = JSON.parse(JSON.stringify(g));
    for (const node of g.nodes) {
      delete node.computed;
    }
    const ws = orderedJSON(g);
    if (ws === backendWorkspace) return;
    console.log('current vs backend', '\n' + ws, '\n' + backendWorkspace);
    backendWorkspace = ws;
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: ws,
    });
    const j = await res.json();
    backendWorkspace = orderedJSON(j);
    nodes.set(j.nodes);
  });
</script>

<div style:height="100vh">
  <SvelteFlow {nodes} {edges} {nodeTypes} fitView
    on:paneclick={toggleNodeSearch}
    proOptions={{ hideAttribution: true }}
    maxZoom={1.5}
    minZoom={0.3}
    >
    <Background patternColor="#39bcf3" />
    <Controls />
    <MiniMap />
    {#if nodeSearchPos}<NodeSearch boxes={$boxes} on:cancel={closeNodeSearch} on:add={addNode} pos={nodeSearchPos} />{/if}
  </SvelteFlow>
</div>
