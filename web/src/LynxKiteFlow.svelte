<script lang="ts">
  import { writable } from 'svelte/store';
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
  import LynxKiteNode from './LynxKiteNode.svelte';
  import NodeSearch from './NodeSearch.svelte';
  import '@xyflow/svelte/dist/style.css';

  const { screenToFlowPosition } = useSvelteFlow();
  const nodeTypes: NodeTypes = {
    basic: LynxKiteNode,
  };

  const nodes = writable<Node[]>([
    {
      id: '1',
      type: 'basic',
      data: { title: 'Compute PageRank', params: { damping: 0.85, iterations: 3 } },
      position: { x: 0, y: 0 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    },
    {
      id: '3',
      type: 'basic',
      data: { title: 'Import Parquet', params: { filename: '/tmp/x.parquet' } },
      position: { x: -300, y: 0 },
      sourcePosition: Position.Right,
    },
  ]);

  const edges = writable<Edge[]>([
    {
      id: '3-1',
      source: '3',
      target: '1',
      markerEnd: {
        type: MarkerType.ArrowClosed,
      },
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

  const boxes = [
    {
      type: 'basic',
      data: { title: 'Import Parquet', params: { filename: '/tmp/x.parquet' } },
      sourcePosition: Position.Right,
    },
    {
      type: 'basic',
      data: { title: 'Export Parquet', params: { filename: '/tmp/x.parquet' } },
      sourcePosition: Position.Right,
    },
    {
      type: 'basic',
      data: { title: 'Export CSV', params: { filename: '/tmp/x.csv' } },
      sourcePosition: Position.Right,
    },
    {
      type: 'basic',
      data: { title: 'Compute PageRank', params: { damping: 0.85, iterations: 3 } },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    },
  ];

  let nodeSearchPos: XYPosition | undefined = undefined
</script>

<div style:height="100vh">
  <SvelteFlow {nodes} {edges} {nodeTypes} fitView
    on:paneclick={toggleNodeSearch}
    proOptions={{ hideAttribution: true }}
    maxZoom={1.5}
    minZoom={0.3}
    >
    <Background />
    <Controls />
    <Background />
    <MiniMap />
    {#if nodeSearchPos}<NodeSearch boxes={boxes} on:cancel={closeNodeSearch} on:add={addNode} pos={nodeSearchPos} />{/if}
  </SvelteFlow>
</div>
