<script lang="ts">
  import { writable } from 'svelte/store';
  import {
    SvelteFlow,
    Controls,
    Background,
    MiniMap,
    MarkerType,
    Position,
    type Node,
    type Edge,
  } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import NodeSearch from './NodeSearch.svelte';
  import '@xyflow/svelte/dist/style.css';

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

  function onPaneContextMenu({ detail: { event } }) {
    event.preventDefault();
    const width = 500;
    const height = 200;
    showNodeSearch = {
      top: event.clientY < height - 200 ? event.clientY : undefined,
      left: event.clientX < width - 200 ? event.clientX : undefined,
      right: event.clientX >= width - 200 ? width - event.clientX : undefined,
      bottom: event.clientY >= height - 200 ? height - event.clientY : undefined
    };
    showNodeSearch = {
      top: event.clientY,
      left: event.clientX - 150,
    };
  }
  function addNode(node: Node) {
    nodes.update((n) => [...n, node]);
  }

  let showNodeSearch;
</script>

<div style:height="100vh">
  <SvelteFlow {nodes} {edges} {nodeTypes} fitView
    on:nodecontextmenu={onPaneContextMenu}
    on:panecontextmenu={onPaneContextMenu}
    on:paneclick={() => showNodeSearch = undefined}
    on:nodeclick={() => showNodeSearch = undefined}
    >
    <Background />
    <Controls />
    <Background />
    <MiniMap />
    {#if showNodeSearch}<NodeSearch on:add={addNode} pos={showNodeSearch} />{/if}
  </SvelteFlow>
</div>
