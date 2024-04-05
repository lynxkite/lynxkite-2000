<script lang="ts">
  import { writable } from 'svelte/store';
  import {
    SvelteFlow,
    Controls,
    Background,
    MiniMap,
    MarkerType,
    type Node,
    type Edge,
  } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
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
    },
    {
      id: '2',
      // type: 'basic',
      data: { label: 'World' },
      position: { x: 150, y: 150 },
    },
  ]);

  const edges = writable<Edge[]>([
    {
      id: '1-2',
      source: '1',
      target: '2',
      markerEnd: {
        type: MarkerType.Arrow
      },
    },
  ]);
</script>

<div style:height="100vh">
  <SvelteFlow {nodes} {edges} {nodeTypes} fitView>
    <Background />
    <Controls />
    <Background />
    <MiniMap />
  </SvelteFlow>
</div>
