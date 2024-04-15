<script lang="ts">
  import { onMount } from 'svelte';
  import { Handle, type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import Sigma from 'sigma';
  import * as graphology from 'graphology';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  let sigmaCanvas: HTMLElement;
  let sigmaInstance: Sigma;

  const graph = new graphology.Graph();
  graph.addNode("1", { label: "Node 1", x: 0, y: 0, size: 10, color: "blue" });
  graph.addNode("2", { label: "Node 2", x: 1, y: 1, size: 20, color: "red" });
  graph.addEdge("1", "2", { size: 5, color: "purple" });

  onMount(async () => {
    sigmaInstance = new Sigma(graph, sigmaCanvas);
  });

</script>

<LynxKiteNode id={id} data={data} {...$$restProps}>
  <div bind:this={sigmaCanvas} style="height: 200px; width: 200px;" >
  </div>
</LynxKiteNode>
<style>
</style>
