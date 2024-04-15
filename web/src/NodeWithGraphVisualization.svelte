<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import Sigma from 'sigma';
  import * as graphology from 'graphology';
  import * as graphologyLibrary from 'graphology-library';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  let sigmaCanvas: HTMLElement;
  let sigmaInstance: Sigma;

  $: if (sigmaCanvas) sigmaInstance = new Sigma(new graphology.Graph(), sigmaCanvas);
  $: if (sigmaInstance && data.graph) {
    const graph = graphology.Graph.from(data.graph);
    graphologyLibrary.layout.random.assign(graph);
    const settings = graphologyLibrary.layoutForceAtlas2.inferSettings(graph);
    graphologyLibrary.layoutForceAtlas2.assign(graph, { iterations: 10, settings });
    graphologyLibrary.layoutNoverlap.assign(graph, { settings: { ratio: 3 } });
    sigmaInstance.graph = graph;
    sigmaInstance.refresh();
  }
</script>

<LynxKiteNode {...$$props}>
  <div bind:this={sigmaCanvas} style="height: 200px; width: 200px;" >
  </div>
</LynxKiteNode>
<style>
</style>
