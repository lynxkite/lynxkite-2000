<script lang="ts">
  import { type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import Sigma from 'sigma';
  import * as graphology from 'graphology';
  import * as graphologyLibrary from 'graphology-library';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  const { updateNodeData } = useSvelteFlow();
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  let sigmaCanvas: HTMLElement;
  let sigmaInstance: Sigma;

  $: if (sigmaCanvas) sigmaInstance = new Sigma(new graphology.Graph(), sigmaCanvas);
  $: if (sigmaInstance && data.view) {
    // Graphology will modify this in place, so we make a copy.
    const view = JSON.parse(JSON.stringify(data.view));
    const graph = graphology.Graph.from(view);
    graphologyLibrary.layout.random.assign(graph);
    const settings = graphologyLibrary.layoutForceAtlas2.inferSettings(graph);
    graphologyLibrary.layoutForceAtlas2.assign(graph, { iterations: 10, settings });
    graphologyLibrary.layoutNoverlap.assign(graph, { settings: { ratio: 3 } });
    sigmaInstance.graph = graph;
    sigmaInstance.refresh();
  }
</script>

<LynxKiteNode {...$$props}>
  {#if data.view}
    <label>Color by
      <select on:change={(evt) => updateNodeData(id, { params: { ...data.params, color_nodes_by: evt.currentTarget.value } })}>
        <option value="">nothing</option>
        {#each data.view.node_attributes as attr}
          <option value={attr} selected={attr === data.params.color_nodes_by}>{attr}</option>
        {/each}
    </select></label>
  {/if}
  <div bind:this={sigmaCanvas} style="height: 200px; width: 200px;" >
  </div>
</LynxKiteNode>
<style>
</style>
