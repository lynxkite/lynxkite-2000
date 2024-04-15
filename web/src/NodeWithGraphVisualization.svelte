<script lang="ts">
  import { onMount } from 'svelte';
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

  const graph = graphology.Graph.from({
    attributes: {
      name: 'My Graph'
    },
    options: {
      allowSelfLoops: true,
      multi: false,
      type: 'mixed'
    },
    nodes: [
      {key: 'Thomas'},
      {key: 'Eric'}
    ],
    edges: [
      {
        key: 'T->E',
        source: 'Thomas',
        target: 'Eric',
      }
    ]
  });
  graphologyLibrary.layout.random.assign(graph);
  const settings = graphologyLibrary.layoutForceAtlas2.inferSettings(graph);
  graphologyLibrary.layoutForceAtlas2.assign(graph, { iterations: 10, settings });
  graphologyLibrary.layoutNoverlap.assign(graph, { settings: { ratio: 3 } });

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
