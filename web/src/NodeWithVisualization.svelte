<script lang="ts">
  import { useNodes, type NodeProps } from '@xyflow/svelte';
  import NodeWithParams from './NodeWithParams.svelte';
  import { Chart } from 'svelte-echarts';
  import { init } from 'echarts';
  type $$Props = NodeProps;
  export let data: $$Props['data'];

  const nodes = useNodes(); // We don't properly get updates to "data". This is a hack.
  $: D = $nodes && data;
</script>

<NodeWithParams {...$$props}>
  <Chart {init} options={D?.display?.value || {}} initOptions={{renderer: 'canvas', width: 250, height: 250}}/>
</NodeWithParams>
<style>
</style>
