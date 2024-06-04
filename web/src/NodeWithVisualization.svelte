<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import NodeWithParams from './NodeWithParams.svelte';
  import { Chart } from 'svelte-echarts';
  import { init } from 'echarts';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
</script>

<NodeWithParams {...$$props}>
  {#if data.display}
    <Chart {init} options={data.display} initOptions={{renderer: 'canvas', width: 250, height: 250}}/>
  {/if}
</NodeWithParams>
<style>
</style>
