<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import NodeWithParams from './NodeWithParams.svelte';
  import { Chart } from 'svelte-echarts';
  import { init } from 'echarts';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
  $: display = JSON.parse(JSON.stringify(data?.display?.value));
</script>

<NodeWithParams {...$$props}>
  {#if data.display}
    <Chart {init} options={display} initOptions={{renderer: 'canvas', width: 250, height: 250}}/>
  {/if}
</NodeWithParams>
<style>
</style>
