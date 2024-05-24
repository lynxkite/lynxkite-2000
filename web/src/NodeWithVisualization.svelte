<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import NodeWithParams from './NodeWithParams.svelte';
  import { Chart } from 'svelte-echarts';
  import { init } from 'echarts';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
</script>

<NodeWithParams {...$$props}>
  <Chart {init} options={data.view} style="width: 250px; height: 250px" />
</NodeWithParams>
<style>
</style>
