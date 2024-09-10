<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import NodeWithParams from './NodeWithParams.svelte';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
</script>

<NodeWithParams {...$$props}>
  {#if data.display}
    <img src={data.display}/>
  {/if}
</NodeWithParams>
<style>
</style>
