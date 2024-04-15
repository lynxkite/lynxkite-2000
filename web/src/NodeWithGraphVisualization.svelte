<script lang="ts">
  import type { ComponentProps } from 'svelte';
  import { Handle, type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  const { updateNodeData } = useSvelteFlow();
</script>

<LynxKiteNode id={id} data={data} {...$$restProps}>
  Graph here.
</LynxKiteNode>
<style>
  .param {
    padding: 8px;
  }
  .param label {
    font-size: 12px;
    display: block;
  }
  .param input {
    width: calc(100% - 8px);
  }
</style>
