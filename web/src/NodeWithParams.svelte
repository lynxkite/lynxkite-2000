<script lang="ts">
  import { type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import NodeParameter from './NodeParameter.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  const { updateNodeData } = useSvelteFlow();
  $: metaParams = data.meta?.params;
</script>

<LynxKiteNode {...$$props}>
  {#each Object.entries(data.params) as [name, value]}
    <NodeParameter
      {name}
      {value}
      meta={metaParams?.[name]}
      onChange={(newValue) => updateNodeData(id, { params: { ...data.params, [name]: newValue } })}
      />
  {/each}
  <slot />
</LynxKiteNode>
