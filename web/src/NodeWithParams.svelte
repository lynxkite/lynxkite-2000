<script lang="ts">
  import { getContext } from 'svelte';
  import { type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import NodeParameter from './NodeParameter.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  const { updateNodeData } = useSvelteFlow();
  $: meta = getContext('LynxKiteFlow').getMeta(data.title);
  $: metaParams = meta && Object.fromEntries(meta.data.params.map((p) => [p.name, p]));
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
