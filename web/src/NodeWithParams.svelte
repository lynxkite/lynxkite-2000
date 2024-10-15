<script lang="ts">
  import { getContext } from 'svelte';
  import { type NodeProps, useNodes } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import NodeParameter from './NodeParameter.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  $: metaParams = data.meta?.params;
  $: store = getContext('LynxKite store');
  function setParam(name, newValue) {
    const i = $store.workspace.nodes.findIndex((n) => n.id === id);
    $store.workspace.nodes[i].data.params[name] = newValue;
  }
  $: params = $nodes && data?.params ? Object.entries(data.params) : [];
  const nodes = useNodes(); // We don't properly get updates to "data". This is a hack.
  $: props = $nodes && $$props;
</script>

<LynxKiteNode {...props}>
  {#each params as [name, value]}
    <NodeParameter
      {name}
      {value}
      meta={metaParams?.[name]}
      onChange={(value) => setParam(name, value)}
      />
  {/each}
  <slot />
</LynxKiteNode>
