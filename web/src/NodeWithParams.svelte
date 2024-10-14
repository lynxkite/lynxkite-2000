<script lang="ts">
  import { getContext } from 'svelte';
  import { type NodeProps, useSvelteFlow, useUpdateNodeInternals } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import NodeParameter from './NodeParameter.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  const { updateNodeData } = useSvelteFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  $: metaParams = data.meta?.params;
  $: store = getContext('LynxKite store');
  function setParam(name, newValue) {
    const i = $store.workspace.nodes.findIndex((n) => n.id === id);
    $store.workspace.nodes[i].data.params[name] = newValue;
    updateNodeInternals();
  }
</script>

<LynxKiteNode {...$$props}>
  {#each Object.entries(data.params) as [name, value]}
    <NodeParameter
      {name}
      {value}
      meta={metaParams?.[name]}
      onChange={(value) => setParam(name, value)}
      />
  {/each}
  <slot />
</LynxKiteNode>
