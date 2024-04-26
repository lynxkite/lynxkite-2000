<script lang="ts">
  import { type NodeProps, useNodes } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  const nodes = useNodes();
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  let isExpanded = true;
  function onToggle({ expanded }) {
    isExpanded = expanded;
    console.log('onToggle', expanded, height);
    nodes.update((n) =>
      n.map((node) =>
        node.parentNode === id
        ? { ...node, hidden: !expanded }
        : node));
  }
  function computeSize(nodes) {
    let width = 200;
    let height = 200;
    for (const node of nodes) {
      if (node.parentNode === id) {
        width = Math.max(width, node.position.x + 300);
        height = Math.max(height, node.position.y + 200);
      }
    }
    return { width, height };
  }
  $: ({ width, height } = computeSize($nodes));
</script>

<LynxKiteNode
  {...$$props}
  width={isExpanded && width} height={isExpanded && height}
  nodeStyle="background: transparent;" containerStyle="max-width: none; max-height: none;" {onToggle}>
</LynxKiteNode>
<style>
</style>
