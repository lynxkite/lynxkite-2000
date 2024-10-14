<script lang="ts">
  import { getContext } from 'svelte';
  import { Handle, useSvelteFlow, useUpdateNodeInternals, type NodeProps, NodeResizeControl } from '@xyflow/svelte';
  import ChevronDownRight from 'virtual:icons/tabler/chevron-down-right';

  const { updateNodeData } = useSvelteFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  type $$Props = NodeProps;

  export let nodeStyle = '';
  export let containerStyle = '';
  export let id: $$Props['id']; id;
  export let data: $$Props['data'];
  export let dragHandle: $$Props['dragHandle'] = undefined; dragHandle;
  export let type: $$Props['type']  = undefined; type;
  export let selected: $$Props['selected'] = undefined; selected;
  export let isConnectable: $$Props['isConnectable'] = undefined; isConnectable;
  export let zIndex: $$Props['zIndex'] = undefined; zIndex;
  export let width: $$Props['width'] = undefined; width;
  export let height: $$Props['height'] = undefined; height;
  export let dragging: $$Props['dragging']; dragging;
  export let targetPosition: $$Props['targetPosition'] = undefined; targetPosition;
  export let sourcePosition: $$Props['sourcePosition'] = undefined; sourcePosition;
  export let positionAbsoluteX: $$Props['positionAbsoluteX'] = undefined; positionAbsoluteX;
  export let positionAbsoluteY: $$Props['positionAbsoluteY'] = undefined; positionAbsoluteY;
  export let onToggle = () => {};

  $: store = getContext('LynxKite store');
  $: expanded = !data.collapsed;
  function titleClicked() {
    const i = $store.workspace.nodes.findIndex((n) => n.id === id);
    $store.workspace.nodes[i].data.collapsed = expanded;
    onToggle({ expanded });
    // Trigger update.
    data = data;
    updateNodeInternals();
  }
  function asPx(n: number | undefined) {
    return n ? n + 'px' : undefined;
  }
  function getHandles(inputs, outputs) {
    const handles: {
      position: 'top' | 'bottom' | 'left' | 'right',
      name: string,
      index: number,
      offsetPercentage: number,
      showLabel: boolean,
    }[] = [];
    for (const e of Object.values(inputs)) {
      handles.push({ ...e, type: 'target' });
    }
    for (const e of Object.values(outputs)) {
      handles.push({ ...e, type: 'source' });
    }
    const counts = { top: 0, bottom: 0, left: 0, right: 0 };
    for (const e of handles) {
      e.index = counts[e.position];
      counts[e.position]++;
    }
    for (const e of handles) {
      e.offsetPercentage = 100 * (e.index + 1) / (counts[e.position] + 1);
      const simpleHorizontal = counts.top === 0 && counts.bottom === 0 && handles.length <= 2;
      const simpleVertical = counts.left === 0 && counts.right === 0 && handles.length <= 2;
      e.showLabel = !simpleHorizontal && !simpleVertical;
    }
    return handles;
  }
  $: handles = getHandles(data.meta?.inputs || {}, data.meta?.outputs || {});
  const handleOffsetDirection = { top: 'left', bottom: 'left', left: 'top', right: 'top' };
</script>

<div class="node-container" class:expanded={expanded}
  style:width={asPx(width)} style:height={asPx(expanded ? height : undefined)} style={containerStyle}>
  <div class="lynxkite-node" style={nodeStyle}>
    <div class="title" on:click={titleClicked}>
      {data.title}
      {#if data.error}<span class="title-icon">⚠️</span>{/if}
      {#if !expanded}<span class="title-icon">⋯</span>{/if}
    </div>
    {#if expanded}
      {#if data.error}
        <div class="error">{data.error}</div>
      {/if}
      <slot />
    {/if}
    {#each handles as handle}
      <Handle
        id={handle.name} type={handle.type} position={handle.position}
        style="{handleOffsetDirection[handle.position]}: {handle.offsetPercentage}%">
        {#if handle.showLabel}<span class="handle-name">{handle.name.replace(/_/g, " ")}</span>{/if}
      </Handle>
    {/each}
  </div>
  {#if expanded}
    <NodeResizeControl
      minWidth={100}
      minHeight={50}
      style="background: transparent; border: none;"
      onResizeStart={() => updateNodeData(id, { beingResized: true })}
      onResizeEnd={() => updateNodeData(id, { beingResized: false })}
      >
      <ChevronDownRight class="node-resizer" />
    </NodeResizeControl>
  {/if}
</div>

<style>
  .error {
    background: #ffdddd;
    padding: 8px;
    font-size: 12px;
  }
  .title-icon {
    margin-left: 5px;
    float: right;
  }
  .node-container {
    padding: 8px;
    position: relative;
  }
  .lynxkite-node {
    box-shadow: 0px 5px 50px 0px rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    background: white;
  }
  .expanded .lynxkite-node {
    overflow-y: auto;
    height: 100%;
  }
  .title {
    background: oklch(75% 0.2 55);
    font-weight: bold;
    padding: 8px;
  }
  .handle-name {
    font-size: 10px;
    color: black;
    letter-spacing: 0.05em;
    text-align: right;
    white-space: nowrap;
    position: absolute;
    top: -5px;
    backdrop-filter: blur(10px);
    padding: 2px 8px;
    border-radius: 4px;
    visibility: hidden;
  }
  :global(.left) .handle-name {
    right: 20px;
  }
  :global(.right) .handle-name {
    left: 20px;
  }
  :global(.top) .handle-name,
  :global(.bottom) .handle-name {
    top: -5px;
    left: 5px;
    backdrop-filter: none;
  }
  .node-container:hover .handle-name {
    visibility: visible;
  }
  :global(.node-resizer) {
    position: absolute;
    bottom: 8px;
    right: 8px;
    cursor: nwse-resize;
    color: var(--bs-border-color);
  }
</style>
