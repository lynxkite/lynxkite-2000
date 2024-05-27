<script lang="ts">
  import { Handle, type NodeProps } from '@xyflow/svelte';

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

  let expanded = true;
  function titleClicked() {
    expanded = !expanded;
    onToggle({ expanded });
  }
  function asPx(n: number | undefined) {
    return n ? n + 'px' : undefined;
  }
  $: inputs = Object.values(data.meta?.inputs || {});
  $: outputs = Object.values(data.meta?.outputs || {});
  const handleOffsetDirection = { top: 'left', bottom: 'left', left: 'top', right: 'top' };
</script>

<div class="node-container" style:width={asPx(width)} style:height={asPx(height)} style={containerStyle}>
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
    {#each inputs as input, i}
      <Handle
        id={input.name} type="target" position={input.position}
        style="{handleOffsetDirection[input.position]}: {100 * (i + 1) / (inputs.length + 1)}%">
        {#if inputs.length>1}<span class="handle-name">{input.name.replace(/_/g, " ")}</span>{/if}
      </Handle>
    {/each}
    {#each outputs as output, i}
      <Handle
        id={output.name} type="source" position={output.position}
        style="{handleOffsetDirection[output.position]}: {100 * (i + 1) / (outputs.length + 1)}%">
        {#if outputs.length>1}<span class="handle-name">{output.name.replace(/_/g, " ")}</span>{/if}
      </Handle>
    {/each}
  </div>
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
    min-width: 200px;
    max-width: 400px;
    max-height: 400px;
  }
  .lynxkite-node {
    box-shadow: 0px 5px 50px 0px rgba(0, 0, 0, 0.3);
    background: white;
    overflow-y: auto;
    border-radius: 4px;
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
</style>
