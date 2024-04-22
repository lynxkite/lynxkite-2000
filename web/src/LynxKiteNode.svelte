<script lang="ts">
  import { Handle, type NodeProps } from '@xyflow/svelte';

  type $$Props = NodeProps;

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

  let expanded = true;
  function titleClicked() {
    expanded = !expanded;
  }
</script>

<div class="node-container">
  <div class="lynxkite-node">
    <div class="title" on:click={titleClicked}>
      {data.title}
      {#if data.error}<span class="error-sign">⚠️</span>{/if}
    </div>
    {#if expanded}
      {#if data.error}
        <div class="error">{data.error}</div>
      {/if}
      <slot />
    {/if}
    {#if sourcePosition}
      <Handle type="source" position={sourcePosition} />
    {/if}
    {#if targetPosition}
      <Handle type="target" position={targetPosition} />
    {/if}
  </div>
</div>

<style>
  .error {
    background: #ffdddd;
    padding: 8px;
    font-size: 12px;
  }
  .error-sign {
    float: right;
  }
  .node-container {
    padding: 8px;
  }
  .lynxkite-node {
    box-shadow: 0px 5px 50px 0px rgba(0, 0, 0, 0.3);
    background: white;
    min-width: 200px;
    max-width: 400px;
    max-height: 400px;
    overflow-y: auto;
    border-radius: 1px;
  }
  .title {
    background: #ff8800;
    font-weight: bold;
    padding: 8px;
  }
</style>
