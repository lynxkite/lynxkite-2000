<script lang="ts">
  import { Handle, Position, type NodeProps, useSvelteFlow } from '@xyflow/svelte';

  type $$Props = NodeProps;

  export let id: $$Props['id'];
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

  const { updateNodeData } = useSvelteFlow();
</script>

<div class="node-container">
  <div class="lynxkite-node">
    <div class="title">{data.title}</div>
    {#each Object.entries(data.params) as [name, value]}
    <div class="param">
      <label>
        {name}<br>
        <input
          value={value}
          on:input={(evt) => updateNodeData(id, { params: { ...data.params, [name]: evt.currentTarget.value } })}
        />
      </label>
    </div>
    {/each}
    <Handle type="source" position={Position.Right} />
    <Handle type="target" position={Position.Left} />
  </div>
</div>

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
  .node-container {
    padding: 5px;
  }
  .lynxkite-node {
    box-shadow: 0px 5px 50px 0px rgba(0, 0, 0, 0.3);
    background: white;
  }
  .title {
    background: #ff8800; /* Brand color. */
    font-weight: bold;
    padding: 8px;
    max-width: 300px;
  }
</style>
