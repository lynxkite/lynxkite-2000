<script lang="ts">
  import { getContext } from 'svelte';
  import { type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import NodeParameter from './NodeParameter.svelte';

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

  function asPx(n: number | undefined) {
    return n ? n + 'px' : undefined;
  }
  const { updateNodeData } = useSvelteFlow();
  $: meta = getContext('LynxKiteFlow').getMeta(data.title);
  $: metaParams = meta && Object.fromEntries(meta.data.params.map((p) => [p.name, p]));
</script>

<div class="area" style:width={asPx(width)} style:height={asPx(height)} style={containerStyle}>
  <div class="title">
    {data.title}
  </div>
  {#each Object.entries(data.params) as [name, value]}
    <NodeParameter
      {name}
      {value}
      meta={metaParams?.[name]}
      onChange={(newValue) => updateNodeData(id, { params: { ...data.params, [name]: newValue } })}
      />
  {/each}
</div>

<style>
  .area {
    border-radius: 10px;
    border: 3px dashed #f80;
    min-width: 400px;
    min-height: 400px;
    max-width: 800px;
    max-height: 800px;
    z-index: 0 !important;
  }
  .title {
    color: #f80;
    width: 100%;
    text-align: center;
    top: -1.5em;
    position: absolute;
    -webkit-text-stroke: 5px white;
    paint-order: stroke fill;
    font-weight: bold;
  }
</style>
