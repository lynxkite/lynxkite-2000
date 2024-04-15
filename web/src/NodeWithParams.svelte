<script lang="ts">
  import { Handle, type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  const { updateNodeData } = useSvelteFlow();
</script>

<LynxKiteNode id={id} data={data} {...$$restProps}>
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
</LynxKiteNode>
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
</style>
