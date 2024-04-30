<script lang="ts">
  import { getContext } from 'svelte';
  import { type NodeProps, useSvelteFlow } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let id: $$Props['id'];
  export let data: $$Props['data'];
  const { updateNodeData } = useSvelteFlow();
  $: meta = getContext('LynxKiteFlow').getMeta(data.title);
  $: metaParams = meta && Object.fromEntries(meta.data.params.map((p) => [p.name, p]));
</script>

<LynxKiteNode {...$$props}>
  {#each Object.entries(data.params) as [name, value]}
    <div class="param">
      <label>
        {name}<br>
        {#if metaParams?.[name]?.type?.enum}
          <select
            value={value}
            on:change={(evt) => updateNodeData(id, { params: { ...data.params, [name]: evt.currentTarget.value } })}
          >
            {#each metaParams[name].type.enum as option}
              <option value={option}>{option}</option>
            {/each}
          </select>
        {:else}
          <input
          value={value}
          on:input={(evt) => updateNodeData(id, { params: { ...data.params, [name]: evt.currentTarget.value } })}
          />
        {/if}
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
  .param input,
  .param select {
    width: calc(100% - 8px);
  }
</style>
