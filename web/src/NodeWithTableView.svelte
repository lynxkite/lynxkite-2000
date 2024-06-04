<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import { Tabulator } from 'tabulator-tables';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import Table from './Table.svelte';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
  const open = {};
  $: single = data.display?.dataframes && Object.keys(data.display.dataframes).length === 1;
</script>

<LynxKiteNode {...$$props}>
  {#if data.display}
    {#each Object.entries(data.display.dataframes) as [name, df]}
      {#if !single}<div class="df-head" on:click={() => open[name] = !open[name]}>{name}</div>{/if}
      {#if single || open[name]}
        <Table columns={df.columns} data={df.data} />
      {/if}
    {/each}
    {#each Object.entries(data.display.others || {}) as [name, o]}
      <div class="df-head" on:click={() => open[name] = !open[name]}>{name}</div>
      {#if open[name]}
      <pre>{o}</pre>
      {/if}
    {/each}
  {/if}
</LynxKiteNode>
<style>
  .df-head {
    font-weight: bold;
    padding: 8px;
    background: #f0f0f0;
    cursor: pointer;
  }
  table {
    table-layout: fixed;
  }
</style>
