<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
  const open = {};
</script>

<LynxKiteNode {...$$props}>
  {#if data.view}
    {#each Object.entries(data.view.dataframes) as [name, df]}
      <div class="df-head" on:click={() => open[name] = !open[name]}>{name}</div>
      {#if open[name]}
      <table>
        <tr>
          {#each df.columns as column}
            <th>{column}</th>
          {/each}
        </tr>
        {#each df.data as row}
          <tr>
            {#each row as cell}
              <td>{cell}</td>
            {/each}
          </tr>
        {/each}
      </table>
      {/if}
    {/each}
    {#each Object.entries(data.view.others || {}) as [name, o]}
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
    margin: 8px;
    border-collapse: collapse;
  }
</style>
