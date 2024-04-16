<script lang="ts">
  import { type NodeProps } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  type $$Props = NodeProps;
  export let data: $$Props['data'];
</script>

<LynxKiteNode {...$$props}>
  {#if data.view}
    {#each Object.entries(data.view.dataframes) as [name, df]}
      <div class="df-head">{name}</div>
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
    {/each}
  {/if}
</LynxKiteNode>
<style>
  .df-head {
    font-weight: bold;
    padding: 8px;
    background: #f0f0f0;
  }
  table {
    margin: 8px;
    border-collapse: collapse;
  }
</style>
