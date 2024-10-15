<script lang="ts">
  import { useNodes, type NodeProps } from '@xyflow/svelte';
  import LynxKiteNode from './LynxKiteNode.svelte';
  import Table from './Table.svelte';
  import SvelteMarkdown from 'svelte-markdown'
  type $$Props = NodeProps;
  export let data: $$Props['data'];
  const nodes = useNodes(); // We don't properly get updates to "data". This is a hack.
  $: D = $nodes && data;
  const open = {};
  $: single = D.display?.value?.dataframes && Object.keys(D.display.value.dataframes).length === 1;
  function toMD(v) {
    if (typeof v === 'string') {
      return v;
    }
    if (Array.isArray(v)) {
      return v.map(toMD).join('\n\n');
    }
    return JSON.stringify(v);
  }
</script>

<LynxKiteNode {...$$props}>
  {#if D?.display?.value}
    {#each Object.entries(D.display.value.dataframes || {}) as [name, df]}
      {#if !single}<div class="df-head" on:click={() => open[name] = !open[name]}>{name}</div>{/if}
      {#if single || open[name]}
        {#if df.data.length > 1}
          <Table columns={df.columns} data={df.data} />
        {:else}
          <dl>
          {#each df.columns as c, i}
            <dt>{c}</dt>
            <dd><SvelteMarkdown source={toMD(df.data[0][i])} /></dd>
          {/each}
          </dl>
        {/if}
      {/if}
    {/each}
    {#each Object.entries(D.display.value.others || {}) as [name, o]}
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
  dl {
    margin: 10px;
  }
</style>
