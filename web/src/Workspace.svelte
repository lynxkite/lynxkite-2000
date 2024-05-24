<script lang="ts">
  // This is the whole LynxKite workspace editor page.
  import { QueryClient, QueryClientProvider } from '@sveltestack/svelte-query'
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import ArrowBack from 'virtual:icons/tabler/arrow-back'
  import Backspace from 'virtual:icons/tabler/backspace'
  import Atom from 'virtual:icons/tabler/Atom'
  import LynxKiteFlow from './LynxKiteFlow.svelte';
  export let path = '';
  $: parent = path.split('/').slice(0, -1).join('/');
  const queryClient = new QueryClient()
</script>

<QueryClientProvider client={queryClient}>
  <div class="page">
    <div class="top-bar">
      <div class="ws-name">
        <a href><img src="/favicon.ico"></a>
        {path}
      </div>
      <div class="tools">
        <a href><Atom /></a>
        <a href><Backspace /></a>
        <a href="#dir?path={parent}"><ArrowBack /></a>
      </div>
    </div>
    <SvelteFlowProvider>
      <LynxKiteFlow path={path} />
    </SvelteFlowProvider>
  </div>
</QueryClientProvider>

<style>
  .top-bar {
    display: flex;
    justify-content: space-between;
    background: oklch(30% 0.13 230);
    color: white;
  }
  .ws-name {
    font-size: 1.5em;
  }
  .ws-name img {
    height: 1.5em;
    vertical-align: middle;
    margin: 4px;
  }
  .page {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .tools {
    display: flex;
    align-items: center;
  }
  .tools a {
    color: oklch(75% 0.13 230);
    font-size: 1.5em;
    padding: 0 10px;
  }
</style>
