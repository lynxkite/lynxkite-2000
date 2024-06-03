<script lang="ts">
  // This is the whole LynxKite workspace editor page.
  import { QueryClient, QueryClientProvider } from '@sveltestack/svelte-query'
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import LynxKiteFlow from './LynxKiteFlow.svelte';
  export let path = '';
  const queryClient = new QueryClient()
</script>

<QueryClientProvider client={queryClient}>
  <SvelteFlowProvider>
    <LynxKiteFlow path={path} />
  </SvelteFlowProvider>
</QueryClientProvider>
