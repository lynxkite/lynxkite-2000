<script lang="ts">
  // This is the whole LynxKite workspace editor page.
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import LynxKiteFlow from './LynxKiteFlow.svelte';
</script>

<SvelteFlowProvider>
  <LynxKiteFlow />
</SvelteFlowProvider>
