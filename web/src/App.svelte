<script lang="ts">
  import Directory from './Directory.svelte';
  import Workspace from './Workspace.svelte';
  let page = '';
  let parameters = {};
  function onHashChange() {
    const parts = location.hash.split('?');
    page = parts[0].substring(1);
    parameters = {};
    if (parts.length > 1) {
      parameters = Object.fromEntries(new URLSearchParams(parts[1]));
    }
    console.log(parameters);
	}
  onHashChange();
</script>

<svelte:window on:hashchange={onHashChange} />
{#if page === 'edit'}
  <Workspace {...parameters} />
{:else}
  <Directory {...parameters} />
{/if}
