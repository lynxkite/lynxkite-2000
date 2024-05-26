<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import Fuse from 'fuse.js'
  const dispatch = createEventDispatcher();
  export let pos;
  export let boxes;
  let searchBox: HTMLInputElement;
  let hits = Object.values(boxes).map(box => ({item: box}));
  let selectedIndex = 0;
  onMount(() => searchBox.focus());
  $: fuse = new Fuse(boxes, {
    keys: ['name']
  })
  function onInput() {
    hits = fuse.search(searchBox.value);
    selectedIndex = Math.max(0, Math.min(selectedIndex, hits.length - 1));
  }
  function onKeyDown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, hits.length - 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
    } else if (e.key === 'Enter') {
      addSelected();
    } else if (e.key === 'Escape') {
      dispatch('cancel');
    }
  }
  function addSelected() {
    const node = {...hits[selectedIndex].item};
    delete node.sub_nodes;
    node.position = pos;
    dispatch('add', node);
  }
  async function lostFocus(e) {
    // If it's a click on a result, let the click handler handle it.
    if (e.relatedTarget && e.relatedTarget.closest('.node-search')) return;
    dispatch('cancel');
  }

</script>

<div class="node-search" style="top: {pos.y}px; left: {pos.x}px;">
  <input
    bind:this={searchBox}
    on:input={onInput}
    on:keydown={onKeyDown}
    on:focusout={lostFocus}
    placeholder="Search for box">
  <div class="matches">
    {#each hits as box, index}
      <div
        tabindex="0"
        on:focus={() => selectedIndex = index}
        on:mouseenter={() => selectedIndex = index}
        on:click={addSelected}
        class="search-result"
        class:selected={index == selectedIndex}>
        {box.item.name}
      </div>
    {/each}
  </div>
</div>

<style>
  input {
    width: calc(100% - 26px);
    font-size: 20px;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #eee;
    margin: 4px;
  }
  .search-result {
    padding: 4px;
    cursor: pointer;
  }
  .search-result.selected {
    background-color: oklch(75% 0.2 55);
    border-radius: 4px;
  }
  .node-search {
    position: fixed;
    width: 300px;
    z-index: 5;
    padding: 4px;
    border-radius: 4px;
    border: 1px solid #888;
    background-color: white;
    max-height: -webkit-fill-available;
    max-height: -moz-available;
    display: flex;
    flex-direction: column;
  }
  .matches {
    overflow-y: auto;
  }
</style>
