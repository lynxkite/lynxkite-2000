<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import Fuse from 'fuse.js'
  const dispatch = createEventDispatcher();
  export let pos;
  export let boxes;
  let searchBox: HTMLInputElement;
  let hits = [];
  let selectedIndex = 0;
  onMount(() => searchBox.focus());
  const fuse = new Fuse(boxes, {
    keys: ['name']
  })
  function onInput() {
    console.log('input', searchBox.value, selectedIndex);
    hits = fuse.search(searchBox.value);
    selectedIndex = Math.min(selectedIndex, hits.length - 1);
  }
  function onKeyDown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, hits.length - 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
    } else if (e.key === 'Enter') {
      dispatch('add', {x: pos.left, y: pos.top, ...hits[selectedIndex].item});
    } else if (e.key === 'Escape') {
      dispatch('cancel');
    }
  }

</script>

<div class="node-search"
style="top: {pos.top}px; left: {pos.left}px; right: {pos.right}px; bottom: {pos.bottom}px;">

  <input
    bind:this={searchBox}
    on:input={onInput}
    on:keydown={onKeyDown}
    on:focusout={() => dispatch('cancel')}
    placeholder="Search for box">
  {#each hits as box, index}
    <div class="search-result" class:selected={index == selectedIndex}>{index} {box.item.name}</div>
  {/each}
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
  }
  .search-result.selected {
    background-color: #f80;
    border-radius: 4px;
  }
  .node-search {
    position: absolute;
    width: 300px;
    z-index: 5;
    padding: 4px;
    border-radius: 4px;
    border: 1px solid #888;
    background-color: white;
  }
</style>
