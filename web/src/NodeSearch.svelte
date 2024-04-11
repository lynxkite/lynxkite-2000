<script lang="ts">
  export let pos;
  console.log({pos});
  function titleClicked() {
    expanded = !expanded;
  }
</script>

<div class="node-search"
style="top: {pos.top}px; left: {pos.left}px; right: {pos.right}px; bottom: {pos.bottom}px;">

  Node Search!
</div>

<style>
  .node-search {
    position: absolute;
    width: 300px;
    z-index: 5;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #888;
    background-color: white;
  }
</style>
