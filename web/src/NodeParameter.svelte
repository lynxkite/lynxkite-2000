<script lang="ts">
  export let name: string;
  export let value;
  export let meta;
  export let onChange;
</script>

<div class="param">
  <label>
    {name}<br>
    {#if meta?.type?.enum}
      <select
        value={value}
        on:change={(evt) => onChange(evt.currentTarget.value)}
      >
        {#each meta.type.enum as option}
          <option value={option}>{option}</option>
        {/each}
      </select>
    {:else}
      <input
      value={value}
      on:input={(evt) => onChange(evt.currentTarget.value)}
      />
    {/if}
  </label>
</div>

<style>
  .param {
    padding: 8px;
  }
  .param label {
    font-size: 12px;
    display: block;
  }
  .param input,
  .param select {
    width: calc(100% - 8px);
  }
</style>
