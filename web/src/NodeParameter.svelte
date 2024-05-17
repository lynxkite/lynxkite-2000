<script lang="ts">
  export let name: string;
  export let value;
  export let meta;
  export let onChange;
</script>

<div class="param">
  <label>
    <span class="param-name">{name.replace('_', ' ')}</span>
    {#if meta?.type?.collapsed}
      <button class="collapsed-param">
        ⋯
      </button>
    {:else if meta?.type?.enum}
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
    padding: 0 8px 8px 8px;
  }
  .param label {
    font-size: 12px;
    display: block;
  }
  .param-name {
    color: #840;
  }
  .param input {
    width: calc(100% - 8px);
  }
  .param select {
    width: 100%;
  }
  .param input,
  .param select,
  .param button {
      border: 1px solid #840;
      border-radius: 4px;
  }
  .collapsed-param {
    width: 100%;
    font-family: auto;
    font-size: 200%;
    line-height: 0.5;
  }
</style>
