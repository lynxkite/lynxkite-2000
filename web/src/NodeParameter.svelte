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
      <button class="collapsed-param form-control form-control-sm">
        ⋯
      </button>
    {:else if meta?.type?.enum}
      <select class="form-select form-select-sm"
        value={value}
        on:change={(evt) => onChange(evt.currentTarget.value)}
      >
        {#each meta.type.enum as option}
          <option value={option}>{option}</option>
        {/each}
      </select>
    {:else}
      <input class="form-control form-control-sm"
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
    letter-spacing: 0.05em;
    margin-left: 4px;
  }
  .param input,
  .param select,
  .param button {
  }
  .collapsed-param {
  }
</style>
