<script lang="ts">
  export let name: string;
  export let value;
  export let meta;
  export let onChange;
</script>

<label class="param">
  <span class="param-name">{name.replace(/_/g, ' ')}</span>
  {#if meta?.type?.collapsed}
    <button class="collapsed-param form-control form-control-sm">
      ⋯
    </button>
  {:else if meta?.type?.enum}
    <select class="form-select form-select-sm"
      value={value || meta.type.enum[0]}
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

<style>
  .param {
    padding: 4px 8px 4px 8px;
    display: block;
  }
  .param-name {
    display: block;
    font-size: 10px;
    letter-spacing: 0.05em;
    margin-left: 10px;
    background: oklch(50% 0.13 230);
    background: oklch(95% 0.2 55);
    background:var(--bs-border-color);
    width: fit-content;
    padding: 2px 8px;
    border-radius: 4px 4px 0 0;
  }
  .collapsed-param {
    min-height: 20px;
    line-height: 10px;
  }
</style>
