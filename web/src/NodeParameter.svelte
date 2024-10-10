<script lang="ts">
  export let name: string;
  export let value;
  export let meta;
  export let onChange;
const BOOLEAN = "<class 'bool'>";
</script>

<label class="param">
  {#if meta?.type?.format === 'collapsed'}
    <span class="param-name">{name.replace(/_/g, ' ')}</span>
    <button class="collapsed-param form-control form-control-sm">
      ⋯
    </button>
  {:else if meta?.type?.format === 'textarea'}
    <span class="param-name">{name.replace(/_/g, ' ')}</span>
    <textarea class="form-control form-control-sm"
      rows="6"
      value={value}
      on:change={(evt) => onChange(evt.currentTarget.value)}
      />
  {:else if meta?.type?.enum}
    <span class="param-name">{name.replace(/_/g, ' ')}</span>
    <select class="form-select form-select-sm"
      value={value || meta.type.enum[0]}
      on:change={(evt) => onChange(evt.currentTarget.value)}
    >
      {#each meta.type.enum as option}
        <option value={option}>{option}</option>
      {/each}
    </select>
  {:else if meta?.type?.type === BOOLEAN}
    <label class="form-check-label">
      <input class="form-check-input"
      type="checkbox"
      checked={value}
      on:change={(evt) => onChange(evt.currentTarget.checked)}
      />
      {name.replace(/_/g, ' ')}
    </label>
  {:else}
    <span class="param-name">{name.replace(/_/g, ' ')}</span>
    <input class="form-control form-control-sm"
      value={value}
      on:change={(evt) => onChange(evt.currentTarget.value)}
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
    background: var(--bs-border-color);
    width: fit-content;
    padding: 2px 8px;
    border-radius: 4px 4px 0 0;
  }
  .collapsed-param {
    min-height: 20px;
    line-height: 10px;
  }
</style>
