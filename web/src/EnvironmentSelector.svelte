<script lang="ts">
  export let options;
  export let value;
  export let onChange;
</script>

<select class="form-select form-select-sm"
  value={value}
  on:change={(evt) => onChange(evt.currentTarget.value)}
  >
  {#each options as option}
    <option value={option}>{option}</option>
  {/each}
</select>
