<script lang="ts">
  export let data, columns;
</script>

<table>
  <thead>
    <tr>
      {#each columns as column}
        <th>{column}</th>
      {/each}
    </tr>
  </thead>
  <tbody>
    {#each data as row}
      <tr>
        {#each columns as column}
          <td>{row[column]}</td>
        {/each}
      </tr>
    {/each}
  </tbody>
</table>
