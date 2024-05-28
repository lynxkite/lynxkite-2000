<script>
  import {TabulatorFull as Tabulator} from 'tabulator-tables';
  import {onMount} from 'svelte';

  export let data, columns;

  let tableComponent;
  let tab;

  onMount(() => {
    console.log(data, columns);
    // The rows in the data are arrays, but Tabulator expects objects.
    const objs = [];
    for (const row of data) {
      const obj = {};
      for (let i = 0; i < columns.length; i++) {
        obj[columns[i]] = row[i];
      }
      objs.push(obj);
    }
    tab = new Tabulator(tableComponent, {
      data: objs,
      columns: columns.map(c => ({title: c, field: c, widthGrow: 1})),
      height: '311px',
      reactiveData: true,
      layout: "fitColumns",
    });
  });
</script>

<div bind:this={tableComponent}></div>
