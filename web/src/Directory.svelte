<script lang="ts">
  // The directory browser.
  import logo from './assets/logo.png';
  export let path = '';
  async function fetchList(path) {
    const encodedPath = encodeURIComponent(path || '');
    const res = await fetch(`/api/dir/list?path=${encodedPath}`);
    const j = await res.json();
    return j;
  }
  $: list = fetchList(path);
  function open(item) {
    if (item.type === 'directory') {
      location.hash = `#dir?path=${item.name}`;
    } else {
      location.hash = `#edit?path=${item.name}`;
    }
  }
  function shortName(item) {
    return item.name.split('/').pop();
  }
</script>

<div class="directory-page">
  <div class="logo">
    <a href="https://lynxkite.com/"><img src="{logo}" class="logo-image"></a>
    <div class="tagline">The Complete Graph Data Science Platform</div>
  </div>
  <div class="entry-list">
    {#await list}
      <div>Loading...</div>
    {:then list}
      {#each list as item}
        <div class="entry" on:click={open(item)}>{shortName(item)}</div>
      {/each}
    {:catch error}
      <p style="color: red">{error.message}</p>
    {/await}
  </div>
</div>

<style>
  @media (min-width: 640px) {
    .directory {
      width: 100%;
    }
  }

  .entry-list {
    width: 100%;
    margin: 10px auto;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0px 2px 4px;
    padding: 10px 0;
  }
  @media (min-width: 768px) {
    .entry-list {
      width: 768px;
    }
  }
  @media (min-width: 960px) {
    .entry-list {
      width: 80%;
    }
  }

  .logo {
    margin: 0;
    padding-top: 50px;
    text-align: center;
  }
  .logo-image {
    max-width: 50%;
  }
  .tagline {
    color: #39bcf3;
    font-size: 14px;
    font-weight: 500;
  }
  @media (min-width: 1400px) {
    .tagline {
      font-size: 18px;
    }
  }

  .entry-list .entry {
    position: relative;
    border-bottom: 1px solid whitesmoke;
  }
  .entry-list .entry {
    padding-left: 40px;
    color: #004165;
    cursor: pointer;
    user-select: none;
    text-decoration: none;
  }
  .entry-list .open .entry,
  .entry-list .entry:hover,
  .entry-list .entry:focus {
    background: #39bcf3;
    color: white;
  }
  .entry-list .entry:last-child {
    border-bottom: none;
  }
  .directory-page {
    background: #002a4c;
    height: 100vh;
  }
</style>
