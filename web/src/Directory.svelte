<script lang="ts">
  // The directory browser.
  import logo from './assets/logo.png';
  import Home from 'virtual:icons/tabler/home'
  import Folder from 'virtual:icons/tabler/folder'
  import FolderPlus from 'virtual:icons/tabler/folder-plus'
  import File from 'virtual:icons/tabler/file'
  import FilePlus from 'virtual:icons/tabler/file-plus'

  export let path = '';
  async function fetchList(path) {
    const encodedPath = encodeURIComponent(path || '');
    const res = await fetch(`/api/dir/list?path=${encodedPath}`);
    const j = await res.json();
    return j;
  }
  $: list = fetchList(path);
  function link(item) {
    if (item.type === 'directory') {
      return `#dir?path=${item.name}`;
    } else {
      return `#edit?path=${item.name}`;
    }
  }
  function shortName(item) {
    return item.name.split('/').pop();
  }
  function newName(list) {
    let i = 0;
    while (true) {
      const name = `Untitled${i ? ` ${i}` : ''}`;
      if (!list.find(item => item.name === name)) {
        return name;
      }
      i++;
    }
  }
  function newWorkspaceIn(path, list) {
    const pathSlash = path ? `${path}/` : '';
    return `#edit?path=${pathSlash}${newName(list)}`;
  }
  async function newFolderIn(path, list) {
    const pathSlash = path ? `${path}/` : '';
    const name = newName(list);
    const res = await fetch(`/api/dir/mkdir`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: pathSlash + name }),
    });
    list = await res.json();
  }
</script>

<div class="directory-page">
  <div class="logo">
    <a href="https://lynxkite.com/"><img src="{logo}" class="logo-image"></a>
    <div class="tagline">The Complete Graph Data Science Platform</div>
  </div>
  <div class="entry-list">
    {#await list}
      <div class="loading spinner-border" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
    {:then list}
      <div class="actions">
        <a href="{newWorkspaceIn(path, list)}"><FilePlus /> New workspace</a>
        <a href on:click="{newFolderIn(path, list)}"><FolderPlus /> New folder</a>
      </div>
      {#if path} <div class="breadcrumbs"><a href="#dir"><Home /></a> {path} </div> {/if}
      {#each list as item}
        <a class="entry" href={link(item)}>
          {#if item.type === 'directory'}
            <Folder />
          {:else}
            <File />
          {/if}
          {shortName(item)}
        </a>
      {/each}
    {:catch error}
      <p style="color: red">{error.message}</p>
    {/await}
  </div>
</div>

<style>
  .entry-list {
    width: 100%;
    margin: 10px auto;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0px 2px 4px;
    padding: 0 0 10px 0;
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

  .actions {
    display: flex;
    justify-content: space-evenly;
    padding: 5px;
  }
  .actions a {
    padding: 2px 10px;
    border-radius: 5px;
  }
  .actions a:hover {
    background: #39bcf3;
    color: white;
  }

  .breadcrumbs {
    padding-left: 10px;
    font-size: 20px;
    background: #002a4c20;
  }
  .breadcrumbs a:hover {
    color: #39bcf3;
  }
  .entry-list .entry {
    display: block;
    border-bottom: 1px solid whitesmoke;
    padding-left: 10px;
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
  a {
    color: black;
    text-decoration: none;
  }
  .loading {
    color: #39bcf3;
    margin: 10px;
  }
</style>
