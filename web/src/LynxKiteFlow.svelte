<script lang="ts">
  import { diff } from 'deep-object-diff';
  import { writable, derived } from 'svelte/store';
  import {
    SvelteFlow,
    Controls,
    MiniMap,
    MarkerType,
    useSvelteFlow,
    type XYPosition,
    type Node,
    type Edge,
    type Connection,
    type NodeTypes,
  } from '@xyflow/svelte';
  import ArrowBack from 'virtual:icons/tabler/arrow-back'
  import Backspace from 'virtual:icons/tabler/backspace'
  import Atom from 'virtual:icons/tabler/Atom'
  import { useQuery } from '@sveltestack/svelte-query';
  import NodeWithParams from './NodeWithParams.svelte';
  import NodeWithVisualization from './NodeWithVisualization.svelte';
  import NodeWithImage from './NodeWithImage.svelte';
  import NodeWithTableView from './NodeWithTableView.svelte';
  import NodeWithSubFlow from './NodeWithSubFlow.svelte';
  import NodeWithArea from './NodeWithArea.svelte';
  import NodeSearch from './NodeSearch.svelte';
  import EnvironmentSelector from './EnvironmentSelector.svelte';
  import '@xyflow/svelte/dist/style.css';
  import { syncedStore, getYjsDoc } from "@syncedstore/core";
  import { svelteSyncedStore } from "@syncedstore/svelte";
  import { WebsocketProvider } from "y-websocket";

  function getCRDTStore(path) {
    const sstore = syncedStore({ workspace: {} });
    console.log('ss', sstore.workspace);
    const doc = getYjsDoc(sstore);
    console.log('doc', doc.toJSON());
    const wsProvider = new WebsocketProvider("ws://localhost:8000/ws/crdt", path, doc);
    wsProvider.on('sync', function(isSynced: boolean) {
      console.log('synced', isSynced, 'ydoc', doc.toJSON());
    });
    return {store: svelteSyncedStore(sstore), doc};
  }
  $: connection = getCRDTStore(path);
  $: store = connection.store;
  $: ws = connection.doc?.getMap('workspace');

  export let path = '';

  const { screenToFlowPosition } = useSvelteFlow();

  const nodeTypes: NodeTypes = {
    basic: NodeWithParams,
    visualization: NodeWithVisualization,
    image: NodeWithImage,
    table_view: NodeWithTableView,
    sub_flow: NodeWithSubFlow,
    area: NodeWithArea,
  };

  function closeNodeSearch() {
    nodeSearchSettings = undefined;
  }
  function toggleNodeSearch({ detail: { event } }) {
    if (nodeSearchSettings) {
      closeNodeSearch();
      return;
    }
    event.preventDefault();
    nodeSearchSettings = {
      pos: { x: event.clientX, y: event.clientY },
      boxes: $catalog.data[ws.env],
    };
  }
  function addNode(e) {
    const meta = {...e.detail};
    store.update((ws) => {
      const node = {
        type: meta.type,
        data: {
          meta: meta,
          title: meta.name,
          params: Object.fromEntries(
            Object.values(meta.params).map((p) => [p.name, p.default])),
        },
      };
      node.position = screenToFlowPosition({x: nodeSearchSettings.pos.x, y: nodeSearchSettings.pos.y});
      const title = node.data.title;
      let i = 1;
      node.id = `${title} ${i}`;
      while (ws.nodes.find((x) => x.id === node.id)) {
        i += 1;
        node.id = `${title} ${i}`;
      }
      node.parentId = nodeSearchSettings.parentId;
      if (node.parentId) {
        node.extent = 'parent';
        const parent = n.find((x) => x.id === node.parentId);
        node.position = { x: node.position.x - parent.position.x, y: node.position.y - parent.position.y };
      }
      return {...ws, nodes: [...n, node]};
    });
    closeNodeSearch();
  }
  const catalog = useQuery(['catalog'], async () => {
    const res = await fetch('/api/catalog');
    return res.json();
  }, {staleTime: 60000, retry: false});

  let nodeSearchSettings: {
    pos: XYPosition,
    boxes: any[],
    parentId: string,
  };

  function nodeClick(e) {
    const node = e.detail.node;
    const meta = node.data.meta;
    if (!meta) return;
    const sub_nodes = meta.sub_nodes;
    if (!sub_nodes) return;
    const event = e.detail.event;
    if (event.target.classList.contains('title')) return;
    nodeSearchSettings = {
      pos: { x: event.clientX, y: event.clientY },
      boxes: sub_nodes,
      parentId: node.id,
    };
  }
  $: parentDir = path.split('/').slice(0, -1).join('/');

</script>

<div class="page">
  <br>doc: {JSON.stringify(connection.doc)}
  <br>w2j: {JSON.parse(JSON.stringify(connection.doc)).workspace}
  <br>ws: {connection.doc?.getMap("workspace")}
  {#if ws !== undefined}
  {{ws}}
  <div class="top-bar">
    <div class="ws-name">
      <a href><img src="/favicon.ico"></a>
      {path}
    </div>
    <div class="tools">
      <EnvironmentSelector
        options={Object.keys($catalog.data || {})}
        value={$store.workspace.env}
        onChange={(env) => {
          console.log('env change', env);
          $store.workspace.env = env;
        }}
        />
      <a href><Atom /></a>
      <a href><Backspace /></a>
      <a href="#dir?path={parentDir}"><ArrowBack /></a>
    </div>
  </div>
  <div style:height="100%">
    <SvelteFlow nodes={ws.nodes} edges={ws.edges} {nodeTypes} fitView
      on:paneclick={toggleNodeSearch}
      on:nodeclick={nodeClick}
      proOptions={{ hideAttribution: true }}
      maxZoom={3}
      minZoom={0.3}
      defaultEdgeOptions={{ markerEnd: { type: MarkerType.Arrow } }}
      >
      <Controls />
      <MiniMap />
      {#if nodeSearchSettings}
      <NodeSearch pos={nodeSearchSettings.pos} boxes={nodeSearchSettings.boxes} on:cancel={closeNodeSearch} on:add={addNode} />
      {/if}
    </SvelteFlow>
  </div>
  {/if}
</div>

<style>
  .top-bar {
    display: flex;
    justify-content: space-between;
    background: oklch(30% 0.13 230);
    color: white;
  }
  .ws-name {
    font-size: 1.5em;
  }
  .ws-name img {
    height: 1.5em;
    vertical-align: middle;
    margin: 4px;
  }
  .page {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .tools {
    display: flex;
    align-items: center;
  }
  .tools a {
    color: oklch(75% 0.13 230);
    font-size: 1.5em;
    padding: 0 10px;
  }
</style>
