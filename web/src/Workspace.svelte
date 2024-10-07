<script lang="ts">
  // This is the whole LynxKite workspace editor page.
  import { QueryClient, QueryClientProvider } from '@sveltestack/svelte-query'
  import { SvelteFlowProvider } from '@xyflow/svelte';
  import LynxKiteFlow from './LynxKiteFlow.svelte';
  // import { syncedStore, getYjsDoc } from "@syncedstore/core";
  // import { svelteSyncedStore } from "@syncedstore/svelte";
  // import { WebsocketProvider } from "y-websocket";
  // const todoStore = syncedStore({ todos: [] });
  // const doc = getYjsDoc(todoStore);
  // const wsProvider = new WebsocketProvider("ws://localhost:8000/ws/crdt", "my-roomname3", doc);
  // wsProvider.on('sync', function(isSynced: boolean) {
  //   console.log('synced test', isSynced, 'ydoc', doc.toJSON());
  // });

  // console.log('ydoc', doc.toJSON());
  // console.log('todoStore', todoStore);
  // todoStore.todos.push('asdfx');
  // console.log('ydoc', doc.toJSON());
  // // console.log(doc.toJSON());
  // // console.log(doc.getMap().toJSON());
  // // console.log(doc.getMap().get('todos'));
  // // doc.getMap().get('todos').observe(() => {
  // //   console.log('todos changed', todoStore.todos);
  // // });
  // // doc.getMap().get('todos').add('hello');

  // export const store = svelteSyncedStore(todoStore);
  export let path = '';
  const queryClient = new QueryClient()
</script>

<QueryClientProvider client={queryClient}>
  <SvelteFlowProvider>
    <LynxKiteFlow path={path} />
  </SvelteFlowProvider>
</QueryClientProvider>
