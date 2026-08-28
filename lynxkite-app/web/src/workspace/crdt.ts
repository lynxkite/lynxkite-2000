// CRDT (via Y.js) is a way to synchronize a document between the backend and the frontend.
// (Or multiple frontends, providing collaborative editing.)
// We need to update the ReactFlow state when we get updates from the backend,
// and we need to update the CRDT state when the user makes changes in the UI.

import {
  applyEdgeChanges,
  applyNodeChanges,
  type Edge,
  type Node,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { useEffect, useRef, useSyncExternalStore } from "react";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import type { WorkspaceEdge, WorkspaceNode, Workspace as WorkspaceType } from "../apiTypes.ts";
import { getWebSocketParams } from "../common.ts";

// How often (in ms) to broadcast a node's position to collaborators while it is being dragged.
const POSITION_BROADCAST_INTERVAL_MS = 16;

function endpointSignature(endpoints: any[] | undefined) {
  return (endpoints || []).map((x) => `${x?.name ?? ""}:${x?.position ?? ""}`).join("|");
}

function needsNodeInternalsUpdate(prevNode: any, nextNode: any) {
  if (!prevNode) return true;
  if (prevNode.width !== nextNode.width || prevNode.height !== nextNode.height) return true;
  if (prevNode.data?.collapsed !== nextNode.data?.collapsed) return true;
  if (
    endpointSignature(prevNode.data?.meta?.inputs) !==
    endpointSignature(nextNode.data?.meta?.inputs)
  ) {
    return true;
  }
  if (
    endpointSignature(prevNode.data?.meta?.outputs) !==
    endpointSignature(nextNode.data?.meta?.outputs)
  ) {
    return true;
  }
  if (prevNode.parentId !== nextNode.parentId) return true;
  if (nextNode.data?.display_version !== prevNode.data?.display_version) return true;
  return false;
}

// What the rest of the app observes as the workspace state. Only mutate it through the methods!
export type CRDTWorkspace = {
  ws?: WorkspaceType;
  feNodes: Node[];
  feEdges: Edge[];
  selectedNodeIds: string[];
  selectedNodeCount: number;
  isAnyGroupSelected: boolean;
  setPausedState: (paused: boolean) => void;
  setEnv: (env: string) => void;
  setExecutionOptions: (options: Record<string, any>) => void;
  setAssistantMessages: (messages: any[]) => void;
  clearAssistantMessages: () => void;
  applyChange: (fn: (conn: CRDTConnection) => void) => void;
  addNode: (node: Partial<WorkspaceNode>) => void;
  addEdge: (edge: Partial<WorkspaceEdge>) => void;
  onFENodesChange?: (changes: any[]) => void;
  onFEEdgesChange?: (changes: any[]) => void;
  undo: () => void;
  redo: () => void;
};

export function nodeToYMap(node: any): Y.Map<WorkspaceNode> {
  const data = node.data ?? {};
  const params = data.params ?? {};
  const yparams = new Y.Map<any>();
  for (const [key, value] of Object.entries(params)) {
    yparams.set(key, value);
  }
  const ydata = new Y.Map<any>();
  for (const [key, value] of Object.entries(data)) {
    ydata.set(key, value);
  }
  ydata.set("params", yparams);
  const ynode = new Y.Map<any>();
  for (const [key, value] of Object.entries(node)) {
    if (key === "measured") continue;
    ynode.set(key, value);
  }
  ynode.set("data", ydata);
  return ynode;
}

// The CRDT connection and keeping it in sync with ReactFlow.
class CRDTConnection {
  doc: Y.Doc;
  ws: Y.Map<any>;
  undoManager: Y.UndoManager;
  wsProvider: WebsocketProvider;
  reactFlow: ReturnType<typeof useReactFlow>;
  updateNodeInternals: (id: string) => void;
  state: CRDTWorkspace;
  observers: Set<() => void> = new Set();
  canWrite = true;
  lastPositionBroadcast: Map<string, number> = new Map();
  nodeMaps: Map<string, Y.Map<any>> = new Map();
  // Remote updates can arrive faster than React can render them. Coalesce them
  // until the next animation frame and read the latest Yjs state then. This
  // prevents a slow client from replaying stale drag positions in sequence.
  pendingRemoteNodeIds: Set<string> = new Set();
  pendingRemoteNodesArrayChanged = false;
  pendingRemoteEdgesChanged = false;
  pendingRemoteHeaderChanged = false;
  remoteFlushScheduled = false;
  started = false;
  destroyTimer: ReturnType<typeof setTimeout> | undefined;
  updateSelectionState = () => {
    const selectedNodeIds = this.state.feNodes.filter((n) => n.selected).map((n) => n.id);
    const prevIds = this.state.selectedNodeIds;
    const sameIds =
      prevIds.length === selectedNodeIds.length &&
      prevIds.every((id, index) => id === selectedNodeIds[index]);
    const selectedNodeCount = selectedNodeIds.length;
    const isAnyGroupSelected = this.state.feNodes.some(
      (n) => n.selected && n.type === "node_group",
    );
    if (
      sameIds &&
      this.state.selectedNodeCount === selectedNodeCount &&
      this.state.isAnyGroupSelected === isAnyGroupSelected
    ) {
      return;
    }
    this.state = {
      ...this.state,
      selectedNodeIds,
      selectedNodeCount,
      isAnyGroupSelected,
    };
  };
  constructor(
    reactFlow: ReturnType<typeof useReactFlow>,
    updateNodeInternals: (id: string) => void,
    path: string,
    canWrite = true,
  ) {
    this.reactFlow = reactFlow;
    this.updateNodeInternals = updateNodeInternals;
    this.canWrite = canWrite;
    this.doc = new Y.Doc();
    this.ws = this.doc.getMap("workspace");
    this.undoManager = new Y.UndoManager(this.ws, { captureTimeout: 600 });
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const encodedPath = path!
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    this.wsProvider = new WebsocketProvider(
      `${proto}//${location.host}/ws/crdt`,
      encodedPath,
      this.doc,
      { connect: false },
    );
    this.ws.observeDeep(this.onBackendChange);
    // The initial document sync applies the whole workspace in one transaction.
    // `observeDeep` fires for it, but to be robust (and to cover the case where
    // the workspace is empty or the event paths are ambiguous) we also do a full
    // reconciliation once the provider reports its first sync.
    this.wsProvider.on("sync", (synced: boolean) => {
      if (synced) {
        this.syncAll();
      }
    });
    this.wsProvider.on("status", ({ status }: { status: string }) => {
      if (status === "connected") {
        this.syncAll();
      }
    });
    this.state = {
      feNodes: [],
      feEdges: [],
      selectedNodeIds: [],
      selectedNodeCount: 0,
      isAnyGroupSelected: false,
      setPausedState: (paused: boolean) => this.setWsKey("paused", paused),
      setEnv: (env: string) => this.setWsKey("env", env),
      setExecutionOptions: (options: Record<string, any>) =>
        this.setWsKey("execution_options", options),
      setAssistantMessages: (messages: any[]) => this.setWsKey("assistant_messages", messages),
      clearAssistantMessages: () => this.setWsKey("assistant_messages", []),
      addNode: (node: Partial<WorkspaceNode>) => {
        if (!this.canWrite) return;
        const ynode = nodeToYMap(node);
        this.doc.transact(() => {
          const wnodes = this.ws.get("nodes") as Y.Array<any>;
          wnodes.push([ynode]);
        });
        this.syncAll();
      },
      addEdge: (edge: Partial<WorkspaceEdge>) => {
        if (!this.canWrite) return;
        const yedge = new Y.Map<any>();
        for (const [key, value] of Object.entries(edge)) {
          yedge.set(key, value);
        }
        this.doc.transact(() => {
          const wedges = this.ws.get("edges") as Y.Array<any>;
          wedges.push([yedge]);
        });
        this.syncAll();
      },
      onFENodesChange: this.onFENodesChange,
      onFEEdgesChange: this.onFEEdgesChange,
      applyChange: (fn: (conn: CRDTConnection) => void) => {
        if (!this.canWrite) return;
        this.doc.transact(() => {
          fn(this);
        });
        this.syncAll();
      },
      undo: () => {
        if (!this.canWrite) return;
        this.undoManager.undo();
        this.applyStateAndInternals();
      },
      redo: () => {
        if (!this.canWrite) return;
        this.undoManager.redo();
        this.applyStateAndInternals();
      },
    };
  }
  start = () => {
    if (this.destroyTimer !== undefined) {
      clearTimeout(this.destroyTimer);
      this.destroyTimer = undefined;
    }
    if (this.started) return;
    this.started = true;
    getWebSocketParams().then((params) => {
      if (!this.started) return;
      this.wsProvider.params = params;
      this.wsProvider.connect();
    });
  };
  stop = () => {
    if (!this.started) return;
    this.started = false;
    this.wsProvider.disconnect();
  };
  scheduleDestroy = () => {
    this.stop();
    this.destroyTimer = setTimeout(() => {
      this.destroyTimer = undefined;
      this.onDestroy();
    }, 0);
  };
  setWsKey = (key: string, value: any) => {
    if (!this.canWrite) return;
    this.ws.set(key, value);
    this.updateHeader();
    this.notifyObservers();
  };
  onDestroy = () => {
    this.started = false;
    if (this.destroyTimer !== undefined) {
      clearTimeout(this.destroyTimer);
      this.destroyTimer = undefined;
    }
    this.stop();
    this.doc.destroy();
    this.wsProvider.destroy();
  };
  setCanWrite = (canWrite: boolean) => {
    this.canWrite = canWrite;
  };
  rebuildAllNodes = (): string[] => {
    const wnodes = this.ws.get("nodes") as Y.Array<any> | undefined;
    if (!wnodes) return [];
    const oldNodes = Object.fromEntries(this.state?.feNodes.map((n) => [n.id, n]) || []);
    const newNodes = [];
    const changedNodeIds = [];
    this.nodeMaps.clear();
    for (const nodeMap of wnodes) {
      const id = nodeMap.get("id");
      this.nodeMaps.set(id, nodeMap);
      const mergedNode = this.nodeFromYMap(nodeMap, oldNodes[id]);
      newNodes.push(mergedNode);
      if (needsNodeInternalsUpdate(oldNodes[id], mergedNode)) {
        changedNodeIds.push(id);
      }
    }
    this.state = { ...this.state, feNodes: newNodes as Node[] };
    return changedNodeIds;
  };
  updateNode = (id: string): string[] => {
    const nodeMap = this.nodeMaps.get(id);
    if (!nodeMap) return [];
    const nodeIndex = this.state.feNodes.findIndex((n) => n.id === id);
    if (nodeIndex === -1) return [];
    const oldNode = this.state.feNodes[nodeIndex];
    const mergedNode = this.nodeFromYMap(nodeMap, oldNode);
    const feNodes = this.state.feNodes.slice();
    feNodes[nodeIndex] = mergedNode;
    this.state = {
      ...this.state,
      feNodes,
    };
    return needsNodeInternalsUpdate(oldNode, mergedNode) ? [id] : [];
  };
  rebuildEdges = () => {
    const wedges = this.ws.get("edges") as Y.Array<any> | undefined;
    if (!wedges) return;
    this.state = { ...this.state, feEdges: wedges.toJSON() as Edge[] };
  };
  // Read only the non node/edge fields of the workspace into `ws`.
  updateHeader = () => {
    const header: Record<string, any> = {};
    for (const key of this.ws.keys()) {
      if (key === "nodes" || key === "edges") continue;
      const v = this.ws.get(key);
      header[key] = v && typeof v.toJSON === "function" ? v.toJSON() : v;
    }
    this.state = { ...this.state, ws: header as WorkspaceType };
  };
  syncAll = (): string[] => {
    const changedNodeIds = this.rebuildAllNodes();
    this.rebuildEdges();
    this.updateSelectionState();
    this.updateHeader();
    this.notifyObservers();
    return changedNodeIds;
  };
  applyStateAndInternals = () => {
    const changedNodeIds = this.syncAll();
    if (changedNodeIds.length > 0) {
      requestAnimationFrame(() => {
        for (const nodeId of changedNodeIds) {
          this.updateNodeInternals(nodeId);
        }
      });
    }
  };
  onBackendChange = (events: Y.YEvent<any>[], transaction: Y.Transaction) => {
    // Only react to remote updates. Local mutations are handled by the methods
    // that perform them (onFENodesChange, addNode, ...) to avoid double work.
    if (transaction.origin !== this.wsProvider) return;
    if (!this.ws) return;
    for (const event of events) {
      const path = event.path;
      if (path[0] === "nodes") {
        if (path.length === 1) {
          // The nodes array itself changed (add/remove/move).
          this.pendingRemoteNodesArrayChanged = true;
        } else {
          const idx = path[1] as number;
          const nodeMap = (this.ws.get("nodes") as Y.Array<any>)?.get(idx);
          if (nodeMap) {
            this.pendingRemoteNodeIds.add(nodeMap.get("id"));
          }
        }
      } else if (path[0] === "edges") {
        this.pendingRemoteEdgesChanged = true;
      } else {
        this.pendingRemoteHeaderChanged = true;
      }
    }
    this.scheduleRemoteFlush();
  };
  scheduleRemoteFlush = () => {
    if (this.remoteFlushScheduled) return;
    this.remoteFlushScheduled = true;
    requestAnimationFrame(() => {
      this.remoteFlushScheduled = false;
      this.flushRemoteChanges();
    });
  };
  flushRemoteChanges = () => {
    const changedNodeIds = this.pendingRemoteNodeIds;
    const nodesArrayChanged = this.pendingRemoteNodesArrayChanged;
    const edgesChanged = this.pendingRemoteEdgesChanged;
    const headerChanged = this.pendingRemoteHeaderChanged;
    this.pendingRemoteNodeIds = new Set();
    this.pendingRemoteNodesArrayChanged = false;
    this.pendingRemoteEdgesChanged = false;
    this.pendingRemoteHeaderChanged = false;

    let internalsChanged: string[] = [];
    if (nodesArrayChanged) {
      internalsChanged = this.rebuildAllNodes();
      this.updateSelectionState();
    } else if (changedNodeIds.size > 0) {
      for (const id of changedNodeIds) {
        internalsChanged.push(...this.updateNode(id));
      }
      this.updateSelectionState();
    }
    if (edgesChanged) this.rebuildEdges();
    if (headerChanged) this.updateHeader();
    if (nodesArrayChanged || changedNodeIds.size > 0 || edgesChanged || headerChanged) {
      this.notifyObservers();
    }
    if (internalsChanged.length > 0) {
      requestAnimationFrame(() => {
        for (const nodeId of internalsChanged) {
          this.updateNodeInternals(nodeId);
        }
      });
    }
  };
  onFENodesChange = (changes: any[]) => {
    // An update from the UI.
    // Selection is always allowed; other mutations need write access.
    const allowed = this.canWrite ? changes : changes.filter((ch) => ch.type === "select");
    if (allowed.length === 0) return;
    // Apply it to the local state...
    this.state.feNodes = applyNodeChanges(allowed, this.state.feNodes);
    // ...and to the CRDT state.
    const wnodes = this.ws.get("nodes") as Y.Array<any>;
    const idToNode = this.nodeMaps;
    const feById = new Map(this.state.feNodes.map((n) => [n.id, n]));
    let wsChanged = false;
    let selectionChanged = false;
    for (const ch of allowed) {
      const node = idToNode.get(ch.id);
      if (!node) continue;
      // Position events sometimes come with NaN values. Ignore them.
      if (ch.type === "position" && !Number.isNaN(ch.position.x) && !Number.isNaN(ch.position.y)) {
        const fe = feById.get(ch.id);
        const pos = fe?.position ?? ch.position;
        const current = node.get("position");
        const moved = current.x !== pos.x || current.y !== pos.y;
        if (ch.dragging) {
          const now = typeof performance !== "undefined" ? performance.now() : Date.now();
          const last = this.lastPositionBroadcast.get(ch.id) ?? 0;
          if (moved && now - last >= POSITION_BROADCAST_INTERVAL_MS) {
            this.lastPositionBroadcast.set(ch.id, now);
            this.doc.transact(() => {
              node.set("position", { x: pos.x, y: pos.y });
            });
          }
          wsChanged = true;
        } else {
          this.lastPositionBroadcast.delete(ch.id);
          if (moved) {
            this.doc.transact(() => {
              node.set("position", { x: pos.x, y: pos.y });
            });
          }
          wsChanged = true;
          // Update edge positions.
          this.updateNodeInternals(ch.id);
        }
      } else if (ch.type === "select") {
        selectionChanged = true;
        wsChanged = true;
      } else if (ch.type === "dimensions") {
        if (
          node.get("width") === ch.dimensions.width &&
          node.get("height") === ch.dimensions.height
        ) {
          continue;
        }
        wsChanged = true;
        this.doc.transact(() => {
          node.set("width", ch.dimensions.width);
          node.set("height", ch.dimensions.height);
        });
        // Update edge positions when node size changes.
        this.updateNodeInternals(ch.id);
      } else if (ch.type === "remove") {
        const nodeIndex = wnodes.map((n: Y.Map<any>) => n.get("id")).indexOf(ch.id);
        if (nodeIndex === -1) continue;
        wnodes.delete(nodeIndex);
        wsChanged = true;
        selectionChanged = true;
        this.lastPositionBroadcast.delete(ch.id);
        this.nodeMaps.delete(ch.id);
      } else if (ch.type === "replace") {
        wsChanged = true;
        this.doc.transact(() => {
          const data = ch.item.data;
          const wdata = node.get("data") as Y.Map<any>;
          if (wdata.get("op_id") !== data.op_id) {
            wdata.set("op_id", data.op_id);
          }
          if (wdata.get("error") !== data.error) {
            wdata.set("error", data.error);
          }
          if (node.get("width") !== ch.item.width) {
            node.set("width", ch.item.width);
          }
          if (node.get("height") !== ch.item.height) {
            node.set("height", ch.item.height);
          }
          if (wdata.get("collapsed") !== data.collapsed) {
            wdata.set("collapsed", data.collapsed);
            // Update edge positions when node collapses/expands.
            setTimeout(() => this.updateNodeInternals(ch.id), 0);
          }
          if (wdata.get("expanded_height") !== data.expanded_height) {
            wdata.set("expanded_height", data.expanded_height);
          }
          wdata.set("__execution_delay", data.__execution_delay);
          let wparams = wdata.get("params") as Y.Map<any>;
          if (!wparams) {
            wparams = new Y.Map<any>();
            wdata.set("params", wparams);
          }
          for (const [key, value] of Object.entries(data.params)) {
            if (wparams.get(key) !== value) {
              wparams.set(key, value);
            }
          }
        });
      } else {
        console.log("Unknown node change", ch);
      }
    }
    if (wsChanged) {
      if (selectionChanged) {
        this.updateSelectionState();
      }
      this.updateFEState();
    }
  };
  onFEEdgesChange = (changes: any[]) => {
    const allowed = this.canWrite ? changes : changes.filter((ch) => ch.type === "select");
    if (allowed.length === 0) return;
    this.state.feEdges = applyEdgeChanges(allowed, this.state.feEdges);
    const wedges = this.ws.get("edges") as Y.Array<any>;
    if (!wedges) return;
    for (const ch of allowed) {
      if (ch.type === "remove") {
        const edgeIndex = wedges.map((n: Y.Map<any>) => n.get("id")).indexOf(ch.id);
        wedges.delete(edgeIndex);
      } else if (ch.type === "select") {
      } else {
        console.log("Unknown edge change", ch);
      }
    }
    this.updateFEState();
  };
  getSnapshot = (): CRDTWorkspace => {
    return this.state;
  };
  subscribe = (onStorageChange: () => void): (() => void) => {
    this.observers.add(onStorageChange);
    return () => {
      this.observers.delete(onStorageChange);
    };
  };
  nodeFromYMap = (nodeMap: Y.Map<any>, oldNode: any) => {
    const n = nodeMap.toJSON();
    delete n.measured;
    if (n.type !== "node_group") {
      n.dragHandle = ".drag-handle";
    }
    const mergedNode = { ...oldNode, ...n };

    // Clean up parent-child properties that may be stale from the old ReactFlow node.
    if (!n.parentId) {
      delete mergedNode.parentId;
    }
    if (!n.extent) {
      delete mergedNode.extent;
    }
    delete mergedNode.measured;
    return mergedNode;
  };
  updateFEState = () => {
    this.state = {
      ...this.state,
    };
    this.notifyObservers();
  };
  notifyObservers = () => {
    for (const observer of this.observers) {
      observer();
    }
  };
}

export const EMPTY_WORKSPACE: CRDTWorkspace = {
  feNodes: [],
  feEdges: [],
  selectedNodeIds: [],
  selectedNodeCount: 0,
  isAnyGroupSelected: false,
  setPausedState: () => {},
  setEnv: () => {},
  setExecutionOptions: () => {},
  setAssistantMessages: () => {},
  clearAssistantMessages: () => {},
  applyChange: () => {},
  addNode: () => {},
  addEdge: () => {},
  undo: () => {},
  redo: () => {},
};

const noopSubscribe = () => () => {};

export function useCRDTWorkspace(path: string, canWrite = true, enabled = true): CRDTWorkspace {
  const reactFlow = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const connection = useRef<CRDTConnection | null>(null);
  if (enabled && !connection.current) {
    connection.current = new CRDTConnection(reactFlow, updateNodeInternals, path, canWrite);
  }
  useEffect(() => {
    if (enabled) {
      connection.current?.setCanWrite(canWrite);
    }
  }, [canWrite, enabled]);
  useEffect(() => {
    const currentConnection = connection.current!;
    currentConnection.start();
    return () => currentConnection.scheduleDestroy();
  }, []);
  return useSyncExternalStore(
    connection.current?.subscribe ?? noopSubscribe,
    connection.current?.getSnapshot ?? (() => EMPTY_WORKSPACE),
  );
}
