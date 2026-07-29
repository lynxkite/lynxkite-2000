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
  if (nextNode.data?.display_version !== prevNode.data?.display_version) return true;
  return false;
}

// What the rest of the app observes as the workspace state. Only mutate it through the methods!
type CRDTWorkspace = {
  ws?: WorkspaceType;
  feNodes: Node[];
  feEdges: Edge[];
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
    this.undoManager = new Y.UndoManager(this.doc, { captureTimeout: 600 });
    this.ws = this.doc.getMap("workspace");
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
    getWebSocketParams().then((params) => {
      this.wsProvider.params = params;
      this.wsProvider.connect();
    });
    this.doc.on("update", this.onBackendChange);
    this.state = {
      feNodes: [],
      feEdges: [],
      setPausedState: (paused: boolean) => {
        if (!this.canWrite) return;
        this.ws.set("paused", paused);
        this.updateState();
      },
      setEnv: (env: string) => {
        if (!this.canWrite) return;
        this.ws.set("env", env);
        this.updateState();
      },
      setExecutionOptions: (options: Record<string, any>) => {
        if (!this.canWrite) return;
        this.ws.set("execution_options", options);
        this.updateState();
      },
      setAssistantMessages: (messages: any[]) => {
        if (!this.canWrite) return;
        this.ws.set("assistant_messages", messages);
        this.updateState();
      },
      clearAssistantMessages: () => {
        if (!this.canWrite) return;
        this.ws.set("assistant_messages", []);
        this.updateState();
      },
      addNode: (node: Partial<WorkspaceNode>) => {
        if (!this.canWrite) return;
        const ynode = nodeToYMap(node);
        this.doc.transact(() => {
          const wnodes = this.ws.get("nodes") as Y.Array<any>;
          wnodes.push([ynode]);
        });
        this.updateState();
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
        this.updateState();
      },
      onFENodesChange: this.onFENodesChange,
      onFEEdgesChange: this.onFEEdgesChange,
      applyChange: (fn: (conn: CRDTConnection) => void) => {
        if (!this.canWrite) return;
        this.doc.transact(() => {
          fn(this);
        });
        this.updateState();
      },
      undo: () => {
        if (!this.canWrite) return;
        this.undoManager.undo();
        this.updateState();
      },
      redo: () => {
        if (!this.canWrite) return;
        this.undoManager.redo();
        this.updateState();
      },
    };
  }
  onDestroy = () => {
    this.doc.destroy();
    this.wsProvider.destroy();
  };
  setCanWrite = (canWrite: boolean) => {
    this.canWrite = canWrite;
  };
  onBackendChange = (_update: any, origin: any, _doc: any, _tr: any) => {
    if (origin === this.wsProvider) {
      if (!this.ws) return;
      const changedNodeIds = this.updateState();
      // Batch DOM updates for better performance
      if (changedNodeIds.length > 0) {
        requestAnimationFrame(() => {
          for (const nodeId of changedNodeIds) {
            this.updateNodeInternals(nodeId);
          }
        });
      }
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
    let wsChanged = false;
    for (const ch of allowed) {
      const nodeIndex = wnodes.map((n: Y.Map<any>) => n.get("id")).indexOf(ch.id);
      if (nodeIndex === -1) continue;
      const node = wnodes.get(nodeIndex) as Y.Map<any>;
      // Position events sometimes come with NaN values. Ignore them.
      if (ch.type === "position" && !Number.isNaN(ch.position.x) && !Number.isNaN(ch.position.y)) {
        if (node.get("position").x === ch.position.x && node.get("position").y === ch.position.y) {
          continue;
        }
        wsChanged = true;
        this.doc.transact(() => {
          node.set("position", { x: ch.position.x, y: ch.position.y });
        });
        // Update edge positions.
        this.updateNodeInternals(ch.id);
      } else if (ch.type === "select") {
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
        wnodes.delete(nodeIndex);
        wsChanged = true;
      } else if (ch.type === "replace") {
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
        wsChanged = true;
      } else {
        console.log("Unknown node change", ch);
      }
    }
    if (wsChanged) {
      this.updateState();
    } else {
      this.updateFEState();
    }
  };
  onFEEdgesChange = (changes: any[]) => {
    const allowed = this.canWrite ? changes : changes.filter((ch) => ch.type === "select");
    if (allowed.length === 0) return;
    this.state.feEdges = applyEdgeChanges(allowed, this.state.feEdges);
    const wedges = this.ws.get("edges") as Y.Array<any>;
    if (!wedges) return;
    let wsChanged = false;
    for (const ch of allowed) {
      if (ch.type === "remove") {
        const edgeIndex = wedges.map((n: Y.Map<any>) => n.get("id")).indexOf(ch.id);
        wedges.delete(edgeIndex);
        wsChanged = true;
      } else if (ch.type === "select") {
      } else {
        console.log("Unknown edge change", ch);
      }
    }
    if (wsChanged) {
      this.updateState();
    } else {
      this.updateFEState();
    }
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
  updateState = (): string[] => {
    const ws = this.ws.toJSON() as WorkspaceType;
    if (!ws.nodes) return [];
    if (!ws.edges) return [];
    // Maintain ReactFlow properties on the nodes even as they pass through CRDT.
    const oldNodes = Object.fromEntries(this.state?.feNodes.map((n) => [n.id, n]) || []);
    const newNodes = [];
    const changedNodeIds = [];
    for (const n of ws.nodes) {
      if (n.type !== "node_group") {
        n.dragHandle = ".drag-handle";
      }
      const mergedNode = { ...oldNodes[n.id], ...n };

      // Clean up parent-child properties that may be stale from the old ReactFlow node.
      if (n.parentId === undefined) {
        delete mergedNode.parentId;
      }
      if (n.extent === undefined) {
        delete mergedNode.extent;
      }

      if (
        n.width != null &&
        n.height != null &&
        (oldNodes[n.id]?.measured?.width !== n.width ||
          oldNodes[n.id]?.measured?.height !== n.height)
      ) {
        mergedNode.measured = { width: n.width, height: n.height };
      }

      newNodes.push(mergedNode);
      if (needsNodeInternalsUpdate(oldNodes[n.id], mergedNode)) {
        changedNodeIds.push(n.id);
      }
    }
    this.state = {
      ...this.state,
      ws,
      feNodes: newNodes as Node[],
      feEdges: ws.edges as Edge[],
    };
    this.notifyObservers();
    return changedNodeIds;
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

export function useCRDTWorkspace(path: string, canWrite = true): CRDTWorkspace {
  const reactFlow = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const connection = useRef<CRDTConnection | null>(null);
  if (!connection.current) {
    connection.current = new CRDTConnection(reactFlow, updateNodeInternals, path, canWrite);
  }
  useEffect(() => {
    connection.current?.setCanWrite(canWrite);
  }, [canWrite]);
  useEffect(() => {
    return () => {
      connection.current!.onDestroy();
      connection.current = null;
    };
  }, []);
  return useSyncExternalStore(connection.current.subscribe, connection.current.getSnapshot);
}
