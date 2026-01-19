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
import type { WorkspaceNode, Workspace as WorkspaceType } from "../apiTypes.ts";

// What the rest of the app observes as the workspace state. Only mutate it through the methods!
type CRDTWorkspace = {
  ws?: WorkspaceType;
  feNodes: Node[];
  feEdges: Edge[];
  setPausedState: (paused: boolean) => void;
  setEnv: (env: string) => void;
  applyChange: (fn: (conn: CRDTConnection) => void) => void;
  addNode: (node: Partial<WorkspaceNode>) => void;
  addEdge: (edge: Edge) => void;
  onFENodesChange?: (changes: any[]) => void;
  onFEEdgesChange?: (changes: any[]) => void;
};

function nodeToYMap(node: any): Y.Map<WorkspaceNode> {
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
  wsProvider: WebsocketProvider;
  reactFlow: ReturnType<typeof useReactFlow>;
  updateNodeInternals: (id: string) => void;
  state: CRDTWorkspace;
  observers: Set<() => void> = new Set();
  constructor(
    reactFlow: ReturnType<typeof useReactFlow>,
    updateNodeInternals: (id: string) => void,
    path: string,
  ) {
    this.reactFlow = reactFlow;
    this.updateNodeInternals = updateNodeInternals;
    this.doc = new Y.Doc();
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
    );
    this.doc.on("update", this.onBackendChange);
    const that = this;
    this.state = {
      feNodes: [],
      feEdges: [],
      setPausedState: (paused: boolean) => {
        that.ws.set("paused", paused);
        that.updateState();
      },
      setEnv: (env: string) => {
        that.ws.set("env", env);
        that.updateState();
      },
      addNode: (node: Partial<WorkspaceNode>) => {
        const ynode = nodeToYMap(node);
        that.doc.transact(() => {
          const wnodes = that.ws.get("nodes") as Y.Array<any>;
          wnodes.push([ynode]);
        });
        that.updateState();
      },
      addEdge(edge) {
        const yedge = new Y.Map<any>();
        for (const [key, value] of Object.entries(edge)) {
          yedge.set(key, value);
        }
        that.doc.transact(() => {
          const wedges = that.ws.get("edges") as Y.Array<any>;
          wedges.push([yedge]);
        });
        that.updateState();
      },
      onFENodesChange: that.onFENodesChange,
      onFEEdgesChange: that.onFEEdgesChange,
      applyChange: (fn: (conn: CRDTConnection) => void) => {
        that.doc.transact(() => {
          fn(that);
        });
        that.updateState();
      },
    };
  }
  onDestroy = () => {
    this.doc.destroy();
    this.wsProvider.destroy();
  };
  onBackendChange = (_update: any, origin: any, _doc: any, _tr: any) => {
    if (origin === this.wsProvider) {
      if (!this.ws) return;
      const ws = this.ws.toJSON() as WorkspaceType;
      if (!ws.nodes) return;
      if (!ws.edges) return;
      for (const n of ws.nodes) {
        if (n.type !== "node_group" && n.dragHandle !== ".drag-handle") {
          n.dragHandle = ".drag-handle";
        }
      }
      const nodes = this.reactFlow.getNodes();
      const selection = nodes.filter((n) => n.selected).map((n) => n.id);
      const newNodes = ws.nodes.map((n) =>
        selection.includes(n.id) ? { ...n, selected: true } : n,
      );
      this.updateState({
        ...this.state,
        ws,
        feNodes: newNodes as Node[],
        feEdges: ws.edges as Edge[],
      });
      for (const node of ws.nodes || []) {
        this.updateNodeInternals(node.id);
      }
    }
  };
  onFENodesChange = (changes: any[]) => {
    // An update from the UI.
    // Apply it to the local state...
    const newFENodes = applyNodeChanges(changes, this.state.feNodes);
    // ...and to the CRDT state.
    const wnodes = this.ws.get("nodes") as Y.Array<any>;
    for (const ch of changes) {
      const nodeIndex = wnodes.map((n: Y.Map<any>) => n.get("id")).indexOf(ch.id);
      if (nodeIndex === -1) continue;
      const node = wnodes.get(nodeIndex) as Y.Map<any>;
      // Position events sometimes come with NaN values. Ignore them.
      if (ch.type === "position" && !Number.isNaN(ch.position.x) && !Number.isNaN(ch.position.y)) {
        this.doc.transact(() => {
          node.set("position", { x: ch.position.x, y: ch.position.y });
        });
        // Update edge positions.
        this.updateNodeInternals(ch.id);
      } else if (ch.type === "select") {
      } else if (ch.type === "dimensions") {
        this.doc.transact(() => {
          node.set("width", ch.dimensions.width);
          node.set("height", ch.dimensions.height);
        });
        // Update edge positions when node size changes.
        this.updateNodeInternals(ch.id);
      } else if (ch.type === "remove") {
        wnodes.delete(nodeIndex);
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
          if (wdata.get("collapsed") !== data.collapsed) {
            wdata.set("collapsed", data.collapsed);
            // Update edge positions when node collapses/expands.
            setTimeout(() => this.updateNodeInternals(ch.id), 0);
          }
          if (wdata.get("expanded_height") !== data.expanded_height) {
            wdata.set("expanded_height", data.expanded_height);
          }
          if (wdata.get("__execution_delay") !== data.__execution_delay) {
            wdata.set("__execution_delay", data.__execution_delay);
          }
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
    this.updateState({
      ...this.state,
      feNodes: newFENodes,
    });
  };
  onFEEdgesChange = (changes: any[]) => {
    const newFEEdges = applyEdgeChanges(changes, this.state.feEdges);
    const wedges = this.state.ws?.edges;
    if (!wedges) return;
    for (const ch of changes) {
      const edgeIndex = wedges.findIndex((e) => e.id === ch.id);
      if (ch.type === "remove") {
        wedges.splice(edgeIndex, 1);
      } else if (ch.type === "select") {
      } else {
        console.log("Unknown edge change", ch);
      }
    }
    this.updateState({
      ...this.state,
      feEdges: newFEEdges,
    });
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
  updateState = (newState?: CRDTWorkspace) => {
    if (!newState) newState = { ...this.state, ws: this.ws.toJSON() as CRDTWorkspace };
    this.state = newState;
    for (const observer of this.observers) {
      observer();
    }
  };
}

export function useCRDTWorkspace(path: string): CRDTWorkspace {
  const reactFlow = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const connection = useRef<CRDTConnection | null>(null);
  if (!connection.current) {
    connection.current = new CRDTConnection(reactFlow, updateNodeInternals, path);
  }
  useEffect(() => {
    return () => {
      connection.current!.onDestroy();
      connection.current = null;
    };
  }, []);
  return useSyncExternalStore(connection.current.subscribe, connection.current.getSnapshot);
}
