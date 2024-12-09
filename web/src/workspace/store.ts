// Like described in https://reactflow.dev/learn/advanced-use/state-management
// but with https://github.com/joebobmiles/zustand-middleware-yjs/ added.
import {
  type Edge,
  type Node,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
} from '@xyflow/react';
import * as apiTypes from "../apiTypes.ts";
import * as Y from "yjs";
import yjs from "zustand-middleware-yjs";
export const doc = new Y.Doc();

export type FlowState = {
  env: string;
  nodes: apiTypes.WorkspaceNode[];
  edges: apiTypes.WorkspaceEdge[];
  onNodesChange: OnNodesChange<Node>;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  setEnv: (env: string) => void;
};

import { create } from 'zustand';
import { addEdge, applyNodeChanges, applyEdgeChanges } from '@xyflow/react';

export const useStore = create<FlowState>(yjs(doc, "shared", (set: any, get: any) => ({
  env: 'LynxKite',
  nodes: [],
  edges: [],
  onNodesChange: (changes: any[]) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
    });
  },
  onEdgesChange: (changes: any[]) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },
  onConnect: (connection: any) => {
    set({
      edges: addEdge(connection, get().edges),
    });
  },
  setNodes: (nodes: Node[]) => {
    console.log("setNodes", { nodes });
    set({ nodes });
  },
  setEdges: (edges: Edge[]) => {
    set({ edges });
  },
  setEnv: (env: string) => {
    set({ env });
  },
})));

export const selector = (state: FlowState) => ({
  nodes: state.nodes,
  edges: state.edges,
  env: state.env,
  onNodesChange: state.onNodesChange,
  onEdgesChange: state.onEdgesChange,
  onConnect: state.onConnect,
  setEnv: state.setEnv,
});
