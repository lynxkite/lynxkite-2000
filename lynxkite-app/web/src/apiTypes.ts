/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export type NodeStatus = "planned" | "active" | "done";

export interface DirectoryEntry {
  name: string;
  type: string;
}
export interface SaveRequest {
  path: string;
  ws: Workspace;
  [k: string]: unknown;
}
/**
 * A workspace is a representation of a computational graph that consists of nodes and edges.
 *
 * Each node represents an operation or task, and the edges represent the flow of data between
 * the nodes. Each workspace is associated with an environment, which determines the operations
 * that can be performed in the workspace and the execution method for the operations.
 */
export interface Workspace {
  env?: string;
  paused?: boolean;
  nodes?: WorkspaceNode[];
  edges?: WorkspaceEdge[];
  [k: string]: unknown;
}
export interface WorkspaceNode {
  id: string;
  type: string;
  data: WorkspaceNodeData;
  position: Position;
  width: number;
  height: number;
  [k: string]: unknown;
}
export interface WorkspaceNodeData {
  title: string;
  params: {
    [k: string]: unknown;
  };
  display?: unknown;
  input_metadata?: unknown;
  error?: string | null;
  status?: NodeStatus;
  [k: string]: unknown;
}
export interface Position {
  x: number;
  y: number;
  [k: string]: unknown;
}
export interface WorkspaceEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
  [k: string]: unknown;
}
