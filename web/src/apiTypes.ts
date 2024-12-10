/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface BaseConfig {
  [k: string]: unknown;
}
export interface Position {
  x: number;
  y: number;
  [k: string]: unknown;
}
export interface Workspace {
  env?: string;
  nodes?: WorkspaceNode[];
  edges?: WorkspaceEdge[];
  [k: string]: unknown;
}
export interface WorkspaceNode {
  id: string;
  type: string;
  data: WorkspaceNodeData;
  position: Position;
  [k: string]: unknown;
}
export interface WorkspaceNodeData {
  title: string;
  params: {
    [k: string]: unknown;
  };
  display?: unknown;
  error?: string | null;
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
