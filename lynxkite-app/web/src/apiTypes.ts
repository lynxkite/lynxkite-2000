/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

/**
 * Defines the position of an input or output in the UI.
 */
export type Position = "left" | "right" | "top" | "bottom";
export type NodeStatus = "planned" | "active" | "done";

export interface DirectoryEntry {
  name: string;
  type: string;
}
export interface Input {
  name: string;
  type: {
    [k: string]: unknown;
  };
  position: Position;
}
export interface Op {
  categories: string[];
  name: string;
  params: (Parameter | ParameterGroup)[];
  inputs: Input[];
  outputs: Output[];
  type?: string;
  color?: string;
  icon?: string | null;
  doc?: unknown[] | null;
  id?: string;
}
/**
 * Defines a parameter for an operation.
 */
export interface Parameter {
  name: string;
  default: unknown;
  type?: {
    [k: string]: unknown;
  };
}
/**
 * Defines a group of parameters for an operation.
 */
export interface ParameterGroup {
  name: string;
  selector: Parameter;
  default: unknown;
  groups: {
    [k: string]: Parameter[];
  };
  type?: string;
}
export interface Output {
  name: string;
  type: {
    [k: string]: unknown;
  };
  position: Position;
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
  execution_options?: {
    [k: string]: unknown;
  };
  nodes?: WorkspaceNode[];
  edges?: WorkspaceEdge[];
  paused?: boolean | null;
  path?: string | null;
  [k: string]: unknown;
}
export interface WorkspaceNode {
  id: string;
  type: string;
  data: WorkspaceNodeData;
  position: Position1;
  width?: number | null;
  height?: number | null;
  [k: string]: unknown;
}
export interface WorkspaceNodeData {
  title: string;
  op_id: string;
  params: {
    [k: string]: unknown;
  };
  display?: unknown;
  input_metadata?:
    | {
        [k: string]: unknown;
      }[]
    | null;
  output_metadata?:
    | {
        [k: string]: unknown;
      }[]
    | null;
  error?: string | null;
  message?: string | null;
  collapsed?: boolean | null;
  expanded_height?: number | null;
  status?: NodeStatus;
  meta?: Op | null;
  [k: string]: unknown;
}
export interface Position1 {
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
