/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

/**
 * Access control list settings for a folder.
 */
export interface ACLSettings {
  /**
   * Users with read access, or `*` for anyone.
   */
  read?: string[];
  /**
   * Users with write access, or `*` for anyone.
   */
  write?: string[];
  [k: string]: unknown;
}
export interface BaseModel {
  [k: string]: unknown;
}
/**
 * Folder-specific settings.
 */
export interface FolderSettings {
  acl?: ACLSettings1;
  /**
   * Overrides for LynxKite Inference Microservices.
   */
  lim?: {
    [k: string]: LIMBoxConfig;
  };
  [k: string]: unknown;
}
/**
 * Access control list settings for a folder.
 */
export interface ACLSettings1 {
  /**
   * Users with read access, or `*` for anyone.
   */
  read?: string[];
  /**
   * Users with write access, or `*` for anyone.
   */
  write?: string[];
  [k: string]: unknown;
}
/**
 * Configuration for one type of LynxKite Inference Microservice.
 */
export interface LIMBoxConfig {
  /**
   * CPU resource request for the container
   */
  cpu?: string | null;
  /**
   * Memory resource request for the container
   */
  memory?: string | null;
  /**
   * Container image to use
   */
  image?: string | null;
  /**
   * Port to expose from the container
   */
  port?: number | null;
  /**
   * Kubernetes namespace for the container
   */
  namespace?: string | null;
  /**
   * Environment variables for the container
   */
  env?:
    | {
        [k: string]: unknown;
      }[]
    | null;
  /**
   * Environment variables to forward from the host
   */
  forward_env?: string[] | null;
  /**
   * Requested storage size for the container
   */
  storage_size?: string | null;
  /**
   * Host path for storage volume
   */
  storage_host_path?: string | null;
  /**
   * PVC claim name for storage volume
   */
  storage_pvc_claim_name?: string | null;
  /**
   * Request timeout in seconds
   */
  timeout_seconds?: number | null;
  /**
   * Heartbeat interval in seconds
   */
  heartbeat_seconds?: number | null;
  /**
   * Telemetry reporting interval in seconds
   */
  telemetry_interval_seconds?: number | null;
  /**
   * Startup timeout in seconds
   */
  startup_timeout?: number | null;
  /**
   * Command-line arguments for the container
   */
  args?: string[] | null;
  [k: string]: unknown;
}
/**
 * Configuration for LynxKite Inference Microservices.
 */
export interface LIMSettings {
  [k: string]: unknown;
}
