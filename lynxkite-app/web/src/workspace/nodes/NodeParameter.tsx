import jmespath from "jmespath";
import React from "react";
import useSWR from "swr";
import type { DirectoryEntry } from "../../apiTypes";
import Tooltip from "../../Tooltip";
import ModelMapping from "./ModelMappingParameter";
import NodeGroupParameter from "./NodeGroupParameter";
import ParameterInput from "./ParameterInput";

const BOOLEAN = "<class 'bool'>";
const MODEL_TRAINING_INPUT_MAPPING =
  "lynxkite_graph_analytics.ml_ops.ModelTrainingInputMapping | None";
const MODEL_INFERENCE_INPUT_MAPPING =
  "lynxkite_graph_analytics.ml_ops.ModelInferenceInputMapping | None";
const MODEL_OUTPUT_MAPPING = "lynxkite_graph_analytics.ml_ops.ModelOutputMapping | None";

function ParamName({ name, doc }: { name: string; doc: string }) {
  return (
    <div className="param-name-row">
      <Tooltip doc={doc}>
        <span className="param-name">{name.replace(/_/g, " ")}</span>
      </Tooltip>
    </div>
  );
}

interface NodeParameterProps {
  name: string;
  value: any;
  meta: any;
  data: any;
  setParam: (name: string, value: any, options: UpdateOptions) => void;
}

export type UpdateOptions = { delay?: number };

function findDocs(docs: any, parameter: string) {
  for (const sec of docs) {
    if (sec.kind === "parameters") {
      for (const p of sec.value) {
        if (p.name === parameter) {
          return p.description;
        }
      }
    }
  }
}

type PathStrInputProps = {
  value: string;
  onChange: (value: string, options?: UpdateOptions) => void;
};

const pathFetcher = (url: string) => fetch(url).then((res) => res.json());

// The root of the file browser — users cannot navigate above this directory.
const ROOT_PATH = "examples";

function deriveInitialPath(value: string | undefined): string {
  if (!value) return ROOT_PATH;
  // For URLs (e.g. http://, s3://) start at the root path.
  if (value.includes("://")) return ROOT_PATH;
  const parts = value.split("/").filter(Boolean);
  if (parts.length <= 1) return ROOT_PATH;
  return parts.slice(0, -1).join("/");
}

function parentPath(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.slice(0, -1).join("/");
}

function shortName(path: string): string {
  return path.split("/").pop() ?? path;
}

function PathStrInput({ value, onChange }: PathStrInputProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [currentPath, setCurrentPath] = React.useState(ROOT_PATH);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const list = useSWR<DirectoryEntry[]>(
    isOpen ? `/api/dir/list?path=${encodeURIComponent(currentPath)}` : null,
    pathFetcher,
    { dedupingInterval: 0 },
  );

  React.useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isOpen]);

  React.useEffect(() => {
    if (!isOpen) return;
    setCurrentPath(deriveInitialPath(value));
  }, [isOpen]);

  const canGoUp = currentPath !== ROOT_PATH;

  async function uploadFile(file: File) {
    setUploading(true);
    setUploadError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Upload failed");
      onChange(`uploads/${file.name}`, { delay: 0 });
      setIsOpen(false);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="path-input-container">
      <div className="path-input">
        <input
          className="input input-bordered w-full path-input-field"
          value={value ?? ""}
          onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
          onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
          onKeyDown={(evt) =>
            evt.code === "Enter" && onChange(evt.currentTarget.value, { delay: 0 })
          }
        />
        <button
          className="btn btn-ghost path-input-button"
          type="button"
          onClick={() => setIsOpen((v) => !v)}
          aria-label="Browse files"
        >
          ...
        </button>
      </div>

      {isOpen && (
        <div className="path-browser-panel">
          <div className="path-browser-path">
            <span className="path-browser-label">Current:</span>
            <span>{`/${currentPath}`}</span>
          </div>
          <div className="path-browser-list">
            {canGoUp && (
              <button
                type="button"
                className="path-browser-row"
                onClick={() => setCurrentPath(parentPath(currentPath))}
              >
                <span className="path-browser-row-name">..</span>
                <span className="path-browser-row-type">Parent</span>
              </button>
            )}
            {list.isLoading && <div className="path-browser-row">Loading...</div>}
            {list.error && <div className="path-browser-row">Failed to load directory.</div>}
            {list.data?.map((entry) => {
              const isDir = entry.type === "directory";
              return (
                <button
                  type="button"
                  key={entry.name}
                  className="path-browser-row"
                  onClick={() => {
                    if (isDir) {
                      setCurrentPath(entry.name);
                    } else {
                      onChange(entry.name, { delay: 0 });
                      setIsOpen(false);
                    }
                  }}
                >
                  <span className="path-browser-row-name">{shortName(entry.name)}</span>
                  <span className="path-browser-row-type">{isDir ? "Folder" : "File"}</span>
                </button>
              );
            })}
          </div>
          <div className="path-browser-upload">
            <button
              type="button"
              className="btn btn-outline btn-sm"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              {uploading ? "Uploading..." : "Upload file"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              style={{ display: "none" }}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                if (file) {
                  uploadFile(file);
                  event.currentTarget.value = "";
                }
              }}
            />
            {uploadError && <div className="path-browser-upload-error">{uploadError}</div>}
          </div>
        </div>
      )}
    </div>
  );
}

export default function NodeParameter({ name, value, meta, data, setParam }: NodeParameterProps) {
  const doc = findDocs(data.meta?.value?.doc ?? [], name);
  function onChange(value: any, opts?: UpdateOptions) {
    setParam(meta.name, value, opts || {});
  }
  return meta?.type?.format === "textarea" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <textarea
        className="textarea textarea-bordered w-full"
        rows={(value ?? "").split("\n").length}
        value={value ?? ""}
        onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
        onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
      />
    </label>
  ) : meta?.type?.format === "dropdown" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <select
        className="select select-bordered appearance-none w-full"
        value={value ?? ""}
        onChange={(evt) => onChange(evt.currentTarget.value)}
      >
        {getDropDownValues(data, meta?.type?.metadata_query).map((option: string) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  ) : meta?.type?.format === "double-dropdown" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <div className="double-dropdown">
        <select
          className="select select-bordered appearance-none double-dropdown-first"
          value={value?.[0] ?? ""}
          onChange={(evt) => onChange([evt.currentTarget.value, value?.[1]])}
        >
          {getDropDownValues(data, meta?.type?.metadata_query1).map((option: string) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <select
          className="select select-bordered appearance-none double-dropdown-second"
          value={value?.[1] ?? ""}
          onChange={(evt) => onChange([value?.[0], evt.currentTarget.value])}
        >
          {getDropDownValues(data, meta?.type?.metadata_query2, { first: value?.[0] }).map(
            (option: string) => (
              <option key={option} value={option}>
                {option}
              </option>
            ),
          )}
        </select>
      </div>
    </label>
  ) : meta?.type === "group" ? (
    <NodeGroupParameter meta={meta} data={data} setParam={setParam} />
  ) : meta?.type?.enum ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <select
        className="select select-bordered appearance-none w-full"
        value={value || meta.type.enum[0]}
        onChange={(evt) => onChange(evt.currentTarget.value)}
      >
        {meta.type.enum.map((option: string) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  ) : meta?.type?.type === BOOLEAN ? (
    <div className="form-control">
      <label className="label cursor-pointer checkbox-param">
        {name.replace(/_/g, " ")}
        <input
          className="checkbox"
          type="checkbox"
          checked={value}
          onChange={(evt) => onChange(evt.currentTarget.checked)}
        />
      </label>
    </div>
  ) : meta?.type?.type === MODEL_TRAINING_INPUT_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ModelMapping value={value} data={data} variant="training input" onChange={onChange} />
    </label>
  ) : meta?.type?.type === MODEL_INFERENCE_INPUT_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ModelMapping value={value} data={data} variant="inference input" onChange={onChange} />
    </label>
  ) : meta?.type?.type === MODEL_OUTPUT_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ModelMapping value={value} data={data} variant="output" onChange={onChange} />
    </label>
  ) : meta?.type?.format === "path" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <PathStrInput value={value ?? ""} onChange={onChange} />
    </label>
  ) : (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ParameterInput value={value} onChange={onChange} />
    </label>
  );
}

function getDropDownValues(
  data: any,
  query: string,
  substitutions?: Record<string, string>,
): string[] {
  const metadata = data.input_metadata;
  if (!metadata || !query) return [];
  const ss = { ...data.params, ...substitutions };
  for (const k in ss) {
    query = query.replace(`<${k}>`, ss[k]);
  }
  try {
    const res = ["", ...jmespath.search(metadata, query)];
    res.sort();
    return res;
  } catch (_) {
    return [""];
  }
}
