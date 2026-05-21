import React from "react";
import useSWR from "swr";
import type { DirectoryEntry } from "../../../apiTypes";
import { parentPath, pathFetcher, shortName, uploadFile } from "../../../common.ts";
import type { UpdateOptions } from "../NodeParameter";

type PathStrInputProps = {
  value: string;
  onChange: (value: string, options?: UpdateOptions) => void;
};

// The root of the file browser — users cannot navigate above this directory.
const ROOT_PATH = "";

function deriveInitialPath(value: string | undefined): string {
  if (!value) return ROOT_PATH;
  const parts = value.split("/").filter(Boolean);
  if (parts.length <= 1) return ROOT_PATH;
  return parts.slice(0, -1).join("/");
}

export default function PathStrInput({ value, onChange }: PathStrInputProps) {
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

  // Close the file browser when Escape is pressed.
  React.useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isOpen]);

  const canGoUp = currentPath !== ROOT_PATH;

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadError(null);
    try {
      await uploadFile(file);
      onChange(`uploads/${file.name}`, { delay: 0 });
      setIsOpen(false);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  React.useEffect(() => {
    if (!isOpen) return;
    setCurrentPath(deriveInitialPath(value));
  }, [isOpen, value]);

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
                  handleUpload(file);
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
