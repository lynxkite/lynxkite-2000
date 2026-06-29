import { useRef, useState } from "react";
// The directory browser.
import { Link, useNavigate } from "react-router";
import useSWR from "swr";
import ChevronDown from "~icons/tabler/chevron-down";
import DotsVertical from "~icons/tabler/dots-vertical";
import File from "~icons/tabler/file";
import FilePlus from "~icons/tabler/file-plus";
import Folder from "~icons/tabler/folder";
import FolderPlus from "~icons/tabler/folder-plus";
import Home from "~icons/tabler/home";
import LayoutGrid from "~icons/tabler/layout-grid";
import LayoutGridAdd from "~icons/tabler/layout-grid-add";
import Plus from "~icons/tabler/plus";
import Upload from "~icons/tabler/upload";
import type { DirectoryEntry } from "./apiTypes.ts";
import { apiFetch, apiJson, useConfig, usePath } from "./common.ts";
import ManagementPage from "./ManagementPage.tsx";
import { Modal, type ModalHandle } from "./Modal.tsx";
import { WorkspaceProgressList } from "./ProgressSidebar.tsx";

const fetcher = (url: string) => apiJson<DirectoryEntry[]>(url);

function Breadcrumbs(props: { path: string }) {
  if (!props.path) {
    return (
      <div className="breadcrumbs">
        <span className="current-folder">Files</span>
        <title>LynxKite 2000:MM</title>
      </div>
    );
  }

  return (
    <div className="breadcrumbs">
      <Link to="/dir/" aria-label="home">
        <Home />
      </Link>
      <span className="current-folder">
        {props.path
          .split("/")
          .filter(Boolean)
          .map((part, index, parts) => {
            const encodedPartPath = parts
              .slice(0, index + 1)
              .map((segment) => encodeURIComponent(segment))
              .join("/");
            const isLast = index === parts.length - 1;
            return (
              <span key={encodedPartPath}>
                {index > 0 ? <span className="path-delimiter">/</span> : null}
                {isLast ? <span>{part}</span> : <Link to={`/dir/${encodedPartPath}`}>{part}</Link>}
              </span>
            );
          })}
      </span>
      <title>{props.path}</title>
    </div>
  );
}

export default function Directory() {
  useConfig();
  const path = usePath().replace(/^[/]$|^[/]dir$|^[/]dir[/]/, "");
  const encodedPath = encodeURIComponent(path || "");
  const config = useConfig();
  const list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher, {
    dedupingInterval: 0,
  });
  const navigate = useNavigate();
  const renameModalRef = useRef<ModalHandle>(null);
  const [renameTarget, setRenameTarget] = useState<DirectoryEntry | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const newWorkspaceModalRef = useRef<ModalHandle>(null);
  const newCodeFileModalRef = useRef<ModalHandle>(null);
  const newFolderModalRef = useRef<ModalHandle>(null);

  function link(item: DirectoryEntry) {
    const encodedName = encodePathSegments(item.name);
    if (item.type === "directory") {
      return `/dir/${encodedName}`;
    }
    if (item.type === "workspace") {
      return `/edit/${encodedName}`;
    }
    return `/code/${encodedName}`;
  }

  function shortName(item: DirectoryEntry) {
    return item.name
      .split("/")
      .pop()
      ?.replace(/[.]lynxkite[.]json$/, "");
  }

  function encodePathSegments(path: string): string {
    const segments = path.split("/");
    return segments.map((segment) => encodeURIComponent(segment)).join("/");
  }

  function newWorkspaceIn(path: string, workspaceName: string) {
    const pathSlash = path ? `${encodePathSegments(path)}/` : "";
    navigate(`/edit/${pathSlash}${encodeURIComponent(workspaceName)}.lynxkite.json`, {
      replace: true,
    });
  }
  function newCodeFile(path: string, name: string) {
    const pathSlash = path ? `${encodePathSegments(path)}/` : "";
    navigate(`/code/${pathSlash}${encodeURIComponent(name)}`, {
      replace: true,
    });
  }
  async function uploadFiles(files: FileList | File[]) {
    const dirParam = encodeURIComponent(path || "");
    for (const file of Array.from(files)) {
      const formData = new FormData();
      formData.append("file", file);
      const res = await apiFetch(`/api/upload?dir=${dirParam}`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        alert(`Failed to upload ${file.name}.`);
        return;
      }
    }
    list.mutate();
  }

  async function newFolderIn(path: string, folderName: string) {
    const pathSlash = path ? `${path}/` : "";
    const res = await apiFetch("/api/dir/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: pathSlash + folderName }),
    });
    list.mutate();
    if (!res.ok) {
      alert("Failed to create folder.");
    }
  }

  async function deleteItem(item: DirectoryEntry) {
    const confirmationEnabled = localStorage.getItem("lynxkite-delete-confirmation") !== "false";
    if (confirmationEnabled && !window.confirm(`Are you sure you want to delete "${item.name}"?`))
      return;
    const apiPath = item.type === "directory" ? "/api/dir/delete" : "/api/delete";
    await apiFetch(apiPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: item.name }),
    });
    list.mutate();
  }

  function openRenameModal(item: DirectoryEntry) {
    setRenameTarget(item);
    renameModalRef.current?.open(shortName(item) ?? "");
  }

  async function submitRename(newName: string) {
    if (!renameTarget) return;
    const oldParts = renameTarget.name.split("/");
    oldParts.pop();
    const parentPath = oldParts.join("/");
    const targetName = renameTarget.type === "workspace" ? `${newName}.lynxkite.json` : newName;
    const newPath = parentPath ? `${parentPath}/${targetName}` : targetName;
    if (newPath === renameTarget.name) {
      setRenameTarget(null);
      return;
    }
    const res = await apiFetch("/api/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_path: renameTarget.name, new_path: newPath }),
    });
    if (!res.ok) {
      alert("Failed to rename item.");
      return;
    }
    setRenameTarget(null);
    list.mutate();
  }

  const [newMenuOpen, setNewMenuOpen] = useState(false);

  const sidebar = (
    <div className="directory-sidebar">
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        multiple
        onChange={(e) => {
          if (e.target.files?.length) uploadFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        className="sidebar-new-button"
        onClick={() => setNewMenuOpen((o) => !o)}
      >
        <Plus />
        <span className="sidebar-label">New</span>
        <ChevronDown className={`new-menu-chevron sidebar-label${newMenuOpen ? " open" : ""}`} />
      </button>
      {newMenuOpen && (
        <div className="sidebar-new-menu">
          <button
            type="button"
            onClick={() => {
              setNewMenuOpen(false);
              newWorkspaceModalRef.current?.open();
            }}
          >
            <LayoutGridAdd /> <span className="sidebar-label">Workspace</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setNewMenuOpen(false);
              newCodeFileModalRef.current?.open();
            }}
          >
            <FilePlus /> <span className="sidebar-label">Code file</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setNewMenuOpen(false);
              newFolderModalRef.current?.open();
            }}
          >
            <FolderPlus /> <span className="sidebar-label">Folder</span>
          </button>
        </div>
      )}
      <button
        type="button"
        className="sidebar-upload-button"
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload />
        <span className="sidebar-label">Upload</span>
      </button>
      {config.data?.enterprise_available && <WorkspaceProgressList />}
    </div>
  );

  return (
    <ManagementPage sidebar={sidebar}>
      {list.error && <p className="error">{list.error.message}</p>}
      {list.isLoading && (
        <output className="loading spinner-border">
          <span className="visually-hidden">Loading...</span>
        </output>
      )}

      {list.data && (
        <>
          <Breadcrumbs path={path} />

          <div
            className={`entry-drop-zone${isDragOver ? " drag-over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) {
                setIsDragOver(false);
              }
            }}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragOver(false);
              if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
            }}
          >
            {list.data.map(
              (item: DirectoryEntry) =>
                !shortName(item)?.startsWith("__") && (
                  <div key={item.name} className="entry">
                    <Link key={link(item)} to={link(item)}>
                      {item.type === "directory" ? (
                        <Folder />
                      ) : item.type === "workspace" ? (
                        <LayoutGrid />
                      ) : (
                        <File />
                      )}
                      <span className="entry-name">{shortName(item)}</span>
                    </Link>
                    <div className="dropdown dropdown-left dropdown-end">
                      <button
                        className="entry-actions-button"
                        tabIndex={0}
                        type="button"
                        aria-label={`Open actions for ${shortName(item)}`}
                      >
                        <DotsVertical />
                      </button>
                      <ul tabIndex={0} className="dropdown-content menu">
                        <li>
                          <button
                            className="delete-button"
                            type="button"
                            onClick={() => {
                              deleteItem(item);
                            }}
                          >
                            Delete
                          </button>
                        </li>
                        <li>
                          <button
                            type="button"
                            onClick={() => {
                              openRenameModal(item);
                            }}
                          >
                            Rename
                          </button>
                        </li>
                        {item.type !== "directory" && (
                          <li>
                            <button
                              type="button"
                              onClick={async () => {
                                const res = await fetch("/api/download", {
                                  method: "POST",
                                  headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify({ path: item.name }),
                                });
                                if (!res.ok) {
                                  alert("Failed to download file.");
                                  return;
                                }
                                const blob = await res.blob();
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement("a");
                                a.href = url;
                                a.download = item.name;
                                a.click();
                                URL.revokeObjectURL(url);
                              }}
                            >
                              Download
                            </button>
                          </li>
                        )}
                      </ul>
                    </div>
                  </div>
                ),
            )}
            {list.data.length === 0 && <div className="entry empty">This folder is empty.</div>}
          </div>
        </>
      )}
      <Modal
        ref={renameModalRef}
        title="Rename item"
        description={renameTarget ? `Current name: ${shortName(renameTarget)}` : ""}
        inputLabel="New name"
        submitLabel="Rename"
        onSubmit={submitRename}
      />
      <Modal
        ref={newWorkspaceModalRef}
        title="New workspace"
        inputLabel="Name"
        submitLabel="Create"
        onSubmit={(name) => newWorkspaceIn(path || "", name)}
      />
      <Modal
        ref={newCodeFileModalRef}
        title="New code file"
        inputLabel="Name"
        submitLabel="Create"
        onSubmit={(name) => newCodeFile(path || "", name)}
      />
      <Modal
        ref={newFolderModalRef}
        title="New folder"
        inputLabel="Name"
        submitLabel="Create"
        onSubmit={(name) => newFolderIn(path || "", name)}
      />
    </ManagementPage>
  );
}
