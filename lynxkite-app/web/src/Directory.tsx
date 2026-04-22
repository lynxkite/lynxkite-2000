import { type ReactElement, useRef, useState } from "react";
// The directory browser.
import { Link, useNavigate } from "react-router";
import useSWR from "swr";
import DotsVertical from "~icons/tabler/dots-vertical";
import File from "~icons/tabler/file";
import FilePlus from "~icons/tabler/file-plus";
import Folder from "~icons/tabler/folder";
import FolderPlus from "~icons/tabler/folder-plus";
import Home from "~icons/tabler/home";
import LayoutGrid from "~icons/tabler/layout-grid";
import LayoutGridAdd from "~icons/tabler/layout-grid-add";
import type { DirectoryEntry } from "./apiTypes.ts";
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";
import { usePath } from "./common.ts";
import { Modal, type ModalHandle } from "./Modal.tsx";

function EntryCreator(props: {
  label: string;
  icon: ReactElement;
  onCreate: (name: string) => void;
}) {
  const modalRef = useRef<ModalHandle>(null);

  return (
    <>
      <button type="button" onClick={() => modalRef.current?.open()}>
        {props.icon} {props.label}
      </button>
      <Modal
        ref={modalRef}
        title={props.label}
        inputLabel="Name"
        submitLabel="Create"
        onSubmit={props.onCreate}
      />
    </>
  );
}

const fetcher = (url: string) => fetch(url).then((res) => res.json());

function Breadcrumbs(props: { path: string }) {
  if (!props.path) {
    return <title>LynxKite 2000:MM</title>;
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
  const path = usePath().replace(/^[/]$|^[/]dir$|^[/]dir[/]/, "");
  const encodedPath = encodeURIComponent(path || "");
  const list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher, {
    dedupingInterval: 0,
  });
  const navigate = useNavigate();
  const renameModalRef = useRef<ModalHandle>(null);
  const [renameTarget, setRenameTarget] = useState<DirectoryEntry | null>(null);

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
  async function newFolderIn(path: string, folderName: string) {
    const pathSlash = path ? `${path}/` : "";
    const res = await fetch("/api/dir/mkdir", {
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
    await fetch(apiPath, {
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
    const res = await fetch("/api/rename", {
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

  return (
    <div className="directory">
      <div className="logo">
        <a href="https://lynxkite.com/">
          <img src={logo} className="logo-image" alt="LynxKite logo" />
        </a>
        <img src={logoSparky} className="logo-image-sparky" alt="LynxKite logo" />
        <div className="tagline">The Complete Graph Data Science Platform</div>
      </div>
      <div className="entry-list">
        {list.error && <p className="error">{list.error.message}</p>}
        {list.isLoading && (
          <output className="loading spinner-border">
            <span className="visually-hidden">Loading...</span>
          </output>
        )}

        {list.data && (
          <>
            <div className="actions">
              <EntryCreator
                onCreate={(name) => {
                  newWorkspaceIn(path || "", name);
                }}
                icon={<LayoutGridAdd />}
                label="New workspace"
              />
              <EntryCreator
                onCreate={(name) => {
                  newCodeFile(path || "", name);
                }}
                icon={<FilePlus />}
                label="New code file"
              />
              <EntryCreator
                onCreate={(name: string) => {
                  newFolderIn(path || "", name);
                }}
                icon={<FolderPlus />}
                label="New folder"
              />
            </div>

            <Breadcrumbs path={path} />

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
                      </ul>
                    </div>
                  </div>
                ),
            )}
            {list.data.length === 0 && <div className="entry empty">This folder is empty.</div>}
          </>
        )}
      </div>

      <Modal
        ref={renameModalRef}
        title="Rename item"
        description={renameTarget ? `Current name: ${shortName(renameTarget)}` : ""}
        inputLabel="New name"
        submitLabel="Rename"
        onSubmit={submitRename}
      />
    </div>
  );
}
