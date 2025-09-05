import { useState } from "react";
// The directory browser.
import { Link, useNavigate } from "react-router";
import useSWR from "swr";
// @ts-expect-error
import File from "~icons/tabler/file";
// @ts-expect-error
import FilePlus from "~icons/tabler/file-plus";
// @ts-expect-error
import Folder from "~icons/tabler/folder";
// @ts-expect-error
import FolderPlus from "~icons/tabler/folder-plus";
// @ts-expect-error
import Home from "~icons/tabler/home";
// @ts-expect-error
import LayoutGrid from "~icons/tabler/layout-grid";
// @ts-expect-error
import LayoutGridAdd from "~icons/tabler/layout-grid-add";
// @ts-expect-error
import Trash from "~icons/tabler/trash";
import type { DirectoryEntry } from "./apiTypes.ts";
import logo from "./assets/logo.png";
import { usePath } from "./common.ts";

function EntryCreator(props: {
  label: string;
  icon: JSX.Element;
  onCreate: (name: string) => void;
}) {
  const [isCreating, setIsCreating] = useState(false);
  const [nameValidationError, setNameValidationError] = useState("");

  function validateName(name: string): boolean {
    if (name.includes("/")) {
      setNameValidationError("Name cannot contain '/' characters");
      return false;
    }
    if (name.trim() === "") {
      setNameValidationError("Name cannot be empty");
      return false;
    }
    setNameValidationError("");
    return true;
  }

  return (
    <>
      {isCreating ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const name = (e.target as HTMLFormElement).entryName.value.trim();
            if (validateName(name)) {
              props.onCreate(name);
              setIsCreating(false);
            }
          }}
        >
          <input
            className={`input input-ghost w-full ${nameValidationError ? "input-error" : ""}`}
            autoFocus
            type="text"
            name="entryName"
            onBlur={() => setIsCreating(false)}
            onChange={(e) => validateName(e.target.value)}
            placeholder={`${props.label} name`}
          />
          {nameValidationError && (
            <div
              className="error-message"
              role="alert"
              style={{ position: "absolute", zIndex: 10 }}
            >
              <span className="error-icon" aria-hidden="true">
                ⚠️
              </span>
              <span className="error-text">{nameValidationError}</span>
            </div>
          )}
        </form>
      ) : (
        <button type="button" onClick={() => setIsCreating(true)}>
          {props.icon} {props.label}
        </button>
      )}
    </>
  );
}

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function Directory() {
  const path = usePath().replace(/^[/]$|^[/]dir$|^[/]dir[/]/, "");
  const encodedPath = encodeURIComponent(path || "");
  const list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher, {
    dedupingInterval: 0,
  });
  const navigate = useNavigate();

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
    navigate(`/code/${pathSlash}${encodeURIComponent(name)}`, { replace: true });
  }
  async function newFolderIn(path: string, folderName: string) {
    const pathSlash = path ? `${path}/` : "";
    const res = await fetch("/api/dir/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: pathSlash + folderName }),
    });
    if (res.ok) {
      const pathSlash = path ? `${encodePathSegments(path)}/` : "";
      navigate(`/dir/${pathSlash}${encodeURIComponent(folderName)}`);
    } else {
      alert("Failed to create folder.");
    }
  }

  async function deleteItem(item: DirectoryEntry) {
    if (!window.confirm(`Are you sure you want to delete "${item.name}"?`)) return;
    const apiPath = item.type === "directory" ? "/api/dir/delete" : "/api/delete";
    await fetch(apiPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: item.name }),
    });
  }

  return (
    <div className="directory">
      <div className="logo">
        <a href="https://lynxkite.com/">
          <img src={logo} className="logo-image" alt="LynxKite logo" />
        </a>
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

            {path ? (
              <div className="breadcrumbs">
                <Link to="/dir/" aria-label="home">
                  <Home />
                </Link>{" "}
                <span className="current-folder">{path}</span>
                <title>{path}</title>
              </div>
            ) : (
              <title>LynxKite 2000:MM</title>
            )}

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
                    <button
                      type="button"
                      onClick={() => {
                        deleteItem(item);
                      }}
                    >
                      <Trash />
                    </button>
                  </div>
                ),
            )}
          </>
        )}
      </div>{" "}
    </div>
  );
}
