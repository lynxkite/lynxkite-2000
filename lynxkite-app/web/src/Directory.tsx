import { useState } from "react";
// The directory browser.
import { Link, useNavigate } from "react-router";
import useSWR from "swr";
import type { DirectoryEntry } from "./apiTypes.ts";
import { usePath } from "./common.ts";

// @ts-ignore
import File from "~icons/tabler/file";
// @ts-ignore
import FilePlus from "~icons/tabler/file-plus";
// @ts-ignore
import Folder from "~icons/tabler/folder";
// @ts-ignore
import FolderPlus from "~icons/tabler/folder-plus";
// @ts-ignore
import Home from "~icons/tabler/home";
// @ts-ignore
import LayoutGrid from "~icons/tabler/layout-grid";
// @ts-ignore
import LayoutGridAdd from "~icons/tabler/layout-grid-add";
// @ts-ignore
import Trash from "~icons/tabler/trash";
import logo from "./assets/logo.png";

function EntryCreator(props: {
  label: string;
  icon: JSX.Element;
  onCreate: (name: string) => void;
}) {
  const [isCreating, setIsCreating] = useState(false);
  return (
    <>
      {isCreating ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            props.onCreate((e.target as HTMLFormElement).entryName.value.trim());
          }}
        >
          <input
            className="input input-ghost w-full"
            autoFocus
            type="text"
            name="entryName"
            onBlur={() => setIsCreating(false)}
            placeholder={`${props.label} name`}
          />
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

export default function () {
  const path = usePath().replace(/^[/]$|^[/]dir$|^[/]dir[/]/, "");
  const encodedPath = encodeURIComponent(path || "");
  const list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher, {
    dedupingInterval: 0,
  });
  const navigate = useNavigate();

  function link(item: DirectoryEntry) {
    if (item.type === "directory") {
      return `/dir/${item.name}`;
    }
    if (item.type === "workspace") {
      return `/edit/${item.name}`;
    }
    return `/code/${item.name}`;
  }

  function shortName(item: DirectoryEntry) {
    return item.name
      .split("/")
      .pop()
      ?.replace(/[.]lynxkite[.]json$/, "");
  }

  function newWorkspaceIn(path: string, workspaceName: string) {
    const pathSlash = path ? `${path}/` : "";
    navigate(`/edit/${pathSlash}${workspaceName}.lynxkite.json`, { replace: true });
  }
  function newCodeFile(path: string, name: string) {
    const pathSlash = path ? `${path}/` : "";
    navigate(`/code/${pathSlash}${name}`, { replace: true });
  }
  async function newFolderIn(path: string, folderName: string) {
    const pathSlash = path ? `${path}/` : "";
    const res = await fetch("/api/dir/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: pathSlash + folderName }),
    });
    if (res.ok) {
      navigate(`/dir/${pathSlash}${folderName}`);
    } else {
      alert("Failed to create folder.");
    }
  }

  async function deleteItem(item: DirectoryEntry) {
    if (!window.confirm(`Are you sure you want to delete "${item.name}"?`)) return;
    const pathSlash = path ? `${path}/` : "";

    const apiPath = item.type === "directory" ? "/api/dir/delete" : "/api/delete";
    await fetch(apiPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: pathSlash + item.name }),
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
