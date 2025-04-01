import { useState } from "react";
// The directory browser.
import { Link, useNavigate, useParams } from "react-router";
import useSWR from "swr";
import type { DirectoryEntry } from "./apiTypes.ts";

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
import Trash from "~icons/tabler/trash";
import logo from "./assets/logo.png";

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function () {
  const { path } = useParams();
  const encodedPath = encodeURIComponent(path || "");
  const list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher, {
    dedupingInterval: 0,
  });
  const navigate = useNavigate();
  const [isCreatingDir, setIsCreatingDir] = useState(false);
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);

  function link(item: DirectoryEntry) {
    if (item.type === "directory") {
      return `/dir/${item.name}`;
    }
    return `/edit/${item.name}`;
  }

  function shortName(item: DirectoryEntry) {
    return item.name.split("/").pop();
  }

  function newName(list: DirectoryEntry[], baseName = "Untitled") {
    let i = 0;
    while (true) {
      const name = `${baseName}${i ? ` ${i}` : ""}`;
      if (!list.find((item) => item.name === name)) {
        return name;
      }
      i++;
    }
  }

  function newWorkspaceIn(path: string, list: DirectoryEntry[], workspaceName?: string) {
    const pathSlash = path ? `${path}/` : "";
    const name = workspaceName || newName(list);
    navigate(`/edit/${pathSlash}${name}`, { replace: true });
  }

  async function newFolderIn(path: string, list: DirectoryEntry[], folderName?: string) {
    const name = folderName || newName(list, "New Folder");
    const pathSlash = path ? `${path}/` : "";

    const res = await fetch("/api/dir/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: pathSlash + name }),
    });
    if (res.ok) {
      navigate(`/dir/${pathSlash}${name}`);
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
              <div className="new-workspace">
                {isCreatingWorkspace && (
                  // @ts-ignore
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      newWorkspaceIn(
                        path || "",
                        list.data,
                        (e.target as HTMLFormElement).workspaceName.value.trim(),
                      );
                    }}
                  >
                    <input
                      type="text"
                      name="workspaceName"
                      defaultValue={newName(list.data)}
                      placeholder={newName(list.data)}
                    />
                  </form>
                )}
                <button type="button" onClick={() => setIsCreatingWorkspace(true)}>
                  <FolderPlus /> New workspace
                </button>
              </div>

              <div className="new-folder">
                {isCreatingDir && (
                  // @ts-ignore
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      newFolderIn(
                        path || "",
                        list.data,
                        (e.target as HTMLFormElement).folderName.value.trim(),
                      );
                    }}
                  >
                    <input
                      type="text"
                      name="folderName"
                      defaultValue={newName(list.data)}
                      placeholder={newName(list.data)}
                    />
                  </form>
                )}
                <button type="button" onClick={() => setIsCreatingDir(true)}>
                  <FolderPlus /> New folder
                </button>
              </div>
            </div>

            {path && (
              <div className="breadcrumbs">
                <Link to="/dir/">
                  <Home />
                </Link>{" "}
                <span className="current-folder">{path}</span>
              </div>
            )}

            {list.data.map((item: DirectoryEntry) => (
              <div key={item.name} className="entry">
                <Link key={link(item)} to={link(item)}>
                  {item.type === "directory" ? <Folder /> : <File />}
                  {shortName(item)}
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
            ))}
          </>
        )}
      </div>{" "}
    </div>
  );
}
