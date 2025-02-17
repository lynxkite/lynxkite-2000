// The directory browser.
import { useParams, useNavigate } from "react-router";
import { useState } from "react";
import useSWR from 'swr'


import logo from './assets/logo.png';
// @ts-ignore
import Home from '~icons/tabler/home'
// @ts-ignore
import Folder from '~icons/tabler/folder'
// @ts-ignore
import FolderPlus from '~icons/tabler/folder-plus'
// @ts-ignore
import File from '~icons/tabler/file'
// @ts-ignore
import FilePlus from '~icons/tabler/file-plus'
// @ts-ignore
import Trash from '~icons/tabler/trash';



const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function () {
  const { path } = useParams();
  const encodedPath = encodeURIComponent(path || '');
  let list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher);
  const navigate = useNavigate();
  const [isCreatingDir, setIsCreatingDir] = useState(false);
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);
  
  
  function link(item: any) {
    if (item.type === 'directory') {
      return `/dir/${item.name}`;
    } else {
      return `/edit/${item.name}`;
    }
  }
  
  function shortName(item: any) {
    return item.name.split('/').pop();
  }

  function newName(list: any[], baseName: string = "Untitled") {
    let i = 0;
    while (true) {
      const name = `${baseName}${i ? ` ${i}` : ''}`;
      if (!list.find(item => item.name === name)) {
        return name;
      }
      i++;
    }
  }
  
  function newWorkspaceIn(path: string, list: any[], workspaceName?: string) {
    const pathSlash = path ? `${path}/` : "";
    const name = workspaceName || newName(list);
    navigate(`/edit/${pathSlash}${name}`, {replace: true});
  }
  

  async function newFolderIn(path: string, list: any[], folderName?: string) {
    const name = folderName || newName(list, "New Folder");
    const pathSlash = path ? `${path}/` : "";
  
    const res = await fetch(`/api/dir/mkdir`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: pathSlash + name }),
    });
    list = await res.json();
    if (res.ok) {
      navigate(`/dir/${pathSlash}${name}`);
    } else {
      alert("Failed to create folder.");
    }
  }
  
  async function deleteItem(item: any) {
    if (!window.confirm(`Are you sure you want to delete "${item.name}"?`)) return;
    const pathSlash = path ? `${path}/` : "";

    const apiPath = item.type === "directory" ? `/api/dir/delete`: `/api/delete`;
    const res = await fetch(apiPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: pathSlash + item.name }),
    });
    list = await res.json();
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
          <div className="loading spinner-border" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        )}

        {list.data && (
          <>
            <div className="actions">
                <div className="new-workspace">
                  {isCreatingWorkspace && 
                    // @ts-ignore
                    <form onSubmit={(e) => {e.preventDefault(); newWorkspaceIn(path || "", list.data, e.target.workspaceName.value.trim())}}>
                      <input
                        type="text"
                        name="workspaceName"
                        defaultValue={newName(list.data)}
                        placeholder={newName(list.data)}
                      />
                    </form>
                  }
                  <button type="button" onClick={() => setIsCreatingWorkspace(true)}>
                    <FolderPlus /> New workspace
                  </button>
                </div>

                <div className="new-folder">
                  {isCreatingDir && 
                    // @ts-ignore
                    <form onSubmit={(e) =>{e.preventDefault(); newFolderIn(path || "", list.data, e.target.folderName.value.trim())}}>
                      <input
                        type="text"
                        name="folderName"
                        defaultValue={newName(list.data)}
                        placeholder={newName(list.data)}
                      />
                    </form>
                  }
                  <button type="button" onClick={() => setIsCreatingDir(true)}>
                    <FolderPlus /> New folder
                  </button>
                </div>
            </div>

            {path && (
              <div className="breadcrumbs">
                <a href="/dir/">
                  <Home />
                </a>{" "}
                <span className="current-folder">{path}</span>
              </div>
            )}

            {list.data.map((item: any) => (
              <div key={item.name} className="entry">
                <a key={link(item)} className="entry" href={link(item)}>
                  {item.type === 'directory' ? <Folder /> : <File />}
                  {shortName(item)}
                </a>
                <button onClick={() => { deleteItem(item) }}>
                  <Trash />
                </button>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
