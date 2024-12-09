// The directory browser.
import { useParams } from "react-router";
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

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function () {
  const { path } = useParams();
  const encodedPath = encodeURIComponent(path || '');
  const list = useSWR(`/api/dir/list?path=${encodedPath}`, fetcher)
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
  function newName(list: any[]) {
    let i = 0;
    while (true) {
      const name = `Untitled${i ? ` ${i}` : ''}`;
      if (!list.find(item => item.name === name)) {
        return name;
      }
      i++;
    }
  }
  function newWorkspaceIn(path: string, list: any[]) {
    const pathSlash = path ? `${path}/` : '';
    return `/edit/${pathSlash}${newName(list)}`;
  }
  async function newFolderIn(path: string, list: any[]) {
    const pathSlash = path ? `${path}/` : '';
    const name = newName(list);
    const res = await fetch(`/api/dir/mkdir`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: pathSlash + name }),
    });
    list = await res.json();
  }

  return (
    <div className="directory">
      <div className="logo">
        <a href="https://lynxkite.com/"><img src={logo} className="logo-image" alt="LynxKite logo" /></a>
        <div className="tagline">The Complete Graph Data Science Platform</div>
      </div>
      <div className="entry-list">
        {list.error && <p className="error">{list.error.message}</p>}
        {list.isLoading &&
          <div className="loading spinner-border" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>}
        {list.data &&
          <>
            <div className="actions">
              <a href={newWorkspaceIn(path || "", list.data)}><FilePlus /> New workspace</a>
              <a href="" onClick={() => newFolderIn(path || "", list.data)}><FolderPlus /> New folder</a>
            </div>
            {path && <div className="breadcrumbs"><a href="/dir/"><Home /></a> {path} </div>}
            {list.data.map((item: any) =>
              <a key={link(item)} className="entry" href={link(item)}>
                {item.type === 'directory' ? <Folder /> : <File />}
                {shortName(item)}
              </a>
            )}
          </>
        }
      </div>
    </div>
  );
}
