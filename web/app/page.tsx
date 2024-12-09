// The directory browser.
import Image from "next/image";
import Link from 'next/link';
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

export default async function Home({
  searchParams,
}: {
  searchParams?: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  ;
  async function fetchList(path) {
    const encodedPath = encodeURIComponent(path || '');
    const res = await fetch(`http://localhost:8000/api/dir/list?path=${encodedPath}`);
    const j = await res.json();
    return j;
  }
  const path = (await searchParams)?.path || '';
  const list = await fetchList(path);
  function link(item) {
    if (item.type === 'directory') {
      return `/dir?path=${item.name}`;
    } else {
      return `/workspace?path=${item.name}`;
    }
  }
  function shortName(item) {
    return item.name.split('/').pop();
  }
  function newName(list) {
    let i = 0;
    while (true) {
      const name = `Untitled${i ? ` ${i}` : ''}`;
      if (!list.find(item => item.name === name)) {
        return name;
      }
      i++;
    }
  }
  function newWorkspaceIn(path, list) {
    const pathSlash = path ? `${path}/` : '';
    return `/workspace?path=${pathSlash}${newName(list)}`;
  }
  async function newFolderIn(path, list) {
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
    <div className="directory-page">
      <div className="logo">
        <Link href="https://lynxkite.com/"><img src='/logo.png' className="logo-image" alt="LynxKite logo" /></Link>
        <div className="tagline">The Complete Graph Data Science Platform</div>
      </div>
      <div className="entry-list">
        <div className="actions">
          <a href={newWorkspaceIn(path, list)}><FilePlus /> New workspace</a>
          <a href="" onClick={() => newFolderIn(path, list)}><FolderPlus /> New folder</a>
        </div>
        {path && <div className="breadcrumbs"><a href="#dir"><Home /></a> {path} </div>}
        {list.map(item =>
          <a className="entry" href={link(item)}>
            {item.type === 'directory' ? <Folder /> : <File />}
            {shortName(item)}
          </a>
        )}
      </div>
    </div>
  );
}
