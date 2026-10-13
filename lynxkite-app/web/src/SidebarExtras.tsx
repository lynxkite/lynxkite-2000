// Search, favorites and recent files for the sidebar.
import { useRef, useState } from "react";
import { Link } from "react-router";
import File from "~icons/tabler/file";
import Folder from "~icons/tabler/folder";
import History from "~icons/tabler/history";
import LayoutGrid from "~icons/tabler/layout-grid";
import SearchIcon from "~icons/tabler/search";
import Star from "~icons/tabler/star";
import type { DirectoryEntry } from "./apiTypes.ts";
import { entryLink, getSearchIndex, shortName, useFavorites, useRecent } from "./sidebarStore.ts";

const MAX_RESULTS = 8;

function EntryIcon(props: { type: string }) {
  if (props.type === "directory") return <Folder />;
  if (props.type === "workspace") return <LayoutGrid />;
  return <File />;
}

function parentPath(item: DirectoryEntry) {
  return item.name.split("/").slice(0, -1).join("/");
}

export function SidebarSearch(props: { open: boolean; onExpand: () => void }) {
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState<DirectoryEntry[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const q = query.trim().toLowerCase();
  const results = q
    ? index.filter((item) => item.name.toLowerCase().includes(q)).slice(0, MAX_RESULTS)
    : [];

  function expandAndFocus() {
    if (props.open) return;
    props.onExpand();
    // Focus once the sidebar has started opening.
    setTimeout(() => inputRef.current?.focus(), 50);
  }

  return (
    <div className="sidebar-search-section">
      {/* biome-ignore lint/a11y/useKeyWithClickEvents: the input inside handles the keyboard */}
      {/* biome-ignore lint/a11y/noStaticElementInteractions: expands the collapsed sidebar */}
      <div className="sidebar-search" title="Search" onClick={expandAndFocus}>
        <SearchIcon />
        <input
          ref={inputRef}
          className="sidebar-label"
          type="search"
          placeholder="Search files..."
          value={query}
          tabIndex={props.open ? 0 : -1}
          onFocus={() => getSearchIndex().then(setIndex)}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") setQuery("");
          }}
        />
      </div>
      {props.open && q && (
        <div className="sidebar-search-results">
          {results.map((item) => (
            <Link
              key={item.name}
              to={entryLink(item)}
              className="sidebar-list-item"
              title={item.name}
              onClick={() => setQuery("")}
            >
              <EntryIcon type={item.type} />
              <span className="sidebar-list-text">
                {shortName(item)}
                {parentPath(item) && <small>{parentPath(item)}</small>}
              </span>
            </Link>
          ))}
          {results.length === 0 && <div className="sidebar-list-empty">No matches</div>}
        </div>
      )}
    </div>
  );
}

function SidebarList(props: {
  title: string;
  icon: React.ReactNode;
  items: DirectoryEntry[];
  emptyText: string;
}) {
  return (
    <div className="sidebar-section">
      <div className="sidebar-section-title">
        {props.icon}
        {props.title}
      </div>
      {props.items.length === 0 && <div className="sidebar-list-empty">{props.emptyText}</div>}
      {props.items.map((item) => (
        <Link key={item.name} to={entryLink(item)} className="sidebar-list-item" title={item.name}>
          <EntryIcon type={item.type} />
          <span className="sidebar-list-text">
            {shortName(item)}
            {parentPath(item) && <small>{parentPath(item)}</small>}
          </span>
        </Link>
      ))}
    </div>
  );
}

export function SidebarFavorites() {
  return (
    <SidebarList
      title="Favorites"
      icon={<Star />}
      items={useFavorites()}
      emptyText="Use the star icon on any file or folder to pin it here."
    />
  );
}

export function SidebarRecent() {
  return (
    <SidebarList
      title="Recent"
      icon={<History />}
      items={useRecent()}
      emptyText="Files you open will be listed here."
    />
  );
}
