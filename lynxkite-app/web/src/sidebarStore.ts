// Sidebar data: browser-local favorites and recently opened files, and the file search request.
import { useSyncExternalStore } from "react";
import type { DirectoryEntry } from "./apiTypes.ts";
import { apiJson } from "./common.ts";

const FAVORITES_KEY = "lynxkite-favorites";
const RECENT_KEY = "lynxkite-recent";
const MAX_RECENT = 5;

export function entryLink(item: DirectoryEntry) {
  const encodedName = item.name.split("/").map(encodeURIComponent).join("/");
  if (item.type === "directory") return `/dir/${encodedName}`;
  if (item.type === "workspace") return `/edit/${encodedName}`;
  return `/code/${encodedName}`;
}

export function shortName(item: DirectoryEntry) {
  return item.name
    .split("/")
    .pop()
    ?.replace(/[.]lynxkite[.]json$/, "");
}

function createListStore(key: string) {
  const listeners = new Set<() => void>();
  function read(): DirectoryEntry[] {
    try {
      const parsed = JSON.parse(localStorage.getItem(key) ?? "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  let snapshot = read();
  function notify() {
    for (const l of listeners) l();
  }
  window.addEventListener("storage", (e) => {
    if (e.key === key) {
      snapshot = read();
      notify();
    }
  });
  return {
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    get: () => snapshot,
    set(items: DirectoryEntry[]) {
      snapshot = items;
      try {
        localStorage.setItem(key, JSON.stringify(items));
      } catch {
        // Storage may be unavailable. The list then only lives until the page is reloaded.
      }
      notify();
    },
  };
}

const favorites = createListStore(FAVORITES_KEY);
const recent = createListStore(RECENT_KEY);

export function useFavorites() {
  return useSyncExternalStore(favorites.subscribe, favorites.get);
}

export function useRecent() {
  return useSyncExternalStore(recent.subscribe, recent.get);
}

export function toggleFavorite(item: DirectoryEntry) {
  const current = favorites.get();
  if (current.some((f) => f.name === item.name)) {
    favorites.set(current.filter((f) => f.name !== item.name));
  } else {
    favorites.set([...current, { name: item.name, type: item.type }]);
  }
}

function isSameOrInside(name: string, path: string) {
  return name === path || name.startsWith(`${path}/`);
}

/** Keeps the lists in sync after a file or folder is renamed. Items inside a folder move with it. */
export function renameEntry(oldPath: string, newPath: string) {
  for (const store of [favorites, recent]) {
    store.set(
      store
        .get()
        .map((item) =>
          isSameOrInside(item.name, oldPath)
            ? { ...item, name: newPath + item.name.slice(oldPath.length) }
            : item,
        ),
    );
  }
}

/** Drops a deleted file or folder, and anything inside it, from the lists. */
export function removeEntry(path: string) {
  for (const store of [favorites, recent]) {
    store.set(store.get().filter((item) => !isSameOrInside(item.name, path)));
  }
}

export function recordVisit(pathname: string) {
  const match = pathname.match(/^[/](edit|code)[/](.+?)[/]?$/);
  if (!match) return;
  let name: string;
  try {
    name = decodeURIComponent(match[2]);
  } catch {
    return;
  }
  const entry = { name, type: match[1] === "edit" ? "workspace" : "file" };
  recent.set([entry, ...recent.get().filter((r) => r.name !== name)].slice(0, MAX_RECENT));
}

export function searchFiles(query: string): Promise<DirectoryEntry[]> {
  return apiJson<DirectoryEntry[]>(`/api/dir/search?q=${encodeURIComponent(query)}`);
}
