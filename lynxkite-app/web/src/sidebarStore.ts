// Browser-local sidebar data: favorites, recently opened files and a name index for search.
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

// Search runs over an index of all names, built by walking the directory listing API.
const MAX_INDEXED = 5000;
let indexCache: { time: number; entries: DirectoryEntry[] } | null = null;
let indexPending: Promise<DirectoryEntry[]> | null = null;

async function buildIndex(): Promise<DirectoryEntry[]> {
  const entries: DirectoryEntry[] = [];
  let level = [""];
  while (level.length && entries.length < MAX_INDEXED) {
    const lists = await Promise.all(
      level.map((dir) =>
        apiJson<DirectoryEntry[]>(`/api/dir/list?path=${encodeURIComponent(dir)}`).catch(
          () => [] as DirectoryEntry[],
        ),
      ),
    );
    level = [];
    for (const item of lists.flat()) {
      if (shortName(item)?.startsWith("__")) continue;
      entries.push(item);
      if (item.type === "directory") level.push(item.name);
    }
  }
  return entries.slice(0, MAX_INDEXED);
}

export async function getSearchIndex(): Promise<DirectoryEntry[]> {
  if (indexCache && Date.now() - indexCache.time < 30_000) return indexCache.entries;
  indexPending ??= buildIndex()
    .then((entries) => {
      indexCache = { time: Date.now(), entries };
      return entries;
    })
    .finally(() => {
      indexPending = null;
    });
  return indexPending;
}
