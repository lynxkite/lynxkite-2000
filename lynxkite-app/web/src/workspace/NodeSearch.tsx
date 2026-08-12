import Fuse from "fuse.js";
import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import ArrowLeftIcon from "~icons/tabler/arrow-left.jsx";
import FolderIcon from "~icons/tabler/folder.jsx";
import type { Op as OpsOp } from "../apiTypes.ts";

export type Catalog = { [op: string]: OpsOp };
export type Catalogs = { [env: string]: Catalog };
type SearchResult = {
  name: string;
  item: OpsOp | Category;
  parentPath: string[];
  score: number;
  description?: string;
  matchedInName?: boolean;
  matchedInDescription?: boolean;
  isCategory?: boolean;
  isBack?: boolean;
};

type SearchableItem = {
  name: string;
  item: OpsOp | Category;
  parentPath: string[];
  isCategory: boolean;
  description: string;
};

export type Category = {
  name: string;
  opsContained: number;
  ops: OpsOp[]; // Operations at this level.
  categories: Category[]; // Subcategories.
};

function sortHierarchy(level: Category): Category {
  const sortedOps = [...level.ops];
  sortedOps.sort((a, b) => a.name.localeCompare(b.name));
  const sortedCategories = level.categories.map(sortHierarchy);
  sortedCategories.sort((a, b) => a.name.localeCompare(b.name));
  const opsContained =
    sortedOps.length + sortedCategories.reduce((sum, cat) => sum + cat.opsContained, 0);
  return { name: level.name, ops: sortedOps, categories: sortedCategories, opsContained };
}

export function buildCategoryHierarchy(boxes: Catalog): Category {
  const hierarchy: Category = { name: "<<root>>", ops: [], categories: [], opsContained: 0 };
  for (const op of Object.values(boxes || {})) {
    const categories = op.categories;
    let currentLevel = hierarchy;
    for (const category of categories) {
      const existingCategory = currentLevel.categories.find((cat) => cat.name === category);
      if (!existingCategory) {
        const newCategory: Category = {
          name: category,
          ops: [],
          categories: [],
          opsContained: 0,
        };
        currentLevel.categories.push(newCategory);
        currentLevel = newCategory;
      } else {
        currentLevel = existingCategory;
      }
    }
    currentLevel.ops.push(op);
  }
  return sortHierarchy(hierarchy);
}

function categoryByPath(rootCategory: Category, categoryPath: string[]): Category | undefined {
  let currentLevel: Category | undefined = rootCategory;
  for (const cat of categoryPath) {
    currentLevel = currentLevel?.categories.find((c) => c.name === cat);
  }
  return currentLevel;
}

function docToSearchText(doc: OpsOp["doc"]): string {
  if (!doc) {
    return "";
  }
  return (
    doc.map?.((section: any) => (section.kind === "text" ? section.value : "")).join("\n") ??
    String(doc)
  );
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function queryTerms(searchTerm: string): string[] {
  return searchTerm
    .trim()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean);
}

function hasWordStartMatch(text: string, term: string): boolean {
  if (!text || !term) {
    return false;
  }
  const matcher = new RegExp(`\\b${escapeRegExp(term)}`, "i");
  return matcher.test(text);
}

function matchesQuery(text: string, searchTerm: string): boolean {
  const terms = queryTerms(searchTerm);
  if (!terms.length) {
    return false;
  }
  return terms.every((term) => hasWordStartMatch(text, term));
}

function descriptionPreview(text: string, searchTerm: string): string {
  if (!text) {
    return "";
  }
  const terms = queryTerms(searchTerm);
  if (!terms.length) {
    return text;
  }

  let firstMatchIndex = -1;
  for (const term of terms) {
    const matcher = new RegExp(`\\b${escapeRegExp(term)}`, "i");
    const match = matcher.exec(text);
    if (match?.index !== undefined) {
      if (firstMatchIndex === -1 || match.index < firstMatchIndex) {
        firstMatchIndex = match.index;
      }
    }
  }

  if (firstMatchIndex === -1) {
    return text;
  }

  const windowSize = 160;
  const contextBefore = 45;
  const start = Math.max(0, firstMatchIndex - contextBefore);
  const end = Math.min(text.length, start + windowSize);
  const prefix = start > 0 ? "..." : "";
  const suffix = end < text.length ? "..." : "";
  return `${prefix}${text.slice(start, end).trim()}${suffix}`;
}

function highlightMatches(text: string, searchTerm: string): React.ReactNode {
  if (!searchTerm || !text) {
    return text;
  }
  const trimmedTerm = searchTerm.trim();
  const escapedTerm = escapeRegExp(trimmedTerm);
  if (!escapedTerm) {
    return text;
  }
  const matcher = new RegExp(`\\b${escapedTerm}`, "ig");
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let matchIndex = 0;
  for (const match of text.matchAll(matcher)) {
    const start = match.index ?? 0;
    const matchedText = match[0] ?? "";
    if (start > lastIndex) {
      nodes.push(<span key={`text-${matchIndex}`}>{text.slice(lastIndex, start)}</span>);
    }
    nodes.push(
      <mark key={`mark-${matchIndex}`} className="search-result-highlight">
        {matchedText}
      </mark>,
    );
    lastIndex = start + matchedText.length;
    matchIndex += 1;
  }
  if (lastIndex < text.length) {
    nodes.push(<span key={`tail-${matchIndex}`}>{text.slice(lastIndex)}</span>);
  }
  return nodes.length ? nodes : text;
}

function rankSearchResult(result: SearchResult, query: string): number {
  const lowerName = result.name.toLowerCase();
  const lowerQuery = query.toLowerCase();
  if (lowerName === lowerQuery) {
    return 0;
  }
  if (lowerName.startsWith(lowerQuery)) {
    return 1;
  }
  if (result.matchedInName) {
    return 2;
  }
  if (result.matchedInDescription) {
    return 3;
  }
  return 4;
}

function filteredList(currentLevel: Category | undefined, searchTerm: string): SearchResult[] {
  if (!currentLevel) {
    return [];
  }

  if (!searchTerm) {
    const categoryMatches = currentLevel.categories.map((cat: Category) => ({
      name: cat.name,
      item: cat,
      parentPath: [],
      isCategory: true as const,
      score: 0,
    }));
    const opMatches = currentLevel.ops.map((op: OpsOp) => ({
      name: op.name,
      item: op,
      parentPath: [],
      score: 0,
    }));
    return [...categoryMatches, ...opMatches];
  }
  function searchAllOperations(level: Category, path: string[] = []): SearchResult[] {
    if (!level) {
      return [];
    }
    const searchableItems: SearchableItem[] = [
      ...level.ops.map(
        (op): SearchableItem => ({
          name: op.name,
          item: op,
          parentPath: [...path],
          isCategory: false,
          description: docToSearchText(op.doc),
        }),
      ),
      ...level.categories.map(
        (category): SearchableItem => ({
          name: category.name,
          item: category,
          parentPath: [...path],
          isCategory: true,
          description: "",
        }),
      ),
    ];
    const fuse = new Fuse(searchableItems, {
      keys: [
        { name: "name", weight: 0.8 },
        { name: "description", weight: 0.2 },
      ],
      threshold: 0.4, // Balanced fuzziness for typos like "Dijkstra" → "Dikstra"
      includeScore: true,
      includeMatches: true,
    });

    const strictMatches = searchableItems
      .filter((item) => {
        if (item.isCategory) {
          return matchesQuery(item.name, searchTerm);
        }
        return matchesQuery(item.name, searchTerm) || matchesQuery(item.description, searchTerm);
      })
      .map(
        (item): SearchResult => ({
          name: item.name,
          item: item.item,
          isCategory: item.isCategory,
          parentPath: item.parentPath,
          score: 0,
          description: item.description,
          matchedInName: matchesQuery(item.name, searchTerm),
          matchedInDescription: !item.isCategory && matchesQuery(item.description, searchTerm),
        }),
      );

    const fuzzyResults = fuse.search(searchTerm);
    const fuzzyMatches = fuzzyResults
      .map((result): SearchResult => {
        const matchedInName =
          result.matches?.some((match) =>
            Array.isArray(match.key) ? match.key.includes("name") : match.key === "name",
          ) ?? false;
        const matchedInDescription =
          result.matches?.some((match) =>
            Array.isArray(match.key)
              ? match.key.includes("description")
              : match.key === "description",
          ) ?? false;

        return {
          name: result.item.name,
          item: result.item.item,
          isCategory: result.item.isCategory,
          parentPath: result.item.parentPath,
          score: result.score ?? 0,
          description: result.item.description,
          matchedInName,
          matchedInDescription,
        };
      })
      .filter((result) => {
        if (result.isCategory) {
          return matchesQuery(result.name, searchTerm);
        }
        return (
          matchesQuery(result.name, searchTerm) ||
          matchesQuery(result.description ?? "", searchTerm)
        );
      });

    const mergedByKey = new Map<string, SearchResult>();
    for (const result of [...strictMatches, ...fuzzyMatches]) {
      const key = `${result.parentPath.join("/")}::${result.name}::${result.isCategory ? "cat" : "op"}`;
      const previous = mergedByKey.get(key);
      if (!previous) {
        mergedByKey.set(key, result);
        continue;
      }
      mergedByKey.set(key, {
        ...previous,
        score: Math.min(previous.score, result.score),
        matchedInName: previous.matchedInName || result.matchedInName,
        matchedInDescription: previous.matchedInDescription || result.matchedInDescription,
      });
    }
    const opsFromThisLevel = [...mergedByKey.values()];
    const opsFromCategories = level.categories.flatMap((cat) =>
      searchAllOperations(cat, [...path, cat.name]),
    );
    return [...opsFromThisLevel, ...opsFromCategories];
  }

  const query = searchTerm.trim();
  const results = searchAllOperations(currentLevel);
  results.sort((a, b) => {
    const rankDiff = rankSearchResult(a, query) - rankSearchResult(b, query);
    if (rankDiff !== 0) {
      return rankDiff;
    }
    return a.score - b.score;
  });
  return results;
}

export default function NodeSearch(props: {
  categoryHierarchy: Category;
  onCancel: () => void;
  onClick: (op: OpsOp) => void;
  pos: { x: number; y: number };
}) {
  // Calculate adjusted position to keep the component visible
  function adjustPosition(pos: { x: number; y: number }) {
    const estimatedHeight = 300; // Approximate height of the search component
    const estimatedWidth = 400; // Approximate width of the search component
    const padding = 20; // Padding from screen edges
    let x = pos.x;
    let y = pos.y;

    // Adjust horizontal position if it would go off-screen
    if (x + estimatedWidth > window.innerWidth - padding) {
      x = window.innerWidth - estimatedWidth - padding;
    }
    if (x < padding) {
      x = padding;
    }

    // Adjust vertical position if it would go off-screen
    if (y + estimatedHeight > window.innerHeight - padding) {
      y = window.innerHeight - estimatedHeight - padding;
    }
    if (y < padding) {
      y = padding;
    }

    return { x, y };
  }
  const adjustedPos = adjustPosition(props.pos);

  return (
    <div
      className="node-search node-search-panel"
      style={{
        top: adjustedPos.y,
        left: adjustedPos.x,
        maxHeight: `calc(100vh - ${adjustedPos.y + 10}px)`,
      }}
      onMouseDown={(e) => e.preventDefault()}
    >
      <NodeSearchInternal {...props} autoFocus={true} />
    </div>
  );
}

export function NodeSearchInternal(props: {
  categoryHierarchy: Category;
  onCancel: any;
  onClick: (op: OpsOp) => void;
  autoFocus?: boolean;
}) {
  const [categoryPath, setCategoryPath] = useState<string[]>([]);
  const currentLevel = useMemo(
    () => categoryByPath(props.categoryHierarchy, categoryPath),
    [props.categoryHierarchy, categoryPath],
  );
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const searchInputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (searchInputRef.current && props.autoFocus) {
      searchInputRef.current.focus();
    }
  }, []);

  function handleCategoryClick(category: SearchResult) {
    setCategoryPath([...categoryPath, ...category.parentPath, category.name]);
    setSearchTerm("");
    setSelectedIndex(0);
  }

  function handleBackClick() {
    if (categoryPath.length > 0) {
      const last = categoryPath.at(-1);
      const newPath = categoryPath.slice(0, -1);
      setCategoryPath(newPath);
      const cat = categoryByPath(props.categoryHierarchy, newPath);
      const results = filteredList(cat, searchTerm);
      let index = results.findIndex((r) => r.isCategory && r.name === last);
      if (newPath.length > 0) index += 1; // Account for the "Back" button.
      setSelectedIndex(index);
    }
  }

  function handleItemClick(op: OpsOp) {
    props.onClick(op);
  }

  useEffect(() => {
    if (!currentLevel && categoryPath.length > 0) {
      setCategoryPath([]);
    }
  }, [currentLevel, categoryPath]);

  const results: SearchResult[] = [
    ...(categoryPath.length > 0
      ? [
          {
            name: "Back",
            item: {} as Category,
            isBack: true,
            parentPath: categoryPath,
            score: 0,
          },
        ]
      : []),
    ...filteredList(currentLevel, searchTerm),
  ];
  useEffect(() => {
    const index = Math.max(0, Math.min(selectedIndex, results.length - 1));
    setSelectedIndex(index);
    itemRefs.current[index]?.scrollIntoView({
      behavior: "instant",
      block: "nearest",
    });
  }, [results.length, selectedIndex]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex(Math.min(selectedIndex + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex(Math.max(selectedIndex - 1, 0));
    } else if (e.key === "Enter") {
      const selected = results[selectedIndex];
      if (selected) {
        handleClick(e, selected);
      }
    } else if (e.key === "Backspace" && searchTerm === "") {
      e.preventDefault();
      if (categoryPath.length > 0) {
        handleBackClick();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      props.onCancel();
    }
  }

  function handleSearchChange(e: React.ChangeEvent<HTMLInputElement>) {
    setSearchTerm(e.target.value);
    setSelectedIndex(0);
  }
  function handleBlur() {
    if (document.activeElement?.closest(".node-search")) return;
    props.onCancel();
  }
  function handleClick(e: { preventDefault: () => void }, result: SearchResult) {
    e.preventDefault();
    if (result.isCategory) {
      handleCategoryClick(result);
    } else if (result.isBack) {
      handleBackClick();
    } else {
      handleItemClick(result.item as OpsOp);
    }
  }
  return (
    <>
      <input
        ref={searchInputRef}
        placeholder="Search for box"
        value={searchTerm}
        onChange={handleSearchChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
      />

      <div className="matches">
        {results.map((result, index) => (
          <button
            key={result.parentPath ? `${result.parentPath.join("-")}-${result.name}` : result.name}
            className={`
              search-result
              ${result.isCategory || result.isBack ? "search-result-category" : "search-result-op"}
              ${index === selectedIndex ? "selected" : ""}`}
            ref={(el) => {
              itemRefs.current[index] = el;
            }}
            onMouseDown={(e) => handleClick(e, result)}
            onMouseEnter={() => setSelectedIndex(index)}
          >
            {result.isCategory ? <FolderIcon /> : result.isBack ? <ArrowLeftIcon /> : null}
            {result.name}{" "}
            {result.isCategory && (
              <span className="category-ops-contained">
                {(result.item as Category).opsContained}
              </span>
            )}
            {result.parentPath.length ? (
              <span className="search-result-path">({result.parentPath.join(" › ")})</span>
            ) : null}
            {!!searchTerm && !result.isCategory && !result.isBack && result.description ? (
              <span className="search-result-description">
                {highlightMatches(descriptionPreview(result.description, searchTerm), searchTerm)}
              </span>
            ) : null}
          </button>
        ))}
      </div>
    </>
  );
}
