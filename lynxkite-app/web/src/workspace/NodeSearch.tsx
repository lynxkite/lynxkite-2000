import Fuse from "fuse.js";
import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
// @ts-expect-error
import ArrowLeftIcon from "~icons/tabler/arrow-left.jsx";
// @ts-expect-error
import FolderIcon from "~icons/tabler/folder.jsx";

export type OpsOp = {
  name: string;
  id: string;
  categories: string[];
  type: string;
  position: { x: number; y: number };
  params: { name: string; default: any }[];
};
export type Catalog = { [op: string]: OpsOp };
export type Catalogs = { [env: string]: Catalog };
type SearchResult = {
  name: string;
  item: OpsOp | Category;
  parentPath: string[];
  score: number;
  isCategory?: boolean;
  isBack?: boolean;
};

export type Category = {
  name: string;
  ops: OpsOp[]; // Operations at this level.
  categories: Category[]; // Subcategories.
};

function sortHierarchy(level: Category): Category {
  const sortedOps = [...level.ops];
  sortedOps.sort((a, b) => a.name.localeCompare(b.name));
  const sortedCategories = level.categories.map(sortHierarchy);
  sortedCategories.sort((a, b) => a.name.localeCompare(b.name));
  return { name: level.name, ops: sortedOps, categories: sortedCategories };
}

export function buildCategoryHierarchy(boxes: Catalog): Category {
  const hierarchy: Category = { name: "<<root>>", ops: [], categories: [] };
  for (const op of Object.values(boxes)) {
    const categories = op.categories;
    let currentLevel = hierarchy;
    for (const category of categories) {
      const existingCategory = currentLevel.categories.find((cat) => cat.name === category);
      if (!existingCategory) {
        const newCategory: Category = { name: category, ops: [], categories: [] };
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
    const fuse = new Fuse([...level.ops, ...level.categories], {
      keys: ["name"],
      threshold: 0.4, // Balanced fuzziness for typos like "Dijkstra" → "Dikstra"
      includeScore: true,
    });
    const fuzzyResults = fuse.search(searchTerm);
    const opsFromThisLevel = fuzzyResults.map((result) => ({
      name: result.item.name,
      item: result.item,
      isCategory: "ops" in result.item,
      parentPath: [...path],
      score: result.score ?? 0,
    }));
    const opsFromCategories = level.categories.flatMap((cat) =>
      searchAllOperations(cat, [...path, cat.name]),
    );
    return [...opsFromThisLevel, ...opsFromCategories];
  }

  const results = searchAllOperations(currentLevel);
  results.sort((a, b) => a.score - b.score);
  return results;
}

export default function NodeSearch(props: {
  categoryHierarchy: Category;
  onCancel: any;
  onAdd: (op: OpsOp) => void;
  pos: { x: number; y: number };
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
    if (searchInputRef.current) {
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
    props.onAdd(op);
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
    <div
      className="node-search"
      style={{ top: props.pos.y, left: props.pos.x }}
      onMouseDown={(e) => e.preventDefault()}
    >
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
            {result.parentPath.length ? (
              <span className="search-result-path">({result.parentPath.join(" › ")})</span>
            ) : null}
          </button>
        ))}
      </div>
    </div>
  );
}
