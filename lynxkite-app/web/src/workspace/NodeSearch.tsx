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

export default function NodeSearch(props: {
  categoryHierarchy: Category;
  onCancel: any;
  onAdd: (op: OpsOp) => void;
  pos: { x: number; y: number };
}) {
  const [categoryPath, setCategoryPath] = useState<string[]>([]);
  const currentLevel = useMemo(() => {
    return categoryPath.reduce((currentLevel: Category | undefined, nextStep: string) => {
      return currentLevel?.categories.find((cat) => cat.name === nextStep);
    }, props.categoryHierarchy);
  }, [props.categoryHierarchy, categoryPath]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, []);

  function handleCategoryClick(category: string) {
    setCategoryPath([...categoryPath, category]);
    setSearchTerm("");
    setSelectedIndex(0);
  }

  function handleBackClick() {
    if (categoryPath.length > 0) {
      setCategoryPath(categoryPath.slice(0, -1));
      setSelectedIndex(0);
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

  function filteredList(): {
    item: string;
    op?: OpsOp;
    category?: string;
    isCategory?: boolean;
  }[] {
    if (!currentLevel) {
      return [];
    }

    if (!searchTerm) {
      const categoryMatches = currentLevel.categories.map((cat: Category) => ({
        item: cat.name,
        isCategory: true as const,
      }));
      const opMatches = currentLevel.ops.map((op: OpsOp) => ({
        item: op.name,
        op,
      }));
      return [...categoryMatches, ...opMatches];
    }
    function searchAllOperations(
      level: Category,
      path: string[] = [],
    ): { item: string; op: OpsOp; category?: string }[] {
      if (!level) {
        return [];
      }
      const fuse = new Fuse(level.ops, {
        keys: ["name"],
        threshold: 0.4, // Balanced fuzziness for typos like "Dijkstra" → "Dikstra"
        includeScore: true,
      });

      const fuzzyResults = fuse.search(searchTerm);
      const opsFromThisLevel = fuzzyResults.map((result) => ({
        item: result.item.name,
        op: result.item,
        category: path.length > 0 ? path.join(" > ") : undefined,
      }));
      const opsFromCategories = level.categories.flatMap((cat) =>
        searchAllOperations(cat, [...path, cat.name]),
      );
      return [...opsFromThisLevel, ...opsFromCategories];
    }

    return searchAllOperations(currentLevel);
  }

  const results = filteredList();

  useEffect(() => {
    setSelectedIndex(Math.max(0, Math.min(selectedIndex, results.length - 1)));
  }, [results.length, selectedIndex]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex(Math.min(selectedIndex + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex(Math.max(selectedIndex - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const selected = results[selectedIndex];
      if (selected) {
        if (selected.op) {
          handleItemClick(selected.op);
        } else if (selected.isCategory) {
          handleCategoryClick(selected.item);
        }
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      if (categoryPath.length > 0) {
        handleBackClick();
      } else {
        props.onCancel();
      }
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

  return (
    <div
      className="node-search"
      style={{ ...styles.container, top: props.pos.y, left: props.pos.x }}
    >
      <style>
        {`
          .node-search-item {
            padding: 6px;
            cursor: pointer;
            border-radius: 4px;
          }
          .node-search-item:hover {
            background-color: orange;
            color: black;
          }
          .node-search-item-category {
            font-weight: bold;
            color: #444;
            background-color: #f8f9fa;
          }
          .node-search-item-selected {
            background-color: rgba(0, 123, 255, 0.2);
            color: black;
          }
        `}
      </style>
      <input
        ref={searchInputRef}
        style={styles.search}
        placeholder="Search for box"
        value={searchTerm}
        onChange={handleSearchChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
      />

      {categoryPath.length > 0 && (
        <div
          style={styles.backButton}
          onMouseDown={(e) => {
            e.preventDefault();
            handleBackClick();
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <ArrowLeftIcon style={{ width: "14px", height: "14px" }} />
            Back
          </span>
        </div>
      )}

      <div style={styles.list}>
        {results.map((result, index) => (
          <div
            key={result.category ? `${result.category}-${result.item}` : result.item}
            className={`node-search-item ${result.isCategory ? "node-search-item-category" : ""} ${index === selectedIndex ? "node-search-item-selected" : ""}`}
            onMouseDown={(e) => {
              e.preventDefault();
              if (result.op) {
                handleItemClick(result.op);
              } else if (result.isCategory) {
                handleCategoryClick(result.item);
              }
            }}
            onMouseEnter={() => {
              setSelectedIndex(index);
            }}
          >
            {result.isCategory ? (
              <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <FolderIcon style={{ width: "16px", height: "16px", color: "#007bff" }} />
                {result.item}
              </span>
            ) : result.category ? (
              <span>
                {result.item}{" "}
                <span style={{ color: "#666", fontSize: "0.8em" }}>({result.category})</span>
              </span>
            ) : (
              result.item
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: "absolute",
    border: "1px solid #aaa",
    width: "300px",
    maxHeight: "400px",
    overflowY: "auto",
    overflowX: "hidden",
    padding: "8px",
    background: "#fff",
    fontFamily: "sans-serif",
    borderRadius: "6px",
    zIndex: 1000,
    resize: "none",
  },
  search: {
    marginBottom: "8px",
    width: "100%",
    padding: "6px",
    boxSizing: "border-box",
    borderRadius: "4px",
    border: "1px solid #ccc",
  },
  backButton: {
    marginBottom: "10px",
    cursor: "pointer",
    color: "#007bff",
    border: "1px solid #007bff",
    padding: "4px 8px",
    borderRadius: "4px",
    display: "inline-block",
    fontSize: "0.9em",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
};
