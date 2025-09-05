import Fuse from "fuse.js";
import type React from "react";
import { useEffect, useRef, useState } from "react";
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
  ops: OpsOp[]; // Operations at this level
  categories: Record<string, Category>; // Subcategories
};

export type CategoryHierarchy = Category;

export function buildCategoryHierarchy(boxes: Catalog): CategoryHierarchy {
  const hierarchy: CategoryHierarchy = { ops: [], categories: {} };

  for (const op of Object.values(boxes)) {
    const categories = op.categories;
    if (!categories || categories.length === 0) {
      if (!hierarchy.ops.find((existing: OpsOp) => existing.id === op.id)) {
        hierarchy.ops.push(op);
      }
      continue;
    }

    let currentLevel = hierarchy;
    for (let i = 0; i < categories.length; i++) {
      const category = categories[i];
      if (!currentLevel.categories[category]) {
        currentLevel.categories[category] = { ops: [], categories: {} };
      }
      if (i === categories.length - 1) {
        if (
          !currentLevel.categories[category].ops.find((existing: OpsOp) => existing.id === op.id)
        ) {
          currentLevel.categories[category].ops.push(op);
        }
      } else {
        currentLevel = currentLevel.categories[category];
      }
    }
  }

  function sortHierarchy(level: CategoryHierarchy): CategoryHierarchy {
    const sortedOps = [...level.ops].sort((a, b) => a.name.localeCompare(b.name));
    const sortedCategories: Record<string, Category> = {};
    for (const key of Object.keys(level.categories).sort((a, b) => a.localeCompare(b))) {
      sortedCategories[key] = sortHierarchy(level.categories[key]);
    }
    return { ops: sortedOps, categories: sortedCategories };
  }

  return sortHierarchy(hierarchy);
}

export default function NodeSearch(props: {
  categoryHierarchy: CategoryHierarchy;
  onCancel: any;
  onAdd: any;
  pos: { x: number; y: number };
}) {
  const [categoryPath, setCategoryPath] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, []);

  const handleCategoryClick = (category: string) => {
    setCategoryPath([...categoryPath, category]);
    setSearchTerm("");
    setSelectedIndex(0);
  };

  const handleBackClick = () => {
    if (categoryPath.length > 0) {
      setCategoryPath(categoryPath.slice(0, -1));
      setSelectedIndex(0);
    }
  };

  const handleItemClick = (op: OpsOp) => {
    props.onAdd(op);
  };

  const getCurrentLevel = (): CategoryHierarchy => {
    try {
      return categoryPath.reduce((currentLevel, category) => {
        if (!currentLevel?.categories[category]) {
          throw new Error("Category not found");
        }
        return currentLevel.categories[category];
      }, props.categoryHierarchy);
    } catch {
      return props.categoryHierarchy;
    }
  };

  useEffect(() => {
    const isValidPath = categoryPath.every((_, index) => {
      const partialPath = categoryPath.slice(0, index + 1);
      const level = partialPath.reduce((currentLevel, cat) => {
        return currentLevel?.categories[cat];
      }, props.categoryHierarchy);
      return level !== undefined;
    });

    if (!isValidPath && categoryPath.length > 0) {
      setCategoryPath([]);
    }
  }, [categoryPath, props.categoryHierarchy]);

  const filteredList = (): {
    item: string;
    op?: OpsOp;
    category?: string;
    isCategory?: boolean;
  }[] => {
    const currentLevel = getCurrentLevel();
    if (!currentLevel || !currentLevel.categories || !currentLevel.ops) {
      return [];
    }

    if (!searchTerm) {
      const categoryMatches = Object.keys(currentLevel.categories)
        .sort((a, b) => a.localeCompare(b))
        .map((key) => ({ item: key, isCategory: true as const }));
      const opMatches = currentLevel.ops.map((op: OpsOp) => ({
        item: op.name,
        op,
      }));
      return [...categoryMatches, ...opMatches];
    }
    function searchAllOperations(
      level: CategoryHierarchy,
      path: string[] = [],
    ): { item: string; op: OpsOp; category?: string }[] {
      if (!level || !level.categories || !level.ops) {
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
      const opsFromCategories = Object.keys(level.categories)
        .sort((a, b) => a.localeCompare(b))
        .flatMap((key) => searchAllOperations(level.categories[key], [...path, key]));
      return [...opsFromThisLevel, ...opsFromCategories];
    }

    return searchAllOperations(props.categoryHierarchy);
  };

  const results = filteredList();

  useEffect(() => {
    setSelectedIndex(Math.max(0, Math.min(selectedIndex, results.length - 1)));
  }, [results.length, selectedIndex]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
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
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    setSelectedIndex(0);
  };
  const handleBlur = () => {
    if (document.activeElement?.closest(".node-search")) return;
    props.onCancel();
  };

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
