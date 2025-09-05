import Fuse from "fuse.js"; // Added back fuzzy search for better user experience with typos
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

// NEW: Better type structure instead of Record<string, any>
// This provides type safety and eliminates the need for __operations magic strings
export type Category = {
  ops: OpsOp[]; // Operations at this level
  categories: Record<string, Category>; // Subcategories
};

export type CategoryHierarchy = Category;

// NEW: Extracted hierarchy building logic for better performance
// This can now be pre-computed in the parent component instead of on every render
export function buildCategoryHierarchy(boxes: Catalog): CategoryHierarchy {
  const hierarchy: CategoryHierarchy = { ops: [], categories: {} };

  // CHANGED: Using for...of loop instead of forEach for better performance
  for (const op of Object.values(boxes)) {
    const categories = op.categories;

    if (!categories || categories.length === 0) {
      // NEW: Always initialize ops array, no need for conditional checks
      if (!hierarchy.ops.find((existing: OpsOp) => existing.id === op.id)) {
        hierarchy.ops.push(op);
      }
      continue;
    }

    let currentLevel = hierarchy;

    // CHANGED: Using regular for loop instead of forEach for better performance
    for (let i = 0; i < categories.length; i++) {
      const category = categories[i];
      if (!currentLevel.categories[category]) {
        // NEW: Always initialize both ops and categories arrays
        currentLevel.categories[category] = { ops: [], categories: {} };
      }
      if (i === categories.length - 1) {
        // NEW: Direct access to ops array, no __operations magic string
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

  // NEW: Pre-sort all operations alphabetically to avoid runtime sorting
  // CHANGED: Using function declaration instead of arrow function for better readability
  function sortHierarchy(level: CategoryHierarchy): CategoryHierarchy {
    const sortedOps = [...level.ops].sort((a, b) => a.name.localeCompare(b.name));
    const sortedCategories: Record<string, Category> = {};

    // CHANGED: Using for...of loop instead of forEach for better performance
    for (const key of Object.keys(level.categories).sort((a, b) => a.localeCompare(b))) {
      sortedCategories[key] = sortHierarchy(level.categories[key]);
    }

    return { ops: sortedOps, categories: sortedCategories };
  }

  return sortHierarchy(hierarchy);
}

// CHANGED: Updated props interface to accept pre-computed hierarchy instead of raw boxes
export default function NodeSearch(props: {
  categoryHierarchy: CategoryHierarchy; // NEW: Pre-computed hierarchy for better performance
  onCancel: any;
  onAdd: any;
  pos: { x: number; y: number };
}) {
  const [categoryPath, setCategoryPath] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  // REMOVED: hoveredItem state - now using CSS :hover for better performance
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

  // NEW: Robust path traversal with error handling
  const getCurrentLevel = (): CategoryHierarchy => {
    try {
      // CHANGED: Using reduce instead of forEach for functional approach
      return categoryPath.reduce((currentLevel, category) => {
        if (!currentLevel?.categories[category]) {
          throw new Error("Category not found");
        }
        // NEW: Direct access to categories property instead of magic string access
        return currentLevel.categories[category];
      }, props.categoryHierarchy);
    } catch {
      // NEW: Fallback to root level if path becomes invalid
      return props.categoryHierarchy;
    }
  };

  // NEW: Proactive validation to reset invalid paths
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

    // NEW: Additional safety check to prevent crashes
    if (!currentLevel || !currentLevel.categories || !currentLevel.ops) {
      return [];
    }

    if (!searchTerm) {
      // CHANGED: Using map instead of forEach + push for functional approach
      const categoryMatches = Object.keys(currentLevel.categories)
        .sort((a, b) => a.localeCompare(b))
        .map((key) => ({ item: key, isCategory: true as const }));

      // NEW: Direct access to ops array, no conditional checks needed
      const opMatches = currentLevel.ops.map((op: OpsOp) => ({
        item: op.name,
        op,
      }));

      // CHANGED: Using spread operator instead of push operations
      return [...categoryMatches, ...opMatches];
    }
    // CHANGED: Using function declaration instead of arrow function
    function searchAllOperations(
      level: CategoryHierarchy,
      path: string[] = [],
    ): { item: string; op: OpsOp; category?: string }[] {
      // NEW: Safety check for each level during recursion
      if (!level || !level.categories || !level.ops) {
        return [];
      }

      // NEW: Fuzzy search using Fuse.js for better user experience
      const fuse = new Fuse(level.ops, {
        keys: ["name"],
        threshold: 0.4, // Balanced fuzziness for typos like "Dijkstra" → "Dikstra"
        includeScore: true,
      });

      const fuzzyResults = fuse.search(searchTerm);
      // CHANGED: Using map instead of filter + map for better performance
      const opsFromThisLevel = fuzzyResults.map((result) => ({
        item: result.item.name,
        op: result.item,
        category: path.length > 0 ? path.join(" > ") : undefined,
      }));

      // CHANGED: Using flatMap for cleaner recursive collection
      const opsFromCategories = Object.keys(level.categories)
        .sort((a, b) => a.localeCompare(b))
        .flatMap((key) => searchAllOperations(level.categories[key], [...path, key]));

      // CHANGED: Using spread operator instead of concatenation
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

  // REMOVED: setTimeout delay - now using onMouseDown to eliminate race conditions
  const handleBlur = () => {
    if (document.activeElement?.closest(".node-search")) return;
    props.onCancel();
  };

  return (
    <div
      className="node-search"
      style={{ ...styles.container, top: props.pos.y, left: props.pos.x }}
    >
      {/* NEW: Moved hover styles to CSS for better performance */}
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

      {/* NEW: Added breadcrumb navigation for better UX */}
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
        {/* CHANGED: Using map instead of forEach for better performance */}
        {results.map((result, index) => (
          <div
            key={result.category ? `${result.category}-${result.item}` : result.item}
            className={`node-search-item ${result.isCategory ? "node-search-item-category" : ""} ${index === selectedIndex ? "node-search-item-selected" : ""}`}
            /* NEW: Using onMouseDown instead of onClick to prevent blur/select race condition */
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
            {/* NEW: Added icons for better visual distinction */}
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

// NEW: Using CSS-in-JS object for better maintainability
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
  // NEW: Added styled back button for better navigation UX
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
  // NEW: Using flexbox for better layout control
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
};
