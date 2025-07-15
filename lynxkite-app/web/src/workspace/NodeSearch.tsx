import React, { useState, useMemo, useEffect, useRef } from "react";
// @ts-ignore
import FolderIcon from "~icons/tabler/folder.jsx";
// @ts-ignore
import ArrowLeftIcon from "~icons/tabler/arrow-left.jsx";

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

export default function NodeSearch(props: {
  boxes: Catalog;
  onCancel: any;
  onAdd: any;
  pos: { x: number; y: number };
}) {
  const [categoryPath, setCategoryPath] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, []);

  const categoryHierarchy: Record<string, any> = useMemo(() => {
    const hierarchy: Record<string, any> = {};

    Object.values(props.boxes).forEach(op => {
      const categories = op.categories;

      if (!categories || categories.length === 0) {
        if (!hierarchy.__operations) hierarchy.__operations = [];
        if (!hierarchy.__operations.find((existing: OpsOp) => existing.id === op.id)) {
          hierarchy.__operations.push(op);
        }
        return;
      }

      let currentLevel = hierarchy;

      categories.forEach((category, index) => {
        if (!currentLevel[category]) currentLevel[category] = {};
        if (index === categories.length - 1) {
          if (!currentLevel[category].__operations) currentLevel[category].__operations = [];
          if (!currentLevel[category].__operations.find((existing: OpsOp) => existing.id === op.id)) {
            currentLevel[category].__operations.push(op);
          }
        } else {
          currentLevel = currentLevel[category];
        }
      });
    });

    return hierarchy;
  }, [props.boxes]);

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

  const getCurrentLevel = () => {
    let currentLevel = categoryHierarchy;
    categoryPath.forEach(category => {
      currentLevel = currentLevel[category];
    });
    return currentLevel;
  };

  const filteredList = (): { item: string; op?: OpsOp; category?: string; isCategory?: boolean }[] => {
    const currentLevel = getCurrentLevel();

    if (!searchTerm) {
      const results: { item: string; op?: OpsOp; category?: string; isCategory?: boolean }[] = [];

      Object.keys(currentLevel)
        .filter(key => key !== '__operations')
        .sort((a, b) => a.localeCompare(b)) // ⬅️ kategóriák ABC sorrendben
        .forEach(key => {
          results.push({ item: key, isCategory: true });
        });

      if (currentLevel.__operations) {
        (currentLevel.__operations as OpsOp[])
          .slice()
          .sort((a: OpsOp, b: OpsOp) => a.name.localeCompare(b.name)) // ⬅️ operátorok ABC sorrendben
          .forEach((op: OpsOp) => {
            results.push({ item: op.name, op });
          });
      }

      return results;
    } else {
      const results: { item: string; op?: OpsOp; category?: string }[] = [];

      const searchAllOperations = (level: any, path: string[] = []) => {
        Object.keys(level)
          .sort((a, b) => a.localeCompare(b))
          .forEach(key => {
            if (key === '__operations') {
              (level[key] as OpsOp[])
                .slice()
                .sort((a: OpsOp, b: OpsOp) => a.name.localeCompare(b.name))
                .forEach((op: OpsOp) => {
                  if (op.name.toLowerCase().includes(searchTerm.toLowerCase())) {
                    results.push({
                      item: op.name,
                      op,
                      category: path.length > 0 ? path.join(' > ') : undefined
                    });
                  }
                });
            } else {
              searchAllOperations(level[key], [...path, key]);
            }
          });
      };

      searchAllOperations(categoryHierarchy);
      return results;
    }
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
    setTimeout(() => {
      if (document.activeElement?.closest('.node-search')) return;
      props.onCancel();
    }, 150);
  };

  return (
    <div className="node-search" style={{ ...styles.container, top: props.pos.y, left: props.pos.x }}>
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
          onMouseDown={(e) => e.preventDefault()}
          onClick={handleBackClick}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ArrowLeftIcon style={{ width: '14px', height: '14px' }} />
            Back
          </span>
        </div>
      )}

      <div style={styles.list}>
        {results.map((result, index) => (
          <div
            key={result.category ? `${result.category}-${result.item}` : result.item}
            style={{
              ...styles.item,
              ...(result.isCategory ? styles.categoryItem : {}),
              ...(index === selectedIndex ? styles.itemSelected : {}),
              ...(hoveredItem === result.item ? styles.itemHover : {}),
            }}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              if (result.op) {
                handleItemClick(result.op);
              } else if (result.isCategory) {
                handleCategoryClick(result.item);
              }
            }}
            onMouseEnter={() => {
              setHoveredItem(result.item);
              setSelectedIndex(index);
            }}
            onMouseLeave={() => setHoveredItem(null)}
          >
            {result.isCategory ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <FolderIcon style={{ width: '16px', height: '16px', color: '#007bff' }} />
                {result.item}
              </span>
            ) : result.category ? (
              <span>{result.item} <span style={{ color: '#666', fontSize: '0.8em' }}>({result.category})</span></span>
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
  item: {
    padding: "6px",
    cursor: "pointer",
    borderRadius: "4px",
  },
  categoryItem: {
    fontWeight: "bold",
    color: "#444",
    backgroundColor: "#f8f9fa",
  },
  itemSelected: {
    backgroundColor: "rgba(0, 123, 255, 0.2)",
    color: "black",
  },
  itemHover: {
    backgroundColor: "orange",
    color: "black",
  },
};
