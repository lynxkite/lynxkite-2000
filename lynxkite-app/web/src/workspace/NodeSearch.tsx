import Fuse from "fuse.js";
import { useEffect, useMemo, useRef, useState } from "react";

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
  const searchBox = useRef(null as unknown as HTMLInputElement);
  const [searchText, setSearchText] = useState("");
  const fuse = useMemo(
    () =>
      new Fuse(Object.values(props.boxes), {
        keys: ["name"],
      }),
    [props.boxes],
  );
  const allOps = useMemo(() => {
    const boxes = Object.values(props.boxes).map((box) => ({ item: box }));
    boxes.sort((a, b) => a.item.name.localeCompare(b.item.name));
    return boxes;
  }, [props.boxes]);
  const hits: { item: OpsOp }[] = searchText ? fuse.search<OpsOp>(searchText) : allOps;
  const [selectedIndex, setSelectedIndex] = useState(0);
  useEffect(() => searchBox.current.focus());
  function typed(text: string) {
    setSearchText(text);
    setSelectedIndex(Math.max(0, Math.min(selectedIndex, hits.length - 1)));
  }
  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex(Math.min(selectedIndex + 1, hits.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex(Math.max(selectedIndex - 1, 0));
    } else if (e.key === "Enter") {
      addSelected();
    } else if (e.key === "Escape") {
      props.onCancel();
    }
  }
  function addSelected() {
    const node = { ...hits[selectedIndex].item };
    node.position = props.pos;
    props.onAdd(node);
  }
  async function lostFocus(e: any) {
    // If it's a click on a result, let the click handler handle it.
    if (e.relatedTarget?.closest(".node-search")) return;
    props.onCancel();
  }

  return (
    <div className="node-search" style={{ top: props.pos.y, left: props.pos.x }}>
      <input
        ref={searchBox}
        value={searchText}
        onChange={(event) => typed(event.target.value)}
        onKeyDown={onKeyDown}
        onBlur={lostFocus}
        placeholder="Search for box"
      />
      <div className="matches">
        {hits.map((box, index) => (
          <div
            key={box.item.name}
            tabIndex={0}
            onFocus={() => setSelectedIndex(index)}
            onMouseEnter={() => setSelectedIndex(index)}
            onClick={addSelected}
            className={`search-result ${index === selectedIndex ? "selected" : ""}`}
          >
            {box.item.categories.map((category) => `${category}\u00A0›\u00A0`)}
            {box.item.name}
          </div>
        ))}
      </div>
    </div>
  );
}
