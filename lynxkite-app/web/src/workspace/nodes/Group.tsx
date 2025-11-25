import { NodeResizeControl, useReactFlow } from "@xyflow/react";
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
// @ts-expect-error
import ChevronDownRight from "~icons/tabler/chevron-down-right.jsx";
// @ts-expect-error
import Palette from "~icons/tabler/palette-filled.jsx";
import { COLORS } from "../../common.ts";
import Tooltip from "../../Tooltip.tsx";

export default function Group({ id, data, width, height, parentId }: any) {
  const rf = useReactFlow();
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const portalRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  const color = data?.params?.color || "gray";

  const setColor = (c: string) => {
    rf.updateNodeData(id, (d: any) => ({ ...d, params: { color: c } }));
    setOpen(false);
  };

  useLayoutEffect(() => {
    if (!open || !btnRef.current || !portalRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ left: r.right - portalRef.current.offsetWidth, top: r.bottom + 6 });
  }, [open]);

  return (
    <div
      className={`node-group ${parentId ? "in-group" : ""}`}
      style={{ width, height, backgroundColor: COLORS[color] }}
    >
      <button
        ref={btnRef}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className="node-group-color-picker-icon"
        aria-label="Change group color"
      >
        <Tooltip doc="Change color">
          <Palette width={30} height={30} />
        </Tooltip>
      </button>

      {open &&
        btnRef.current &&
        createPortal(
          <div
            ref={portalRef}
            className="menu p-2 shadow-sm bg-base-100 rounded-box"
            style={{
              position: "absolute",
              left: pos.left,
              top: pos.top,
              zIndex: 9999,
            }}
          >
            <div className="flex gap-2">
              {Object.keys(COLORS)
                .filter((c) => c !== color)
                .map((c) => (
                  <button
                    key={c}
                    style={{ backgroundColor: COLORS[c] }}
                    className="w-7 h-7 rounded"
                    onClick={() => setColor(c)}
                  />
                ))}
            </div>
          </div>,
          document.body,
        )}

      <NodeResizeControl minWidth={100} minHeight={100}>
        <ChevronDownRight className="node-resizer" />
      </NodeResizeControl>
    </div>
  );
}
