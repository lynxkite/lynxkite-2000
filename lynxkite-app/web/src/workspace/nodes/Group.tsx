import { NodeResizeControl, useReactFlow, useViewport } from "@xyflow/react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
// @ts-expect-error
import ChevronDownRight from "~icons/tabler/chevron-down-right.jsx";
// @ts-expect-error
import Palette from "~icons/tabler/palette-filled.jsx";
import { COLORS } from "../../common.ts";
import Tooltip from "../../Tooltip.tsx";

export default function Group(props: any) {
  const reactFlow = useReactFlow();
  const [displayingColorPicker, setDisplayingColorPicker] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);

  function setColor(newValue: string) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { color: newValue },
    }));
    setDisplayingColorPicker(false);
  }

  function toggleColorPicker(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    e.stopPropagation();
    setDisplayingColorPicker((s) => !s);
  }

  const currentColor = props.data?.params?.color || "gray";

  return (
    <div
      className={`node-group ${props.parentId ? "in-group" : ""}`}
      style={{
        width: props.width,
        height: props.height,
        backgroundColor: COLORS[currentColor],
        position: "relative",
      }}
    >
      <button
        ref={buttonRef}
        onClick={toggleColorPicker}
        aria-label="Change group color"
        style={{
          background: "transparent",
          border: "none",
          cursor: "pointer",
          zIndex: 10,
          transform: "scale(1.25)",
          transformOrigin: "center",
        }}
      >
        <Tooltip doc="Change color">
          <Palette />
        </Tooltip>
      </button>

      {displayingColorPicker && (
        <ColorPickerPortal
          anchorRef={buttonRef}
          currentColor={currentColor}
          onPick={setColor}
          onRequestClose={() => setDisplayingColorPicker(false)}
        />
      )}

      <NodeResizeControl minWidth={100} minHeight={100}>
        <ChevronDownRight className="node-resizer" />
      </NodeResizeControl>
    </div>
  );
}

function ColorPickerPortal({
  anchorRef,
  currentColor,
  onPick,
  onRequestClose,
}: {
  anchorRef: React.RefObject<HTMLElement | null>;
  currentColor: string;
  onPick: (c: string) => void;
  onRequestClose: () => void;
}) {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  const { zoom } = useViewport();

  useEffect(() => {
    const anchor = anchorRef.current;
    const portalEl = portalRef.current;
    if (!anchor || !portalEl) return;

    const update = () => {
      const aRect = anchor.getBoundingClientRect();
      const pRect = portalEl.getBoundingClientRect();
      const margin = 6;

      const left = aRect.right - pRect.width;

      const top = aRect.bottom + margin;

      setPos({ left, top });
    };

    update();
    window.addEventListener("scroll", update, true);
    window.addEventListener("resize", update);

    return () => {
      window.removeEventListener("scroll", update, true);
      window.removeEventListener("resize", update);
    };
  }, [anchorRef]);

  useEffect(() => {
    function onDocMouseDown(e: MouseEvent) {
      const portalEl = portalRef.current;
      const anchorEl = anchorRef.current;
      if (portalEl?.contains(e.target as Node)) return;
      if (anchorEl?.contains(e.target as Node)) return;

      onRequestClose();
    }

    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [onRequestClose, anchorRef]);

  return createPortal(
    <div
      ref={portalRef}
      style={{
        position: "fixed",
        left: pos.left,
        top: pos.top,
        zIndex: 200000,

        transform: `scale(${zoom})`,
        transformOrigin: "top left",
      }}
    >
      <ColorPicker currentColor={currentColor} onPick={onPick} />
    </div>,
    document.body,
  );
}

function ColorPicker({
  currentColor,
  onPick,
}: {
  currentColor: string;
  onPick: (color: string) => void;
}) {
  const colors = Object.keys(COLORS).filter((c) => c !== currentColor);

  return (
    <div
      style={{
        display: "flex",
        gap: 8,
        padding: "10px 12px",
        background: "white",
        borderRadius: 12,
        boxShadow: "0 2px 6px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.08)",
        border: "1px solid rgba(0,0,0,0.06)",
      }}
    >
      {colors.map((color) => (
        <button
          key={color}
          onClick={() => onPick(color)}
          aria-label={`Pick ${color}`}
          style={{
            width: 30,
            height: 30,
            borderRadius: 6,
            background: COLORS[color],
            border: "none",
            cursor: "pointer",
          }}
        />
      ))}
    </div>
  );
}
