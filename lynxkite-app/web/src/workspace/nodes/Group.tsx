import { NodeResizeControl, useReactFlow } from "@xyflow/react";
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ChevronDownRight from "~icons/tabler/chevron-down-right.jsx";
import Palette from "~icons/tabler/palette-filled.jsx";
import { COLORS } from "../../common.ts";
import Tooltip from "../../Tooltip.tsx";

export default function Group(props: any) {
  const reactFlow = useReactFlow();
  const [displayingColorPicker, setDisplayingColorPicker] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const portalRef = useRef<HTMLDivElement | null>(null);
  const [portalPos, setPortalPos] = useState({ left: 0, top: 0 });

  const currentColor = props.data?.params?.color || "gray";

  function setColor(newColor: string) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { color: newColor },
    }));
    setDisplayingColorPicker(false);
  }

  function toggleColorPicker(e: React.MouseEvent<HTMLButtonElement>) {
    e.stopPropagation();
    setDisplayingColorPicker((s) => !s);
  }

  useLayoutEffect(() => {
    if (!displayingColorPicker || !buttonRef.current || !portalRef.current) return;

    const buttonRect = buttonRef.current.getBoundingClientRect();
    setPortalPos({
      left: buttonRect.right - portalRef.current.offsetWidth,
      top: buttonRect.bottom + 6,
    });
  }, [displayingColorPicker]);

  return (
    <div
      className={`node-group ${props.parentId ? "in-group" : ""}`}
      style={{
        width: props.width,
        height: props.height,
        backgroundColor: COLORS[currentColor],
      }}
    >
      <button
        ref={buttonRef}
        onClick={toggleColorPicker}
        className="node-group-color-picker-icon"
        aria-label="Change group color"
      >
        <Tooltip doc="Change color">
          <Palette width={30} height={30} />
        </Tooltip>
      </button>

      {displayingColorPicker &&
        buttonRef.current &&
        createPortal(
          <div
            ref={portalRef}
            className="dropdown-content menu p-2 shadow-sm bg-base-100 rounded-box"
            style={{
              position: "absolute",
              left: portalPos.left,
              top: portalPos.top,
              zIndex: 9999,
            }}
          >
            <ColorPicker currentColor={currentColor} onPick={setColor} />
          </div>,
          document.body,
        )}

      <NodeResizeControl minWidth={100} minHeight={100}>
        <ChevronDownRight className="node-resizer" />
      </NodeResizeControl>
    </div>
  );
}

function ColorPicker(props: { currentColor: string; onPick: (color: string) => void }) {
  const colors = Object.keys(COLORS).filter((color) => color !== props.currentColor);

  return (
    <div className="flex gap-2">
      {colors.map((color) => (
        <button
          key={color}
          style={{ backgroundColor: COLORS[color] }}
          className="w-7 h-7 rounded"
          onClick={() => props.onPick(color)}
        />
      ))}
    </div>
  );
}
