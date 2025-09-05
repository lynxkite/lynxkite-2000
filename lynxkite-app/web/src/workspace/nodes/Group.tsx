import { useReactFlow } from "@xyflow/react";
import { useState } from "react";
// @ts-expect-error
import Palette from "~icons/tabler/palette-filled.jsx";
// @ts-expect-error
import Square from "~icons/tabler/square-filled.jsx";
import { COLORS } from "../../common.ts";
import Tooltip from "../../Tooltip.tsx";

export default function Group(props: any) {
  const reactFlow = useReactFlow();
  const [displayingColorPicker, setDisplayingColorPicker] = useState(false);
  function setColor(newValue: string) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { color: newValue },
    }));
    setDisplayingColorPicker(false);
  }
  function toggleColorPicker(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    e.stopPropagation();
    setDisplayingColorPicker(!displayingColorPicker);
  }
  const currentColor = props.data?.params?.color || "gray";
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
        className="node-group-color-picker-icon"
        onClick={toggleColorPicker}
        aria-label="Change group color"
      >
        <Tooltip doc="Change color">
          <Palette />
        </Tooltip>
      </button>
      {displayingColorPicker && <ColorPicker currentColor={currentColor} onPick={setColor} />}
    </div>
  );
}

function ColorPicker(props: { currentColor: string; onPick: (color: string) => void }) {
  const colors = Object.keys(COLORS).filter((color) => color !== props.currentColor);
  return (
    <div className="color-picker">
      {colors.map((color) => (
        <button
          key={color}
          className="color-picker-button"
          style={{ color: COLORS[color] }}
          onClick={() => props.onPick(color)}
        >
          <Square />
        </button>
      ))}
    </div>
  );
}
