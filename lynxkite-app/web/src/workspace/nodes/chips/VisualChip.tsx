import type React from "react";
import { useState } from "react";
import type { BaseChip } from "./Chips.tsx";

interface VisualChipProps {
  chip: BaseChip;
  index: number;
  onEdit: (e: React.MouseEvent, index: number) => void;
  onToggleDisable: (e: React.MouseEvent, index: number) => void;
  onDelete: (index: number) => void;
  onInteractiveChange: () => void;
}

const THEME = {
  border: "#e2e8f0",
  deleteBtn: { bg: "#fee2e2", text: "#ef4444", hoverBg: "#fecaca" },
  disableBtn: {
    bg: "#ffffff",
    text: "#1f2937",
    hoverBg: "#f3f4f6",
    activeBg: "#1f2937",
    activeText: "#ffffff",
    activeHoverBg: "#111827",
    border: "#e5e7eb",
  },
};

const USER_SELECT_NONE_STYLE: React.CSSProperties = {
  userSelect: "none",
  WebkitUserSelect: "none",
};

export default function VisualChip({
  chip,
  index,
  onEdit,
  onToggleDisable,
  onDelete,
  onInteractiveChange,
}: VisualChipProps) {
  const [deleteHovered, setDeleteHovered] = useState(false);
  const [disableHovered, setDisableHovered] = useState(false);

  const getDisableBg = () => {
    if (chip.disabled) {
      return disableHovered ? THEME.disableBtn.activeHoverBg : THEME.disableBtn.activeBg;
    }
    return disableHovered ? THEME.disableBtn.hoverBg : THEME.disableBtn.bg;
  };

  return (
    <div
      onClick={(e) => onEdit(e, index)}
      style={{
        display: "inline-flex",
        alignItems: "center",
        background: chip.bg,
        color: chip.text,
        padding: "5px 8px 5px 12px",
        borderRadius: 10,
        gap: 10,
        fontSize: 12,
        fontWeight: 600,
        cursor: "pointer",
        border: `1px solid color-mix(in srgb, ${chip.text} 25%, transparent)`,
        opacity: chip.disabled ? 0.5 : 1,
        transition: "opacity 0.15s ease",
        ...USER_SELECT_NONE_STYLE,
      }}
    >
      <span style={{ textDecoration: chip.disabled ? "line-through" : "none" }}>
        {chip.getLabel()}
      </span>

      {!chip.disabled && chip.render(onInteractiveChange)}

      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <span
          onClick={(e) => onToggleDisable(e, index)}
          onMouseEnter={() => setDisableHovered(true)}
          onMouseLeave={() => setDisableHovered(false)}
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 20,
            height: 20,
            borderRadius: "50%",
            border: `1px solid ${THEME.disableBtn.border}`,
            background: getDisableBg(),
            color: chip.disabled ? THEME.disableBtn.activeText : THEME.disableBtn.text,
            fontWeight: "bold",
            fontSize: 12,
            transition: "background-color 0.15s ease, color 0.15s ease",
            lineHeight: 1,
          }}
          title={chip.disabled ? "Enable setting" : "Disable setting"}
        />

        <span
          onClick={(e) => {
            e.stopPropagation();
            onDelete(index);
          }}
          onMouseEnter={() => setDeleteHovered(true)}
          onMouseLeave={() => setDeleteHovered(false)}
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: deleteHovered ? THEME.deleteBtn.hoverBg : THEME.deleteBtn.bg,
            color: THEME.deleteBtn.text,
            fontWeight: "bold",
            fontSize: 14,
            transition: "background-color 0.15s ease",
            lineHeight: 1,
          }}
        >
          ×
        </span>
      </div>
    </div>
  );
}
