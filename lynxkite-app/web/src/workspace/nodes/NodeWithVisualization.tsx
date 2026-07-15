import * as echarts from "echarts";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { useDisplay } from "../../common.ts";
import LynxKiteNode from "./LynxKiteNode.tsx";

const getPastelColor = (s: string): string => {
  let h = 0,
    t = String(s ?? "");
  for (let i = 0; i < t.length; i++) h = t.charCodeAt(i) + ((h << 5) - h);
  return `hsl(${Math.abs(h * 131) % 360}, 80%, 60%)`;
};

export interface FormFieldConfig {
  key: string;
  label?: string;
}

export abstract class BaseChip {
  abstract type: string;
  disabled: boolean;
  bg!: string;
  text!: string;

  static type: string;
  static displayName: string;
  static target: string;
  static formFields: FormFieldConfig[];

  constructor(disabled: boolean = false) {
    this.disabled = disabled;
  }

  abstract getLabel(): string;
  abstract getFormData(): Record<string, string>;
  abstract apply(series: any): void;
}

export class NodeColorChip extends BaseChip {
  static type = "node_color";
  type = NodeColorChip.type;
  static displayName = "Color Setting (Nodes)";
  static target = "node";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(disabled);
    this.attribute = data.attribute || "";
    this.bg = "#e0f2fe";
    this.text = "#0369a1";
  }

  getLabel(): string {
    return `Node Color by: ${this.attribute}`;
  }

  getFormData() {
    return { attribute: this.attribute };
  }

  apply(series: any): void {
    if (!series?.data) return;
    series.data.forEach((node: any) => {
      const val = node.attributes?.[this.attribute];
      if (val !== undefined && val !== null && val !== "") {
        node.itemStyle = node.itemStyle || {};
        node.itemStyle.color = getPastelColor(String(val));
      }
    });
  }
}

export class EdgeColorChip extends BaseChip {
  static type = "edgeColor";
  type = EdgeColorChip.type;
  static displayName = "Color Setting (Edges)";
  static target = "edge";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(disabled);
    this.attribute = data.attribute || "";
    this.bg = "#fae8ff";
    this.text = "#86198f";
  }

  getLabel(): string {
    return `Edge Color by: ${this.attribute}`;
  }

  getFormData() {
    return { attribute: this.attribute };
  }

  apply(series: any): void {
    const edges = series?.links || series?.edges;
    if (!edges) return;

    edges.forEach((edge: any) => {
      const val = edge.attributes?.[this.attribute];
      if (val !== undefined && val !== null && val !== "") {
        edge.lineStyle = edge.lineStyle || {};
        edge.lineStyle.color = getPastelColor(String(val));
      }
    });
  }
}

export class PositionChip extends BaseChip {
  static type = "position";
  type = PositionChip.type;
  static displayName = "Position Setting";
  static target = "node";
  static formFields: FormFieldConfig[] = [
    { key: "xAttr", label: "X:" },
    { key: "yAttr", label: "Y:" },
  ];

  xAttr: string;
  yAttr: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(disabled);
    this.xAttr = data.xAttr || "";
    this.yAttr = data.yAttr || "";
    this.bg = "#e2ffac";
    this.text = "rgb(18 93 53 / 0.78)";
  }

  getLabel(): string {
    return `Position: X(${this.xAttr}) Y(${this.yAttr})`;
  }

  getFormData() {
    return { xAttr: this.xAttr, yAttr: this.yAttr };
  }

  apply(series: any): void {
    if (!series?.data) return;
    series.layout = "none";
    series.data.forEach((node: any) => {
      const xVal = Number(node.attributes?.[this.xAttr]);
      const yVal = Number(node.attributes?.[this.yAttr]);
      if (!Number.isNaN(xVal) && !Number.isNaN(yVal)) {
        node.x = xVal;
        node.y = yVal;
      }
    });
  }
}

export class LabelChip extends BaseChip {
  static type = "label";
  type = LabelChip.type;
  static displayName = "Label Setting";
  static target: "node" | "edge" = "node";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(disabled);
    this.attribute = data.attribute || "";
    this.bg = "#fef08a";
    this.text = "#a16207";
  }

  getLabel(): string {
    return `Label by: ${this.attribute}`;
  }

  getFormData() {
    return { attribute: this.attribute };
  }

  apply(series: any): void {
    if (!series?.data) return;
    series.data.forEach((node: any) => {
      const val = node.attributes?.[this.attribute];
      node.label = {
        ...node.label,
        show: val !== undefined && val !== null && val !== "",
        formatter: String(val ?? ""),
        position: "top",
      };
    });
  }
}

const CHIP_REGISTRY = [NodeColorChip, EdgeColorChip, PositionChip, LabelChip];

const BORDER_RADIUS_MAIN = 10;
const BORDER_RADIUS_BUTTON = 20;

const THEME = {
  button: { bg: "rgb(33 168 96 / 0.78)", text: "#ffffff" },
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
  WebkitUserSelect: "none",
  MozUserSelect: "none",
  msUserSelect: "none",
  userSelect: "none",
};

// Extends and separates node and edge parameters explicitly
const extractNodeAttributes = (opts: any): string[] => {
  const nodes = opts?.series?.[0]?.data || [];
  const keysSet = new Set<string>();
  for (const node of nodes) {
    if (node?.attributes) {
      for (const key of Object.keys(node.attributes)) {
        keysSet.add(key);
      }
    }
  }
  return Array.from(keysSet);
};

const extractEdgeAttributes = (opts: any): string[] => {
  const edges = opts?.series?.[0]?.links || opts?.series?.[0]?.edges || [];
  const keysSet = new Set<string>();
  for (const edge of edges) {
    if (edge?.attributes) {
      for (const key of Object.keys(edge.attributes)) {
        keysSet.add(key);
      }
    }
  }
  return Array.from(keysSet);
};

interface ChipFormProps {
  nodeAttrs: string[];
  edgeAttrs: string[];
  initialChip: BaseChip | null;
  onSubmit: (newChip: BaseChip) => void;
}

function ChipForm({ nodeAttrs, edgeAttrs, initialChip, onSubmit }: ChipFormProps) {
  const [formType, setFormType] = useState<string>(initialChip?.type || CHIP_REGISTRY[0].type);
  const [formData, setFormData] = useState<Record<string, string>>({});

  const ActiveClass = CHIP_REGISTRY.find((c) => c.type === formType) || CHIP_REGISTRY[0];
  const targetAttrs = ActiveClass.target === "edge" ? edgeAttrs : nodeAttrs;

  useEffect(() => {
    const defaultData: Record<string, string> = {};
    const initialData = initialChip?.type === formType ? initialChip.getFormData() : {};

    ActiveClass.formFields.forEach((fieldConfig, index) => {
      const fieldKey = fieldConfig.key;
      defaultData[fieldKey] = initialData[fieldKey] || targetAttrs[index] || targetAttrs[0] || "";
    });

    setFormData(defaultData);
  }, [formType, initialChip, targetAttrs, ActiveClass]);

  const handleFieldChange = (fieldKey: string, value: string) => {
    setFormData((prev) => ({ ...prev, [fieldKey]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newChip = new (ActiveClass as any)(formData, initialChip?.disabled);
    onSubmit(newChip);
  };

  const selectStyle = {
    padding: "4px 8px",
    borderRadius: 6,
    border: `1px solid ${THEME.border}`,
    outline: "none",
    backgroundColor: "#fff",
    fontSize: 12,
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "#fff",
        padding: "6px 12px",
        borderRadius: BORDER_RADIUS_MAIN,
        boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
        border: `1px solid ${THEME.border}`,
      }}
    >
      <select value={formType} onChange={(e) => setFormType(e.target.value)} style={selectStyle}>
        {CHIP_REGISTRY.map((c) => (
          <option key={c.type} value={c.type}>
            {c.displayName}
          </option>
        ))}
      </select>

      {ActiveClass.formFields.map((fieldConfig) => (
        <div key={fieldConfig.key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {fieldConfig.label && (
            <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}>
              {fieldConfig.label}
            </span>
          )}
          <select
            value={formData[fieldConfig.key] || ""}
            onChange={(e) => handleFieldChange(fieldConfig.key, e.target.value)}
            style={selectStyle}
          >
            {targetAttrs.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
      ))}

      <button
        type="submit"
        style={{
          padding: "4px 12px",
          fontSize: 12,
          background: THEME.button.bg,
          color: THEME.button.text,
          border: "none",
          borderRadius: 6,
          cursor: "pointer",
          fontWeight: "bold",
        }}
      >
        {initialChip ? "Save" : "Apply"}
      </button>
    </form>
  );
}

interface VisualChipProps {
  chip: BaseChip;
  index: number;
  onEdit: (e: React.MouseEvent, index: number) => void;
  onToggleDisable: (e: React.MouseEvent, index: number) => void;
  onDelete: (index: number) => void;
}

function VisualChip({ chip, index, onEdit, onToggleDisable, onDelete }: VisualChipProps) {
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
        borderRadius: BORDER_RADIUS_MAIN,
        gap: 10,
        fontSize: 12,
        fontWeight: 600,
        cursor: "pointer",
        border: `1px solid ${chip.text}20`,
        opacity: chip.disabled ? 0.5 : 1,
        transition: "opacity 0.15s ease",
        ...USER_SELECT_NONE_STYLE,
      }}
    >
      <span style={{ textDecoration: chip.disabled ? "line-through" : "none" }}>
        {chip.getLabel()}
      </span>

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

export function NodeWithVisualization({ data, id }: { data: any; id: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const opts = useDisplay(data?.display_version, id);

  const [chips, setChips] = useState<BaseChip[]>([]);
  const [open, setOpen] = useState(false);
  const [nodeAttrs, setNodeAttrs] = useState<string[]>([]);
  const [edgeAttrs, setEdgeAttrs] = useState<string[]>([]);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [mainBtnHover, setMainBtnHover] = useState(false);

  useEffect(() => {
    setNodeAttrs(extractNodeAttributes(opts));
    setEdgeAttrs(extractEdgeAttributes(opts));
  }, [opts]);

  useEffect(() => {
    if (!opts || !ref.current) return;

    const chartOpts = JSON.parse(JSON.stringify(opts));
    const series = chartOpts.series?.[0];

    if (series) {
      if (series.data) {
        series.data = series.data.map((n: any) => ({
          ...n,
          itemStyle: { ...n.itemStyle },
        }));
      }

      const edgeKey = series.links ? "links" : "edges";
      if (series[edgeKey]) {
        series[edgeKey] = series[edgeKey].map((e: any) => ({
          ...e,
          lineStyle: { ...e.lineStyle },
        }));
      }

      chips
        .filter((c) => !c.disabled)
        .forEach((c) => {
          c.apply(series);
        });
    }

    const inst = echarts.init(ref.current, undefined, {
      renderer: "canvas",
      width: "auto",
      height: "auto",
    });
    inst.setOption(chartOpts, true);

    const obs = new ResizeObserver(() => inst.resize());
    obs.observe(ref.current);

    return () => {
      obs.disconnect();
      inst.dispose();
    };
  }, [opts, chips]);

  const handleFormSubmit = (newChip: BaseChip) => {
    if (editingIdx !== null) {
      const updated = [...chips];
      updated[editingIdx] = newChip;
      setChips(updated);
      setEditingIdx(null);
    } else {
      setChips([...chips, newChip]);
    }
    setOpen(false);
  };

  const startEdit = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    setEditingIdx(index);
    setOpen(true);
  };

  const handleToggleDisable = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    const updated = [...chips];
    const current = updated[index];

    const TargetClass = CHIP_REGISTRY.find((c) => c.type === current.type) || CHIP_REGISTRY[0];
    updated[index] = new (TargetClass as any)(current.getFormData(), !current.disabled);

    setChips(updated);
  };

  const handleToggleForm = (e: React.MouseEvent) => {
    e.stopPropagation();
    setOpen(!open);
    setEditingIdx(null);
  };

  const handleDeleteChip = (index: number) => {
    setChips(chips.filter((_, idx) => idx !== index));
    if (editingIdx === index) {
      setEditingIdx(null);
      setOpen(false);
    }
  };

  const hasAttributes = nodeAttrs.length > 0 || edgeAttrs.length > 0;

  return (
    <div
      style={{ flex: 1, position: "relative", width: "100%", height: "100%", minHeight: "350px" }}
    >
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          zIndex: 99,
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
          ...USER_SELECT_NONE_STYLE,
        }}
      >
        {hasAttributes && (
          <button
            onClick={handleToggleForm}
            onMouseEnter={() => setMainBtnHover(true)}
            onMouseLeave={() => setMainBtnHover(false)}
            style={{
              width: 28,
              height: 28,
              borderRadius: BORDER_RADIUS_BUTTON,
              border: `1px solid ${open ? `${THEME.deleteBtn.text}40` : THEME.border}`,
              background: open
                ? mainBtnHover
                  ? THEME.deleteBtn.hoverBg
                  : THEME.deleteBtn.bg
                : "#fff",
              color: open ? THEME.deleteBtn.text : "#555",
              cursor: "pointer",
              fontWeight: "bold",
              fontSize: 16,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 0,
              outline: "none",
              boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
              transition: "background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease",
              ...USER_SELECT_NONE_STYLE,
            }}
          >
            {open ? "×" : "+"}
          </button>
        )}

        {open && hasAttributes && (
          <ChipForm
            nodeAttrs={nodeAttrs}
            edgeAttrs={edgeAttrs}
            initialChip={editingIdx !== null ? chips[editingIdx] : null}
            onSubmit={handleFormSubmit}
          />
        )}

        {chips.map((c, i) => {
          if (editingIdx === i) return null;
          return (
            <VisualChip
              key={i}
              chip={c}
              index={i}
              onEdit={startEdit}
              onToggleDisable={handleToggleDisable}
              onDelete={handleDeleteChip}
            />
          );
        })}
      </div>
      <div
        ref={ref}
        style={{ width: "100%", height: "100%", minHeight: "350px", ...USER_SELECT_NONE_STYLE }}
      />
    </div>
  );
}

export default LynxKiteNode(NodeWithVisualization);
