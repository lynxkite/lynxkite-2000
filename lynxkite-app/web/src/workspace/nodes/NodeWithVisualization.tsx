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
  type?: "select" | "number";
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

  constructor(_data: Record<string, string>, disabled = false) {
    this.disabled = disabled;
  }

  static getInitialData(
    attribute: string,
    _rawItems: any[],
    previousData?: Record<string, string>,
  ): Record<string, string> {
    const data: Record<string, string> = {};
    BaseChip.formFields?.forEach((fieldConfig) => {
      const key = fieldConfig.key;
      if (key === "attribute") {
        data[key] = attribute;
      } else {
        data[key] = previousData?.[key] || "";
      }
    });
    return data;
  }

  abstract getLabel(): string;
  abstract getFormData(): Record<string, string>;
  abstract apply(series: any): void;

  render(_onChange?: () => void): React.ReactNode {
    return null;
  }
}

export class NodeColorChip extends BaseChip {
  static type = "node_color";
  type = NodeColorChip.type;
  static displayName = "Node color by";
  static target = "node";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled);
    this.attribute = data.attribute || "";
    this.bg = "#e0f2fe";
    this.text = "#0369a1";
  }

  getLabel() {
    return `Node color by: ${this.attribute}`;
  }

  getFormData() {
    return { attribute: this.attribute };
  }

  apply(series: any) {
    if (!this.attribute) return;
    series?.data?.forEach((node: any) => {
      const val = node.attributes?.[this.attribute];
      if (val !== undefined && val !== null && val !== "") {
        node.itemStyle = { ...node.itemStyle, color: getPastelColor(String(val)) };
      }
    });
  }
}

export class EdgeColorChip extends BaseChip {
  static type = "edgeColor";
  type = EdgeColorChip.type;
  static displayName = "Edge color by";
  static target = "edge";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled);
    this.attribute = data.attribute || "";
    this.bg = "#fae8ff";
    this.text = "#86198f";
  }

  getLabel() {
    return `Edge color by: ${this.attribute}`;
  }

  getFormData() {
    return { attribute: this.attribute };
  }

  apply(series: any) {
    if (!this.attribute) return;
    const edgeKey = series.links ? "links" : "edges";
    const edges = series?.[edgeKey];
    edges?.forEach((edge: any) => {
      const val = edge.attributes?.[this.attribute];
      if (val !== undefined && val !== null && val !== "") {
        edge.lineStyle = { ...edge.lineStyle, color: getPastelColor(String(val)) };
      }
    });
  }
}

export class PositionChip extends BaseChip {
  static type = "position";
  type = PositionChip.type;
  static displayName = "Position";
  static target = "node";
  static formFields: FormFieldConfig[] = [
    { key: "xAttr", label: "X:" },
    { key: "yAttr", label: "Y:" },
  ];

  xAttr: string;
  yAttr: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled);
    this.xAttr = data.xAttr || "";
    this.yAttr = data.yAttr || "";
    this.bg = "#e2ffac";
    this.text = "rgb(18 93 53 / 0.78)";
  }

  getLabel() {
    return `Position: X(${this.xAttr}) Y(${this.yAttr})`;
  }

  getFormData() {
    return { xAttr: this.xAttr, yAttr: this.yAttr };
  }

  apply(series: any) {
    if (!series?.data || !this.xAttr || !this.yAttr) return;
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
  static displayName = "Label by";
  static target = "node";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled);
    this.attribute = data.attribute || "";
    this.bg = "#fef08a";
    this.text = "#a16207";
  }

  getLabel() {
    return `Label by: ${this.attribute}`;
  }

  getFormData() {
    return { attribute: this.attribute };
  }

  apply(series: any) {
    if (!this.attribute) return;
    series?.data?.forEach((node: any) => {
      const val = node.attributes?.[this.attribute];
      const hasValue = val !== undefined && val !== null && val !== "";
      node.label = {
        ...node.label,
        show: hasValue,
        formatter: hasValue ? String(val) : "",
        position: "top",
      };
    });
  }
}

export class SliderChip extends BaseChip {
  static type = "slider";
  type = SliderChip.type;
  static displayName = "Slider";
  static target = "node";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;
  limitMin: number;
  limitMax: number;
  currentMin: number;
  currentMax: number;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled);
    this.attribute = data.attribute || "";
    this.limitMin = Number(data.min) || 0;
    this.limitMax = Number(data.max) || 100;

    this.currentMin = data.currentMin !== undefined ? Number(data.currentMin) : this.limitMin;
    this.currentMax = data.currentMax !== undefined ? Number(data.currentMax) : this.limitMax;

    this.bg = "#fee2e2";
    this.text = "#dc2626";
  }

  static override getInitialData(
    attribute: string,
    rawItems: any[],
    previousData?: Record<string, string>,
  ): Record<string, string> {
    const bounds = calculateAttributeBounds(rawItems, attribute);
    const isSameAttribute = previousData?.attribute === attribute;

    return {
      attribute,
      min: String(bounds.min),
      max: String(bounds.max),
      currentMin:
        isSameAttribute && previousData?.currentMin !== undefined
          ? previousData.currentMin
          : String(bounds.min),
      currentMax:
        isSameAttribute && previousData?.currentMax !== undefined
          ? previousData.currentMax
          : String(bounds.max),
    };
  }

  getLabel() {
    return `Filter: ${this.attribute}`;
  }

  getFormData() {
    return {
      attribute: this.attribute,
      min: String(this.limitMin),
      max: String(this.limitMax),
      currentMin: String(this.currentMin),
      currentMax: String(this.currentMax),
    };
  }

  apply(series: any) {
    if (!series?.data || !this.attribute) return;
    const keptNodeIds = new Set<string>();

    series.data = series.data.filter((node: any) => {
      const val = Number(node.attributes?.[this.attribute]);
      if (Number.isNaN(val)) return true;
      const keep = val >= this.currentMin && val <= this.currentMax;
      if (keep) keptNodeIds.add(node.id);
      return keep;
    });

    const edgeKey = series.links ? "links" : "edges";
    if (series[edgeKey]) {
      series[edgeKey] = series[edgeKey].filter(
        (edge: any) => keptNodeIds.has(edge.source) && keptNodeIds.has(edge.target),
      );
    }
  }

  override render(onChange?: () => void) {
    const handleMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = Math.min(Number(e.target.value), this.currentMax);
      this.currentMin = val;
      if (onChange) onChange();
    };

    const handleMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = Math.max(Number(e.target.value), this.currentMin);
      this.currentMax = val;
      if (onChange) onChange();
    };

    const rangeDiff = this.limitMax - this.limitMin || 1;
    const leftPercent = ((this.currentMin - this.limitMin) / rangeDiff) * 100;
    const rightPercent = ((this.currentMax - this.limitMin) / rangeDiff) * 100;

    const rangeInputStyle = (isMin: boolean): React.CSSProperties => ({
      position: "absolute",
      width: "100%",
      height: 2,
      pointerEvents: "none",
      appearance: "none",
      WebkitAppearance: "none",
      background: "none",
      outline: "none",
      margin: 0,
      zIndex: isMin ? 3 : 2,
    });

    return (
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginLeft: 4,
          background: "rgba(255,255,255,0.45)",
          padding: "2px 8px",
          borderRadius: 6,
        }}
      >
        <span style={{ fontSize: 9, fontFamily: "monospace", opacity: 0.8 }}>
          {this.currentMin}
        </span>

        <div
          style={{
            position: "relative",
            width: 60,
            height: 16,
            display: "flex",
            alignItems: "center",
          }}
        >
          <div
            style={{
              position: "absolute",
              width: "100%",
              height: 2,
              borderRadius: 1,
              background: `linear-gradient(to right, #fca5a5 ${leftPercent}%, ${this.text} ${leftPercent}%, ${this.text} ${rightPercent}%, #fca5a5 ${rightPercent}%)`,
            }}
          />

          <style>{`
            input[type="range"] {
              pointer-events: none !important;
            }
            input[type="range"]::-webkit-slider-thumb {
              pointer-events: auto !important;
              appearance: none !important;
              width: 8px !important;
              height: 8px !important;
              border-radius: 50% !important;
              background-color: ${this.text} !important;
              cursor: pointer !important;
              transition: transform 0.1s ease !important;
            }
            input[type="range"]::-webkit-slider-thumb:hover {
              transform: scale(1.25) !important;
            }
            input[type="range"]::-moz-range-thumb {
              pointer-events: auto !important;
              width: 8px !important;
              height: 8px !important;
              border: none !important;
              border-radius: 50% !important;
              background-color: ${this.text} !important;
              cursor: pointer !important;
              transition: transform 0.1s ease !important;
            }
            input[type="range"]::-moz-range-thumb:hover {
              transform: scale(1.25) !important;
            }
          `}</style>

          <input
            type="range"
            min={this.limitMin}
            max={this.limitMax}
            value={this.currentMin}
            onChange={handleMinChange}
            style={rangeInputStyle(true)}
          />
          <input
            type="range"
            min={this.limitMin}
            max={this.limitMax}
            value={this.currentMax}
            onChange={handleMaxChange}
            style={rangeInputStyle(false)}
          />
        </div>

        <span style={{ fontSize: 9, fontFamily: "monospace", opacity: 0.8 }}>
          {this.currentMax}
        </span>
      </div>
    );
  }
}

interface ChipConstructor {
  type: string;
  displayName: string;
  target: string;
  formFields: FormFieldConfig[];
  new (data: Record<string, string>, disabled?: boolean): BaseChip;
  getInitialData(
    attribute: string,
    rawItems: any[],
    previousData?: Record<string, string>,
  ): Record<string, string>;
}

const CHIP_REGISTRY: ChipConstructor[] = [
  NodeColorChip,
  EdgeColorChip,
  PositionChip,
  LabelChip,
  SliderChip,
];

const BORDER_RADIUS_MAIN = 10;

const THEME = {
  button: { bg: "rgb(33 168 96 / 0.78)", text: "#ffffff", disabledBg: "#cbd5e1" },
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

const extractAttributes = (items: any[]): string[] => {
  const keysSet = new Set<string>();
  items.forEach((item) => {
    if (item?.attributes) {
      Object.keys(item.attributes).forEach((k) => {
        keysSet.add(k);
      });
    }
  });
  return Array.from(keysSet);
};

const calculateAttributeBounds = (
  items: any[],
  attribute: string,
): { min: number; max: number } => {
  let min = Infinity;
  let max = -Infinity;

  items?.forEach((item) => {
    const val = Number(item?.attributes?.[attribute]);
    if (!Number.isNaN(val)) {
      if (val < min) min = val;
      if (val > max) max = val;
    }
  });

  return min === Infinity ? { min: 0, max: 100 } : { min, max };
};

interface ChipFormProps {
  nodeAttrs: string[];
  edgeAttrs: string[];
  initialChip: BaseChip | null;
  onSubmit: (newChip: BaseChip) => void;
  rawElements: { nodes: any[]; edges: any[] };
}

function ChipForm({ nodeAttrs, edgeAttrs, initialChip, onSubmit, rawElements }: ChipFormProps) {
  const [formType, setFormType] = useState(initialChip?.type || CHIP_REGISTRY[0].type);
  const [formData, setFormData] = useState<Record<string, string>>({});

  const ActiveClass = CHIP_REGISTRY.find((c) => c.type === formType) || CHIP_REGISTRY[0];
  const targetAttrs = ActiveClass.target === "edge" ? edgeAttrs : nodeAttrs;

  // Initialize form state once when form type or initial chip changes
  useEffect(() => {
    const initialData: Record<string, string> = {};
    ActiveClass.formFields.forEach((field) => {
      if (initialChip && initialChip.type === formType) {
        initialData[field.key] = initialChip.getFormData()[field.key] || "";
      } else {
        initialData[field.key] = ""; // Keep fields clear/unselected by default
      }
    });
    setFormData(initialData);
  }, [formType, initialChip]);

  const handleFieldChange = (key: string, value: string) => {
    const activeRawItems = ActiveClass.target === "edge" ? rawElements.edges : rawElements.nodes;

    setFormData((prev) => {
      const nextData = { ...prev, [key]: value };

      // If updating the primary "attribute" field, generate the initial bounds automatically if subclass supports it
      if (key === "attribute" || ActiveClass.formFields.length === 1) {
        const defaultData = ActiveClass.getInitialData(value, activeRawItems);
        return { ...nextData, ...defaultData };
      }

      return nextData;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(new ActiveClass(formData, initialChip?.disabled));
  };

  const isFormInvalid = ActiveClass.formFields.some((field) => {
    const val = formData[field.key];
    return !val || val === "";
  });

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

      {ActiveClass.formFields.map((field) => (
        <div key={field.key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {field.label && (
            <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}>{field.label}</span>
          )}
          {field.type === "number" ? (
            <input
              type="number"
              value={formData[field.key] ?? ""}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              style={{ ...selectStyle, width: 60 }}
            />
          ) : (
            <select
              value={formData[field.key] || ""}
              onChange={(e) => handleFieldChange(field.key, e.target.value)}
              style={selectStyle}
            >
              <option value=""></option>
              {targetAttrs.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          )}
        </div>
      ))}

      <button
        type="submit"
        disabled={isFormInvalid}
        style={{
          padding: "4px 12px",
          fontSize: 12,
          background: isFormInvalid ? THEME.button.disabledBg : THEME.button.bg,
          color: THEME.button.text,
          border: "none",
          borderRadius: 6,
          cursor: isFormInvalid ? "not-allowed" : "pointer",
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
  onInteractiveChange: () => void;
}

function VisualChip({
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

export function NodeWithVisualization({ data, id }: { data: any; id: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const opts = useDisplay(data?.display_version, id);

  const [chips, setChips] = useState<BaseChip[]>([]);
  const [open, setOpen] = useState(false);
  const [nodeAttrs, setNodeAttrs] = useState<string[]>([]);
  const [edgeAttrs, setEdgeAttrs] = useState<string[]>([]);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [mainBtnHover, setMainBtnHover] = useState(false);

  const [interactiveTick, setInteractiveTick] = useState(0);

  useEffect(() => {
    setNodeAttrs(extractAttributes(opts?.series?.[0]?.data || []));
    setEdgeAttrs(extractAttributes(opts?.series?.[0]?.links || opts?.series?.[0]?.edges || []));
  }, [opts]);

  useEffect(() => {
    if (!opts || !ref.current) return;

    const chartOpts = JSON.parse(JSON.stringify(opts));
    const series = chartOpts.series?.[0];

    if (series) {
      if (series.data) {
        series.data = series.data.map((n: any) => ({
          ...n,
          label: { ...n.label },
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

    const inst = echarts.init(ref.current);
    inst.setOption(chartOpts, true);

    const obs = new ResizeObserver(() => inst.resize());
    obs.observe(ref.current);

    return () => {
      obs.disconnect();
      inst.dispose();
    };
  }, [opts, chips, interactiveTick]);

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

  const handleToggleDisable = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    const updated = [...chips];
    const current = updated[index];
    const TargetClass = CHIP_REGISTRY.find((c) => c.type === current.type) || CHIP_REGISTRY[0];
    updated[index] = new TargetClass(current.getFormData(), !current.disabled);
    setChips(updated);
  };

  const hasAttributes = nodeAttrs.length > 0 || edgeAttrs.length > 0;

  const rawNodes = opts?.series?.[0]?.data || [];
  const rawEdges = opts?.series?.[0]?.links || opts?.series?.[0]?.edges || [];

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
            onClick={(e) => {
              e.stopPropagation();
              setOpen(!open);
              setEditingIdx(null);
            }}
            onMouseEnter={() => setMainBtnHover(true)}
            onMouseLeave={() => setMainBtnHover(false)}
            style={{
              width: 28,
              height: 28,
              borderRadius: 20,
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
            rawElements={{ nodes: rawNodes, edges: rawEdges }}
          />
        )}

        {chips.map((c, i) => {
          if (editingIdx === i) return null;
          return (
            <VisualChip
              key={i}
              chip={c}
              index={i}
              onEdit={(e) => {
                e.stopPropagation();
                setEditingIdx(i);
                setOpen(true);
              }}
              onToggleDisable={handleToggleDisable}
              onInteractiveChange={() => setInteractiveTick((prev) => prev + 1)}
              onDelete={(idx) => {
                setChips(chips.filter((_, ci) => ci !== idx));
                if (editingIdx === idx) {
                  setEditingIdx(null);
                  setOpen(false);
                }
              }}
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
