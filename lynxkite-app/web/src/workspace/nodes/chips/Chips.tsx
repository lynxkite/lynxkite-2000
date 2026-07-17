import type React from "react";

export interface FormFieldConfig {
  key: string;
  label?: string;
  type?: "select" | "number";
}

const getColor = (s: string): string => {
  let h = 0,
    t = String(s ?? "");
  for (let i = 0; i < t.length; i++) h = t.charCodeAt(i) + ((h << 5) - h);
  return `hsl(${Math.abs(h * 131) % 360}, 80%, 60%)`;
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
        node.itemStyle = { ...node.itemStyle, color: getColor(String(val)) };
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
        edge.lineStyle = { ...edge.lineStyle, color: getColor(String(val)) };
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

export const CHIP_REGISTRY = [NodeColorChip, EdgeColorChip, PositionChip, LabelChip, SliderChip];
