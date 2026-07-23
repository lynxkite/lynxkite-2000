import type React from "react";
import { BaseChip, type ChipApplyContext, type ChipData, type FormFieldConfig } from "./ChipCore";

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

  private static getBounds(items: any[], attribute: string): { min: number; max: number } {
    let min = Infinity;
    let max = -Infinity;

    items?.forEach((item) => {
      const value = Number(item?.attributes?.[attribute]);
      if (!Number.isNaN(value)) {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    });

    return min === Infinity ? { min: 0, max: 0 } : { min, max };
  }

  constructor(data: ChipData, disabled?: boolean) {
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
    previousData?: ChipData,
  ): ChipData {
    const bounds = SliderChip.getBounds(rawItems, attribute);
    const sameAttr = previousData?.attribute === attribute;

    return {
      attribute,
      min: String(bounds.min),
      max: String(bounds.max),
      currentMin:
        sameAttr && previousData?.currentMin !== undefined
          ? previousData.currentMin
          : String(bounds.min),
      currentMax:
        sameAttr && previousData?.currentMax !== undefined
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

  apply(context: ChipApplyContext) {
    const series = context.series;
    if (!series?.data || !this.attribute) return;

    const keptIds = new Set<string>();
    series.data = series.data.filter((node: any) => {
      const value = Number(node.attributes?.[this.attribute]);
      if (Number.isNaN(value)) return true;
      const keep = value >= this.currentMin && value <= this.currentMax;
      if (keep) keptIds.add(node.id);
      return keep;
    });

    if (series.links) {
      series.links = series.links.filter(
        (edge: any) => keptIds.has(edge.source) && keptIds.has(edge.target),
      );
    }
  }

  override render(onChange?: () => void) {
    const onMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      this.currentMin = Math.min(Number(e.target.value), this.currentMax);
      onChange?.();
    };

    const onMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      this.currentMax = Math.max(Number(e.target.value), this.currentMin);
      onChange?.();
    };

    const diff = this.limitMax - this.limitMin || 1;
    const left = ((this.currentMin - this.limitMin) / diff) * 100;
    const right = ((this.currentMax - this.limitMin) / diff) * 100;

    const rangeStyle = (isMin: boolean): React.CSSProperties => ({
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
              background: `linear-gradient(to right, #fca5a5 ${left}%, ${this.text} ${left}%, ${this.text} ${right}%, #fca5a5 ${right}%)`,
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
            onChange={onMinChange}
            style={rangeStyle(true)}
          />
          <input
            type="range"
            min={this.limitMin}
            max={this.limitMax}
            value={this.currentMax}
            onChange={onMaxChange}
            style={rangeStyle(false)}
          />
        </div>

        <span style={{ fontSize: 9, fontFamily: "monospace", opacity: 0.8 }}>
          {this.currentMax}
        </span>
      </div>
    );
  }
}
