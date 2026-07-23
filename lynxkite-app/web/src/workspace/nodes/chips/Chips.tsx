import L from "leaflet";
import type React from "react";
import "leaflet/dist/leaflet.css";

export interface FormFieldConfig {
  key: string;
  label?: string;
  type?: "select" | "number";
}

export interface ChipFormRenderContext {
  formData: ChipData;
  setFormData: React.Dispatch<React.SetStateAction<ChipData>>;
}

export type ChipTarget = "node" | "edge";
export type ChipRenderer = "echarts" | string;

export interface ChipApplyContext {
  renderer: ChipRenderer;
  series: any;
  surfaceDiv: HTMLDivElement | null;
}

export type ChipData = Record<string, string>;
export type PositionMode = "xy" | "map";

export interface ChipClass {
  new (data: ChipData, disabled?: boolean): BaseChip;
  type: string;
  displayName: string;
  target: ChipTarget;
  formFields: FormFieldConfig[];
  getInitialData(attribute: string, rawItems: any[], previousData?: ChipData): ChipData;
  initFormData?: (formData: ChipData) => ChipData;
  getFormFieldLabel?: (field: FormFieldConfig, formData: ChipData) => string | undefined;
  renderFormExtra?: (context: ChipFormRenderContext) => React.ReactNode;
}

const ATTRIBUTE_FORM_FIELDS: FormFieldConfig[] = [{ key: "attribute" }];

const hasAttributeValue = (value: unknown): boolean =>
  value !== undefined && value !== null && value !== "";

const getColor = (s: string): string => {
  let h = 0;
  const t = String(s ?? "");
  for (let i = 0; i < t.length; i++) {
    h = t.charCodeAt(i) + ((h << 5) - h);
  }
  return `hsl(${Math.abs(h * 131) % 360}, 80%, 60%)`;
};

const getPositionMode = (data: ChipData): PositionMode =>
  data.mode === "map" || (data.latAttr !== undefined && data.lngAttr !== undefined) ? "map" : "xy";

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

  static formFields: FormFieldConfig[];

  constructor(_data: ChipData, disabled = false) {
    this.disabled = disabled;
  }

  static getInitialData(attribute: string, _rawItems: any[], previousData?: ChipData): ChipData {
    const data: ChipData = {};
    BaseChip.formFields.forEach((fieldConfig) => {
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
  abstract getFormData(): ChipData;
  abstract apply(context: ChipApplyContext): void;

  getApplyOrder(): number {
    return 0;
  }

  getRenderer(): ChipRenderer {
    return "echarts";
  }

  cleanup(): void {}

  render(_onChange?: () => void): React.ReactNode {
    return null;
  }
}

abstract class SingleAttributeChip extends BaseChip {
  attribute: string;

  constructor(data: ChipData, disabled: boolean | undefined, bg: string, text: string) {
    super(data, disabled);
    this.attribute = data.attribute || "";
    this.bg = bg;
    this.text = text;
  }

  getFormData() {
    return { attribute: this.attribute };
  }
}

export class NodeColorChip extends SingleAttributeChip {
  static type = "node_color";
  type = NodeColorChip.type;
  static displayName = "Node color by";
  static target: ChipTarget = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FORM_FIELDS;

  constructor(data: ChipData, disabled?: boolean) {
    super(data, disabled, "#e0f2fe", "#0369a1");
  }

  getLabel() {
    return `Node color by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    context.series?.data?.forEach((node: any) => {
      const val = node.attributes?.[this.attribute];
      if (hasAttributeValue(val)) {
        node.itemStyle = { ...node.itemStyle, color: getColor(String(val)) };
      }
    });
  }
}

export class EdgeColorChip extends SingleAttributeChip {
  static type = "edgeColor";
  type = EdgeColorChip.type;
  static displayName = "Edge color by";
  static target: ChipTarget = "edge";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FORM_FIELDS;

  constructor(data: ChipData, disabled?: boolean) {
    super(data, disabled, "#fae8ff", "#86198f");
  }

  getLabel() {
    return `Edge color by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    context.series?.links?.forEach((edge: any) => {
      const val = edge.attributes?.[this.attribute];
      if (hasAttributeValue(val)) {
        edge.lineStyle = { ...edge.lineStyle, color: getColor(String(val)) };
      }
    });
  }
}

export class PositionChip extends BaseChip {
  static type = "position";
  type = PositionChip.type;
  static displayName = "Position";
  static target: ChipTarget = "node";
  static formFields: FormFieldConfig[] = [
    { key: "xAttr", label: "X:" },
    { key: "yAttr", label: "Y:" },
  ];

  mode: PositionMode;
  xAttr: string;
  yAttr: string;

  private _map: L.Map | null = null;
  private _mapDiv: HTMLDivElement | null = null;
  private _resizeObserver: ResizeObserver | null = null;

  constructor(data: ChipData, disabled?: boolean) {
    super(data, disabled);
    this.mode = getPositionMode(data);
    this.xAttr = data.xAttr || data.lngAttr || "";
    this.yAttr = data.yAttr || data.latAttr || "";
    this.setColors();
  }

  getLabel() {
    return this.mode === "map"
      ? `Position: Lon(${this.xAttr}) Lat(${this.yAttr})`
      : `Position: X(${this.xAttr}) Y(${this.yAttr})`;
  }

  getFormData() {
    return {
      mode: this.mode,
      xAttr: this.xAttr,
      yAttr: this.yAttr,
      lngAttr: this.xAttr,
      latAttr: this.yAttr,
    };
  }

  override getRenderer(): ChipRenderer {
    return this.mode === "map" ? "leaflet" : "echarts";
  }

  override cleanup(): void {
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this._map?.remove();
    this._map = null;
    this._mapDiv = null;
  }

  private setColors() {
    if (this.mode === "map") {
      this.bg = "#ccfbf1";
      this.text = "#0f766e";
      return;
    }
    this.bg = "#e2ffac";
    this.text = "rgb(18 93 53 / 0.78)";
  }

  private toggleMode() {
    const wasMap = this.mode === "map";
    this.mode = wasMap ? "xy" : "map";
    this.setColors();
    if (wasMap) this.cleanup();
  }

  private applyMap(context: ChipApplyContext) {
    const surfaceDiv = context.surfaceDiv;
    const series = context.series;
    if (
      context.renderer !== this.getRenderer() ||
      !surfaceDiv ||
      !series?.data ||
      !this.xAttr ||
      !this.yAttr
    )
      return;

    if (!this._map || this._mapDiv !== surfaceDiv) {
      this._map?.remove();
      this._map = L.map(surfaceDiv, { zoomControl: false });
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(this._map);
      this._mapDiv = surfaceDiv;
    }

    const map = this._map;

    this._resizeObserver?.disconnect();
    this._resizeObserver = new ResizeObserver(() => map.invalidateSize());
    this._resizeObserver.observe(surfaceDiv);

    map.eachLayer((layer) => {
      if (!(layer instanceof L.TileLayer)) map.removeLayer(layer);
    });

    const edges: any[] = series.links || [];

    const nodePositions = new Map<string, L.LatLng>();
    series.data.forEach((node: any) => {
      const lng = Number(node.attributes?.[this.xAttr]);
      const lat = Number(node.attributes?.[this.yAttr]);
      if (!Number.isNaN(lat) && !Number.isNaN(lng)) {
        nodePositions.set(String(node.id ?? node.name), L.latLng(lat, lng));
      }
    });

    edges.forEach((edge: any) => {
      const src = nodePositions.get(String(edge.source));
      const tgt = nodePositions.get(String(edge.target));
      if (!src || !tgt) return;
      L.polyline([src, tgt], {
        color: edge.lineStyle?.color || "#94a3b8",
        weight: 1.5,
        opacity: 0.7,
      }).addTo(map);
    });

    const validPoints: L.LatLng[] = [];
    series.data.forEach((node: any) => {
      const pt = nodePositions.get(String(node.id ?? node.name));
      if (!pt) return;
      validPoints.push(pt);

      const labelText = node.label?.show ? String(node.label?.formatter ?? node.name ?? "") : "";

      const circle = L.circleMarker(pt, {
        radius: 7,
        fillColor: node.itemStyle?.color || "#4f46e5",
        color: "#fff",
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.85,
      });

      if (labelText) {
        circle.bindTooltip(labelText, {
          permanent: true,
          direction: "top",
          offset: [0, -8],
          className: "lk-map-node-label",
        });
      }

      circle.addTo(map);
    });

    if (validPoints.length > 0) {
      try {
        map.fitBounds(L.latLngBounds(validPoints), { padding: [30, 30], maxZoom: 12 });
      } catch (_) {}
    }
  }

  private applyXY(series: any) {
    if (!series?.data || !this.xAttr || !this.yAttr) return;

    series.coordinateSystem = undefined;
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

  apply(context: ChipApplyContext) {
    if (this.mode === "map") {
      this.applyMap(context);
      return;
    }
    this.applyXY(context.series);
  }

  override getApplyOrder(): number {
    return this.mode === "map" ? 100 : 0;
  }

  override render(onChange?: () => void): React.ReactNode {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          this.toggleMode();
          onChange?.();
        }}
        style={{
          border: "none",
          borderRadius: 999,
          padding: "2px 8px",
          fontSize: 10,
          fontWeight: 700,
          cursor: "pointer",
          background: "rgba(255,255,255,0.55)",
          color: this.text,
          textTransform: "uppercase",
          letterSpacing: 0.3,
        }}
      >
        {this.mode === "map" ? "Switch to X/Y" : "Switch to Lon/Lat"}
      </button>
    );
  }
}

export class LabelChip extends SingleAttributeChip {
  static type = "label";
  type = LabelChip.type;
  static displayName = "Label by";
  static target: ChipTarget = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FORM_FIELDS;

  constructor(data: ChipData, disabled?: boolean) {
    super(data, disabled, "#fef08a", "#a16207");
  }

  getLabel() {
    return `Label by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    context.series?.data?.forEach((node: any) => {
      const val = node.attributes?.[this.attribute];
      const hasValue = hasAttributeValue(val);
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
  static target: ChipTarget = "node";
  static formFields: FormFieldConfig[] = [{ key: "attribute" }];

  attribute: string;
  limitMin: number;
  limitMax: number;
  currentMin: number;
  currentMax: number;

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

  apply(context: ChipApplyContext) {
    const series = context.series;
    if (!series?.data || !this.attribute) return;
    const keptNodeIds = new Set<string>();

    series.data = series.data.filter((node: any) => {
      const val = Number(node.attributes?.[this.attribute]);
      if (Number.isNaN(val)) return true;
      const keep = val >= this.currentMin && val <= this.currentMax;
      if (keep) keptNodeIds.add(node.id);
      return keep;
    });

    if (series.links) {
      series.links = series.links.filter(
        (edge: any) => keptNodeIds.has(edge.source) && keptNodeIds.has(edge.target),
      );
    }
  }

  override render(onChange?: () => void) {
    const handleMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      this.currentMin = Math.min(Number(e.target.value), this.currentMax);
      if (onChange) onChange();
    };

    const handleMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      this.currentMax = Math.max(Number(e.target.value), this.currentMin);
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

export const CHIP_REGISTRY: ChipClass[] = [
  NodeColorChip,
  EdgeColorChip,
  PositionChip,
  LabelChip,
  SliderChip,
];

export const CHIP_CLASS_BY_TYPE = new Map<string, ChipClass>([
  ...CHIP_REGISTRY.map((chipClass) => [chipClass.type, chipClass] as [string, ChipClass]),
  ["map", PositionChip],
]);

export const normalizeChipType = (type?: string): string | undefined =>
  type === "map" ? PositionChip.type : type;

export const getChipClass = (type: string): ChipClass =>
  CHIP_CLASS_BY_TYPE.get(type) || CHIP_REGISTRY[0];

export const getActiveRenderer = (chips: BaseChip[]): ChipRenderer =>
  chips.find((chip) => !chip.disabled && chip.getRenderer() !== "echarts")?.getRenderer() ||
  "echarts";
