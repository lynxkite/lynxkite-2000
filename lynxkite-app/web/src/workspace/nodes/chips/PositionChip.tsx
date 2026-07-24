import L from "leaflet";
import type React from "react";
import "leaflet/dist/leaflet.css";
import { BaseChip, type ChipApplyContext, type ChipData, type FormFieldConfig } from "./ChipCore";

export class PositionChip extends BaseChip {
  static type = "position";
  type = PositionChip.type;
  static displayName = "Position";
  static target = "node";
  static formFields: FormFieldConfig[] = [
    { key: "xAttr", label: "X:" },
    { key: "yAttr", label: "Y:" },
  ];

  mode: string;
  xAttr: string;
  yAttr: string;

  private _map: L.Map | null = null;
  private _mapDiv: HTMLDivElement | null = null;
  private _resizeObserver: ResizeObserver | null = null;

  private static getMode(data: ChipData): string {
    return data.mode === "map" ? "map" : "xy";
  }

  constructor(data: ChipData, disabled?: boolean) {
    super(data, disabled, "#b4fbb6", "#045b15");
    this.mode = PositionChip.getMode(data);
    this.xAttr = data.xAttr || "";
    this.yAttr = data.yAttr || "";
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
      type: this.type,
      disabled: String(this.disabled),
    };
  }

  override getRenderer(): string {
    return this.mode === "map" ? "leaflet" : "echarts";
  }

  override cleanup(): void {
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this._map?.remove();
    this._map = null;
    this._mapDiv = null;
  }

  private toggleMode() {
    const wasMap = this.mode === "map";
    this.mode = wasMap ? "xy" : "map";
    if (wasMap) this.cleanup();
  }

  private applyMap(context: ChipApplyContext) {
    const { surfaceDiv, series } = context;
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
      L.control.zoom({ position: "bottomleft" }).addTo(this._map);
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

    const nodePositions = new Map<string, L.LatLng>();
    series.data.forEach((node: any) => {
      const lng = Number(node.attributes?.[this.xAttr]);
      const lat = Number(node.attributes?.[this.yAttr]);
      if (!Number.isNaN(lat) && !Number.isNaN(lng)) {
        nodePositions.set(String(node.id ?? node.name), L.latLng(lat, lng));
      }
    });

    (series.links || []).forEach((edge: any) => {
      const source = nodePositions.get(String(edge.source));
      const target = nodePositions.get(String(edge.target));
      if (!source || !target) return;
      L.polyline([source, target], {
        color: edge.lineStyle?.color || "#94a3b8",
        weight: 1.5,
        opacity: 0.7,
      }).addTo(map);
    });

    const points: L.LatLng[] = [];
    series.data.forEach((node: any) => {
      const point = nodePositions.get(String(node.id ?? node.name));
      if (!point) return;
      points.push(point);

      const labelText = node.label?.show ? String(node.label?.formatter ?? node.name ?? "") : "";

      const circle = L.circleMarker(point, {
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

    if (points.length > 0) {
      map.fitBounds(L.latLngBounds(points), { padding: [30, 30], maxZoom: 12 });
    }
  }

  private applyXY(series: any) {
    if (!series?.data || !this.xAttr || !this.yAttr) return;
    series.coordinateSystem = undefined;
    series.layout = "none";

    series.data.forEach((node: any) => {
      const x = Number(node.attributes?.[this.xAttr]);
      const y = Number(node.attributes?.[this.yAttr]);
      if (!Number.isNaN(x) && !Number.isNaN(y)) {
        node.x = x;
        node.y = y;
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
        {this.mode === "map" ? "Map" : "X/Y"}
      </button>
    );
  }
}
