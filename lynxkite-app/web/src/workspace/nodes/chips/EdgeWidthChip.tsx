import type React from "react";
import { BaseChip, type ChipApplyContext, type ChipData, type FormFieldConfig } from "./ChipCore";

export class EdgeWidthChip extends BaseChip {
  static type = "edge_width";
  static displayName = "Set edge width";
  static formFields: FormFieldConfig[] = [];

  type = EdgeWidthChip.type;
  edgeWidth: number;

  constructor(data: ChipData) {
    super(data, "#e0f2fe", "#0369a1");
    this.edgeWidth = Number(data.edgeWidth);
  }

  getLabel() {
    return "Edge width:";
  }

  override getFormData(): ChipData {
    return {
      edgeWidth: String(this.edgeWidth),
      type: this.type,
      disabled: String(this.disabled),
    };
  }

  apply(context: ChipApplyContext) {
    if (!this.edgeWidth) return;
    context.series.lineStyle.width = this.edgeWidth;
  }

  override render(onChange?: () => void): React.ReactNode {
    return (
      <input
        min={1}
        max={100}
        value={this.edgeWidth || undefined}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          this.edgeWidth = Number(e.target.value);
          onChange?.();
        }}
        style={{
          width: 48,
          height: 24,
          textAlign: "center",
          alignItems: "center",
          fontSize: 12,
          fontWeight: 600,
          borderRadius: 10,
          border: "1px solid #cccccc",
          background: "#dddddd",
          color: this.text,
        }}
      />
    );
  }
}
