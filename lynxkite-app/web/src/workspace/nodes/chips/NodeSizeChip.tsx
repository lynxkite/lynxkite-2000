import type React from "react";
import { BaseChip, type ChipApplyContext, type ChipData, type FormFieldConfig } from "./ChipCore";

export class NodeSizeChip extends BaseChip {
  static type = "node_size";
  static displayName = "Set node size";
  static formFields: FormFieldConfig[] = [];

  type = NodeSizeChip.type;
  nodeSize: number;

  constructor(data: ChipData) {
    super(data, "#e0f2fe", "#0369a1");
    console.log(data.id);
    this.nodeSize = Number(data.symbolSize);
  }

  getLabel() {
    return "Node size:";
  }

  override getFormData(): ChipData {
    return {
      symbolSize: String(this.nodeSize),
      type: this.type,
      disabled: String(this.disabled),
    };
  }

  apply(context: ChipApplyContext) {
    context.series?.data?.forEach((node: any) => {
      if (this.nodeSize) {
        node.symbolSize = this.nodeSize;
      }
    });
  }

  override render(onChange?: () => void): React.ReactNode {
    return (
      <input
        min={1}
        max={100}
        value={this.nodeSize || undefined}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          this.nodeSize = Number(e.target.value);
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
