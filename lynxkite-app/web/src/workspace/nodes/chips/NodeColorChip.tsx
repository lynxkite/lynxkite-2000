import type React from "react";
import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  ColorMap,
  type FormFieldConfig,
  hasValue,
  SingleAttributeChip,
} from "./ChipCore";

export class NodeColorChip extends SingleAttributeChip {
  static type = "node_color";
  static displayName = "Node color by";
  static target = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;
  type = NodeColorChip.type;
  continuous = false;
  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled, "#e0f2fe", "#0369a1");
  }

  getLabel() {
    return `Node color by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    const colormap = new ColorMap(context.series?.data, this.attribute);
    context.series?.data?.forEach((node: any) => {
      const value = node.attributes?.[this.attribute];
      if (hasValue(value)) {
        node.itemStyle = {
          ...node.itemStyle,
          color: this.continuous ? colormap.getContinuous(node) : colormap.getCategories(node),
        };
      }
    });
  }

  private toggleMode() {
    this.continuous = !this.continuous;
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
        {this.continuous ? "Continuous" : "Categorical"}
      </button>
    );
  }
}
