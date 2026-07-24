import type React from "react";
import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  ColorMap,
  type FormFieldConfig,
  hasValue,
  SingleAttributeChip,
} from "./ChipCore";

export class EdgeColorChip extends SingleAttributeChip {
  static type = "edgeColor";
  type = EdgeColorChip.type;
  static displayName = "Edge color by";
  static target = "edge";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;
  private continuous: boolean = false;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled, "#fae8ff", "#86198f");
    this.continuous = data.continuous === "true";
  }

  getLabel() {
    return `Edge color by: ${this.attribute}`;
  }

  override getFormData() {
    return {
      attribute: this.attribute,
      continuous: String(this.continuous),
      type: this.type,
      disabled: String(this.disabled),
    };
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    const colormap = new ColorMap(context.series?.links, this.attribute);
    context.series?.links?.forEach((edge: any) => {
      const value = edge.attributes?.[this.attribute];
      if (hasValue(value)) {
        edge.lineStyle = {
          ...edge.lineStyle,
          color: this.continuous ? colormap.getContinuous(edge) : colormap.getCategories(edge),
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
