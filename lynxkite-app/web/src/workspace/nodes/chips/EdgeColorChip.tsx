import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  colorFromText,
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

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled, "#fae8ff", "#86198f");
  }

  getLabel() {
    return `Edge color by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    context.series?.links?.forEach((edge: any) => {
      const value = edge.attributes?.[this.attribute];
      if (hasValue(value)) {
        edge.lineStyle = { ...edge.lineStyle, color: colorFromText(String(value)) };
      }
    });
  }
}
