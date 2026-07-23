import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  colorFromText,
  type FormFieldConfig,
  hasValue,
  SingleAttributeChip,
} from "./ChipCore";

export class NodeColorChip extends SingleAttributeChip {
  static type = "node_color";
  type = NodeColorChip.type;
  static displayName = "Node color by";
  static target = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled, "#e0f2fe", "#0369a1");
  }

  getLabel() {
    return `Node color by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    context.series?.data?.forEach((node: any) => {
      const value = node.attributes?.[this.attribute];
      if (hasValue(value)) {
        node.itemStyle = { ...node.itemStyle, color: colorFromText(String(value)) };
      }
    });
  }
}
