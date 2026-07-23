import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  type FormFieldConfig,
  hasValue,
  SingleAttributeChip,
} from "./ChipCore";

export class LabelChip extends SingleAttributeChip {
  static type = "label";
  type = LabelChip.type;
  static displayName = "Label by";
  static target = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;

  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled, "#fef08a", "#a16207");
  }

  getLabel() {
    return `Label by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;
    context.series?.data?.forEach((node: any) => {
      const value = node.attributes?.[this.attribute];
      const show = hasValue(value);
      node.label = {
        ...node.label,
        show,
        formatter: show ? String(value) : "",
        position: "top",
      };
    });
  }
}
