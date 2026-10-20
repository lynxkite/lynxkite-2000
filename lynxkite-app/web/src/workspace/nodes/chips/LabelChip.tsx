// A chip for labeling graph nodes with the value of an attribute.
import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  type FormFieldConfig,
  SingleAttributeChip,
} from "./ChipCore";

export class LabelChip extends SingleAttributeChip {
  static type = "label";
  type = LabelChip.type;
  static displayName = "Label by";
  static target = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;

  constructor(data: Record<string, string>) {
    super(data, "#fef08a", "#a16207");
  }

  getLabel() {
    return `Label by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;

    context.series?.data?.forEach((node: any) => {
      const value = node.attributes?.[this.attribute];
      const text =
        value === "" || Number.isNaN(Number(value)) ? value : String(+Number(value).toFixed(2));
      node.label = {
        ...node.label,
        show: true,
        formatter: text,
        position: "top",
      };
    });
  }
}
