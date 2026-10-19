// A chip for setting the graph node hover tooltips to an attribute.
import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  type FormFieldConfig,
  SingleAttributeChip,
} from "./ChipCore";

export class NodeTooltipChip extends SingleAttributeChip {
  static type = "node_tooltip";
  type = NodeTooltipChip.type;
  static displayName = "Node tooltip by";
  static target = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;

  constructor(data: Record<string, string>) {
    super(data, "#dbeafe", "#1d4ed8");
  }

  getLabel() {
    return `Node tooltip by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;

    context.series?.data?.forEach((node: any) => {
      const value = node.attributes?.[this.attribute];
      const text =
        value === "" || Number.isNaN(Number(value)) ? value : String(+Number(value).toFixed(2));
      node.tooltip = {
        ...node.tooltip,
        show: true,
        formatter: text,
      };
    });
  }
}
