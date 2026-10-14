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
      let text = "";
      if (value !== undefined && value !== null) {
        text = String(value);
        if (value !== "" && !Number.isNaN(Number(value))) {
          text = String(Math.round(Number(value) * 100) / 100);
        }
      }
      node.tooltip = {
        ...node.tooltip,
        show: true,
        formatter: text,
      };
    });
  }
}
