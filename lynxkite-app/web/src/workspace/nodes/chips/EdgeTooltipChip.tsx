// A chip for setting the graph edge hover tooltips to an attribute.
import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  type FormFieldConfig,
  SingleAttributeChip,
} from "./ChipCore";

export class EdgeTooltipChip extends SingleAttributeChip {
  static type = "edge_tooltip";
  type = EdgeTooltipChip.type;
  static displayName = "Edge tooltip by";
  static target = "edge";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;

  constructor(data: Record<string, string>) {
    super(data, "#fae8ff", "#86198f");
  }

  getLabel() {
    return `Edge tooltip by: ${this.attribute}`;
  }

  apply(context: ChipApplyContext) {
    if (!this.attribute) return;

    context.series?.links?.forEach((edge: any) => {
      const value = edge.attributes?.[this.attribute];
      let text = "";
      if (value !== undefined && value !== null) {
        text = String(value);
        if (value !== "" && !Number.isNaN(Number(value))) {
          text = String(Math.round(Number(value) * 100) / 100);
        }
      }
      edge.tooltip = {
        ...edge.tooltip,
        show: true,
        formatter: text,
      };
    });
  }
}
