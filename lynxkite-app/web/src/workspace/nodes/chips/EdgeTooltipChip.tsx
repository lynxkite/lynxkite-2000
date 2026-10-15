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
      const text =
        value === "" || Number.isNaN(Number(value)) ? value : String(+Number(value).toFixed(2));
      edge.tooltip = {
        ...edge.tooltip,
        show: true,
        formatter: text,
      };
    });
  }
}
