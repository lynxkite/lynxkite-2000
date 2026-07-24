import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  ColorMap,
  type FormFieldConfig,
  hasValue,
  ToggleChip,
} from "./ChipCore";

export class NodeColorChip extends ToggleChip {
  static type = "node_color";
  static displayName = "Node color by";
  static target = "node";
  static formFields: FormFieldConfig[] = ATTRIBUTE_FIELD;
  type = NodeColorChip.type;
  continuous = false;
  constructor(data: Record<string, string>, disabled?: boolean) {
    super(data, disabled, "#e0f2fe", "#0369a1");
    this.continuous = data.continuous === "true";
  }

  getLabel() {
    return `Node color by: ${this.attribute}`;
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

  override toggleMode(): void {
    this.continuous = !this.continuous;
  }

  override toggleText(): string {
    return this.continuous ? "Continuous" : "Categorical";
  }
}
