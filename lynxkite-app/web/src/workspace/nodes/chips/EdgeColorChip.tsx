import {
  ATTRIBUTE_FIELD,
  type ChipApplyContext,
  ColorMap,
  type FormFieldConfig,
  hasValue,
  ToggleChip,
} from "./ChipCore";

export class EdgeColorChip extends ToggleChip {
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

  override toggleMode(): void {
    this.continuous = !this.continuous;
  }

  override toggleText(): string {
    return this.continuous ? "Continuous" : "Categorical";
  }
}
