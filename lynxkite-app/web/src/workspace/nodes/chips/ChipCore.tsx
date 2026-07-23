import type React from "react";

export interface FormFieldConfig {
  key: string;
  label?: string;
  type?: string;
}

export interface ChipFormRenderContext {
  formData: ChipData;
  setFormData: React.Dispatch<React.SetStateAction<ChipData>>;
}

export interface ChipApplyContext {
  renderer: string;
  series: any;
  surfaceDiv: HTMLDivElement | null;
}

export type ChipData = Record<string, string>;

export interface ChipClass {
  new (data: ChipData, disabled?: boolean): BaseChip;
  type: string;
  displayName: string;
  target: string;
  formFields: FormFieldConfig[];
  getInitialData(attribute: string, rawItems: any[], previousData?: ChipData): ChipData;
  initFormData?: (formData: ChipData) => ChipData;
  getFormFieldLabel?: (field: FormFieldConfig, formData: ChipData) => string | undefined;
  renderFormExtra?: (context: ChipFormRenderContext) => React.ReactNode;
}

export const ATTRIBUTE_FIELD: FormFieldConfig[] = [{ key: "attribute" }];

export const hasValue = (value: unknown): boolean =>
  value !== undefined && value !== null && value !== "";

export const colorFromText = (text: string): string => {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash);
  }
  return `hsl(${Math.abs(hash * 131) % 360}, 80%, 60%)`;
};

export abstract class BaseChip {
  abstract type: string;
  disabled: boolean;
  bg!: string;
  text!: string;

  static formFields: FormFieldConfig[];

  constructor(_data: ChipData, disabled = false) {
    this.disabled = disabled;
  }

  static getInitialData(attribute: string, _rawItems: any[], previousData?: ChipData): ChipData {
    const data: ChipData = {};
    BaseChip.formFields.forEach((fieldConfig) => {
      const key = fieldConfig.key;
      data[key] = key === "attribute" ? attribute : previousData?.[key] || "";
    });
    return data;
  }

  abstract getLabel(): string;
  abstract getFormData(): ChipData;
  abstract apply(context: ChipApplyContext): void;

  getApplyOrder(): number {
    return 0;
  }

  getRenderer(): string {
    return "echarts";
  }

  cleanup(): void {}

  render(_onChange?: () => void): React.ReactNode {
    return null;
  }
}

export abstract class SingleAttributeChip extends BaseChip {
  attribute: string;

  constructor(data: ChipData, disabled: boolean | undefined, bg: string, text: string) {
    super(data, disabled);
    this.attribute = data.attribute || "";
    this.bg = bg;
    this.text = text;
  }

  static getInitialData(attribute: string, _rawItems: any[], _previousData?: ChipData): ChipData {
    return { attribute };
  }

  getFormData() {
    return { attribute: this.attribute };
  }
}
