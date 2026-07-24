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

export function hasValue(value: unknown): boolean {
  return value !== undefined && value !== null && value !== "";
}

export function getBounds(items: any[], attribute: string): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;

  items?.forEach((item) => {
    const value = Number(item?.attributes?.[attribute]);
    if (!Number.isNaN(value)) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
  });

  return min === Infinity ? { min: 0, max: 0 } : { min, max };
}

export class ColorMap {
  private readonly min: number;
  private readonly max: number;
  private readonly attribute: string;

  constructor(items: any, attribute: string) {
    const bounds = getBounds(items, attribute);
    this.min = bounds.min;
    this.max = bounds.max;
    this.attribute = attribute;
  }
  getContinuous(item: any) {
    const value = Number(item?.attributes?.[this.attribute]);
    if (Number.isNaN(value)) return "#000000";
    const t = (value - this.min) / (this.max - this.min);
    return `hsl(${(1 - t) * 240}, 90%, 60%)`;
  }
  getCategories(item: any) {
    let hash = 0;
    const text = String(item?.attributes?.[this.attribute]);
    for (let i = 0; i < text.length; i++) {
      hash = text.charCodeAt(i) + ((hash << 5) - hash);
    }
    return `hsl(${Math.abs(hash * 131) % 360}, 90%, 60%)`;
  }
}

export abstract class BaseChip {
  abstract type: string;
  disabled: boolean;
  bg!: string;
  text!: string;

  static formFields: FormFieldConfig[];

  constructor(_data: ChipData, disabled = false, bg: string, text: string) {
    this.disabled = disabled;
    this.bg = bg;
    this.text = text;
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
    super(data, disabled, bg, text);
    this.attribute = data.attribute || "";
  }

  static getInitialData(attribute: string, _rawItems: any[], _previousData?: ChipData): ChipData {
    return { attribute };
  }

  getFormData() {
    return {
      attribute: this.attribute,
      type: this.type,
      disabled: String(this.disabled),
    };
  }
}
