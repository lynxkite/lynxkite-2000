export type {
  ChipApplyContext,
  ChipData,
  ChipFormRenderContext,
  FormFieldConfig,
} from "./ChipCore";
export { BaseChip, type ChipClass } from "./ChipCore";
export { EdgeColorChip } from "./EdgeColorChip";
export { LabelChip } from "./LabelChip";
export { NodeColorChip } from "./NodeColorChip";
export { PositionChip } from "./PositionChip";
export { SliderChip } from "./SliderChip";

import type { BaseChip, ChipClass } from "./ChipCore";
import { EdgeColorChip } from "./EdgeColorChip";
import { LabelChip } from "./LabelChip";
import { NodeColorChip } from "./NodeColorChip";
import { PositionChip } from "./PositionChip";
import { SliderChip } from "./SliderChip";

export const CHIP_REGISTRY: ChipClass[] = [
  NodeColorChip,
  EdgeColorChip,
  PositionChip,
  LabelChip,
  SliderChip,
];

export const CHIP_CLASS_BY_TYPE = new Map<string, ChipClass>([
  ...CHIP_REGISTRY.map((chipClass) => [chipClass.type, chipClass] as [string, ChipClass]),
  ["map", PositionChip],
]);

export const normalizeChipType = (type?: string): string | undefined =>
  type === "map" ? PositionChip.type : type;

export const getChipClass = (type: string): ChipClass =>
  CHIP_CLASS_BY_TYPE.get(type) || CHIP_REGISTRY[0];

export const getActiveRenderer = (chips: BaseChip[]): string =>
  chips.find((chip) => !chip.disabled && chip.getRenderer() !== "echarts")?.getRenderer() ||
  "echarts";
