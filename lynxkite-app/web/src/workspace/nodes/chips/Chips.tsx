// Central chip registry, and helper functions
import type { BaseChip, ChipClass } from "./ChipCore";
import { EdgeColorChip } from "./EdgeColorChip";
import { EdgeWidthChip } from "./EdgeWidthChip.tsx";
import { LabelChip } from "./LabelChip";
import { NodeColorChip } from "./NodeColorChip";
import { NodeSizeChip } from "./NodeSizeChip.tsx";
import { PositionChip } from "./PositionChip";
import { SliderChip } from "./SliderChip";

export const CHIP_REGISTRY: ChipClass[] = [
  NodeColorChip,
  EdgeColorChip,
  PositionChip,
  LabelChip,
  SliderChip,
  NodeSizeChip,
  EdgeWidthChip,
];

export const CHIP_CLASS_BY_TYPE = new Map<string, ChipClass>([
  ...CHIP_REGISTRY.map((chipClass) => [chipClass.type, chipClass] as [string, ChipClass]),
  ["map", PositionChip],
]);

export const normalizeChipType = (type?: string): string | undefined =>
  type === "map" ? PositionChip.type : type;

export const getChipClass = (type: string): ChipClass => {
  const ChipClass = CHIP_CLASS_BY_TYPE.get(type);
  if (!ChipClass) {
    throw new Error(`Unknown chip type: ${type}`);
  }
  return ChipClass;
};

export const getActiveRenderer = (chips: BaseChip[]): string =>
  chips.find((chip) => !chip.disabled && chip.getRenderer() !== "echarts")?.getRenderer() ||
  "echarts";
