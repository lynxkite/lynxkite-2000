import * as echarts from "echarts";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { useDisplay } from "../../common.ts";
import ChipForm from "./chips/ChipForm";
import {
  type BaseChip,
  type ChipApplyContext,
  getActiveRenderer,
  getChipClass,
} from "./chips/Chips.tsx";
import VisualChip from "./chips/VisualChip";
import LynxKiteNode from "./LynxKiteNode.tsx";

const USER_SELECT_NONE_STYLE: React.CSSProperties = {
  userSelect: "none",
  WebkitUserSelect: "none",
};

const THEME = {
  border: "#e2e8f0",
  deleteBtn: { bg: "#fee2e2", text: "#ef4444", hoverBg: "#fecaca" },
};

const collectAttrs = (items: any[]): string[] => {
  const keys = new Set<string>();
  items.forEach((item) => {
    if (item?.attributes) {
      Object.keys(item.attributes).forEach((k) => {
        keys.add(k);
      });
    }
  });
  return Array.from(keys);
};

const getLinks = (series: any): any[] => series?.links || [];

const copySeries = (series: any) => {
  if (!series) return null;
  return {
    ...series,
    data: (series.data || []).map((node: any) => ({
      ...node,
      label: { ...node.label },
      itemStyle: { ...node.itemStyle },
    })),
    links: getLinks(series).map((edge: any) => ({
      ...edge,
      lineStyle: { ...edge.lineStyle },
    })),
  };
};

export function NodeWithVisualization({ data, id }: { data: any; id: string }) {
  const echartsRef = useRef<HTMLDivElement>(null);
  const surfaceDivRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const viewOpts = useDisplay(data?.display_version, id);

  const [chips, setChips] = useState<BaseChip[]>([]);
  const [open, setOpen] = useState(false);
  const [nodeAttrs, setNodeAttrs] = useState<string[]>([]);
  const [edgeAttrs, setEdgeAttrs] = useState<string[]>([]);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [mainBtnHover, setMainBtnHover] = useState(false);
  const [interactiveTick, setInteractiveTick] = useState(0);
  const chipRef = useRef<BaseChip[]>([]);

  const activeRenderer = getActiveRenderer(chips);
  const usesCustomRenderer = activeRenderer !== "echarts";

  useEffect(() => {
    chipRef.current = chips;
  }, [chips]);

  useEffect(() => {
    setNodeAttrs(collectAttrs(viewOpts?.series?.[0]?.data || []));
    setEdgeAttrs(collectAttrs(getLinks(viewOpts?.series?.[0])));
  }, [viewOpts]);

  useEffect(() => {
    if (!viewOpts) return;

    const chartOpts = JSON.parse(JSON.stringify(viewOpts));
    const series = copySeries(chartOpts.series?.[0]);
    if (series) chartOpts.series[0] = series;

    const context: ChipApplyContext = {
      renderer: activeRenderer,
      series,
      surfaceDiv: surfaceDivRef.current,
    };

    chips
      .filter((c) => !c.disabled)
      .slice()
      .sort((a, b) => a.getApplyOrder() - b.getApplyOrder())
      .forEach((c) => {
        c.apply(context);
      });

    if (!usesCustomRenderer && echartsRef.current) {
      if (!chartRef.current) {
        chartRef.current = echarts.init(echartsRef.current, null, { renderer: "canvas" });
      }
      chartRef.current.setOption(chartOpts, true);

      const obs = new ResizeObserver(() => chartRef.current?.resize());
      obs.observe(echartsRef.current);
      return () => obs.disconnect();
    }
  }, [viewOpts, chips, usesCustomRenderer, activeRenderer, interactiveTick]);

  useEffect(() => {
    return () => {
      chartRef.current?.dispose();
      chartRef.current = null;
      chipRef.current.forEach((chip) => {
        chip.cleanup();
      });
    };
  }, []);

  const saveChip = (newChip: BaseChip) => {
    if (editingIdx !== null) {
      const updated = [...chips];
      updated[editingIdx]?.cleanup();
      updated[editingIdx] = newChip;
      setChips(updated);
      setEditingIdx(null);
    } else {
      setChips([...chips, newChip]);
    }
    setOpen(false);
  };

  const toggleChip = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    const updated = [...chips];
    const current = updated[index];
    updated[index]?.cleanup();
    const TargetClass = getChipClass(current.type);
    updated[index] = new TargetClass(current.getFormData(), !current.disabled);
    setChips(updated);
  };

  const hasAttrs = nodeAttrs.length > 0 || edgeAttrs.length > 0;
  const rawNodes = viewOpts?.series?.[0]?.data || [];
  const rawEdges = getLinks(viewOpts?.series?.[0]);

  return (
    <div
      style={{ flex: 1, position: "relative", width: "100%", height: "100%", minHeight: "350px" }}
    >
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          zIndex: 999,
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
          ...USER_SELECT_NONE_STYLE,
        }}
      >
        {hasAttrs && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setOpen(!open);
              setEditingIdx(null);
            }}
            onMouseEnter={() => setMainBtnHover(true)}
            onMouseLeave={() => setMainBtnHover(false)}
            style={{
              width: 28,
              height: 28,
              borderRadius: 20,
              border: `1px solid ${open ? `${THEME.deleteBtn.text}40` : THEME.border}`,
              background: open
                ? mainBtnHover
                  ? THEME.deleteBtn.hoverBg
                  : THEME.deleteBtn.bg
                : "#fff",
              color: open ? THEME.deleteBtn.text : "#555",
              cursor: "pointer",
              fontWeight: "bold",
              fontSize: 16,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 0,
              outline: "none",
              boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
              transition: "background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease",
            }}
          >
            {open ? "×" : "+"}
          </button>
        )}

        {open && hasAttrs && (
          <ChipForm
            nodeAttrs={nodeAttrs}
            edgeAttrs={edgeAttrs}
            initialChip={editingIdx !== null ? chips[editingIdx] : null}
            onSubmit={saveChip}
            rawElements={{ nodes: rawNodes, edges: rawEdges }}
          />
        )}

        {chips.map((c, i) => {
          if (editingIdx === i) return null;
          return (
            <VisualChip
              key={i}
              chip={c}
              index={i}
              onEdit={(e) => {
                e.stopPropagation();
                setEditingIdx(i);
                setOpen(true);
              }}
              onToggleDisable={toggleChip}
              onInteractiveChange={() => setInteractiveTick((prev) => prev + 1)}
              onDelete={(idx) => {
                chips[idx]?.cleanup();
                setChips(chips.filter((_, ci) => ci !== idx));
                if (editingIdx === idx) {
                  setEditingIdx(null);
                  setOpen(false);
                }
              }}
            />
          );
        })}
      </div>

      <div
        ref={echartsRef}
        style={{
          width: "100%",
          height: "100%",
          minHeight: "350px",
          display: usesCustomRenderer ? "none" : "block",
          ...USER_SELECT_NONE_STYLE,
        }}
      />

      <div
        ref={surfaceDivRef}
        style={{
          width: "100%",
          height: "100%",
          minHeight: "350px",
          display: usesCustomRenderer ? "block" : "none",
        }}
      />
    </div>
  );
}

export default LynxKiteNode(NodeWithVisualization);
