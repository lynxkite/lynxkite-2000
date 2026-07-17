import * as echarts from "echarts";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { useDisplay } from "../../common.ts";
import ChipForm from "./chips/ChipForm";
import { type BaseChip, CHIP_REGISTRY } from "./chips/Chips.tsx";
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

const extractAttributes = (items: any[]): string[] => {
  const keysSet = new Set<string>();
  items.forEach((item) => {
    if (item?.attributes) {
      Object.keys(item.attributes).forEach((k) => {
        keysSet.add(k);
      });
    }
  });
  return Array.from(keysSet);
};

export function NodeWithVisualization({ data, id }: { data: any; id: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const opts = useDisplay(data?.display_version, id);

  const [chips, setChips] = useState<BaseChip[]>([]);
  const [open, setOpen] = useState(false);
  const [nodeAttrs, setNodeAttrs] = useState<string[]>([]);
  const [edgeAttrs, setEdgeAttrs] = useState<string[]>([]);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [mainBtnHover, setMainBtnHover] = useState(false);

  const [interactiveTick, setInteractiveTick] = useState(0);

  useEffect(() => {
    setNodeAttrs(extractAttributes(opts?.series?.[0]?.data || []));
    setEdgeAttrs(extractAttributes(opts?.series?.[0]?.links || opts?.series?.[0]?.edges || []));
  }, [opts]);

  useEffect(() => {
    if (!opts || !ref.current) return;

    const chartOpts = JSON.parse(JSON.stringify(opts));
    const series = chartOpts.series?.[0];

    if (series) {
      if (series.data) {
        series.data = series.data.map((n: any) => ({
          ...n,
          label: { ...n.label },
          itemStyle: { ...n.itemStyle },
        }));
      }
      const edgeKey = series.links ? "links" : "edges";
      if (series[edgeKey]) {
        series[edgeKey] = series[edgeKey].map((e: any) => ({
          ...e,
          lineStyle: { ...e.lineStyle },
        }));
      }
      chips
        .filter((c) => !c.disabled)
        .forEach((c) => {
          c.apply(series);
        });
    }

    const inst = echarts.init(ref.current, null, { renderer: "canvas" });
    inst.setOption(chartOpts, true);

    const obs = new ResizeObserver(() => inst.resize());
    obs.observe(ref.current);

    return () => {
      obs.disconnect();
      inst.dispose();
    };
  }, [opts, chips, interactiveTick]);

  const handleFormSubmit = (newChip: BaseChip) => {
    if (editingIdx !== null) {
      const updated = [...chips];
      updated[editingIdx] = newChip;
      setChips(updated);
      setEditingIdx(null);
    } else {
      setChips([...chips, newChip]);
    }
    setOpen(false);
  };

  const handleToggleDisable = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    const updated = [...chips];
    const current = updated[index];
    const TargetClass = CHIP_REGISTRY.find((c) => c.type === current.type) || CHIP_REGISTRY[0];
    updated[index] = new TargetClass(current.getFormData(), !current.disabled);
    setChips(updated);
  };

  const hasAttributes = nodeAttrs.length > 0 || edgeAttrs.length > 0;

  const rawNodes = opts?.series?.[0]?.data || [];
  const rawEdges = opts?.series?.[0]?.links || opts?.series?.[0]?.edges || [];

  return (
    <div
      style={{ flex: 1, position: "relative", width: "100%", height: "100%", minHeight: "350px" }}
    >
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          zIndex: 99,
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
          ...USER_SELECT_NONE_STYLE,
        }}
      >
        {hasAttributes && (
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

        {open && hasAttributes && (
          <ChipForm
            nodeAttrs={nodeAttrs}
            edgeAttrs={edgeAttrs}
            initialChip={editingIdx !== null ? chips[editingIdx] : null}
            onSubmit={handleFormSubmit}
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
              onToggleDisable={handleToggleDisable}
              onInteractiveChange={() => setInteractiveTick((prev) => prev + 1)}
              onDelete={(idx) => {
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
        ref={ref}
        style={{ width: "100%", height: "100%", minHeight: "350px", ...USER_SELECT_NONE_STYLE }}
      />
    </div>
  );
}

export default LynxKiteNode(NodeWithVisualization);
