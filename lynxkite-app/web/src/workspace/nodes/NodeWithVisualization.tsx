import { useEffect, useRef, useState } from "react";
import { useDisplay } from "../../common.ts";
import LynxKiteNode from "./LynxKiteNode";

const echarts = await import("echarts");
// Fast, stable hash function to map varying row text strings into vibrant individual HSL colors

const getColor = (s: string) => {
  let h = 0,
    t = String(s ?? "");
  for (let i = 0; i < t.length; i++) h = t.charCodeAt(i) + ((h << 5) - h);
  return `hsl(${Math.abs(h * 131) % 360}, 90%, 50%)`;
};

export function NodeWithVisualization({ data, id }: any) {
  const ref = useRef<HTMLDivElement>(null);
  const opts = useDisplay(data?.display_version, id);
  const [chips, setChips] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [attr, setAttr] = useState("");
  const [attrs, setAttrs] = useState<string[]>([]);

  useEffect(() => {
    const nodes = opts?.series?.[0]?.data || [];
    if (!nodes.length) return;
    const keys = Array.from(
      new Set<string>(nodes.flatMap((n: any) => Object.keys(n.attributes || {}))),
    );
    setAttrs(keys);
    setAttr((p) => (keys.includes(p) ? p : keys[0] || ""));
  }, [opts]);

  useEffect(() => {
    if (!opts || !ref.current) return;
    const chartOpts = JSON.parse(JSON.stringify(opts));
    const active = chips[chips.length - 1];

    if (chartOpts.series?.[0]?.data && active) {
      chartOpts.series[0].data = chartOpts.series[0].data.map((n: any) => {
        const v = n.attributes?.[active];
        if (v === undefined || v === null || v === "") return n;
        return { ...n, itemStyle: { ...n.itemStyle, color: getColor(String(v)) } };
      });
    }

    const inst = echarts.init(ref.current, null, {
      renderer: "canvas",
      width: "auto",
      height: "auto",
    });
    inst.setOption(chartOpts, true);

    const obs = new ResizeObserver(() => inst.resize());
    obs.observe(ref.current);

    return () => {
      obs.disconnect();
      inst.dispose();
    };
  }, [opts, chips]);

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
        }}
      >
        {attrs.length > 0 && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setOpen(!open);
            }}
            style={{
              width: 28,
              height: 28,
              borderRadius: 6,
              border: "1px solid #ccc",
              background: "#fff",
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            {open ? "×" : "+"}
          </button>
        )}

        {open && attrs.length > 0 && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (attr) {
                setChips([...chips, attr]);
                setOpen(false);
              }
            }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: "#fff",
              padding: "6px 10px",
              borderRadius: 6,
              boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
              border: "1px solid #ddd",
            }}
          >
            <select
              value={attr}
              onChange={(e) => setAttr(e.target.value)}
              style={{ padding: "4px 6px", borderRadius: 4, border: "1px solid #ccc" }}
            >
              {attrs.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
            <button
              type="submit"
              style={{
                padding: "4px 10px",
                fontSize: 12,
                background: "#3498db",
                color: "#fff",
                border: "none",
                borderRadius: 4,
                cursor: "pointer",
                fontWeight: "bold",
              }}
            >
              Hash Color
            </button>
          </form>
        )}

        {chips.map((c, i) => (
          <div
            key={i}
            style={{
              display: "inline-flex",
              alignItems: "center",
              background: "#2c3e50",
              padding: "4px 10px",
              borderRadius: 6,
              gap: 8,
              color: "#fff",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            <span>Hashed Field: {c}</span>
            <span
              onClick={(e) => {
                e.stopPropagation();
                setChips(chips.filter((_, idx) => idx !== i));
              }}
              style={{ cursor: "pointer", color: "#e74c3c", fontWeight: "bold", fontSize: 14 }}
            >
              ×
            </span>
          </div>
        ))}
      </div>
      <div style={{ width: "100%", height: "100%", minHeight: "350px" }} ref={ref} />
    </div>
  );
}

export default LynxKiteNode(NodeWithVisualization);
