import React, { useEffect } from "react";
import NodeWithParams from "./NodeWithParams";
const echarts = await import("echarts");

const NodeWithVisualization = (props: any) => {
  const chartsRef = React.useRef<HTMLDivElement>(null);
  const chartsInstanceRef = React.useRef<echarts.ECharts>();
  useEffect(() => {
    const opts = props.data?.display?.value;
    if (!opts || !chartsRef.current) return;
    chartsInstanceRef.current = echarts.init(chartsRef.current, null, {
      renderer: "canvas",
      width: 250,
      height: 250,
    });
    chartsInstanceRef.current.setOption(opts);
    const onResize = () => chartsInstanceRef.current?.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chartsInstanceRef.current?.dispose();
    };
  }, [props.data?.display?.value]);
  return (
    <NodeWithParams {...props}>
      <div className="box" draggable={false} ref={chartsRef} />
    </NodeWithParams>
  );
};

export default NodeWithVisualization;
