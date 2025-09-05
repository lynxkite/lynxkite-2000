import React, { useEffect } from "react";
import LynxKiteNode from "./LynxKiteNode";
import { NodeWithParams } from "./NodeWithParams";

const echarts = await import("echarts");

function NodeWithVisualization(props: any) {
  const chartsRef = React.useRef<HTMLDivElement>(null);
  const chartsInstanceRef = React.useRef<echarts.ECharts>();
  useEffect(() => {
    const opts = props.data?.display?.value;
    if (!opts || !chartsRef.current) return;
    if (opts.tooltip?.formatter === "GET_THIRD_VALUE") {
      // We can't pass a function from the backend, and can't get good tooltips otherwise.
      opts.tooltip.formatter = (params: any) => params.value[2];
    }
    chartsInstanceRef.current = echarts.init(chartsRef.current, null, {
      renderer: "canvas",
      width: "auto",
      height: "auto",
    });
    chartsInstanceRef.current.setOption(opts);
    const resizeObserver = new ResizeObserver(() => {
      const e = chartsRef.current;
      if (!e) return;
      e.style.padding = "1px";
      chartsInstanceRef.current?.resize();
      e.style.padding = "0";
    });
    const observed = chartsRef.current;
    resizeObserver.observe(observed);
    return () => {
      resizeObserver.unobserve(observed);
      chartsInstanceRef.current?.dispose();
    };
  }, [props.data?.display?.value]);
  return (
    <NodeWithParams collapsed {...props}>
      <div style={{ flex: 1 }} ref={chartsRef} />
    </NodeWithParams>
  );
}

export default LynxKiteNode(NodeWithVisualization);
