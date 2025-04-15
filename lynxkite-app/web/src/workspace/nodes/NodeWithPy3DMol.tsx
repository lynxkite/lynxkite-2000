import React, { useEffect } from "react";
import NodeWithParams from "./NodeWithParams";
const $3Dmol = await import("3dmol");

const NodeWithPy3Dmol = (props: any) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const viewerRef = React.useRef<any>(null);

  useEffect(() => {
    const config = props.data?.display?.value;
    if (!config || !containerRef.current) return;

    try {
      // Initialize viewer only once
      if (!viewerRef.current) {
        viewerRef.current = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: "white",
        });
      }

      const viewer = viewerRef.current;

      // Clear previous models
      viewer.clear();

      // Add new model and style it
      viewer.addModel(config.data, config.format);
      viewer.setStyle({}, { stick: {} });
      viewer.zoomTo();
      viewer.render();
    } catch (error) {
      console.error("Error rendering 3D molecule:", error);
    }

    const resizeObserver = new ResizeObserver(() => {
      viewerRef.current?.resize();
    });

    const observed = containerRef.current;
    resizeObserver.observe(observed);

    return () => {
      resizeObserver.unobserve(observed);
      if (viewerRef.current) {
        viewerRef.current.clear();
      }
    };
  }, [props.data?.display?.value]);

  const nodeStyle = { display: "flex", flexDirection: "column", height: "100%" };
  const vizStyle = {
    flex: 1,
    minHeight: "300px",
    border: "1px solid #ddd",
    borderRadius: "4px",
    overflow: "hidden",
    position: "relative",
  };

  return (
    <NodeWithParams nodeStyle={nodeStyle} collapsed {...props}>
      <div style={vizStyle} ref={containerRef} />
    </NodeWithParams>
  );
};

export default NodeWithPy3Dmol;
