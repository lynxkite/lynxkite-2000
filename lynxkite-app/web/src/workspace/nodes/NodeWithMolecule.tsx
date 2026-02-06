import React, { type CSSProperties, useEffect } from "react";
import LynxKiteNode from "./LynxKiteNode";
import { NodeWithParams } from "./NodeWithParams";

const NodeWithMolecule = (props: any) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const viewerRef = React.useRef<any>(null);

  useEffect(() => {
    const config = props.data?.display;
    if (!config || !containerRef.current) return;

    const observed = containerRef.current;

    async function run() {
      const $3Dmol = await import("3dmol");

      try {
        // Initialize viewer only once
        if (!viewerRef.current) {
          viewerRef.current = $3Dmol.createViewer(observed, {
            backgroundColor: "white",
          });
        }

        const viewer = viewerRef.current;

        // Clear previous models
        viewer.clear();

        // Add new model and style it
        viewer.addModel(config.data);
        viewer.setStyle({}, { stick: {} });
        viewer.zoomTo();
        viewer.render();
      } catch (error) {
        console.error("Error rendering 3D molecule:", error);
      }
    }
    run();

    const resizeObserver = new ResizeObserver(() => {
      viewerRef.current?.resize();
    });

    resizeObserver.observe(observed);

    // Block ALL wheel events and implement our own zoom
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();

      if (viewerRef.current) {
        const viewer = viewerRef.current;
        const factor = e.deltaY > 0 ? 0.95 : 1.05;
        viewer.zoom(factor);
        viewer.render();
      }
    };

    // Capture phase with passive false to completely block 3Dmol
    observed.addEventListener("wheel", handleWheel, { passive: false, capture: true });

    return () => {
      resizeObserver.unobserve(observed);
      observed.removeEventListener("wheel", handleWheel, { capture: true });
      if (viewerRef.current) {
        viewerRef.current.clear();
      }
    };
  }, [props.data?.display]);

  const vizStyle: CSSProperties = {
    flex: 1,
    minHeight: "300px",
    border: "1px solid #ddd",
    borderRadius: "4px",
    overflow: "hidden",
    position: "relative",
  };

  return (
    <NodeWithParams collapsed {...props}>
      <div style={vizStyle} ref={containerRef} />
    </NodeWithParams>
  );
};

export default LynxKiteNode(NodeWithMolecule);
