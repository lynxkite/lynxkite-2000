import React, { type CSSProperties, useEffect } from "react";
import "molstar/build/viewer/molstar.css";
import { PluginCommands } from "molstar/lib/mol-plugin/commands";
import AdjustmentsHorizontal from "~icons/tabler/adjustments-horizontal.jsx";
import LynxKiteNode from "./LynxKiteNode";
import { NodeWithParams } from "./NodeWithParams";

function inferFormat(data: string): "pdb" | "mmcif" | "sdf" | "mol" | "mol2" {
  const trimmed = data.trimStart();
  if (trimmed.startsWith("data_")) return "mmcif";
  if (trimmed.includes("@<TRIPOS>MOLECULE")) return "mol2";
  if (trimmed.includes("M  END")) {
    if (trimmed.includes("$$$$")) return "sdf";
    return "mol";
  }
  return "pdb";
}

const NodeWithMolecule = (props: any) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const viewerRef = React.useRef<any>(null);

  useEffect(() => {
    const config = props.data?.display;
    if (!config || !containerRef.current) return;

    const observed = containerRef.current;
    let active = true;
    let isInitializing = false;

    async function run() {
      if (isInitializing) return; // Prevent concurrent initializations
      isInitializing = true;

      const { Viewer } = await import("molstar/lib/apps/viewer/app");

      try {
        // Dispose old viewer if it exists
        if (viewerRef.current?.dispose) {
          viewerRef.current.dispose();
          viewerRef.current = null;
        }

        if (!active) return; // Check before clearing DOM

        observed.innerHTML = "";

        if (!active) return; // Check again before creating

        const viewer = await Viewer.create(observed, {
          layoutShowControls: false,
          layoutShowSequence: false,
          layoutShowLog: false,
          layoutShowLeftPanel: false,
          viewportShowExpand: false,
          viewportShowControls: false,
          viewportShowSettings: false,
          viewportShowAnimation: false,
          collapseLeftPanel: true,
          collapseRightPanel: true,
        });

        if (!active) {
          viewer.dispose();
          return;
        }

        viewerRef.current = viewer;

        if (config.data && active) {
          await viewer.loadStructureFromData(config.data, inferFormat(config.data));
        }

        if (config.ligand && active) {
          await viewer.loadStructureFromData(config.ligand, inferFormat(config.ligand));
        }
      } catch (error) {
        console.error("Error rendering Mol* molecule:", error);
      } finally {
        isInitializing = false;
      }
    }

    run();

    const resizeObserver = new ResizeObserver(() => {
      if (viewerRef.current?.plugin?.canvas3d?.requestResize) {
        viewerRef.current.plugin.canvas3d.requestResize();
      }
      viewerRef.current?.handleResize?.();
    });

    resizeObserver.observe(observed);

    return () => {
      active = false;
      resizeObserver.unobserve(observed);
      if (viewerRef.current?.dispose) {
        viewerRef.current.dispose();
      }
      viewerRef.current = null;
    };
  }, [props.data?.display]);

  const toggleRightPanel = () => {
    if (!viewerRef.current?.plugin) return;
    const plugin = viewerRef.current.plugin;
    const currentLayoutState = plugin.layout.state;
    const rightIsOpen =
      currentLayoutState.showControls && currentLayoutState.regionState.right === "full";

    PluginCommands.Layout.Update(plugin, {
      state: {
        ...currentLayoutState,
        showControls: true,
        regionState: {
          ...currentLayoutState.regionState,
          right: rightIsOpen ? "hidden" : "full",
        },
      },
    });
  };

  const vizStyle: CSSProperties = {
    flex: 1,
    minHeight: "300px",
    border: "1px solid #ddd",
    borderRadius: "4px",
    overflow: "hidden",
    position: "relative",
    display: "flex",
    flexDirection: "column",
    height: "100%",
  };

  return (
    <NodeWithParams collapsed {...props}>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px", height: "100%" }}>
        <button
          onClick={toggleRightPanel}
          className="btn-link cursor-pointer self-end mr-1"
          title="Toggle control panel"
          aria-label="Toggle control panel"
        >
          <AdjustmentsHorizontal />
        </button>
        <div style={vizStyle} ref={containerRef} className="msp-lynxkite-container" />
      </div>
    </NodeWithParams>
  );
};

export default LynxKiteNode(NodeWithMolecule);
