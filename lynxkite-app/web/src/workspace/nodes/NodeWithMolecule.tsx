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
        viewer.clear();

        // Load main structure (3dmol will auto-detect format)
        if (config.data) {
          try {
            viewer.addModel(config.data, "");
          } catch (error) {
            console.error("Failed to load molecular data:", error);
          }
        }

        // Try to load ligand if present
        if (config.ligand) {
          try {
            viewer.addModel(config.ligand, "");
            console.log("Ligand loaded");
          } catch (error) {
            console.log("Could not load ligand:", error);
          }
        }

        // Apply styling
        const model = viewer.getModel();
        if (model?.atoms) {
          // Define colors for each chain
          const chainColors: { [key: string]: number } = {
            A: 0xff00ff, // Magenta
            B: 0x00ff00, // Green
            G: 0xff8c00, // Orange
            R: 0x0000ff, // Blue
            S: 0xff0000, // Red
          };

          // Set default cartoon style
          viewer.setStyle({}, { cartoon: { color: "spectrum" } });

          // Get unique chains and apply chain-specific colors
          const chains = new Set<string>();
          for (const atom of model.atoms) {
            if (atom.chain) {
              chains.add(atom.chain);
            }
          }

          for (const chain of chains) {
            const color = chainColors[chain] || 0x888888;
            viewer.setStyle(
              { chain: chain },
              {
                cartoon: {
                  color: color,
                  thickness: 0.9,
                },
                tube: {
                  color: color,
                  radius: 0.3,
                },
              },
            );
          }

          // Style heteroatoms (ligands, ions, etc) as ball-and-stick
          viewer.setStyle(
            { hetflag: true },
            {
              stick: {
                radius: 0.25,
                colorscheme: "default",
              },
              sphere: {
                scale: 0.35,
                colorscheme: "default",
              },
            },
          );

          // Style non-standard residues as stick-and-sphere
          const nonStandardResidues = ["LIG", "AVE", "SDF", "MOL"];
          for (const res of nonStandardResidues) {
            viewer.setStyle(
              { resn: res },
              {
                stick: {
                  radius: 0.25,
                  colorscheme: "Jmol",
                },
                sphere: {
                  scale: 0.35,
                  colorscheme: "Jmol",
                },
              },
            );
          }
        }

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

    // Block wheel events and implement custom zoom
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

    observed.addEventListener("wheel", handleWheel, {
      passive: false,
      capture: true,
    });

    return () => {
      resizeObserver.unobserve(observed);
      observed.removeEventListener("wheel", handleWheel, { capture: true });
      viewerRef.current?.clear();
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
