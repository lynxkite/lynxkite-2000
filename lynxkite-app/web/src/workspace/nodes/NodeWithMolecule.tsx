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

        // Detect format and load data
        if (config.data) {
          const dataStr = String(config.data);
          let format = "pdb"; // default

          // Detect format from content
          if (dataStr.trim().startsWith("data_")) {
            format = "cif"; // mmCIF format
          } else if (dataStr.includes("@<tripos>MOLECULE")) {
            format = "mol2";
          } else if (dataStr.includes("V2000") || dataStr.includes("V3000")) {
            format = "sdf";
          }

          console.log(`Detected format: ${format}`);

          try {
            viewer.addModel(config.data, format);
          } catch (_formatError) {
            console.log(`Failed to load as ${format}, trying as pdb...`);
            viewer.addModel(config.data, "pdb");
          }
        }

        // Try to load ligand if present
        if (config.ligand) {
          const ligandStr = String(config.ligand);
          let ligandFormat = "sdf"; // default for ligands

          if (ligandStr.trim().startsWith("data_")) {
            ligandFormat = "cif";
          } else if (ligandStr.includes("@<tripos>MOLECULE")) {
            ligandFormat = "mol2";
          } else if (ligandStr.includes("HEADER")) {
            ligandFormat = "pdb";
          }

          try {
            viewer.addModel(config.ligand, ligandFormat);
            console.log(`Ligand loaded as ${ligandFormat}`);
          } catch (e) {
            console.log("Could not load ligand:", e);
          }
        }

        // Apply styling to everything
        const model = viewer.getModel();
        if (model?.atoms) {
          const chainColors: { [key: string]: number } = {
            A: 0xff00ff,
            B: 0x00ff00,
            G: 0xff8c00,
            R: 0x0000ff,
            S: 0xff0000,
          };

          const chains = new Set<string>();
          model.atoms.forEach((atom: any) => {
            if (atom.chain) chains.add(atom.chain);
          });

          console.log("Chains found:", Array.from(chains));

          const chainAtomCounts: { [key: string]: number } = {};
          model.atoms.forEach((atom: any) => {
            if (atom.chain) {
              chainAtomCounts[atom.chain] = (chainAtomCounts[atom.chain] || 0) + 1;
            }
          });
          console.log("Atoms per chain:", chainAtomCounts);

          viewer.addSurface($3Dmol.SurfaceType.VDW, {
            opacity: 0.4,
            color: 0xcccccc,
            wireframe: false,
          });

          viewer.setStyle(
            {},
            {
              cartoon: {
                color: "spectrum",
              },
            },
          );

          chains.forEach((chain) => {
            const color = chainColors[chain] || 0x888888;

            console.log(`Styling chain ${chain} with color ${color.toString(16)}`);

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
          });

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

          const nonStandardResidues = ["LIG", "AVE", "SDF", "MOL"];
          nonStandardResidues.forEach((res) => {
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
          });
        }

        viewer.zoomTo();
        viewer.render();
      } catch (error) {
        console.error("Error rendering 3D molecule:", error);
      }
    }

    run();

    const resizeObserver = new ResizeObserver(() => {
      if (viewerRef.current) {
        viewerRef.current.resize();
      }
    });

    resizeObserver.observe(observed);

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
