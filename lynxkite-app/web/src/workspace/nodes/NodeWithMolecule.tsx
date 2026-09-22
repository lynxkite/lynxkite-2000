import { useReactFlow } from "@xyflow/react";
import React, { useEffect } from "react";
import "molstar/build/viewer/molstar.css";
import { useDisplay } from "../../common";
import LynxKiteNode from "./LynxKiteNode";
import { NodeWithParams } from "./NodeWithParams";

const MOLSTAR_STATE_PARAM = "_molstar_state";
const SAVE_DEBOUNCE_MS = 1000;

function inferFormat(
  data: string,
  format?: string,
): "gro" | "pdb" | "mmcif" | "sdf" | "mol" | "mol2" {
  const trimmed = data.trimStart();
  if (trimmed.startsWith("data_")) return "mmcif";
  if (trimmed.includes("@<TRIPOS>MOLECULE")) return "mol2";
  if (trimmed.includes("M  END")) {
    if (trimmed.includes("$$$$")) return "sdf";
    return "mol";
  }
  if (format === "gro") return format;
  return "pdb";
}

function parseStoredViewState(raw: unknown): any | null {
  if (!raw) return null;
  try {
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

/** Capture representation/canvas settings without embedding molecule bytes. */
function captureViewState(viewer: any) {
  const representations: any[] = [];
  for (const s of viewer.plugin.managers.structure.hierarchy.current.structures ?? []) {
    for (const c of s.components ?? []) {
      for (const r of c.representations ?? []) {
        const params = r.cell?.transform?.params;
        if (params) representations.push(params);
      }
    }
  }
  const snapshot = viewer.plugin.state.getSnapshot({ data: false });
  return {
    representations,
    canvas3d: snapshot.canvas3d,
    canvas3dContext: snapshot.canvas3dContext,
    structureComponentManager: snapshot.structureComponentManager,
  };
}

async function applyViewState(viewer: any, stored: any) {
  const saved = Array.isArray(stored?.representations) ? stored.representations : [];
  const structures = viewer.plugin.managers.structure.hierarchy.current.structures ?? [];
  let ri = 0;
  for (const s of structures) {
    for (const c of s.components ?? []) {
      for (let i = 0; i < (c.representations?.length ?? 0); i++) {
        const params = saved[ri++] ?? saved[0];
        if (!params) continue;
        try {
          await viewer.plugin.managers.structure.component.updateRepresentations(
            [c],
            c.representations[i],
            {
              ...c.representations[i].cell.transform.params,
              type: params.type,
              colorTheme: params.colorTheme,
              sizeTheme: params.sizeTheme,
            },
          );
        } catch (error) {
          console.error("Failed to apply Mol* representation:", error);
        }
      }
    }
  }

  await viewer.plugin.state.setSnapshot({
    id: stored.id,
    canvas3d: stored.canvas3d,
    canvas3dContext: stored.canvas3dContext,
    structureComponentManager: stored.structureComponentManager,
  });
}

const NodeWithMolecule = (props: any) => {
  const reactFlow = useReactFlow();
  const containerRef = React.useRef<HTMLDivElement>(null);
  const wrapperRef = React.useRef<HTMLDivElement>(null);
  const viewerRef = React.useRef<any>(null);
  const suppressSaveRef = React.useRef(false);
  const saveTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const suppressTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const paramsRef = React.useRef(props.data?.params);
  paramsRef.current = props.data?.params;
  const config = useDisplay(props.data?.display_version, props.id);

  function setParam(name: string, newValue: any) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { ...prevData.params, [name]: newValue },
    }));
  }

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const container = containerRef.current;
    if (!config || !container || !wrapper) return;
    const wrap = wrapper;
    let active = true;
    let isInitializing = false;

    async function run() {
      if (isInitializing) return; // Prevent concurrent initializations
      const container = containerRef.current;
      if (!container) return;
      isInitializing = true;

      const { Viewer } = await import("molstar/lib/apps/viewer/app");

      try {
        // Dispose old viewer if it exists
        if (viewerRef.current?.dispose) {
          viewerRef.current.dispose();
          viewerRef.current = null;
        }

        if (!active) return; // Check before clearing DOM

        container.innerHTML = "";

        const viewer = await Viewer.create(container, {
          layoutIsExpanded: false,
          layoutShowControls: false,
          layoutShowRemoteState: false,
          layoutShowLog: false,
          collapseLeftPanel: true,
        });

        if (!active) {
          viewer.dispose();
          return;
        }

        viewerRef.current = viewer;
        const stored = parseStoredViewState(paramsRef.current?.[MOLSTAR_STATE_PARAM]);
        suppressSaveRef.current = true;
        // Hide until saved styles are applied so the default preset does not flash.
        if (stored) wrap.style.opacity = "0";
        try {
          if (config.data && active) {
            await viewer.loadStructureFromData(
              config.data,
              inferFormat(config.data, config.format),
            );
          }

          if (config.ligand && active) {
            await viewer.loadStructureFromData(
              config.ligand,
              inferFormat(config.ligand, config.format),
            );
          }

          if (config.model && config.coordinates && active) {
            await viewer.loadTrajectory({
              model: {
                kind: "model-url",
                url: config.model,
                format: "gro",
              },
              coordinates: {
                kind: "coordinates-url",
                url: config.coordinates,
                isBinary: true,
                format: "xtc",
              },
              preset: "default",
            });
          }

          if (stored && active) {
            await applyViewState(viewer, stored);
          }
        } finally {
          if (stored) {
            wrap.style.opacity = "";
            viewer.handleResize?.();
            viewer.plugin.managers.camera.reset(undefined, 0);
          }
          if (suppressTimerRef.current) clearTimeout(suppressTimerRef.current);
          suppressTimerRef.current = setTimeout(() => {
            if (active) suppressSaveRef.current = false;
          }, 500);
        }

        if (!active) return;

        const scheduleSave = () => {
          if (!active || suppressSaveRef.current) return;
          if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
          saveTimerRef.current = setTimeout(() => {
            if (!active || suppressSaveRef.current || !viewerRef.current) return;
            try {
              const payload = JSON.stringify(captureViewState(viewerRef.current));
              const prev = paramsRef.current?.[MOLSTAR_STATE_PARAM];
              if (prev === payload) return;
              setParam(MOLSTAR_STATE_PARAM, payload);
            } catch (error) {
              console.error("Failed to save Mol* state:", error);
            }
          }, SAVE_DEBOUNCE_MS);
        };

        // Representation changes only — camera is refit per molecule, not persisted.
        viewer.subscribe(viewer.plugin.state.data.events.changed, scheduleSave);
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
    resizeObserver.observe(container);
    const handleWheel = (e: WheelEvent) => {
      e.stopPropagation();
    };
    wrap.addEventListener("wheel", handleWheel, {
      passive: false,
    });

    return () => {
      active = false;
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      if (suppressTimerRef.current) clearTimeout(suppressTimerRef.current);
      wrap.style.opacity = "";
      resizeObserver.unobserve(container);
      wrap.removeEventListener("wheel", handleWheel);
      viewerRef.current?.dispose?.();
      viewerRef.current = null;
    };
  }, [config]);

  return (
    <NodeWithParams collapsed {...props}>
      <div ref={wrapperRef} className="msp-lynxkite-wrapper">
        <div ref={containerRef} />
      </div>
    </NodeWithParams>
  );
};

export default LynxKiteNode(NodeWithMolecule);
