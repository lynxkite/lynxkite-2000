import { Handle, NodeResizeControl, type Position, useReactFlow } from "@xyflow/react";
import Color from "colorjs.io";
import React, { useContext } from "react";
import { ErrorBoundary } from "react-error-boundary";
import AlertTriangle from "~icons/tabler/alert-triangle-filled.jsx";
import ChevronDownRight from "~icons/tabler/chevron-down-right.jsx";
import Dots from "~icons/tabler/dots.jsx";
import Skull from "~icons/tabler/skull.jsx";
import type { Op as OpsOp, Workspace, WorkspaceNodeData } from "../../apiTypes.ts";
import { COLORS, useCategoryHierarchy } from "../../common.ts";
import InlineSVG from "../../InlineSVG.tsx";
import Tooltip from "../../Tooltip";
import { LynxKiteState } from "../LynxKiteState.ts";
import { NodeSearchInternal } from "../NodeSearch.tsx";

interface LynxKiteNodeProps {
  id: string;
  width: number;
  height: number;
  nodeStyle: any;
  data: any;
  children: any;
  parentId?: string;
  dragging?: boolean;
}

function paramSummary(data: WorkspaceNodeData): string {
  const lines = [];
  for (const [key, value] of Object.entries(data.params || {})) {
    const displayValue = value;
    if (typeof value === "object") {
      continue;
    }
    lines.push(`${key}: ${displayValue}`);
  }
  return lines.join(", ");
}

function docToString(doc: any): string {
  if (!doc) return "";
  return (
    doc.map?.((section: any) => (section.kind === "text" ? section.value : "")).join("\n") ??
    String(doc)
  );
}

function getHandles(ws: Workspace, id: string, inputs: any[], outputs: any[]) {
  const handles: {
    position: "top" | "bottom" | "left" | "right";
    name: string;
    index: number;
    offsetPercentage: number;
    showLabel: boolean;
    type: "source" | "target";
  }[] = [];
  for (const e of inputs) {
    handles.push({ ...e, type: "target" });
  }
  for (const e of outputs) {
    handles.push({ ...e, type: "source" });
  }
  const counts = { top: 0, bottom: 0, left: 0, right: 0 };
  for (const e of handles) {
    e.index = counts[e.position];
    counts[e.position]++;
  }
  const simpleHorizontal =
    counts.top === 0 && counts.bottom === 0 && counts.left <= 1 && counts.right <= 1;
  const simpleVertical =
    counts.left === 0 && counts.right === 0 && counts.top <= 1 && counts.bottom <= 1;
  for (const e of handles) {
    e.offsetPercentage = (100 * (e.index + 1)) / (counts[e.position] + 1);
    e.showLabel = !simpleHorizontal && !simpleVertical;
  }
  // Add handles for connections that exist but are not defined in inputs/outputs.
  // This can happen on unknown operations, or when the inputs/outputs are renamed.
  for (const e of ws.edges ?? []) {
    if (e.target === id && !handles.find((h) => h.name === e.targetHandle)) {
      handles.push({
        position: "left",
        name: e.targetHandle,
        index: counts.left,
        offsetPercentage: 50,
        showLabel: true,
        type: "target",
      });
      counts.left++;
    }
    if (e.source === id && !handles.find((h) => h.name === e.sourceHandle)) {
      handles.push({
        position: "right",
        name: e.sourceHandle,
        index: counts.right,
        offsetPercentage: 50,
        showLabel: true,
        type: "source",
      });
      counts.right++;
    }
  }
  return handles;
}

function canScrollX(element: HTMLElement) {
  const style = getComputedStyle(element);
  return style.overflowX === "auto" || style.overflow === "auto";
}
function canScrollY(element: HTMLElement) {
  const style = getComputedStyle(element);
  return style.overflowY === "auto" || style.overflow === "auto";
}
function canScrollUp(e: HTMLElement) {
  return canScrollY(e) && e.scrollTop > 0;
}
function canScrollDown(e: HTMLElement) {
  return canScrollY(e) && e.scrollTop < e.scrollHeight - e.clientHeight - 1;
}
function canScrollLeft(e: HTMLElement) {
  return canScrollX(e) && e.scrollLeft > 0;
}
function canScrollRight(e: HTMLElement) {
  return canScrollX(e) && e.scrollLeft < e.scrollWidth - e.clientWidth - 1;
}

function onWheel(e: WheelEvent) {
  if (e.ctrlKey) return; // Zoom, not scroll.
  let t = e.target as HTMLElement;
  // If we find an element inside the node container that can apply this scroll event, we stop propagation.
  // Otherwise ReactFlow can have it and pan the workspace.
  while (t && !t.classList.contains("node-container")) {
    if (
      (e.deltaX < 0 && canScrollLeft(t)) ||
      (e.deltaX > 0 && canScrollRight(t)) ||
      (e.deltaY < 0 && canScrollUp(t)) ||
      (e.deltaY > 0 && canScrollDown(t))
    ) {
      e.stopPropagation();
      return;
    }
    t = t.parentElement as HTMLElement;
  }
}

function formatTime(secondsStr: number | string | undefined | null): string {
  const seconds = Number(secondsStr);
  if (!seconds || !isFinite(seconds) || seconds < 0) return "00:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function NodeProgress({ telemetry, color }: { telemetry: any; color: string }) {
  if (!telemetry || Object.keys(telemetry).length === 0) return null;

  const { n = 0, total = 0, elapsed = 0, rate, prefix, postfix, unit = "it" } = telemetry;

  const hasTotal = typeof total === "number" && total > 0;
  // If we don't have a total, we can show an indeterminate animated bar or full bar
  const percentage = hasTotal ? Math.min(100, Math.max(0, (n / total) * 100)) : 100;

  // Estimate time remaining
  const eta = hasTotal && rate && rate > 0 ? (total - n) / rate : null;
  const itersPerSec = rate ? `${rate.toFixed(2)}${unit}/s` : "";

  return (
    <div
      className="node-progress"
      style={{
        padding: "8px 12px",
        fontSize: "12px",
        background: "var(--bg-secondary, rgba(0,0,0,0.05))",
        borderBottom: "1px solid var(--border-color, rgba(128,128,128,0.2))",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
          {prefix || "Execution Progress"}
        </span>
        <span style={{ fontWeight: 500 }}>
          {hasTotal ? `${percentage.toFixed(1)}%` : `${n} ${unit}`}
        </span>
      </div>

      <div
        style={{
          height: "6px",
          background: "rgba(128, 128, 128, 0.2)",
          borderRadius: "3px",
          overflow: "hidden",
          marginBottom: "6px",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${percentage}%`,
            backgroundColor: telemetry.colour || color,
            transition: "width 0.2s ease-out",
            opacity: 0.9,
          }}
        />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          color: "var(--text-secondary, #888)",
        }}
      >
        <span>
          {n} {hasTotal ? `/ ${total}` : ""} {unit} {postfix ? `— ${postfix}` : ""}
        </span>
        <span style={{ display: "flex", gap: "8px", textAlign: "right" }}>
          {itersPerSec && <span>{itersPerSec}</span>}
          <span>
            {eta !== null && eta > 0
              ? `ETA: ${formatTime(eta)}`
              : `Elapsed: ${formatTime(elapsed)}`}
          </span>
        </span>
      </div>
    </div>
  );
}

function LynxKiteNodeComponent(props: LynxKiteNodeProps) {
  const reactFlow = useReactFlow();
  const containerRef = React.useRef<HTMLDivElement>(null);
  const data = props.data;
  const state = useContext(LynxKiteState);
  const handles = getHandles(
    state.workspace,
    props.id,
    data.meta?.inputs || [],
    data.meta?.outputs || [],
  );
  React.useEffect(() => {
    // ReactFlow handles wheel events to zoom/pan and this would prevent scrolling inside the node.
    // To stop the event from reaching ReactFlow, we stop propagation on the wheel event.
    // This must be done with a "passive: false" listener, which we can only register like this.
    containerRef.current?.addEventListener("wheel", onWheel, {
      passive: false,
    });
    return () => {
      containerRef.current?.removeEventListener("wheel", onWheel);
    };
  }, [containerRef]);
  const node = reactFlow.getNode(props.id);
  function titleClicked() {
    const dataUpdate = {
      collapsed: !data.collapsed,
      expanded_height: data.expanded_height,
    };
    if (data.collapsed) {
      reactFlow.updateNode(props.id, {
        height: data.expanded_height || 200,
      });
    } else {
      dataUpdate.expanded_height = props.height;
    }
    reactFlow.updateNodeData(props.id, dataUpdate);
  }
  function setNewOpId(newOpId: string) {
    reactFlow.updateNodeData(props.id, {
      op_id: newOpId,
      error: undefined,
    });
  }
  const height = Math.max(67, node?.height ?? props.height ?? 200);
  const meta = data.meta ?? {};
  const summary: string = data.error
    ? `Error: ${data.error}`
    : (data.collapsed && paramSummary(data)) || docToString(meta.doc);
  const handleOffsetDirection = {
    top: "left",
    bottom: "left",
    left: "top",
    right: "top",
  };
  const color = new Color(COLORS[meta.color] ?? meta.color ?? "oklch(75% 0.2 55)");
  const titleStyle = { backgroundColor: color.toString() };
  color.lch[0] = 20;
  color.alpha = 0.5;
  const borderColor = color.toString();
  color.lch[1] = 50;
  color.alpha = 0.25;
  const nodeStyle = {
    ...props.nodeStyle,
    borderColor,
    boxShadow: `0px 5px 30px 0px ${color.toString()}`,
  };
  const titleTooltip = data.collapsed ? "Click to expand node" : summary;

  return (
    <div
      className={`node-container ${data.collapsed ? "collapsed" : "expanded"}`}
      style={{
        width: props.width || 200,
        height: data.collapsed ? undefined : height,
      }}
      ref={containerRef}
    >
      <div className="lynxkite-node" style={nodeStyle}>
        <Tooltip doc={titleTooltip} disabled={props.dragging}>
          <div
            style={titleStyle}
            className={`title drag-handle ${data.status}`}
            onClick={titleClicked}
          >
            <Icon name={meta.icon} />
            <div className="title-right-side">
              <div className="title-right-side-top">
                <span className="title-title">{data.title}</span>
                {data.error && <AlertTriangle />}
                {data.collapsed && <Dots />}
              </div>
              {summary && <span className="title-summary">{summary}</span>}
            </div>
          </div>
        </Tooltip>
        {!data.collapsed && (
          <>
            <div className="node-content">
              {data.telemetry && (
                <NodeProgress telemetry={data.telemetry} color={titleStyle.backgroundColor} />
              )}
              {data.message && data.message.length > 0 && (
                <div className="node-message" title="Execution status">
                  {data.message}
                </div>
              )}
              {data.error === "Unknown operation." ? (
                <UnknownOperationNode op_id={data.op_id} onChange={setNewOpId} />
              ) : (
                <>
                  {data.error && <div className="error">{data.error}</div>}
                  <ErrorBoundary
                    resetKeys={[props]}
                    fallback={
                      <p
                        className="error"
                        style={{ display: "flex", alignItems: "center", gap: 8 }}
                      >
                        <Skull style={{ fontSize: 20 }} />
                        Failed to display this node.
                      </p>
                    }
                  >
                    {props.children}
                  </ErrorBoundary>
                </>
              )}
            </div>
            <NodeResizeControl
              minWidth={100}
              minHeight={50}
              style={{ background: "transparent", border: "none" }}
            >
              <ChevronDownRight className="node-resizer" />
            </NodeResizeControl>
          </>
        )}
        {handles.map((handle) => (
          <Handle
            key={`${handle.name} on ${handle.position}`}
            id={handle.name}
            type={handle.type}
            position={handle.position as Position}
            style={{
              [handleOffsetDirection[handle.position]]: `${handle.offsetPercentage}% `,
            }}
          >
            {handle.showLabel && (
              <span className="handle-name">{handle.name.replace(/_/g, " ")}</span>
            )}
          </Handle>
        ))}
      </div>
    </div>
  );
}

function Icon({ name }: { name: string }) {
  if (!name) {
    return <div className="title-icon-placeholder" />;
  }
  if (name.startsWith("<svg")) {
    return <span className="title-icon" dangerouslySetInnerHTML={{ __html: name }} />;
  }
  return <InlineSVG className="title-icon" src={`/api/icons/${name}`} />;
}

export default function LynxKiteNode(Component: React.ComponentType<any>) {
  return (props: any) => {
    return (
      <LynxKiteNodeComponent {...props}>
        <Component {...props} />
      </LynxKiteNodeComponent>
    );
  };
}

function UnknownOperationNode(props: { op_id: string; onChange: (newName: string) => void }) {
  const categoryHierarchy = useCategoryHierarchy();
  return (
    categoryHierarchy && (
      <div className="node-search" style={{ overflowY: "auto" }}>
        <div style={{ marginBottom: 20 }}>
          {props.op_id ? (
            <>
              "{props.op_id}" is not a known box. You may need to install an extension, or fix an
              issue with a code file that defined it. Or perhaps the box has a new name. You can
              choose a replacement from the list below:
            </>
          ) : (
            <>Choose a replacement from the list below:</>
          )}
        </div>
        <NodeSearchInternal
          onCancel={() => {}}
          onClick={(op: OpsOp) => op.id && props.onChange(op.id)}
          categoryHierarchy={categoryHierarchy}
        />
      </div>
    )
  );
}
