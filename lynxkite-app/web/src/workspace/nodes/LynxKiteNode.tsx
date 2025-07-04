import { Handle, NodeResizeControl, type Position, useReactFlow } from "@xyflow/react";
import React from "react";
import { ErrorBoundary } from "react-error-boundary";
// @ts-ignore
import AlertTriangle from "~icons/tabler/alert-triangle-filled.jsx";
// @ts-ignore
import ChevronDownRight from "~icons/tabler/chevron-down-right.jsx";
// @ts-ignore
import Dots from "~icons/tabler/dots.jsx";
// @ts-ignore
import Help from "~icons/tabler/question-mark.jsx";
// @ts-ignore
import Skull from "~icons/tabler/skull.jsx";
import Tooltip from "../../Tooltip";
import { COLORS } from "../../common.ts";

interface LynxKiteNodeProps {
  id: string;
  width: number;
  height: number;
  nodeStyle: any;
  data: any;
  children: any;
  parentId?: string;
}

function getHandles(inputs: any[], outputs: any[]) {
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
  return handles;
}

const stopPropagation = (e: Event) => e.stopPropagation();

function LynxKiteNodeComponent(props: LynxKiteNodeProps) {
  const reactFlow = useReactFlow();
  const containerRef = React.useRef<HTMLDivElement>(null);
  const data = props.data;
  const expanded = !data.collapsed;
  const handles = getHandles(data.meta?.value?.inputs || [], data.meta?.value?.outputs || []);
  React.useEffect(() => {
    // ReactFlow handles wheel events to zoom/pan and this would prevent scrolling inside the node.
    // To stop the event from reaching ReactFlow, we stop propagation on the wheel event.
    // This must be done with a "passive: false" listener, which we can only register like this.
    containerRef.current?.addEventListener("wheel", stopPropagation, { passive: false });
    return () => {
      containerRef.current?.removeEventListener("wheel", stopPropagation);
    };
  }, [containerRef]);
  function titleClicked() {
    reactFlow.updateNodeData(props.id, { collapsed: expanded });
  }
  const handleOffsetDirection = {
    top: "left",
    bottom: "left",
    left: "top",
    right: "top",
  };
  const titleStyle: { backgroundColor?: string } = {};
  if (data.meta?.value?.color) {
    titleStyle.backgroundColor = COLORS[data.meta.value.color] || data.meta.value.color;
  }
  return (
    <div
      className={`node-container ${expanded ? "expanded" : "collapsed"} ${props.parentId ? "in-group" : ""}`}
      style={{
        width: props.width || 200,
        height: expanded ? props.height || 200 : undefined,
      }}
      ref={containerRef}
    >
      <div className="lynxkite-node" style={props.nodeStyle}>
        <div
          className={`title bg-primary drag-handle ${data.status}`}
          style={titleStyle}
          onClick={titleClicked}
        >
          <span className="title-title">{data.title}</span>
          {data.error && (
            <Tooltip doc={`Error: ${data.error}`}>
              <AlertTriangle />
            </Tooltip>
          )}
          {expanded || (
            <Tooltip doc="Click to expand node">
              <Dots />
            </Tooltip>
          )}
          <Tooltip doc={data.meta?.value?.doc}>
            <Help />
          </Tooltip>
        </div>
        {expanded && (
          <>
            {data.error && <div className="error">{data.error}</div>}
            <ErrorBoundary
              resetKeys={[props]}
              fallback={
                <p className="error" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Skull style={{ fontSize: 20 }} />
                  Failed to display this node.
                </p>
              }
            >
              <div className="node-content">{props.children}</div>
            </ErrorBoundary>
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

export default function LynxKiteNode(Component: React.ComponentType<any>) {
  return (props: any) => {
    return (
      <LynxKiteNodeComponent {...props}>
        <Component {...props} />
      </LynxKiteNodeComponent>
    );
  };
}
