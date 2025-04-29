import { BaseEdge } from "@xyflow/react";
import { getBetterBezierPath } from "reactflow-better-bezier-edge";

export default function LynxKiteEdge(props: any) {
  const [edgePath] = getBetterBezierPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
    offset: 0.3 * Math.hypot(props.targetX - props.sourceX, props.targetY - props.sourceY),
  });

  return (
    <>
      <BaseEdge
        id={props.id}
        path={edgePath}
        {...props}
        style={{
          strokeWidth: 2,
          stroke: "black",
        }}
      />
    </>
  );
}
