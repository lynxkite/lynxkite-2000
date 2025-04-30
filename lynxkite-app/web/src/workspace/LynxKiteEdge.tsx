import { BaseEdge, Position } from "@xyflow/react";

function addOffset(x: number, y: number, p: Position, offset: number) {
  if (p === Position.Top) return `${x},${y - offset}`;
  if (p === Position.Bottom) return `${x},${y + offset}`;
  if (p === Position.Left) return `${x - offset},${y}`;
  return `${x + offset},${y}`;
}

export default function LynxKiteEdge(props: any) {
  const offset = 0.3 * Math.hypot(props.targetX - props.sourceX, props.targetY - props.sourceY);
  const s = addOffset(props.sourceX, props.sourceY, props.sourcePosition, 0);
  const sc = addOffset(props.sourceX, props.sourceY, props.sourcePosition, offset);
  const tc = addOffset(props.targetX, props.targetY, props.targetPosition, offset);
  const t = addOffset(props.targetX, props.targetY, props.targetPosition, 0);
  const path = `M${s} C${sc} ${tc} ${t}`;
  return (
    <>
      <BaseEdge
        id={props.id}
        path={path}
        {...props}
        style={{
          strokeWidth: 2,
          stroke: "black",
        }}
      />
    </>
  );
}
