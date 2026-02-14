import { BaseEdge, Position } from "@xyflow/react";
import React, { useContext } from "react";
import { LynxKiteState } from "./LynxKiteState";

function addOffset(x: number, y: number, p: Position, offset: number) {
  if (p === Position.Top) return `${x},${y - offset}`;
  if (p === Position.Bottom) return `${x},${y + offset}`;
  if (p === Position.Left) return `${x - offset},${y}`;
  return `${x + offset},${y}`;
}

export default function LynxKiteEdge(props: any) {
  const state = useContext(LynxKiteState);
  const time = Date.now();
  const dtime = 250 + 0.2 * ((props.targetX ?? 0) + 0.3 * (props.targetY ?? 0));
  const timeLeft = state.creationTime + dtime - time;
  const [show, setShow] = React.useState(timeLeft <= 0);
  if (!show) {
    setTimeout(() => {
      setShow(true);
    }, timeLeft);
  }

  const offset = 0.3 * Math.hypot(props.targetX - props.sourceX, props.targetY - props.sourceY);
  const s = addOffset(props.sourceX, props.sourceY, props.sourcePosition, 0);
  const sc = addOffset(props.sourceX, props.sourceY, props.sourcePosition, offset);
  const tc = addOffset(props.targetX, props.targetY, props.targetPosition, offset);
  const t = addOffset(props.targetX, props.targetY, props.targetPosition, 0);
  const path = `M${s} C${sc} ${tc} ${t}`;
  return (
    show && (
      <BaseEdge
        path={path}
        labelX={props.labelX}
        labelY={props.labelY}
        markerStart={props.markerStart}
        markerEnd={props.markerEnd}
        style={{
          strokeWidth: 2,
          stroke: "#888",
        }}
      />
    )
  );
}
