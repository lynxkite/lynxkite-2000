import { useContext } from 'react';
import { LynxKiteState } from '../LynxKiteState';
import { useReactFlow, Handle, NodeResizeControl, Position } from '@xyflow/react';
// @ts-ignore
import ChevronDownRight from '~icons/tabler/chevron-down-right.jsx';

interface LynxKiteNodeProps {
  id: string;
  width: number;
  height: number;
  nodeStyle: any;
  data: any;
  children: any;
}

function getHandles(inputs: object, outputs: object) {
  const handles: {
    position: 'top' | 'bottom' | 'left' | 'right',
    name: string,
    index: number,
    offsetPercentage: number,
    showLabel: boolean,
    type: 'source' | 'target',
  }[] = [];
  for (const e of Object.values(inputs)) {
    handles.push({ ...e, type: 'target' });
  }
  for (const e of Object.values(outputs)) {
    handles.push({ ...e, type: 'source' });
  }
  const counts = { top: 0, bottom: 0, left: 0, right: 0 };
  for (const e of handles) {
    e.index = counts[e.position];
    counts[e.position]++;
  }
  for (const e of handles) {
    e.offsetPercentage = 100 * (e.index + 1) / (counts[e.position] + 1);
    const simpleHorizontal = counts.top === 0 && counts.bottom === 0 && handles.length <= 2;
    const simpleVertical = counts.left === 0 && counts.right === 0 && handles.length <= 2;
    e.showLabel = !simpleHorizontal && !simpleVertical;
  }
  return handles;
}

export default function LynxKiteNode(props: LynxKiteNodeProps) {
  const reactFlow = useReactFlow();
  const data = props.data;
  const state = useContext(LynxKiteState);
  const expanded = !data.collapsed;
  const handles = getHandles(data.meta?.inputs || {}, data.meta?.outputs || {});
  function asPx(n: number | undefined) {
    return (n ? n + 'px' : undefined) || '200px';
  }
  function titleClicked() {
    reactFlow.updateNodeData(props.id, { collapsed: expanded });
  }
  const handleOffsetDirection = { top: 'left', bottom: 'left', left: 'top', right: 'top' };

  return (
    <div className={'node-container ' + (expanded ? 'expanded' : 'collapsed')}
      style={{ width: asPx(props.width), height: asPx(expanded ? props.height : undefined) }}>
      <div className="lynxkite-node" style={props.nodeStyle}>
        <div className="title bg-primary" onClick={titleClicked}>
          {data.title}
          {data.error && <span className="title-icon">⚠️</span>}
          {expanded || <span className="title-icon">⋯</span>}
        </div>
        {expanded && <>
          {data.error &&
            <div className="error">{data.error}</div>
          }
          {props.children}
          {handles.map(handle => (
            <Handle
              key={handle.name}
              id={handle.name} type={handle.type} position={handle.position as Position}
              style={{ [handleOffsetDirection[handle.position]]: handle.offsetPercentage + '%' }}>
              {handle.showLabel && <span className="handle-name">{handle.name.replace(/_/g, " ")}</span>}
            </Handle >
          ))}
          <NodeResizeControl
            minWidth={100}
            minHeight={50}
            style={{ 'background': 'transparent', 'border': 'none' }}
          >
            <ChevronDownRight className="node-resizer" />
          </NodeResizeControl>
        </>}
      </div>
    </div>
  );
}
