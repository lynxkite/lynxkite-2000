import { useContext } from 'react';
import { LynxKiteState } from '../LynxKiteState';
import LynxKiteNode from './LynxKiteNode';
import { useNodesState } from '@xyflow/react';
import NodeParameter from './NodeParameter';

function NodeWithParams(props) {
  const metaParams = props.data.meta?.params;
  const state = useContext(LynxKiteState);
  function setParam(name, newValue) {
    const i = state.workspace.nodes.findIndex((n) => n.id === props.id);
    state.workspace.nodes[i].data.params[name] = newValue;
  }
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const params = nodes && props.data?.params ? Object.entries(props.data.params) : [];

  return (
    <LynxKiteNode {...props}>
      {params.map(([name, value]) =>
        <NodeParameter
          name={name}
          key={name}
          value={value}
          meta={metaParams?.[name]}
          onChange={(value) => setParam(name, value)}
        />
      )}
    </LynxKiteNode >
  );
}

export default NodeWithParams;
