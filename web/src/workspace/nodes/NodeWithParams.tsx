import { useContext } from 'react';
import { LynxKiteState } from '../LynxKiteState';
import LynxKiteNode from './LynxKiteNode';
import { useReactFlow } from '@xyflow/react';
import NodeParameter from './NodeParameter';

function NodeWithParams(props: any) {
  const reactFlow = useReactFlow();
  const metaParams = props.data.meta?.params;
  const state = useContext(LynxKiteState);
  function setParam(name: string, newValue: any) {
    reactFlow.updateNodeData(props.id, { params: { ...props.data.params, [name]: newValue } });
  }
  const params = props.data?.params ? Object.entries(props.data.params) : [];

  return (
    <LynxKiteNode {...props}>
      {params.map(([name, value]) =>
        <NodeParameter
          name={name}
          key={name}
          value={value}
          meta={metaParams?.[name]}
          onChange={(value: any) => setParam(name, value)}
        />
      )}
    </LynxKiteNode >
  );
}

export default NodeWithParams;
