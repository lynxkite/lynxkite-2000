import LynxKiteNode from './LynxKiteNode';
import { useReactFlow } from '@xyflow/react';
import NodeParameter from './NodeParameter';

export type UpdateOptions = { delay?: number };

function NodeWithParams(props: any) {
  const reactFlow = useReactFlow();
  const metaParams = props.data.meta?.params;
  function setParam(name: string, newValue: any, opts: UpdateOptions) {
    reactFlow.updateNodeData(props.id, { params: { ...props.data.params, [name]: newValue }, __execution_delay: opts.delay || 0 });
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
          onChange={(value: any, opts?: UpdateOptions) => setParam(name, value, opts || {})}
        />
      )}
    </LynxKiteNode >
  );
}

export default NodeWithParams;
