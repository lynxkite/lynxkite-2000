import { useReactFlow } from "@xyflow/react";
import React from "react";
// @ts-expect-error
import Triangle from "~icons/tabler/triangle-inverted-filled.jsx";
import LynxKiteNode from "./LynxKiteNode";
import NodeParameter, { type UpdateOptions } from "./NodeParameter";

export function NodeWithParams(props: any) {
  const reactFlow = useReactFlow();
  const metaParams = props.data.meta?.value?.params ?? [];
  const [collapsed, setCollapsed] = React.useState(props.collapsed);

  function setParam(name: string, newValue: any, opts: UpdateOptions) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { ...prevData.data.params, [name]: newValue },
      __execution_delay: opts.delay || 0,
    }));
  }

  return (
    <>
      {props.collapsed && metaParams.length > 0 && (
        <div className="params-expander" onClick={() => setCollapsed(!collapsed)}>
          <Triangle className={`flippy ${collapsed ? "flippy-90" : ""}`} />
        </div>
      )}
      {!collapsed &&
        metaParams.map((meta: any) => (
          <NodeParameter
            name={meta.name}
            key={meta.name}
            value={props.data.params[meta.name]}
            data={props.data}
            meta={meta}
            setParam={setParam}
          />
        ))}
      {props.children}
    </>
  );
}

export default LynxKiteNode(NodeWithParams);
