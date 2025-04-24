import { useReactFlow } from "@xyflow/react";
import React from "react";
// @ts-ignore
import Triangle from "~icons/tabler/triangle-inverted-filled.jsx";
import LynxKiteNode from "./LynxKiteNode";
import NodeGroupParameter from "./NodeGroupParameter";
import NodeParameter from "./NodeParameter";

export type UpdateOptions = { delay?: number };

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

  function deleteParam(name: string, opts: UpdateOptions) {
    if (props.data.params[name] === undefined) {
      return;
    }
    delete props.data.params[name];
    reactFlow.updateNodeData(props.id, {
      params: { ...props.data.params },
      __execution_delay: opts.delay || 0,
    });
  }
  return (
    <>
      {props.collapsed && metaParams.length > 0 && (
        <div className="params-expander" onClick={() => setCollapsed(!collapsed)}>
          <Triangle className={`flippy ${collapsed ? "flippy-90" : ""}`} />
        </div>
      )}
      {!collapsed &&
        metaParams.map((meta: any) =>
          meta.type === "group" ? (
            <NodeGroupParameter
              key={meta.name}
              value={props.data.params[meta.name]}
              data={props.data}
              meta={meta}
              setParam={(name: string, value: any, opts?: UpdateOptions) =>
                setParam(name, value, opts || {})
              }
              deleteParam={(name: string, opts?: UpdateOptions) => deleteParam(name, opts || {})}
            />
          ) : (
            <NodeParameter
              name={meta.name}
              key={meta.name}
              value={props.data.params[meta.name]}
              data={props.data}
              meta={meta}
              onChange={(value: any, opts?: UpdateOptions) =>
                setParam(meta.name, value, opts || {})
              }
            />
          ),
        )}
      {props.children}
    </>
  );
}

export default LynxKiteNode(NodeWithParams);
