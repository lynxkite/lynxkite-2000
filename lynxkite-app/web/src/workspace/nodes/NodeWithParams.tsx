import { useReactFlow } from "@xyflow/react";
import React from "react";
// @ts-ignore
import Triangle from "~icons/tabler/triangle-inverted-filled.jsx";
import LynxKiteNode from "./LynxKiteNode";
import NodeGroupParameter from "./NodeGroupParameter";
import NodeParameter from "./NodeParameter";

export type UpdateOptions = { delay?: number };

function NodeWithParams(props: any) {
  const reactFlow = useReactFlow();
  const metaParams = props.data.meta?.params;
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
  const params = props.data?.params ? Object.entries(props.data.params) : [];

  return (
    <LynxKiteNode {...props}>
      {props.collapsed && params.length > 0 && (
        <div className="params-expander" onClick={() => setCollapsed(!collapsed)}>
          <Triangle className={`flippy ${collapsed ? "flippy-90" : ""}`} />
        </div>
      )}
      {!collapsed &&
        params.map(([name, value]) =>
          metaParams?.[name]?.type === "group" ? (
            <NodeGroupParameter
              key={name}
              value={value}
              data={props.data}
              meta={metaParams?.[name]}
              setParam={(name: string, value: any, opts?: UpdateOptions) =>
                setParam(name, value, opts || {})
              }
              deleteParam={(name: string, opts?: UpdateOptions) => deleteParam(name, opts || {})}
            />
          ) : (
            <NodeParameter
              name={name}
              key={name}
              value={value}
              data={props.data}
              meta={metaParams?.[name]}
              onChange={(value: any, opts?: UpdateOptions) => setParam(name, value, opts || {})}
            />
          ),
        )}
      {props.children}
    </LynxKiteNode>
  );
}

export default NodeWithParams;
