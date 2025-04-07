import { useReactFlow } from "@xyflow/react";
import { useState } from "react";
import React from "react";
import Markdown from "react-markdown";
import LynxKiteNode from "./LynxKiteNode";
import Table from "./Table";

function toMD(v: any): string {
  if (typeof v === "string") {
    return v;
  }
  if (Array.isArray(v)) {
    return v.map(toMD).join("\n\n");
  }
  return JSON.stringify(v);
}

export default function NodeWithTableView(props: any) {
  const reactFlow = useReactFlow();
  const [open, setOpen] = useState(
    props.data?.params?.tables_open ?? ({} as { [name: string]: boolean }),
  );
  const display = props.data.display?.value;
  const single = display?.dataframes && Object.keys(display?.dataframes).length === 1;
  const dfs = Object.entries(display?.dataframes || {});
  dfs.sort();
  function setParam(name: string, newValue: any) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { ...prevData.data.params, [name]: newValue },
    }));
  }
  function toggleTable(name: string) {
    setOpen((prevOpen) => ({ ...prevOpen, [name]: !prevOpen[name] }));
    setParam("tables_open", { ...open, [name]: !open[name] });
  }
  return (
    <LynxKiteNode {...props}>
      {display && [
        dfs.map(([name, df]: [string, any]) => (
          <React.Fragment key={name}>
            {!single && (
              <div key={`${name}-header`} className="df-head" onClick={() => toggleTable(name)}>
                {name}
              </div>
            )}
            {(single || open[name]) &&
              (df.data.length > 1 ? (
                <Table key={`${name}-table`} columns={df.columns} data={df.data} />
              ) : df.data.length ? (
                <dl key={`${name}-dl`}>
                  {df.columns.map((c: string, i: number) => (
                    <React.Fragment key={`${name}-${c}`}>
                      <dt>{c}</dt>
                      <dd>
                        <Markdown>{toMD(df.data[0][i])}</Markdown>
                      </dd>
                    </React.Fragment>
                  ))}
                </dl>
              ) : (
                JSON.stringify(df.data)
              ))}
          </React.Fragment>
        )),
        Object.entries(display.others || {}).map(([name, o]) => (
          <>
            <div key={`${name}-header`} className="df-head" onClick={() => toggleTable(name)}>
              {name}
            </div>
            {open[name] && <pre>{(o as any).toString()}</pre>}
          </>
        )),
      ]}
    </LynxKiteNode>
  );
}
