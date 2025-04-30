import { useRef } from "react";
// @ts-ignore
import ArrowsHorizontal from "~icons/tabler/arrows-horizontal.jsx";
// @ts-ignore
import Help from "~icons/tabler/question-mark.jsx";
import Tooltip from "../../Tooltip";
import NodeGroupParameter from "./NodeGroupParameter";

const BOOLEAN = "<class 'bool'>";
const MODEL_TRAINING_INPUT_MAPPING =
  "<class 'lynxkite_graph_analytics.ml_ops.ModelTrainingInputMapping'>";
const MODEL_INFERENCE_INPUT_MAPPING =
  "<class 'lynxkite_graph_analytics.ml_ops.ModelInferenceInputMapping'>";
const MODEL_OUTPUT_MAPPING = "<class 'lynxkite_graph_analytics.ml_ops.ModelOutputMapping'>";

function ParamName({ name, doc }: { name: string; doc: string }) {
  const help = doc && (
    <Tooltip doc={doc} width={200}>
      <Help />
    </Tooltip>
  );
  return (
    <div className="param-name-row">
      <span className="param-name bg-base-200">{name.replace(/_/g, " ")}</span>
      {help}
    </div>
  );
}

function Input({
  value,
  onChange,
  inputRef,
}: {
  value: string;
  onChange: (value: string, options?: { delay: number }) => void;
  inputRef?: React.Ref<HTMLInputElement>;
}) {
  return (
    <input
      className="input input-bordered w-full"
      ref={inputRef}
      value={value ?? ""}
      onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
      onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
      onKeyDown={(evt) => evt.code === "Enter" && onChange(evt.currentTarget.value, { delay: 0 })}
    />
  );
}

type Bindings = {
  [key: string]: {
    df: string;
    column: string;
  };
};

function getModelBindings(
  data: any,
  variant: "training input" | "inference input" | "output",
): string[] {
  function bindingsOfModel(m: any): string[] {
    switch (variant) {
      case "training input":
        return [...m.inputs, ...m.loss_inputs.filter((i: string) => !m.outputs.includes(i))];
      case "inference input":
        return m.inputs;
      case "output":
        return m.outputs;
    }
  }
  const bindings = new Set<string>();
  const inputs = data?.input_metadata?.value ?? data?.input_metadata ?? [];
  for (const input of inputs) {
    const other = input.other ?? {};
    for (const e of Object.values(other) as any[]) {
      if (e.type === "model") {
        for (const b of bindingsOfModel(e.model)) {
          bindings.add(b);
        }
      }
    }
  }
  const list = [...bindings];
  list.sort();
  return list;
}

function parseJsonOrEmpty(json: string): object {
  try {
    const j = JSON.parse(json);
    if (j !== null && typeof j === "object") {
      return j;
    }
  } catch (e) {}
  return {};
}

function ModelMapping({ value, onChange, data, variant }: any) {
  const dfsRef = useRef({} as { [binding: string]: HTMLSelectElement | null });
  const columnsRef = useRef(
    {} as { [binding: string]: HTMLSelectElement | HTMLInputElement | null },
  );
  const v: any = parseJsonOrEmpty(value);
  v.map ??= {};
  const dfs: { [df: string]: string[] } = {};
  const inputs = data?.input_metadata?.value ?? data?.input_metadata ?? [];
  for (const input of inputs) {
    if (!input.dataframes) continue;
    const dataframes = input.dataframes as {
      [df: string]: { columns: string[] };
    };
    for (const [df, { columns }] of Object.entries(dataframes)) {
      dfs[df] = columns;
    }
  }
  const bindings = getModelBindings(data, variant);
  function getMap() {
    const map: Bindings = {};
    for (const binding of bindings) {
      const df = dfsRef.current[binding]?.value ?? "";
      const column = columnsRef.current[binding]?.value ?? "";
      if (df.length || column.length) {
        map[binding] = { df, column };
      }
    }
    return map;
  }
  return (
    <table className="model-mapping-param">
      <tbody>
        {bindings.length > 0 ? (
          bindings.map((binding: string) => (
            <tr key={binding}>
              <td>{binding}</td>
              <td>
                <ArrowsHorizontal />
              </td>
              <td>
                <select
                  className="select select-ghost"
                  value={v.map?.[binding]?.df}
                  ref={(el) => {
                    dfsRef.current[binding] = el;
                  }}
                  onChange={() => onChange(JSON.stringify({ map: getMap() }))}
                >
                  <option key="" value="" />
                  {Object.keys(dfs).map((df: string) => (
                    <option key={df} value={df}>
                      {df}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                {variant === "output" ? (
                  <Input
                    inputRef={(el) => {
                      columnsRef.current[binding] = el;
                    }}
                    value={v.map?.[binding]?.column}
                    onChange={(column, options) => {
                      const map = getMap();
                      // At this point the <input> has not been updated yet. We use the value from the event.
                      const df = dfsRef.current[binding]?.value ?? "";
                      map[binding] ??= { df, column };
                      map[binding].column = column;
                      onChange(JSON.stringify({ map }), options);
                    }}
                  />
                ) : (
                  <select
                    className="select select-ghost"
                    value={v.map?.[binding]?.column}
                    ref={(el) => {
                      columnsRef.current[binding] = el;
                    }}
                    onChange={() => onChange(JSON.stringify({ map: getMap() }))}
                  >
                    <option key="" value="" />
                    {dfs[v.map?.[binding]?.df]?.map((col: string) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                )}
              </td>
            </tr>
          ))
        ) : (
          <tr>
            <td>no bindings</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

interface NodeParameterProps {
  name: string;
  value: any;
  meta: any;
  data: any;
  setParam: (name: string, value: any, options: UpdateOptions) => void;
}

export type UpdateOptions = { delay?: number };

function findDocs(docs: any, parameter: string) {
  for (const sec of docs) {
    if (sec.kind === "parameters") {
      for (const p of sec.value) {
        if (p.name === parameter) {
          return p.description;
        }
      }
    }
  }
}

export default function NodeParameter({ name, value, meta, data, setParam }: NodeParameterProps) {
  const doc = findDocs(data.meta?.value?.doc ?? [], name);
  function onChange(value: any, opts?: UpdateOptions) {
    setParam(meta.name, value, opts || {});
  }
  return meta?.type?.format === "collapsed" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <button className="collapsed-param">⋯</button>
    </label>
  ) : meta?.type?.format === "textarea" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <textarea
        className="textarea textarea-bordered w-full"
        rows={6}
        value={value}
        onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
        onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
      />
    </label>
  ) : meta?.type === "group" ? (
    <NodeGroupParameter meta={meta} data={data} setParam={setParam} />
  ) : meta?.type?.enum ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <select
        className="select select-bordered w-full"
        value={value || meta.type.enum[0]}
        onChange={(evt) => onChange(evt.currentTarget.value)}
      >
        {meta.type.enum.map((option: string) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  ) : meta?.type?.type === BOOLEAN ? (
    <div className="form-control">
      <label className="label cursor-pointer">
        {name.replace(/_/g, " ")}
        <input
          className="checkbox"
          type="checkbox"
          checked={value}
          onChange={(evt) => onChange(evt.currentTarget.checked)}
        />
      </label>
    </div>
  ) : meta?.type?.type === MODEL_TRAINING_INPUT_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ModelMapping value={value} data={data} variant="training input" onChange={onChange} />
    </label>
  ) : meta?.type?.type === MODEL_INFERENCE_INPUT_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ModelMapping value={value} data={data} variant="inference input" onChange={onChange} />
    </label>
  ) : meta?.type?.type === MODEL_OUTPUT_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ModelMapping value={value} data={data} variant="output" onChange={onChange} />
    </label>
  ) : (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <Input value={value} onChange={onChange} />
    </label>
  );
}
