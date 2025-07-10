// @ts-ignore
import ArrowsHorizontal from "~icons/tabler/arrows-horizontal.jsx";
// @ts-ignore
import Help from "~icons/tabler/question-mark.jsx";
import Tooltip from "../../Tooltip";
import ModelMapping from "./ModelMappingParameter";
import NodeGroupParameter from "./NodeGroupParameter";
import ParameterInput from "./ParameterInput";

const BOOLEAN = "<class 'bool'>";
const MODEL_TRAINING_INPUT_MAPPING =
  "<class 'lynxkite_graph_analytics.ml_ops.ModelTrainingInputMapping'>";
const MODEL_INFERENCE_INPUT_MAPPING =
  "<class 'lynxkite_graph_analytics.ml_ops.ModelInferenceInputMapping'>";
const MODEL_OUTPUT_MAPPING = "<class 'lynxkite_graph_analytics.ml_ops.ModelOutputMapping'>";
const ATRIBUTE_BASED_BATCHING_MAPPING =
  "<class 'lynxkite_graph_analytics.ml_ops.AttributeBasedBatchingMapping'>";
const STRING_TRIPLE = "tuple[str, str, str]";

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

// Helper for editing a mapping from master columns to dataframes
function AttrributeBasedBatchingMapping({
  value,
  onChange,
  data,
}: {
  value: string;
  onChange: (value: string, options?: { delay: number }) => void;
  data: any;
}) {
  // Parse the value as JSON or use empty object
  const v: any = parseJsonOrEmpty(value);
  v.master_df_name ??= "";
  v.map ??= {};

  // Get available dataframes
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
  const nonMasterDfs = Object.keys(dfs).filter((df) => df !== v.master_df_name);
  console.log(dfs);
  // Get columns of the selected master dataframe
  const masterColumns: string[] =
    v.master_df_name && dfs[v.master_df_name] ? dfs[v.master_df_name] : [];

  function updateMasterDfName(dfName: string) {
    // Reset mapping if master df changes
    onChange(JSON.stringify({ master_df_name: dfName, map: {} }));
  }

  function updateMapping(col: string, df: string) {
    const newMapping = {
      ...v.map,
      [col]: { df: df, column: "" },
    };
    onChange(JSON.stringify({ master_df_name: v.master_df_name, map: newMapping }));
  }

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <label>
          Master dataframe:&nbsp;
          <select
            className="select select-ghost"
            value={v.master_df_name}
            onChange={(evt) => updateMasterDfName(evt.currentTarget.value)}
          >
            <option key="" value="" />
            {Object.keys(dfs).map((df) => (
              <option key={df} value={df}>
                {df}
              </option>
            ))}
          </select>
        </label>
      </div>
      {v.master_df_name && (
        <table className="model-mapping-param">
          <tbody>
            {masterColumns.length > 0 ? (
              masterColumns.map((col) => (
                <tr key={col}>
                  <td>{col}</td>
                  <td>
                    <ArrowsHorizontal />
                  </td>
                  <td>
                    <select
                      className="select select-ghost"
                      value={v.map[col]?.df || ""}
                      onChange={(evt) => updateMapping(col, evt.currentTarget.value)}
                    >
                      <option key="" value="" />
                      {nonMasterDfs.map((df) => (
                        <option key={df} value={df}>
                          {df}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td>no columns</td>
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
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
        value={value || ""}
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
  ) : meta?.type?.type === ATRIBUTE_BASED_BATCHING_MAPPING ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <AttrributeBasedBatchingMapping value={value} data={data} onChange={onChange} />
    </label>
  ) : meta?.type?.type?.startsWith(STRING_TRIPLE) ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      {(() => {
        const parts = (value ?? "  ,  ,  ").split(",").map((p: string) => p.trim());
        const update = (idx: number, val: string, opts?: UpdateOptions) => {
          parts[idx] = val;
          onChange(parts.join(","), opts);
        };
        return (
          <div className="flex gap-1">
            <Input value={parts[0]} onChange={(v, o) => update(0, v, o)} />
            <Input value={parts[1]} onChange={(v, o) => update(1, v, o)} />
            <Input value={parts[2]} onChange={(v, o) => update(2, v, o)} />
          </div>
        );
      })()}
    </label>
  ) : (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ParameterInput value={value} onChange={onChange} />
    </label>
  );
}
