// @ts-ignore
import ArrowsHorizontal from "~icons/tabler/arrows-horizontal.jsx";

const BOOLEAN = "<class 'bool'>";
const MODEL_TRAINING_INPUT_MAPPING =
  "<class 'lynxkite_graph_analytics.lynxkite_ops.ModelTrainingInputMapping'>";
const MODEL_INFERENCE_INPUT_MAPPING =
  "<class 'lynxkite_graph_analytics.lynxkite_ops.ModelInferenceInputMapping'>";
const MODEL_OUTPUT_MAPPING = "<class 'lynxkite_graph_analytics.lynxkite_ops.ModelOutputMapping'>";
function ParamName({ name }: { name: string }) {
  return <span className="param-name bg-base-200">{name.replace(/_/g, " ")}</span>;
}

function Input({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string, options?: { delay: number }) => void;
}) {
  return (
    <input
      className="input input-bordered w-full"
      value={value || ""}
      onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
      onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
      onKeyDown={(evt) => evt.code === "Enter" && onChange(evt.currentTarget.value, { delay: 0 })}
    />
  );
}

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
  const v: any = parseJsonOrEmpty(value);
  v.map ??= {};
  const dfs: { [df: string]: string[] } = {};
  const inputs = data?.input_metadata?.value ?? data?.input_metadata ?? [];
  for (const input of inputs) {
    const dataframes = input.dataframes as {
      [df: string]: { columns: string[] };
    };
    for (const [df, { columns }] of Object.entries(dataframes)) {
      dfs[df] = columns;
    }
  }
  const bindings = getModelBindings(data, variant);
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
                  onChange={(evt) => {
                    const df = evt.currentTarget.value;
                    if (df === "") {
                      const map = { ...v.map, [binding]: undefined };
                      onChange(JSON.stringify({ map }));
                    } else {
                      const columnSpec = {
                        column: dfs[df][0],
                        ...(v.map?.[binding] || {}),
                        df,
                      };
                      const map = { ...v.map, [binding]: columnSpec };
                      onChange(JSON.stringify({ map }));
                    }
                  }}
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
                    value={v.map?.[binding]?.column}
                    onChange={(column, options) => {
                      const columnSpec = {
                        ...(v.map?.[binding] || {}),
                        column,
                      };
                      const map = { ...v.map, [binding]: columnSpec };
                      onChange(JSON.stringify({ map }), options);
                    }}
                  />
                ) : (
                  <select
                    className="select select-ghost"
                    value={v.map?.[binding]?.column}
                    onChange={(evt) => {
                      const column = evt.currentTarget.value;
                      const columnSpec = {
                        ...(v.map?.[binding] || {}),
                        column,
                      };
                      const map = { ...v.map, [binding]: columnSpec };
                      onChange(JSON.stringify({ map }));
                    }}
                  >
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
  onChange: (value: any, options?: { delay: number }) => void;
}

export default function NodeParameter({ name, value, meta, data, onChange }: NodeParameterProps) {
  return (
    // biome-ignore lint/a11y/noLabelWithoutControl: Most of the time there is a control.
    <label className="param">
      {meta?.type?.format === "collapsed" ? (
        <>
          <ParamName name={name} />
          <button className="collapsed-param">⋯</button>
        </>
      ) : meta?.type?.format === "textarea" ? (
        <>
          <ParamName name={name} />
          <textarea
            className="textarea textarea-bordered w-full"
            rows={6}
            value={value}
            onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
            onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
          />
        </>
      ) : meta?.type?.enum ? (
        <>
          <ParamName name={name} />
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
        </>
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
        <>
          <ParamName name={name} />
          <ModelMapping value={value} data={data} variant="training input" onChange={onChange} />
        </>
      ) : meta?.type?.type === MODEL_INFERENCE_INPUT_MAPPING ? (
        <>
          <ParamName name={name} />
          <ModelMapping value={value} data={data} variant="inference input" onChange={onChange} />
        </>
      ) : meta?.type?.type === MODEL_OUTPUT_MAPPING ? (
        <>
          <ParamName name={name} />
          <ModelMapping value={value} data={data} variant="output" onChange={onChange} />
        </>
      ) : (
        <>
          <ParamName name={name} />
          <Input value={value} onChange={onChange} />
        </>
      )}
    </label>
  );
}
