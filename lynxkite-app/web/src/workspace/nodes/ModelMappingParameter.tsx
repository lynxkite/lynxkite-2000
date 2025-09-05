import { useRef } from "react";
// @ts-expect-error
import ArrowsHorizontal from "~icons/tabler/arrows-horizontal.jsx";
import ParameterInput from "./ParameterInput";

type Bindings = {
  [key: string]: {
    df: string;
    column: string;
  };
};

type NamedId = {
  name: string;
  id: string;
};

function getModelBindings(
  data: any,
  variant: "training input" | "inference input" | "output",
): NamedId[] {
  function bindingsOfModel(m: any): string[] {
    switch (variant) {
      case "training input":
        return [
          ...m.model_inputs,
          ...m.loss_inputs.filter((i: string) => !m.model_outputs.includes(i)),
        ];
      case "inference input":
        return m.model_inputs;
      case "output":
        return m.model_outputs;
    }
  }
  const bindings = new Set<NamedId>();
  const inputs = data?.input_metadata?.value ?? data?.input_metadata ?? [];
  for (const input of inputs) {
    const other = input.other ?? {};
    for (const e of Object.values(other) as any[]) {
      if (e.type === "pytorch-model") {
        for (const id of bindingsOfModel(e.model)) {
          bindings.add({ id, name: e.model.input_output_names[id] ?? id });
        }
      }
    }
  }
  const list = [...bindings];
  list.sort((a, b) => {
    if (a.name < b.name) return -1;
    if (a.name > b.name) return 1;
    return 0;
  });
  return list;
}

function parseJsonOrEmpty(json: string): object {
  try {
    const j = JSON.parse(json);
    if (j !== null && typeof j === "object") {
      return j;
    }
  } catch (_) {}
  return {};
}

export default function ModelMapping({ value, onChange, data, variant }: any) {
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
      const df = dfsRef.current[binding.id]?.value ?? "";
      const column = columnsRef.current[binding.id]?.value ?? "";
      if (df.length || column.length) {
        map[binding.id] = { df, column };
      }
    }
    return map;
  }
  return (
    <table className="model-mapping-param">
      <tbody>
        {bindings.length > 0 ? (
          bindings.map((binding: NamedId) => (
            <tr key={binding.id}>
              <td>{binding.name}</td>
              <td>
                <ArrowsHorizontal />
              </td>
              <td>
                <select
                  className="select select-ghost"
                  value={v.map?.[binding.id]?.df}
                  ref={(el) => {
                    dfsRef.current[binding.id] = el;
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
                  <ParameterInput
                    inputRef={(el) => {
                      columnsRef.current[binding.id] = el;
                    }}
                    value={v.map?.[binding.id]?.column}
                    onChange={(column, options) => {
                      const map = getMap();
                      // At this point the <input> has not been updated yet. We use the value from the event.
                      const df = dfsRef.current[binding.id]?.value ?? "";
                      map[binding.id] ??= { df, column };
                      map[binding.id].column = column;
                      onChange(JSON.stringify({ map }), options);
                    }}
                  />
                ) : (
                  <select
                    className="select select-ghost"
                    value={v.map?.[binding.id]?.column}
                    ref={(el) => {
                      columnsRef.current[binding.id] = el;
                    }}
                    onChange={() => onChange(JSON.stringify({ map: getMap() }))}
                  >
                    <option key="" value="" />
                    {dfs[v.map?.[binding.id]?.df]?.map((col: string) => (
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
