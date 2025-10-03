// @ts-expect-error
import Close from "~icons/tabler/x.jsx";
import type { WorkspaceNodeData } from "../../apiTypes";
import NodeParameter from "./NodeParameter";

type NamedId = {
  name: string;
  id: string;
};

function getInputs(data: WorkspaceNodeData): any[] {
  return (data?.input_metadata as any)?.value ?? data?.input_metadata ?? [];
}

function getAllModels(data: WorkspaceNodeData): any[] {
  const models: any[] = [];
  for (const input of getInputs(data)) {
    const other = input.other ?? {};
    for (const e of Object.values(other) as any[]) {
      if (e.type === "pytorch-model") {
        models.push(e.model);
      }
    }
  }
  return models;
}

function getHandlers(data: WorkspaceNodeData): any {
  const handlers = {};
  for (const model of getAllModels(data)) {
    Object.assign(handlers, model.input_handlers);
  }
  return handlers;
}

function getModelBindings(
  data: WorkspaceNodeData,
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
  for (const model of getAllModels(data)) {
    for (const id of bindingsOfModel(model)) {
      bindings.add({ id, name: model.input_output_names[id] ?? id });
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

function BindingMapping({ meta, params, data, setParam }: any) {
  params = { ...params };
  for (const p of meta?.params ?? []) {
    params[p.name] = params[p.name] ?? p.default;
  }
  return meta.params.map((paramMeta: any) => (
    <NodeParameter
      name={paramMeta.name}
      key={paramMeta.name}
      value={params[paramMeta.name]}
      data={{ ...data, params, meta: { value: meta } }}
      meta={paramMeta}
      setParam={setParam}
    />
  ));
}

export default function ModelMapping({ value, onChange, data, variant }: any) {
  const v: any = parseJsonOrEmpty(value);
  v.map ??= {};
  const bindings = getModelBindings(data, variant);
  const handlers = getHandlers(data);
  function setBindingParam(bindingId: string, name: string, newValue: any) {
    const newMap = {
      ...v.map,
      [bindingId]: { ...v.map[bindingId], [name]: newValue },
    };
    onChange(JSON.stringify({ map: newMap }));
  }
  const outputBindingMeta = {
    params: [
      {
        name: "table_name",
        default: "",
        type: {
          format: "dropdown",
          metadata_query: "[].dataframes[].keys(@)[]",
        },
      },
      {
        name: "column",
        default: "",
        type: "str",
      },
    ],
  };
  return (
    <div className="model-mapping-param">
      <div className="model-mapping-param-tools">
        <button onClick={() => onChange('{"map":{}}')}>
          <Close />
        </button>
      </div>
      <table>
        <tbody>
          {bindings.length > 0 ? (
            bindings.map((binding: NamedId) => (
              <tr key={binding.id}>
                <td className="model-mapping-param-label">{binding.name}:</td>
                <td className="model-mapping-param-inputs">
                  <BindingMapping
                    data={data}
                    meta={variant === "output" ? outputBindingMeta : handlers[binding.id]}
                    params={v.map[binding.id] ?? {}}
                    setParam={(name: string, newValue: any) =>
                      setBindingParam(binding.id, name, newValue)
                    }
                  />
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
    </div>
  );
}
