import jmespath from "jmespath";
import Tooltip from "../../Tooltip";
import ModelMapping from "./ModelMappingParameter";
import NodeGroupParameter from "./NodeGroupParameter";
import ParameterInput from "./ParameterInput";
import PathStrInput from "./parameters/PathStrInput";

const BOOLEAN = "<class 'bool'>";
const MODEL_TRAINING_INPUT_MAPPING =
  "lynxkite_graph_analytics.operations.ml_ops.ModelTrainingInputMapping | None";
const MODEL_INFERENCE_INPUT_MAPPING =
  "lynxkite_graph_analytics.operations.ml_ops.ModelInferenceInputMapping | None";
const MODEL_OUTPUT_MAPPING = "lynxkite_graph_analytics.operations.ml_ops.ModelOutputMapping | None";

function ParamName({ name, doc }: { name: string; doc: string }) {
  return (
    <div className="param-name-row">
      <Tooltip doc={doc}>
        <span className="param-name">{name.replace(/_/g, " ")}</span>
      </Tooltip>
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

function DropdownTextAdder({
  value,
  onChange,
  options,
}: {
  value: [string, string][];
  onChange: (v: [string, string][], opts?: UpdateOptions) => void;
  options: string[];
}) {
  const safeValue: [string, string][] = Array.isArray(value) ? value : [];

  const addRow = () => {
    onChange([...safeValue, ["", ""]]);
  };

  const updateFirst = (index: number, v: string, opts?: UpdateOptions) => {
    onChange(
      safeValue.map((row, i) => (i === index ? [v, row[1]] : row)),
      opts,
    );
  };

  const updateSecond = (index: number, v: string, opts?: UpdateOptions) => {
    onChange(
      safeValue.map((row, i) => (i === index ? [row[0], v] : row)),
      opts,
    );
  };

  const removeRow = (index: number) => {
    onChange(safeValue.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      {safeValue.map((val, index) => (
        <div key={index} className="flex gap-2 items-start">
          <select
            className="select select-bordered w-full"
            value={val[0]}
            onChange={(e) => updateFirst(index, e.target.value)}
          >
            <option value="" disabled>
              Select...
            </option>
            {options.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>

          <input
            className="input input-bordered w-full"
            value={val[1] ?? ""}
            onChange={(evt) => updateSecond(index, evt.currentTarget.value, { delay: 2 })}
            onBlur={(evt) => updateSecond(index, evt.currentTarget.value, { delay: 0 })}
            onKeyDown={(evt) => {
              if (evt.code === "Enter") {
                updateSecond(index, evt.currentTarget.value, { delay: 0 });
              }
            }}
          />

          <button
            type="button"
            className="btn btn-error btn-sm mt-1"
            onClick={() => removeRow(index)}
          >
            ✕
          </button>
        </div>
      ))}

      <button type="button" className="btn btn-sm" onClick={addRow}>
        + Add
      </button>
    </div>
  );
}

function DoubleTextAdder({
  value,
  onChange,
}: {
  value: [string, string][];
  onChange: (v: [string, string][], opts?: UpdateOptions) => void;
}) {
  const safeValue: [string, string][] = Array.isArray(value) ? value : [];

  const addRow = () => {
    onChange([...safeValue, ["", ""]]);
  };

  const updateFirst = (index: number, v: string, opts?: UpdateOptions) => {
    onChange(
      safeValue.map((row, i) => (i === index ? [v, row[1]] : row)),
      opts,
    );
  };

  const updateSecond = (index: number, v: string, opts?: UpdateOptions) => {
    onChange(
      safeValue.map((row, i) => (i === index ? [row[0], v] : row)),
      opts,
    );
  };

  const removeRow = (index: number) => {
    onChange(safeValue.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      {safeValue.map((val, index) => (
        <div key={index} className="flex gap-2 items-start">
          <input
            type="text"
            className="input input-bordered w-full"
            value={val[0] ?? ""}
            onChange={(evt) => updateFirst(index, evt.currentTarget.value, { delay: 2 })}
            onBlur={(evt) => updateFirst(index, evt.currentTarget.value, { delay: 0 })}
            onKeyDown={(evt) => {
              if (evt.code === "Enter") {
                updateFirst(index, evt.currentTarget.value, { delay: 0 });
              }
            }}
          />

          <input
            type="text"
            className="input input-bordered w-full"
            value={val[1] ?? ""}
            onChange={(evt) => updateSecond(index, evt.currentTarget.value, { delay: 2 })}
            onBlur={(evt) => updateSecond(index, evt.currentTarget.value, { delay: 0 })}
            onKeyDown={(evt) => {
              if (evt.code === "Enter") {
                updateSecond(index, evt.currentTarget.value, { delay: 0 });
              }
            }}
          />

          <button
            type="button"
            className="btn btn-error btn-sm mt-1"
            onClick={() => removeRow(index)}
          >
            ✕
          </button>
        </div>
      ))}

      <button type="button" className="btn btn-sm" onClick={addRow}>
        + Add
      </button>
    </div>
  );
}

export default function NodeParameter({ name, value, meta, data, setParam }: NodeParameterProps) {
  const doc = findDocs(data.meta?.doc ?? [], name);
  function onChange(value: any, opts?: UpdateOptions) {
    setParam(meta.name, value, opts || {});
  }
  return meta?.type?.format === "textarea" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <textarea
        className="textarea textarea-bordered w-full"
        rows={(value ?? "").split("\n").length}
        value={value ?? ""}
        onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
        onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
      />
    </label>
  ) : meta?.type?.format === "dropdown" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <select
        className="select select-bordered appearance-none w-full"
        value={value ?? ""}
        onChange={(evt) => onChange(evt.currentTarget.value)}
      >
        {getDropDownValues(data, meta?.type?.metadata_query).map((option: string) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  ) : meta?.type?.format === "multi-dropdown" ? (
    <div className="param">
      <ParamName name={name} doc={doc} />

      <div className="dropdown dropdown-bottom w-full">
        <button
          tabIndex={0}
          className="border border-base-300 bg-base-100 rounded-lg h-12 px-4 w-full flex items-center justify-between cursor-pointer"
        >
          <span className="truncate text-sm text-base-content/90">
            {(Array.isArray(value) ? value : []).filter(Boolean).join(", ")}
          </span>
          <span className="text-[10px] opacity-60 text-base-content">▼</span>
        </button>

        <div
          tabIndex={0}
          className="dropdown-content left-0 mt-1 w-full bg-white z-[999] rounded-lg border p-2 max-h-60 overflow-y-auto"
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          {(getDropDownValues(data, meta?.type?.metadata_query) as string[])
            ?.filter((option: string) => option && option.trim() !== "")
            ?.map((option: string) => {
              const currentSelection: string[] = Array.isArray(value) ? value : [];
              const isChecked: boolean = currentSelection.includes(option);

              return (
                <label
                  key={option}
                  className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 p-2 rounded text-gray-900"
                >
                  <input
                    type="checkbox"
                    className="checkbox checkbox-sm checkbox-primary"
                    checked={isChecked}
                    onChange={(evt) => {
                      const target = evt.target as HTMLInputElement;
                      const nextSelection: string[] = target.checked
                        ? [...currentSelection, option]
                        : currentSelection.filter((v: string) => v !== option);

                      onChange(nextSelection);
                    }}
                  />
                  <span className="text-sm select-none">{option}</span>
                </label>
              );
            })}
        </div>
      </div>
    </div>
  ) : meta?.type?.format === "dropdown-textbox_adder" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <DropdownTextAdder
        value={value ?? []}
        onChange={onChange}
        options={getDropDownValues(data, meta?.type?.metadata_query1)}
      />
    </label>
  ) : meta?.type?.format === "double-textbox_adder" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <DoubleTextAdder value={value ?? []} onChange={onChange} />
    </label>
  ) : meta?.type?.format === "double-dropdown" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <div className="double-dropdown">
        <select
          className="select select-bordered appearance-none double-dropdown-first"
          value={value?.[0] ?? ""}
          onChange={(evt) => onChange([evt.currentTarget.value, value?.[1]])}
        >
          {getDropDownValues(data, meta?.type?.metadata_query1).map((option: string) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <select
          className="select select-bordered appearance-none double-dropdown-second"
          value={value?.[1] ?? ""}
          onChange={(evt) => onChange([value?.[0], evt.currentTarget.value])}
        >
          {getDropDownValues(data, meta?.type?.metadata_query2, { first: value?.[0] }).map(
            (option: string) => (
              <option key={option} value={option}>
                {option}
              </option>
            ),
          )}
        </select>
      </div>
    </label>
  ) : meta?.type === "group" ? (
    <NodeGroupParameter meta={meta} data={data} setParam={setParam} />
  ) : meta?.type?.enum ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <select
        className="select select-bordered appearance-none w-full"
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
      <label className="label cursor-pointer checkbox-param">
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
  ) : meta?.type?.format === "path" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <PathStrInput value={value ?? ""} onChange={onChange} />
    </label>
  ) : (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ParameterInput value={value} onChange={onChange} />
    </label>
  );
}

function getDropDownValues(
  data: any,
  query: string,
  substitutions?: Record<string, string>,
): string[] {
  const metadata = data.input_metadata;
  if (!metadata || !query) return [];
  const ss = { ...data.params, ...substitutions };
  for (const k in ss) {
    query = query.replace(`<${k}>`, ss[k]);
  }
  try {
    const res = ["", ...jmespath.search(metadata, query)];
    res.sort();
    return res;
  } catch (_) {
    return [""];
  }
}
