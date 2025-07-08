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
  return meta?.type?.format === "textarea" ? (
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
  ) : meta?.type?.format === "dropdown" ? (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <select
        className="select select-bordered w-full"
        value={value || getDropDownValues(data, meta)[0]}
        onChange={(evt) => onChange(evt.currentTarget.value)}
      >
        {getDropDownValues(data, meta).map((option: string) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
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
      <ParameterInput value={value} onChange={onChange} />
    </label>
  );
}

// We have a little "language" for describing which part of the input_metadata
// to use in the dropdown.
function getDropDownValues(data: any, meta: any): string[] {
  const metadata = data.input_metadata.value;
  const { metadata_path, metadata_filter_key, metadata_filter_value } = meta.type;
  let o = [metadata];
  for (const path of metadata_path) {
    o = o.flatMap((x: any) => {
      if (x === undefined || x === null) {
        return [];
      }
      if (typeof x === "object") {
        if (path === "*") {
          return Object.values(x);
        }
        return [x[path]];
      }
      if (path === "*") {
        return x;
      }
      return [x[Number.parseInt(path)]];
    });
  }
  o = o.flatMap((x: any) => {
    if (x === undefined || x === null) {
      return [];
    }
    if (typeof x === "object") {
      if (metadata_filter_key && metadata_filter_value) {
        const keys = [];
        for (const key in x) {
          if (x[key][metadata_filter_key] === metadata_filter_value) {
            keys.push(key);
          }
        }
        return keys;
      }
      return Object.keys(x);
    }
    return x;
  });
  return o;
}
