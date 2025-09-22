import jmespath from "jmespath";
// @ts-expect-error
import Help from "~icons/tabler/question-mark.jsx";
import Tooltip from "../../Tooltip";
import ModelMapping from "./ModelMappingParameter";
import NodeGroupParameter from "./NodeGroupParameter";
import ParameterInput from "./ParameterInput";

const BOOLEAN = "<class 'bool'>";
const MODEL_TRAINING_INPUT_MAPPING =
  "lynxkite_graph_analytics.ml_ops.ModelTrainingInputMapping | None";
const MODEL_INFERENCE_INPUT_MAPPING =
  "lynxkite_graph_analytics.ml_ops.ModelInferenceInputMapping | None";
const MODEL_OUTPUT_MAPPING = "lynxkite_graph_analytics.ml_ops.ModelOutputMapping | None";

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
        className="select select-bordered w-full"
        value={value ?? ""}
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
  ) : (
    <label className="param">
      <ParamName name={name} doc={doc} />
      <ParameterInput value={value} onChange={onChange} />
    </label>
  );
}

function getDropDownValues(data: any, meta: any): string[] {
  const metadata = data.input_metadata?.value;
  let query = meta?.type?.metadata_query;
  if (!metadata || !query) {
    return [];
  }
  // Substitute parameters in the query.
  for (const p in data.params) {
    query = query.replace(`<${p}>`, data.params[p]);
  }
  try {
    const res = ["", ...jmespath.search(metadata, query)];
    res.sort();
    return res;
  } catch (_) {
    return [""];
  }
}
