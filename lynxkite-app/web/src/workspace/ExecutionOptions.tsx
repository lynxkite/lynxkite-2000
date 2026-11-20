import Tooltip from "../Tooltip";

export default function ExecutionOptions(props: {
  env: string;
  value: any;
  onChange: (val: any) => void;
}) {
  // Different environments may have different options. For now, just GPUs for graph analytics.
  if (props.env !== "LynxKite Graph Analytics") return null;
  return (
    <>
      <h3>Execution options</h3>
      <Tooltip doc="Maximum number of GPUs used by the workspace.">
        <label className="top-bar-gpus">
          <input
            type="number"
            className="input"
            required
            min="1"
            value={props.value?.gpus ?? 1}
            onChange={(evt) =>
              props.onChange({ ...props.value, gpus: parseInt(evt.currentTarget.value, 10) })
            }
          />
          GPUs
        </label>
      </Tooltip>
    </>
  );
}
