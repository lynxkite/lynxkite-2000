import React, { memo } from "react";
import Settings from "~icons/tabler/settings-filled.jsx";
import Tooltip from "../Tooltip";

const SettingsIcon = memo(Settings);

export default function ExecutionOptions(props: {
  env: string;
  value: any;
  onChange: (val: any) => void;
}) {
  const [show, setShow] = React.useState(false);
  // Different environments may have different options. For now, just GPUs for graph analytics.
  if (props.env !== "LynxKite Graph Analytics") return null;
  return (
    <>
      <button className="icon-button" onClick={() => setShow((s) => !s)}>
        <SettingsIcon />
      </button>
      {show && <LynxKiteGraphAnalyticsExecutionOptions {...props} />}
    </>
  );
}

function LynxKiteGraphAnalyticsExecutionOptions(props: {
  env: string;
  value: any;
  onChange: (val: any) => void;
}) {
  return (
    <Tooltip doc="Maximum number of GPUs used by the workspace.">
      <label className="top-bar-gpus">
        <input
          dir="rtl"
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
  );
}
