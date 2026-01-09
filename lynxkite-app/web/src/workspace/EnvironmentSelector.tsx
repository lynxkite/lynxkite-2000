import Tooltip from "../Tooltip";

export default function EnvironmentSelector(props: {
  options: string[];
  value: string;
  onChange: (val: string) => void;
}) {
  return (
    <Tooltip doc="Select workspace environment">
      <select
        className="env-select select w-full max-w-xs"
        name="workspace-env"
        value={props.value}
        onChange={(evt) => props.onChange(evt.currentTarget.value)}
      >
        {props.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </Tooltip>
  );
}
