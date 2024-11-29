export default function EnvironmentSelector(props: { options: string[], value: string, onChange: (val: string) => void }) {
  return (
    <select className="form-select form-select-sm"
      value={props.value}
      onChange={(evt) => props.onChange(evt.currentTarget.value)}
    >
      {props.options.map(option =>
        <option key={option} value={option}>{option}</option>
      )}
    </select>
  );
}
