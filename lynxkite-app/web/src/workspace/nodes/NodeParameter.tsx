const BOOLEAN = "<class 'bool'>";

function ParamName({ name }: { name: string }) {
  return (
    <span className="param-name bg-base-200">{name.replace(/_/g, " ")}</span>
  );
}

interface NodeParameterProps {
  name: string;
  value: any;
  meta: any;
  onChange: (value: any, options?: { delay: number }) => void;
}

export default function NodeParameter({
  name,
  value,
  meta,
  onChange,
}: NodeParameterProps) {
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
            className="textarea textarea-bordered w-full max-w-xs"
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
            className="select select-bordered w-full max-w-xs"
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
            <input
              className="checkbox"
              type="checkbox"
              checked={value}
              onChange={(evt) => onChange(evt.currentTarget.checked)}
            />
            {name.replace(/_/g, " ")}
          </label>
        </div>
      ) : (
        <>
          <ParamName name={name} />
          <input
            className="input input-bordered w-full max-w-xs"
            value={value || ""}
            onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
            onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
          />
        </>
      )}
    </label>
  );
}
