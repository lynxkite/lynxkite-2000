const BOOLEAN = "<class 'bool'>";

function ParamName({ name }) {
  return <span className="param-name bg-base-200">{name.replace(/_/g, ' ')}</span>;
}

export default function NodeParameter({ name, value, meta, onChange }) {
  return (
    <label className="param">
      {meta?.type?.format === 'collapsed' ? <>
        <ParamName name={name} />
        <button className="collapsed-param">
          ⋯
        </button>
      </> : meta?.type?.format === 'textarea' ? <>
        <ParamName name={name} />
        <textarea className="textarea textarea-bordered w-full max-w-xs"
          rows={6}
          value={value}
          onChange={(evt) => onChange(evt.currentTarget.value)}
        />
      </> : meta?.type?.enum ? <>
        <ParamName name={name} />
        <select className="select select-bordered w-full max-w-xs"
          value={value || meta.type.enum[0]}
          onChange={(evt) => onChange(evt.currentTarget.value)}
        >
          {meta.type.enum.map(option =>
            <option key={option} value={option}>{option}</option>
          )}
        </select>
      </> : meta?.type?.type === BOOLEAN ? <div className="form-control">
        <label className="label cursor-pointer">
          <input className="checkbox"
            type="checkbox"
            checked={value}
            onChange={(evt) => onChange(evt.currentTarget.checked)}
          />
          {name.replace(/_/g, ' ')}
        </label>
      </div> : <>
        <ParamName name={name} />
        <input className="input input-bordered w-full max-w-xs"
          value={value || ""}
          onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
          onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
        />
      </>
      }
    </label >
  );
}
