import type { UpdateOptions } from "./NodeParameter";

export function DropdownTextAdder({
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

export function DoubleTextAdder({
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
