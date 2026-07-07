import type { UpdateOptions } from "./NodeParameter";

function SelectDropdown({
  options,
  value,
  onChange,
  multi = false,
  placeholder = "",
}: {
  options: string[];
  value: string | string[];
  onChange: (v: string | string[]) => void;
  multi?: boolean;
  placeholder?: string;
}) {
  const visibleOptions = options.filter((option) => option && option.trim() !== "");
  const selectedValues = Array.isArray(value) ? value : value ? [value] : [];
  const selectedLabel = selectedValues.length ? selectedValues.join(", ") : placeholder;

  return (
    <details
      className="relative w-full min-w-0 flex-1 basis-0"
      style={{ width: "100%" }}
      onBlur={(evt) => {
        const next = evt.relatedTarget as Node | null;
        if (!next || !evt.currentTarget.contains(next)) {
          evt.currentTarget.removeAttribute("open");
        }
      }}
    >
      <summary className="border border-base-300 bg-base-100 rounded-lg h-12 min-h-12 max-h-12 px-4 w-full min-w-0 flex items-center justify-between cursor-pointer overflow-hidden whitespace-nowrap list-none [&::-webkit-details-marker]:hidden">
        <span className="truncate text-sm text-base-content/90 min-w-0">{selectedLabel}</span>
        <span className="text-[10px] opacity-60 text-base-content">▼</span>
      </summary>
      <div
        className="absolute left-0 top-full mt-1 w-full min-w-0 bg-white z-[999] rounded-lg border p-2 max-h-60 overflow-y-auto"
        style={{ width: "100%", minWidth: "0" }}
      >
        {visibleOptions.map((option) => {
          const isSelected = selectedValues.includes(option);
          return (
            <button
              key={option}
              type="button"
              style={{ backgroundColor: "white", minWidth: "0" }}
              className="flex items-center justify-between w-full text-left p-2 rounded text-gray-900 outline-none focus:outline-none focus:ring-0 active:outline-none transition-colors duration-150 bg-white !bg-white hover:bg-gray-100"
              onClick={() => {
                if (multi) {
                  const next = isSelected
                    ? selectedValues.filter((v) => v !== option)
                    : [...selectedValues, option];
                  onChange(next);
                  return;
                }
                // Single mode supports deselect by clicking the selected option again.
                onChange(isSelected ? "" : option);
              }}
            >
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="checkbox checkbox-sm checkbox-primary"
                  checked={isSelected}
                  aria-checked={isSelected}
                  readOnly
                />
                <span className="text-sm select-none">{option}</span>
              </div>
            </button>
          );
        })}
      </div>
    </details>
  );
}

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

export function DropdownMultiDropdownAdder({
  value,
  onChange,
  options1,
  options2,
}: {
  value: [string, string[]][];
  onChange: (v: [string, string[]][], opts?: UpdateOptions) => void;
  options1: string[];
  options2: string[];
}) {
  const safeValue: [string, string[]][] = Array.isArray(value) ? value : [];

  const addRow = () => {
    onChange([...safeValue, ["", []]]);
  };

  const updateFirst = (index: number, v: string, opts?: UpdateOptions) => {
    onChange(
      safeValue.map((row, i) => (i === index ? [v, row[1]] : row) as [string, string[]]),
      opts,
    );
  };

  const updateSecond = (index: number, v: string[], opts?: UpdateOptions) => {
    onChange(
      safeValue.map((row, i) => (i === index ? [row[0], v] : row) as [string, string[]]),
      opts,
    );
  };

  const removeRow = (index: number) => {
    onChange(safeValue.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      {safeValue.map((val, index) => {
        const selected = Array.isArray(val[1]) ? val[1] : [];
        return (
          <div key={index} className="flex gap-2 items-start">
            <div className="flex-1 basis-0 min-w-0">
              <SelectDropdown
                options={options1}
                value={val[0] ?? ""}
                onChange={(v) => updateFirst(index, v as string)}
              />
            </div>
            <div className="flex-1 basis-0 min-w-0">
              <SelectDropdown
                options={options2}
                value={selected}
                multi
                onChange={(v) => updateSecond(index, v as string[])}
              />
            </div>

            <button
              type="button"
              className="btn btn-error btn-sm mt-1"
              onClick={() => removeRow(index)}
            >
              ✕
            </button>
          </div>
        );
      })}

      <button type="button" className="btn btn-sm" onClick={addRow}>
        + Add
      </button>
    </div>
  );
}
