export default function ParameterInput({
  value,
  onChange,
  inputRef,
}: {
  value: string;
  onChange: (value: string, options?: { delay: number }) => void;
  inputRef?: React.Ref<HTMLInputElement>;
}) {
  return (
    <input
      className="input input-bordered w-full"
      ref={inputRef}
      value={value ?? ""}
      onChange={(evt) => onChange(evt.currentTarget.value, { delay: 2 })}
      onBlur={(evt) => onChange(evt.currentTarget.value, { delay: 0 })}
      onKeyDown={(evt) => evt.code === "Enter" && onChange(evt.currentTarget.value, { delay: 0 })}
    />
  );
}
