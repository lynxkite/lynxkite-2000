// Modal for asking a single user input.
import { forwardRef, type PropsWithChildren, useImperativeHandle, useRef, useState } from "react";

export interface ModalHandle {
  open: (initialValue?: string) => void;
  close: () => void;
}

export interface ModalProps {
  title: string;
  description?: string;
  inputLabel?: string;
  submitLabel?: string;
  validate?: (name: string) => string;
  onSubmit: (value: string) => void;
}

export const Modal = forwardRef<ModalHandle, PropsWithChildren<ModalProps>>((props, ref) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [value, setValue] = useState("");
  const [error, setError] = useState("");

  const defaultValidate = (_name: string): string => {
    return "";
  };

  const validationFn = props.validate || defaultValidate;

  const handleValidate = (input: string) => {
    const errorMsg = validationFn(input);
    setError(errorMsg);
    return errorMsg === "";
  };

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (handleValidate(trimmed)) {
      props.onSubmit(trimmed);
      dialogRef.current?.close();
      setValue("");
      setError("");
    }
  };

  const handleCancel = () => {
    dialogRef.current?.close();
    setValue("");
    setError("");
  };

  useImperativeHandle(ref, () => ({
    open: (initialValue = "") => {
      setValue(initialValue);
      setError("");
      dialogRef.current?.showModal();
    },
    close: () => {
      dialogRef.current?.close();
      setValue("");
      setError("");
    },
  }));

  return (
    <dialog className="modal" ref={dialogRef}>
      <div className="modal-box">
        <h1 className="title">{props.title}</h1>
        {props.description && <p className="description">{props.description}</p>}

        <label className="form-control w-full">
          <span className="label-text">{props.inputLabel}</span>
          <input
            className={`input input-bordered w-full ${error ? "input-error" : ""}`}
            type="text"
            value={value}
            onChange={(e) => {
              const newValue = e.target.value;
              setValue(newValue);
              if (newValue.trim() !== "") handleValidate(newValue);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleSubmit();
              }
            }}
            autoFocus
          />
        </label>
        {error ? <p className="text-error mt-2">{error}</p> : null}

        <div className="modal-action">
          <button className="btn btn-ghost btn-primary" type="button" onClick={handleCancel}>
            Cancel
          </button>
          <button className="btn btn-primary" type="button" onClick={handleSubmit}>
            {props.submitLabel}
          </button>
        </div>
      </div>
    </dialog>
  );
});
