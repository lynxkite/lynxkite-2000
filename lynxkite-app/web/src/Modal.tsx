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
  onSubmit: (value: string) => void;
}

export const Modal = forwardRef<ModalHandle, PropsWithChildren<ModalProps>>((props, ref) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [value, setValue] = useState("");

  function handleSubmit() {
    const trimmed = value.trim();
    props.onSubmit(trimmed);
    dialogRef.current?.close();
    setValue("");
  }

  function handleCancel() {
    dialogRef.current?.close();
    setValue("");
  }

  useImperativeHandle(ref, () => ({
    open: (initialValue = "") => {
      setValue(initialValue);
      dialogRef.current?.showModal();
    },
    close: () => {
      dialogRef.current?.close();
      setValue("");
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
            className="input input-bordered w-full"
            type="text"
            value={value}
            onChange={(e) => {
              const newValue = e.target.value;
              setValue(newValue);
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
