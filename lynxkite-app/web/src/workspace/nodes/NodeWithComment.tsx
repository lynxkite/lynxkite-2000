import { useReactFlow } from "@xyflow/react";
import { useState } from "react";
import Markdown from "react-markdown";
import type { UpdateOptions } from "./NodeParameter";

export default function NodeWithComment(props: any) {
  const reactFlow = useReactFlow();
  const [editing, setEditing] = useState(false);
  function setComment(newValue: string, opts?: UpdateOptions) {
    reactFlow.updateNodeData(props.id, (prevData: any) => ({
      ...prevData,
      params: { text: newValue },
      __execution_delay: opts?.delay || 0,
    }));
  }
  function onClick(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    // Start editing on double-click.
    if (e.detail === 2) {
      setEditing(true);
    }
  }
  function finishEditing(el: HTMLTextAreaElement) {
    setComment(el.value);
    setEditing(false);
  }
  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      finishEditing(e.currentTarget);
    }
  }
  function onInput(el: HTMLTextAreaElement | null) {
    if (!el) return;
    el.focus();
    // Resize the textarea to the content.
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }
  if (editing) {
    return (
      <textarea
        className="comment-editor"
        onBlur={(e) => finishEditing(e.currentTarget)}
        onKeyDown={onKeyDown}
        onInput={(e) => onInput(e.currentTarget)}
        ref={(el) => onInput(el)}
        defaultValue={props.data.params.text}
        onClick={(e) => e.stopPropagation()}
        placeholder="Enter workspace comment"
      />
    );
  }
  const text = props.data.params.text || "_double-click to edit_";
  return (
    <div className="comment-view drag-handle prose" onClick={onClick}>
      <Markdown>{text}</Markdown>
    </div>
  );
}
