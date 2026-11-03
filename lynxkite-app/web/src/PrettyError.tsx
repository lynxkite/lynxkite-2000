import type React from "react";
import { useState } from "react";
import Check from "~icons/tabler/check.jsx";
import Copy from "~icons/tabler/copy.jsx";
import Tooltip from "./Tooltip";

export default function PrettyError(props: { error: string }) {
  return (
    <div className="error">
      <span className="error-line">{props.error}</span>
      <CopyErrorButton text={props.error} />
    </div>
  );
}

function CopyErrorButton(props: { text: string }) {
  const [copied, setCopied] = useState(false);

  function copy(e: React.MouseEvent) {
    e.stopPropagation();
    navigator.clipboard?.writeText(props.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <Tooltip doc={copied ? "Copied!" : "Copy error to clipboard"}>
      <button
        className="error-copy-button"
        onClick={copy}
        aria-label="Copy error to clipboard"
        type="button"
      >
        {copied ? <Check /> : <Copy />}
      </button>
    </Tooltip>
  );
}
