import { useId } from "react";
import Markdown from "react-markdown";
import { Tooltip as ReactTooltip } from "react-tooltip";

export default function Tooltip(props: any) {
  const id = useId();
  if (!props.doc) return null;
  return (
    <>
      <a data-tooltip-id={id} tabIndex={0}>
        {props.children}
      </a>
      <ReactTooltip id={id} className="tooltip" place="top-end">
        {props.doc.map?.(
          (section: any, i: number) =>
            section.kind === "text" && <Markdown key={i}>{section.value}</Markdown>,
        ) ?? <Markdown>{props.doc}</Markdown>}
      </ReactTooltip>
    </>
  );
}
