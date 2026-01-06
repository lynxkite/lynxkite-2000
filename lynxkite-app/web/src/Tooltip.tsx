import { useMemo } from "react";
import Markdown from "react-markdown";
import { Tooltip as ReactTooltip } from "react-tooltip";

export default function Tooltip(props: any) {
  const id = useMemo(
    () => props.id || `tooltip-${JSON.stringify(props.doc).substring(0, 20)}`,
    [props.id, props.doc],
  );

  if (!props.doc) return props.children;

  return (
    <>
      <a data-tooltip-id={id}>{props.children}</a>
      <ReactTooltip id={id} delayShow={1000}>
        {props.doc.map && typeof props.doc.map === "function" ? (
          props.doc.map(
            (section: any, i: number) =>
              section.kind === "text" && <Markdown key={i}>{section.value}</Markdown>,
          )
        ) : (
          <Markdown>{props.doc}</Markdown>
        )}
      </ReactTooltip>
    </>
  );
}
