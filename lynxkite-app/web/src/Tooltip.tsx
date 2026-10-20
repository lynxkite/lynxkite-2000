import { useMemo } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";

export default function Tooltip(props: any) {
  const html = useMemo(() => {
    if (!props.doc) return null;
    const md =
      props.doc.map && typeof props.doc.map === "function"
        ? props.doc.map((section: any) => (section.kind === "text" ? section.value : "")).join("\n")
        : String(props.doc);
    return renderToStaticMarkup(<Markdown>{md}</Markdown>);
  }, [props.doc]);

  if (!html) return props.children;

  return (
    <div
      data-tooltip-id="tooltip-global"
      data-tooltip-delay-show={1000}
      data-tooltip-html={html}
      data-tooltip-hidden={props.disabled}
    >
      {props.children}
    </div>
  );
}
