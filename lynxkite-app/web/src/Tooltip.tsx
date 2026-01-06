import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";

export default function Tooltip(props: any) {
  if (!props.doc) return props.children;
  const md =
    props.doc.map && typeof props.doc.map === "function"
      ? props.doc.map((section: any) => (section.kind === "text" ? section.value : "")).join("\n")
      : String(props.doc);
  const html = renderToStaticMarkup(<Markdown>{md}</Markdown>);
  return (
    <div data-tooltip-id="tooltip-global" data-tooltip-delay-show={1000} data-tooltip-html={html}>
      {props.children}
    </div>
  );
}
