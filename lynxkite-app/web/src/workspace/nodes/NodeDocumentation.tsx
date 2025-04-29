import Markdown from "react-markdown";

export default function NodeDocumentation(props: any) {
  if (!props.doc) return null;
  return (
    <div className="dropdown dropdown-hover dropdown-top dropdown-end title-icon">
      {props.children}
      <div className="node-documentation dropdown-content" style={{ width: props.width }}>
        {props.doc.map?.(
          (section: any, i: number) =>
            section.kind === "text" && <Markdown key={i}>{section.value}</Markdown>,
        ) ?? <Markdown>{props.doc}</Markdown>}
      </div>
    </div>
  );
}
