// @ts-expect-error
import WindowMaximize from "~icons/tabler/window-maximize.jsx";
import LynxKiteNode from "./LynxKiteNode";

// @ts-expect-error
await import("https://gradio.s3-us-west-2.amazonaws.com/5.49.1/gradio.js");

declare global {
  namespace JSX {
    interface IntrinsicElements {
      "gradio-app": React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        src?: string;
        theme_mode?: string;
        container?: string;
      };
    }
  }
}

function NodeWithGradio(props: any) {
  const path = props.data?.display?.value?.backend;
  if (!path) {
    return <div style={{ margin: "16px" }}>nothing yet...</div>;
  }
  const basePath = `${window.location.protocol}//${window.location.host}`;
  const src = `${basePath}${path}/`;
  return (
    <div style={{ margin: "16px" }}>
      <div style={{ marginBottom: "16px" }}>
        <a href={src} target="_blank">
          <WindowMaximize style={{ marginRight: "5px" }} />
          Pop out
        </a>
      </div>
      <gradio-app src={src} theme_mode="light" container="false"></gradio-app>
    </div>
  );
}

export default LynxKiteNode(NodeWithGradio);
