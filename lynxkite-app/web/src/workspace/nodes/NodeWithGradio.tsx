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
      };
    }
  }
}

function NodeWithGradio(props: any) {
  const path = props.data?.display?.value?.backend;
  const basePath = `${window.location.protocol}//${window.location.host}`;
  const src = `${basePath}${path}/`;
  return (
    <>
      <div style={{ margin: "16px" }}>
        <a href={src} target="_blank">
          <WindowMaximize style={{ marginRight: "5px" }} />
          Pop out
        </a>
      </div>
      <gradio-app src={src}></gradio-app>
    </>
  );
}

export default LynxKiteNode(NodeWithGradio);
