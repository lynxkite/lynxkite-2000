// Full-page editor for code files.

import Editor from "@monaco-editor/react";
import { loader } from "@monaco-editor/react";
import { useEffect } from "react";
import { useParams } from "react-router";
import useSWR, { type Fetcher } from "swr";
// @ts-ignore
import Atom from "~icons/tabler/atom.jsx";
// @ts-ignore
import Backspace from "~icons/tabler/backspace.jsx";
// @ts-ignore
import Close from "~icons/tabler/x.jsx";
import favicon from "./assets/favicon.ico";
import theme from "./code-theme.ts";

export default function Code() {
  useEffect(() => {
    const initMonaco = async () => {
      const monaco = await loader.init();
      monaco.editor.defineTheme("lynxkite", theme);
    };
    initMonaco();
  }, []);
  const { path } = useParams();
  const parentDir = path!.split("/").slice(0, -1).join("/");
  const fetcher: Fetcher<{ code: string }> = (resource: string, init?: RequestInit) =>
    fetch(resource, init).then((res) => res.json());
  const code = useSWR(`/api/getCode?path=${path}`, fetcher);
  return (
    <div className="workspace">
      <div className="top-bar bg-neutral">
        <a className="logo" href="">
          <img alt="" src={favicon} />
        </a>
        <div className="ws-name">{path}</div>
        <div className="tools text-secondary">
          <a href="">
            <Atom />
          </a>
          <a href="">
            <Backspace />
          </a>
          <a href={`/dir/${parentDir}`}>
            <Close />
          </a>
        </div>
      </div>
      {code.isLoading && (
        <div className="loading">
          <div className="loading-text">Loading...</div>
        </div>
      )}
      {code.error && (
        <div className="error">
          <div className="error-text">Error: {code.error}</div>
        </div>
      )}
      {code.data && (
        <Editor
          defaultLanguage="python"
          defaultValue={code.data.code}
          theme="lynxkite"
          options={{
            cursorStyle: "block",
            cursorBlinking: "solid",
            minimap: { enabled: false },
            renderLineHighlight: "none",
          }}
        />
      )}
    </div>
  );
}
