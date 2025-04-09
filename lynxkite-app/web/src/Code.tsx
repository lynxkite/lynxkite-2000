// Full-page editor for code files.

import Editor, { type Monaco } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import { useEffect, useRef } from "react";
import { useParams } from "react-router";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
// @ts-ignore
import Atom from "~icons/tabler/atom.jsx";
// @ts-ignore
import Backspace from "~icons/tabler/backspace.jsx";
// @ts-ignore
import Close from "~icons/tabler/x.jsx";
import favicon from "./assets/favicon.ico";
import theme from "./code-theme.ts";
// For some reason y-monaco is gigantic. The other Monaco packages are small.
const MonacoBinding = await import("y-monaco").then((m) => m.MonacoBinding);

export default function Code() {
  const { path } = useParams();
  const parentDir = path!.split("/").slice(0, -1).join("/");
  const ydoc = useRef<any>();
  const wsProvider = useRef<any>();
  const monacoBinding = useRef<any>();
  function beforeMount(monaco: Monaco) {
    monaco.editor.defineTheme("lynxkite", theme);
  }
  function onMount(_editor: editor.IStandaloneCodeEditor) {
    ydoc.current = new Y.Doc();
    const text = ydoc.current.getText("text");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    wsProvider.current = new WebsocketProvider(
      `${proto}//${location.host}/ws/code/crdt`,
      path!,
      ydoc.current,
    );
    monacoBinding.current = new MonacoBinding(
      text,
      _editor.getModel()!,
      new Set([_editor]),
      wsProvider.current.awareness,
    );
  }
  useEffect(() => {
    return () => {
      ydoc.current?.destroy();
      wsProvider.current?.destroy();
      monacoBinding.current?.destroy();
    };
  });
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
      <Editor
        defaultLanguage="python"
        theme="lynxkite"
        path={path}
        beforeMount={beforeMount}
        onMount={onMount}
        options={{
          cursorStyle: "block",
          cursorBlinking: "solid",
          minimap: { enabled: false },
          renderLineHighlight: "none",
        }}
      />
    </div>
  );
}
