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

export default function Code() {
  const { path } = useParams();
  const parentDir = path!.split("/").slice(0, -1).join("/");
  const yDocRef = useRef<any>();
  const wsProviderRef = useRef<any>();
  const monacoBindingRef = useRef<any>();
  const yMonacoRef = useRef<any>();
  const editorRef = useRef<any>();
  useEffect(() => {
    const loadMonaco = async () => {
      // y-monaco is gigantic. The other Monaco packages are small.
      yMonacoRef.current = await import("y-monaco");
      initCRDT();
    };
    loadMonaco();
  }, []);
  function beforeMount(monaco: Monaco) {
    monaco.editor.defineTheme("lynxkite", theme);
  }
  function onMount(_editor: editor.IStandaloneCodeEditor) {
    editorRef.current = _editor;
    initCRDT();
  }
  function initCRDT() {
    if (!yMonacoRef.current || !editorRef.current) return;
    if (yDocRef.current) return;
    yDocRef.current = new Y.Doc();
    const text = yDocRef.current.getText("text");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    wsProviderRef.current = new WebsocketProvider(
      `${proto}//${location.host}/ws/code/crdt`,
      path!,
      yDocRef.current,
    );
    monacoBindingRef.current = new yMonacoRef.current.MonacoBinding(
      text,
      editorRef.current.getModel()!,
      new Set([editorRef.current]),
      wsProviderRef.current.awareness,
    );
  }
  useEffect(() => {
    return () => {
      yDocRef.current?.destroy();
      wsProviderRef.current?.destroy();
      monacoBindingRef.current?.destroy();
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
        loading={null}
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
