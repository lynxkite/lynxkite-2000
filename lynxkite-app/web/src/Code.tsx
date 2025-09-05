// Full-page editor for code files.

import Editor, { type Monaco } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import { useEffect, useRef } from "react";
import { Link } from "react-router";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
// @ts-expect-error
import Atom from "~icons/tabler/atom.jsx";
// @ts-expect-error
import Backspace from "~icons/tabler/backspace.jsx";
// @ts-expect-error
import Close from "~icons/tabler/x.jsx";
import favicon from "./assets/favicon.ico";
import theme from "./code-theme.ts";
import { usePath } from "./common.ts";

export default function Code() {
  const path = usePath().replace(/^[/]code[/]/, "");
  const parentDir = path!.split("/").slice(0, -1).join("/");
  const yDocRef = useRef<any>();
  const wsProviderRef = useRef<any>();
  const monacoBindingRef = useRef<any>();
  const yMonacoRef = useRef<any>();
  const yMonacoLoadingRef = useRef(false);
  const editorRef = useRef<any>();
  useEffect(() => {
    const loadMonaco = async () => {
      if (yMonacoLoadingRef.current) return;
      yMonacoLoadingRef.current = true;
      // y-monaco is gigantic. The other Monaco packages are small.
      yMonacoRef.current = await import("y-monaco");
      initCRDT();
    };
    loadMonaco();
  }, []);
  function beforeMount(monaco: Monaco) {
    monaco.editor.defineTheme("lynxkite", theme);
  }
  function onMount(_editor: editor.IStandaloneCodeEditor, monaco: Monaco) {
    // Do nothing on Ctrl+S. We save after every keypress anyway.
    _editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {});
    editorRef.current = _editor;
    initCRDT();
  }
  function initCRDT() {
    if (!yMonacoRef.current || !editorRef.current) return;
    if (yDocRef.current) return;
    yDocRef.current = new Y.Doc();
    const text = yDocRef.current.getText("text");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const encodedPath = path!
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    wsProviderRef.current = new WebsocketProvider(
      `${proto}//${location.host}/ws/code/crdt`,
      encodedPath!,
      yDocRef.current,
    );
    editorRef.current.getModel()!.setEOL(0); // https://github.com/yjs/y-monaco/issues/6
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
        <Link className="logo" to="/">
          <img alt="" src={favicon} />
        </Link>
        <div className="ws-name">{path}</div>
        <title>{path}</title>
        <div className="tools text-secondary">
          <button className="btn btn-link">
            <Atom />
          </button>
          <button className="btn btn-link">
            <Backspace />
          </button>
          <Link
            to={`/dir/${parentDir
              .split("/")
              .map((segment) => encodeURIComponent(segment))
              .join("/")}`}
            className="btn btn-link"
          >
            <Close />
          </Link>
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
