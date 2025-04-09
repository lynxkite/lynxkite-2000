// A simple theme using the LynxKite colors.

import type { editor } from "monaco-editor/esm/vs/editor/editor.api";

const theme: editor.IStandaloneThemeData = {
  base: "vs-dark",
  inherit: true,
  rules: [
    {
      foreground: "ff8800",
      token: "keyword",
    },
    {
      foreground: "0088ff",
      fontStyle: "italic",
      token: "comment",
    },
    {
      foreground: "39bcf3",
      token: "string",
    },
    {
      foreground: "ffc600",
      token: "",
    },
  ],
  colors: {
    "editor.foreground": "#FFFFFF",
    "editor.background": "#002a4c",
    "editor.selectionBackground": "#0050a4",
    "editor.lineHighlightBackground": "#1f4662",
    "editorCursor.foreground": "#ffc600",
    "editorWhitespace.foreground": "#7f7f7fb2",
    "editorIndentGuide.background": "#3b5364",
    "editorIndentGuide.activeBackground": "#ffc600",
  },
};
export default theme;
