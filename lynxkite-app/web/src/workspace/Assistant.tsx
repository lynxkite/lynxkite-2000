import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import useSWR from "swr";
import { pathFetcher } from "../common.ts";

type AssistantConfig = {
  assistant_available: boolean;
};

export default function Assistant(props: { workspace: string }) {
  const { data: config } = useSWR<AssistantConfig>("/api/config", pathFetcher);
  const [input, setInput] = useState("");
  const editorRef = useRef<HTMLDivElement>(null);
  const { messages, sendMessage, status, error, stop } = useChat({
    transport: new TextStreamChatTransport({
      api: "/api/assistant/stream",
      body: { workspace: props.workspace },
    }),
  });
  const isGenerating = status === "submitted" || status === "streaming";
  const disabled = config?.assistant_available !== true;
  return (
    <aside className="assistant-panel prose">
      <div className="assistant-messages">
        {messages.length === 0 && !isGenerating && (
          <div className="assistant-empty">
            Ask to make changes to the workspace, create custom boxes, or for general help.
          </div>
        )}

        {messages.map((message, idx) => (
          <div
            key={`${message.role}-${idx}`}
            className={`chat ${message.role === "user" ? "chat-end" : "chat-start"}`}
          >
            <div className={` ${message.role === "user" ? "chat-bubble chat-bubble-primary" : ""}`}>
              {message.parts.map((part, index) =>
                part.type === "text" ? (
                  <Markdown remarkPlugins={[remarkGfm]} key={`${message.id}-${index}`}>
                    {part.text}
                  </Markdown>
                ) : null,
              )}
            </div>
          </div>
        ))}

        {isGenerating && (
          <div className="chat chat-start">
            <div className="chat-bubble">
              <span className="loading loading-dots loading-sm" />
            </div>
          </div>
        )}
      </div>

      {error && <div className="assistant-error">{error.message}</div>}

      <form
        className="assistant-input-row"
        onSubmit={async (event) => {
          event.preventDefault();
          const prompt = input.trim();
          if (disabled || !prompt || status !== "ready") return;
          sendMessage({ text: prompt });
          setInput("");
          if (editorRef.current) {
            editorRef.current.textContent = "";
          }
        }}
      >
        <div
          ref={editorRef}
          className="assistant-editor"
          aria-disabled={disabled || status !== "ready"}
          data-placeholder={disabled ? "Assistant unavailable" : "Ask the assistant..."}
          contentEditable={!disabled && status === "ready"}
          suppressContentEditableWarning
          onInput={(event) => setInput(event.currentTarget.textContent ?? "")}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.closest("form")?.requestSubmit();
            }
          }}
        />
        <div className="assistant-actions">
          {isGenerating && (
            <button className="btn btn-sm" type="button" onClick={stop}>
              Stop
            </button>
          )}
          <button
            className="btn btn-primary btn-sm"
            type="submit"
            disabled={disabled || status !== "ready"}
          >
            Send
          </button>
        </div>
      </form>
    </aside>
  );
}
