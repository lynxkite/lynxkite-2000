// The LynxKite workspace chat assistant panel.
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import RobotIcon from "~icons/tabler/robot.jsx";

export default function Assistant(props: { workspace: string }) {
  const [input, setInput] = useState("");
  const editorRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const { messages, sendMessage, status, error, stop } = useChat({
    transport: new TextStreamChatTransport({
      api: "/api/assistant/stream",
      body: { workspace: props.workspace },
    }),
  });
  const isGenerating = status === "submitted" || status === "streaming";

  useEffect(() => {
    const container = messagesRef.current;
    if (!container || !shouldAutoScrollRef.current) return;
    container.scrollTop = container.scrollHeight;
  }, [messages, isGenerating]);

  return (
    <aside className="assistant-panel prose">
      <div
        ref={messagesRef}
        className="assistant-messages"
        onScroll={(event) => {
          const container = event.currentTarget;
          const distanceFromBottom =
            container.scrollHeight - container.scrollTop - container.clientHeight;
          shouldAutoScrollRef.current = distanceFromBottom <= 16;
        }}
      >
        {messages.length === 0 && !isGenerating && (
          <div className="assistant-empty">
            <RobotIcon />
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
          if (!prompt || status !== "ready") return;
          sendMessage({ text: prompt });
          setInput("");
          if (editorRef.current) {
            editorRef.current.textContent = "";
            editorRef.current.focus();
          }
        }}
      >
        <div
          ref={editorRef}
          className="assistant-editor"
          aria-disabled={status !== "ready"}
          data-placeholder="Ask to make changes to the workspace, create custom boxes, or for general help."
          contentEditable
          suppressContentEditableWarning
          onInput={(event) => setInput(event.currentTarget.textContent ?? "")}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.closest("form")?.requestSubmit();
              editorRef.current?.focus();
            }
          }}
        />
        <div className="assistant-actions">
          {isGenerating && (
            <button className="btn btn-sm" type="button" onClick={stop}>
              Stop
            </button>
          )}
          <button className="btn btn-primary btn-sm" type="submit" disabled={status !== "ready"}>
            Send
          </button>
        </div>
      </form>
    </aside>
  );
}
