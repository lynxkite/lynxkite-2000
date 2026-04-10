import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import useSWR from "swr";
import { pathFetcher } from "../common.ts";

type AssistantConfig = {
  assistant_available: boolean;
};

export default function Assistant() {
  const { data: config } = useSWR<AssistantConfig>("/api/config", pathFetcher);
  const [input, setInput] = useState("");

  const { messages, sendMessage, status, error, stop } = useChat({
    transport: new TextStreamChatTransport({
      api: "/api/assistant/stream",
    }),
  });
  const isGenerating = status === "submitted" || status === "streaming";

  const disabled = config?.assistant_available !== true;
  const statusText = useMemo(() => {
    if (config == null) return "Checking assistant availability...";
    if (disabled) return "Assistant is currently unavailable.";
    return "Assistant is ready.";
  }, [config, disabled]);

  return (
    <aside className="assistant-panel prose">
      <div className="assistant-messages">
        {messages.length === 0 && !isGenerating && (
          <div className="assistant-empty">
            Ask about this workspace, operations, or next steps.
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
              <span className="loading loading-dots loading-sm" /> ...
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
        }}
      >
        <textarea
          className="textarea textarea-bordered"
          rows={3}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={disabled ? "Assistant unavailable" : "Ask the assistant..."}
          disabled={disabled || status !== "ready"}
        />
        <div className="assistant-actions">
          <button className="btn btn-sm" type="button" onClick={stop} disabled={!isGenerating}>
            Stop
          </button>
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
