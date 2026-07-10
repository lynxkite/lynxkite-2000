import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import RobotIcon from "~icons/tabler/robot.jsx";
import type { useCRDTWorkspace } from "./crdt";

export default function Assistant(props: { crdtWorkspace: ReturnType<typeof useCRDTWorkspace> }) {
  const { crdtWorkspace } = props;
  const crdtWorkspaceRef = useRef(crdtWorkspace);
  crdtWorkspaceRef.current = crdtWorkspace;
  const [input, setInput] = useState("");
  const [includeSelectedNodes, setIncludeSelectedNodes] = useState(false);
  const editorRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const selectedNodeIds = crdtWorkspace.feNodes
    .filter((node) => node.selected)
    .map((node) => node.id);

  const workspacePath = crdtWorkspace.ws?.path || "";
  const persistedMessages = crdtWorkspace.ws?.assistant_messages || [];
  const [messagesLoaded, setMessagesLoaded] = useState(false);

  const { messages, sendMessage, status, error, stop, setMessages } = useChat({
    transport: new TextStreamChatTransport({
      api: "/api/assistant/stream",
      body: { workspace: workspacePath },
    }),
  });

  // Initialize with persisted messages on first load
  useEffect(() => {
    if (!messagesLoaded && persistedMessages.length > 0 && messages.length === 0) {
      setMessages(persistedMessages);
      setMessagesLoaded(true);
    }
  }, [messagesLoaded, persistedMessages, messages.length, setMessages]);
  const isGenerating = status === "submitted" || status === "streaming";

  useEffect(() => {
    const container = messagesRef.current;
    if (!container || !shouldAutoScrollRef.current) return;
    container.scrollTop = container.scrollHeight;
  }, [messages, isGenerating]);

  function clearChatHistory() {
    setMessages([]);
    setMessagesLoaded(true);
    crdtWorkspaceRef.current.clearAssistantMessages();
  }

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
            className={`chat ${message.role === "assistant" ? "chat-start" : "chat-end"}`}
          >
            <div
              className={` ${
                message.role === "user"
                  ? "chat-bubble chat-bubble-primary"
                  : message.role === "system"
                    ? "chat-system-message"
                    : ""
              }`}
            >
              {message.parts?.some((p) => p.type === "text") ? (
                message.parts.map((part, index) =>
                  part.type === "text" ? (
                    <Markdown remarkPlugins={[remarkGfm]} key={`${message.id}-${index}`}>
                      {part.text}
                    </Markdown>
                  ) : null,
                )
              ) : (
                <Markdown remarkPlugins={[remarkGfm]}>{(message as any).content ?? ""}</Markdown>
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

      {error && (
        <div className="assistant-error">
          {(() => {
            try {
              // For HTTPException, the error message is a JSON string with a "detail" field.
              return JSON.parse(error.message).detail;
            } catch {
              return error.message;
            }
          })()}
        </div>
      )}

      <form
        className="assistant-input-row"
        onSubmit={async (event) => {
          event.preventDefault();
          const prompt = input.trim();
          if (!prompt || status !== "ready") return;
          if (includeSelectedNodes && selectedNodeIds.length > 0) {
            setMessages([
              ...persistedMessages,
              {
                role: "system",
                content: `Referencing·box(es):·${selectedNodeIds.join(",·")}`,
              },
            ]);
          }
          sendMessage({
            text: prompt,
            metadata: { selected_node_ids: includeSelectedNodes ? selectedNodeIds : undefined },
          });
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
          <button
            className={`btn btn-sm ${includeSelectedNodes ? "btn-primary" : ""}`}
            type="button"
            onClick={() => setIncludeSelectedNodes((value) => !value)}
          >
            Reference selected boxes
          </button>
          <button
            className="assistant-clear-button btn btn-sm"
            type="button"
            onClick={clearChatHistory}
          >
            Clear
          </button>
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
