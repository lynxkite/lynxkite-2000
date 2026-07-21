import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useEffect, useRef, useState } from "react";
import useSpeechToText from "react-hook-speech-to-text";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import MicrophoneIcon from "~icons/tabler/microphone.jsx";
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
  const dictationBaseRef = useRef("");

  const {
    error: speechError,
    interimResult,
    isRecording,
    results,
    setResults,
    startSpeechToText,
    stopSpeechToText,
  } = useSpeechToText({
    continuous: true,
    useLegacyResults: false,
    speechRecognitionProperties: {
      interimResults: true,
      lang: "en-US",
    },
  });

  const hasSpeechRecognitionApi =
    typeof window !== "undefined" &&
    ("SpeechRecognition" in window || "webkitSpeechRecognition" in (window as any));
  const isSpeechToTextUnsupported =
    !hasSpeechRecognitionApi ||
    speechError === "SpeechRecognition API is not available in this browser";
  const microphoneTitle = isSpeechToTextUnsupported
    ? "Voice input is unavailable: this browser does not support the Web Speech API."
    : speechError
      ? speechError
      : isRecording
        ? "Stop voice input"
        : "Start voice input";

  const transcriptText = results
    .map((result) => (typeof result === "string" ? result : result.transcript))
    .join(" ")
    .trim();

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

  useEffect(() => {
    const textFromSpeech = `${transcriptText}${interimResult ? ` ${interimResult}` : ""}`.trim();
    if (!textFromSpeech) return;
    const nextInput = `${dictationBaseRef.current}${textFromSpeech}`.trim();
    setInput(nextInput);
  }, [interimResult, transcriptText]);

  useEffect(() => {
    if (!editorRef.current) return;
    if (editorRef.current.textContent !== input) {
      editorRef.current.textContent = input;
    }
  }, [input]);

  function clearChatHistory() {
    setMessages([]);
    setMessagesLoaded(true);
    crdtWorkspaceRef.current.clearAssistantMessages();
  }

  function toggleVoiceInput() {
    if (isSpeechToTextUnsupported) return;
    if (isRecording) {
      stopSpeechToText();
      return;
    }
    dictationBaseRef.current = input ? `${input.trim()} ` : "";
    setResults([]);
    void startSpeechToText();
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
              ...messages,
              {
                id: `system-${Date.now()}`,
                role: "system",
                parts: [
                  { type: "text", text: `Referencing·box(es):·${selectedNodeIds.join(",·")}` },
                ],
              },
            ]);
          }
          sendMessage({
            text: prompt,
            metadata: { selected_node_ids: includeSelectedNodes ? selectedNodeIds : undefined },
          });
          if (isRecording) {
            stopSpeechToText();
          }
          dictationBaseRef.current = "";
          setResults([]);
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
          <span className="assistant-mic-wrapper" title={microphoneTitle}>
            <button
              className={`btn btn-sm btn-square assistant-mic-button ${
                isRecording ? "is-recording" : ""
              } ${isSpeechToTextUnsupported ? "is-disabled" : ""}`}
              type="button"
              aria-label={isRecording ? "Stop voice input" : "Start voice input"}
              aria-disabled={isSpeechToTextUnsupported}
              onClick={toggleVoiceInput}
            >
              <MicrophoneIcon />
            </button>
          </span>
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

      <SpeechToTextDemo />
    </aside>
  );
}

function SpeechToTextDemo() {
  const { error, interimResult, isRecording, results, startSpeechToText, stopSpeechToText } =
    useSpeechToText({
      continuous: true,
      useLegacyResults: false,
      speechRecognitionProperties: {
        lang: "en-US", // 👈 Explicitly define your language
        interimResults: true,
      },
    });

  useEffect(() => {
    console.log("🟢 [DEBUG] Component Mounted");
    return () => {
      console.log("🔴 [DEBUG] Component Unmounted!");
    };
  }, []);

  useEffect(() => {
    if (error) console.log("Speech-to-text error details:", error);
  }, [error]);

  if (error)
    return (
      <p style={{ padding: "8px", color: "red" }}>
        Web Speech API is not available in this browser 🤷
      </p>
    );

  return (
    <div style={{ padding: "8px", borderTop: "1px solid #d8dde3", fontSize: "12px" }}>
      <p>Demo — Recording: {isRecording.toString()}</p>
      <button
        className="btn btn-sm"
        type="button"
        onClick={isRecording ? stopSpeechToText : startSpeechToText}
      >
        {isRecording ? "Stop Recording" : "Start Recording"}
      </button>
      <ul>
        {(results as { timestamp: number; transcript: string }[]).map((result) => (
          <li key={result.timestamp}>{result.transcript}</li>
        ))}
        {interimResult && <li>{interimResult}</li>}
      </ul>
    </div>
  );
}
