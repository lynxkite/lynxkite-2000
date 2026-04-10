"""FastAPI router exposing a Deep Agents assistant."""

import os
import fastapi
import pydantic
from typing import cast
from fastapi.responses import StreamingResponse
import deepagents
from .workspace_backend import WorkspaceBackend

router = fastapi.APIRouter()


class AssistantMessage(pydantic.BaseModel):
    role: str
    content: str | None = None
    parts: list[dict] | None = None


class AssistantCompletionRequest(pydantic.BaseModel):
    workspace: str
    messages: list[AssistantMessage]


def _extract_text_content(message: AssistantMessage) -> str:
    if message.content:
        return message.content

    if not message.parts:
        return ""

    text_parts: list[str] = []
    for part in message.parts:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])

    return "".join(text_parts)


def _extract_token_text(token_content: object) -> str:
    if token_content is None:
        return ""
    if isinstance(token_content, list):
        chunks: list[str] = []
        for item in token_content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                item_dict = cast(dict[str, object], item)
                text = item_dict.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


@router.post("/api/assistant/stream")
async def assistant_stream(req: AssistantCompletionRequest) -> StreamingResponse:
    model = os.environ.get("LYNXKITE_ASSISTANT_MODEL")
    backend = WorkspaceBackend(req.workspace)
    agent = deepagents.create_deep_agent(model=model, backend=backend)
    request_messages: list[dict[str, str]] = []
    for msg in req.messages:
        content = _extract_text_content(msg).strip()
        if not content:
            continue
        request_messages.append({"role": msg.role, "content": content})

    async def generate():
        for chunk in agent.stream(
            {"messages": request_messages},
            stream_mode="messages",
            subgraphs=False,
            version="v2",
        ):
            if chunk["type"] != "messages":
                continue
            token, _metadata = chunk["data"]
            delta = _extract_token_text(token.content)
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/event-stream; charset=utf-8")
