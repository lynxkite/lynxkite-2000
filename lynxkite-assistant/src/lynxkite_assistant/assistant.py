"""FastAPI router exposing a Deep Agents assistant."""

import os
import fastapi
import pydantic
from typing import cast
from fastapi.responses import StreamingResponse
import deepagents
from .workspace_backend import WorkspaceBackend

router = fastapi.APIRouter()

SYSTEM_PROMPT = """
You are an assistant for the LynxKite no-code AI workflow builder.
The user sees the workflow in a visual representation. You have access to it as a file in `workspace.py`.
Edit this file to implement the user's requests. `workspace.py` must only contain function calls.
Keyword arguments must be constants or previous results. Positional arguments are not allowed.
New boxes can be added by editing `boxes.py`. Follow the existing conventions in `boxes.py` when defining a new box.
The new box can be used in `workspace.py` by calling the function from `boxes.py`. No extra imports are needed.
"""


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
    agent = deepagents.create_deep_agent(model=model, backend=backend, system_prompt=SYSTEM_PROMPT)
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
