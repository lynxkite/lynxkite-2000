"""FastAPI router exposing a Deep Agents assistant."""

import os
import fastapi
import openai
import pydantic
from typing import cast
from fastapi.responses import StreamingResponse
import deepagents
from deepagents import backends
from .workspace_backend import WorkspaceBackend

router = fastapi.APIRouter()

SYSTEM_PROMPT = """
You are an assistant for the LynxKite no-code AI workflow builder.
The user sees the workflow in a visual representation. You have access to it as a file in `workspace.py`, which the user does not see.
Each function call in `workspace.py` corresponds to a box in the visual representation.
Edit this file to implement the user's requests. `workspace.py` must only contain function calls.
DO NOT REMOVE any existing code or comments! (unless asked explicitly by the user)
Keyword arguments must be constants or previous results. Positional arguments are not allowed.
New boxes can be added by editing `boxes.py`. Follow the existing conventions in `boxes.py` when defining a new box.
The new box can be used in `workspace.py` by calling the function from `boxes.py`. The functions are available under a custom module name, specified at the beginning of `boxes.py`.
You must use existing boxes directly in `workspace.py` without adding them to the `boxes` module.
You can see any errors that occurred in the boxes in `errors.txt`. Before finishing a task you must fix all errors in the new boxes.
If a custom box returns an 'Unknown operation' error message, check if you are using the correct module name for the new box.
The module name and usage examples are specified at the beginning of `boxes.py`.
If you cannot fix an error, you must ask the user for clarification.
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
    workspace_backend = WorkspaceBackend(req.workspace)
    backend = backends.CompositeBackend(
        default=workspace_backend,
        routes={
            "/skills/": backends.FilesystemBackend(
                root_dir=("../.agents/skills"), virtual_mode=True
            )
        },
    )
    agent = deepagents.create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills"],
        system_prompt=SYSTEM_PROMPT,
    )
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

    try:
        gen = generate()
        first_chunk = (
            await gen.__anext__()
        )  # peek the first chunk to check for authentication errors

        async def chained_generator():
            yield first_chunk
            async for chunk in gen:
                yield chunk

        return StreamingResponse(chained_generator(), media_type="text/event-stream; charset=utf-8")
    except openai.AuthenticationError:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI Authentication failed. Check your API key.",
        )
