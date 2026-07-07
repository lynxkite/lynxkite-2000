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
from lynxkite_core import workspace

try:
    from lynxkite_app import crdt
except ImportError:
    crdt = None  # type: ignore

router = fastapi.APIRouter()

SYSTEM_PROMPT = """
You are an assistant for the LynxKite no-code AI workflow builder.
The user sees the workflow in a visual representation. You have access to it as a file in `workspace.py`, which the user does not see.
Each function call in `workspace.py` corresponds to a box in the visual representation. Boxes can be connected to each other by their inputs and outputs.
You may change the layout of the boxes in the visual representation by editing `layout.json`. The user does not see this file, but they will see the updated layout in the visual representation.
When editing the layout, keep in mind that boxes that are connected should be placed closer to each other, as these boxes are connected by arrows in the visual representation.
Edit this file to implement the user's requests. `workspace.py` must only contain function calls.
Keyword arguments must be constants or previous results. Positional arguments are not allowed.
DO NOT REMOVE or edit any existing code or comments! (unless asked explicitly by the user).
When adding new comments, make sure to add them above the relevant line of code, so they appear above the box they are associated with.
New boxes can be added by editing `boxes.py`. Follow the existing conventions in `boxes.py` when defining a new box.
The new box can be used in `workspace.py` by calling the function from `boxes.py`. The functions are available under a custom module name, specified at the beginning of `boxes.py`.
You must use existing boxes directly in `workspace.py` without adding them to `boxes.py`.
You can see any errors that occurred in the boxes in `errors.txt`. Before finishing a task you MUST FIX ALL ERRORS in the new boxes.
If a custom box returns an 'Unknown operation' error message, check if you are using the correct module name for the new box.
The module name and usage examples are specified at the beginning of `boxes.py`.
Attempt to fix any errors in the boxes you add, and if you cannot, explain to the user what went wrong and how to fix it.
If workspace.py is empty, you can still add new boxes by editing boxes.py or using the pre-defined boxes.
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
async def assistant_stream(
    req: AssistantCompletionRequest, skill_root="../.agents/skills"
) -> StreamingResponse:
    model = os.environ.get("LYNXKITE_ASSISTANT_MODEL")
    workspace_backend = WorkspaceBackend(req.workspace)
    backend = backends.CompositeBackend(
        default=workspace_backend,
        routes={
            "/skills/": backends.FilesystemBackend(
                root_dir=skill_root, virtual_mode=True
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
    ws = workspace.Workspace.load(req.workspace)
    ws.assistant_messages = request_messages.copy()
    ws.save(req.workspace)
    if crdt:
        room = crdt.get_room_or_none(req.workspace)
        if room is not None:
            crdt.update_workspace(room.ws, ws)

    async def generate():
        response_message = []
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
                response_message.append(delta)
        ws = workspace.Workspace.load(req.workspace)
        if not ws.assistant_messages:
            ws.assistant_messages = []
        ws.assistant_messages.append(
            {"role": "assistant", "content": "".join(response_message)}
        )
        ws.save(req.workspace)
        if crdt:
            room = crdt.get_room_or_none(req.workspace)
            if room is not None:
                crdt.update_workspace(room.ws, ws)

    try:
        gen = generate()
        first_chunk = (
            await gen.__anext__()
        )  # peek the first chunk to check for authentication errors

        async def chained_generator():
            yield first_chunk
            async for chunk in gen:
                yield chunk

        return StreamingResponse(
            chained_generator(), media_type="text/event-stream; charset=utf-8"
        )
    except openai.AuthenticationError:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI Authentication failed. Check your API key.",
        )
