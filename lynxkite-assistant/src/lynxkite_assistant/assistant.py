"""FastAPI router exposing a Deep Agents assistant."""

import os
import fastapi
import openai
import pydantic
import requests
from typing import cast
from pathlib import Path
from fastapi.responses import StreamingResponse
import deepagents
from deepagents import backends
from .workspace_backend import WorkspaceBackend
from lynxkite_core import workspace
from .instructions import SYSTEM_PROMPT, INTERNET_ACCESS_INFO

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


web_access_url = os.environ.get("LYNXKITE_WEB_ACCESS_URL")
headers = None
if os.environ.get("LYNXKITE_WEB_ACCESS_API_KEY"):
    headers = {
        "Authorization": f"Bearer {os.environ.get('LYNXKITE_WEB_ACCESS_API_KEY')}"
    }


def internet_search(query: str, **kwargs: object):
    """Run a web search. kwargs: compatible with the FireCrawl API, will be passed in the request body."""
    # sometimes kwargs is passed as a dict under the key "kwargs", sometimes it's passed directly as keyword arguments. Handle both cases.
    other_params = kwargs.get("kwargs", kwargs)
    return requests.post(
        f"{web_access_url}/v1/search",
        json={"query": query, **other_params},
        headers=headers,
    ).text


def scrape_web_page(url: str, **kwargs: object):
    """Scrape a web page. kwargs: compatible with the FireCrawl API, will be passed in the request body."""
    other_params = kwargs.get("kwargs", kwargs)
    return requests.post(
        f"{web_access_url}/v1/scrape",
        json={"url": url, **other_params},
        headers=headers,
    ).text


def map_web_page(url: str, **kwargs: object):
    """Input a website and get all the urls on the website. kwargs: compatible with the FireCrawl API, will be passed in the request body."""
    other_params = kwargs.get("kwargs", kwargs)
    return requests.post(
        f"{web_access_url}/v1/map",
        json={"url": url, **other_params},
        headers=headers,
    ).text


@router.post("/api/assistant/stream")
async def assistant_stream(
    req: AssistantCompletionRequest, skill_root="../.agents/skills"
) -> StreamingResponse:
    model = os.environ.get("LYNXKITE_ASSISTANT_MODEL")
    workspace_backend = WorkspaceBackend(req.workspace)
    routes = {
        "/skills/": backends.FilesystemBackend(root_dir=skill_root, virtual_mode=True)
    }
    workspace_files_path = (
        Path(req.workspace).parent / ".workspace_files" / Path(req.workspace).name
    )
    if workspace_files_path.exists():
        routes["/workspace_files/"] = backends.FilesystemBackend(
            root_dir=str(workspace_files_path), virtual_mode=True
        )
    backend = backends.CompositeBackend(
        default=workspace_backend,
        routes=routes,
    )
    tools = [internet_search, scrape_web_page, map_web_page] if web_access_url else []
    agent = deepagents.create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills"],
        tools=tools,
        system_prompt=(SYSTEM_PROMPT + INTERNET_ACCESS_INFO)
        if web_access_url
        else SYSTEM_PROMPT,
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

    async def generate():
        response_message = []
        async for chunk in agent.astream(
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
