"""Ask mode must not change the workspace or anything on disk."""

import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from lynxkite_assistant import assistant
from lynxkite_core import ops, workspace
from lynxkite_core.workspace_paths import workspace_data_dir

FILES = Path(__file__).parent / "files"


class ToolCallingFakeModel(GenericFakeChatModel):
    """Replays the given messages, including their tool calls. Records the prompts."""

    prompts: list[list[BaseMessage]] = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


def fake_model(*messages: AIMessage) -> ToolCallingFakeModel:
    # The streaming implementation of GenericFakeChatModel drops the tool calls.
    return ToolCallingFakeModel(messages=iter(messages), disable_streaming=True)


def text(s: str) -> list[str | dict]:
    # assistant_stream only streams list content.
    return [{"type": "text", "text": s}]


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def ws_path(tmp_path, monkeypatch) -> str:
    # Workspace paths are relative to the data directory.
    monkeypatch.chdir(tmp_path)
    path = Path("test.lynxkite")
    shutil.copytree(FILES / "original.lynxkite", path)
    workspace_data_dir(path).mkdir()
    (workspace_data_dir(path) / "result.txt").write_text("the answer is 42")
    skills = tmp_path / "skills"
    (skills / "some-skill").mkdir(parents=True)
    (skills / "some-skill" / "SKILL.md").write_text(
        "---\nname: some-skill\ndescription: A skill.\n---\nDetails."
    )
    monkeypatch.setattr(assistant, "SKILL_ROOT", str(skills))
    return str(path)


# These writes would all change the workspace or files on disk if they went through.
WRITE_ATTEMPTS = [
    {
        "name": "edit_file",
        "args": {
            "file_path": "/workspace.py",
            "old_string": "#! ## A beautiful comment",
            "new_string": "#! ## An edited comment",
        },
    },
    {
        "name": "edit_file",
        "args": {
            "file_path": "/layout.json",
            "old_string": '"automatic_layout": false',
            "new_string": '"automatic_layout": true',
        },
    },
    {
        "name": "edit_file",
        "args": {
            "file_path": "/boxes.py",
            "old_string": "# Add new box definitions here.",
            "new_string": "# Add new box definitions here.\nx = 1",
        },
    },
    {
        "name": "write_file",
        "args": {"file_path": "/workspace_files/new.txt", "content": "x"},
    },
    {
        "name": "edit_file",
        "args": {
            "file_path": "/workspace_files/result.txt",
            "old_string": "42",
            "new_string": "43",
        },
    },
    {"name": "write_file", "args": {"file_path": "/skills/new.md", "content": "x"}},
    # Hidden paths are easy to miss with glob-based checks.
    {
        "name": "write_file",
        "args": {"file_path": "/workspace_files/.hidden.txt", "content": "x"},
    },
    {
        "name": "write_file",
        "args": {"file_path": "/skills/.hidden/new.md", "content": "x"},
    },
]


def tool_calls(calls: list[dict]) -> list[dict]:
    return [{**c, "id": f"call_{i}"} for i, c in enumerate(calls)]


def tool_results(result: dict) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}


async def test_ask_mode_cannot_write(ws_path, tmp_path):
    before = snapshot(tmp_path)
    model = fake_model(
        AIMessage(content=text("Let me try."), tool_calls=tool_calls(WRITE_ATTEMPTS)),
        AIMessage(content=text("Done.")),
    )
    agent = assistant.create_agent(ws_path, "ask", model)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hi"}]})
    results = tool_results(result)
    assert len(results) == len(WRITE_ATTEMPTS)
    for call_id, message in results.items():
        assert message.status == "error", (call_id, message.content)
    assert snapshot(tmp_path) == before
    # The previous states are not initialized in Ask mode.
    assert not (tmp_path / ".assistant_memory").exists()


@pytest.mark.parametrize(
    "attempt", WRITE_ATTEMPTS, ids=lambda a: f"{a['name']} {a['args']['file_path']}"
)
async def test_agent_mode_can_write(ws_path, tmp_path, monkeypatch, attempt):
    # Makes sure that each write in test_ask_mode_cannot_write would really change something.
    monkeypatch.setattr(ops, "EXECUTORS", defaultdict(lambda: _skip_execution))
    monkeypatch.setattr(ops, "CATALOGS", defaultdict(dict))
    before = snapshot(tmp_path)
    model = fake_model(
        AIMessage(content=text("Let me try."), tool_calls=tool_calls([attempt])),
        AIMessage(content=text("Done.")),
    )
    agent = assistant.create_agent(ws_path, "agent", model)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hi"}]})
    [message] = tool_results(result).values()
    assert message.status == "success", message.content
    after = snapshot(tmp_path)
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    # Agent mode saves the previous states. We are interested in the other changes.
    assert {k for k in changed if not k.startswith(".assistant_memory")}


async def _skip_execution(ws, catalog):
    pass


async def test_ask_mode_can_read(ws_path):
    reads = [
        {"name": "read_file", "args": {"file_path": "/workspace.py"}},
        {"name": "read_file", "args": {"file_path": "/workspace_files/result.txt"}},
        {"name": "ls", "args": {"path": "/skills/some-skill"}},
        {"name": "grep", "args": {"pattern": "answer", "path": "/workspace_files"}},
    ]
    model = fake_model(
        AIMessage(content=text("Let me look."), tool_calls=tool_calls(reads)),
        AIMessage(content=text("Done.")),
    )
    agent = assistant.create_agent(ws_path, "ask", model)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hi"}]})
    results = tool_results(result)
    assert "enter_table_data" in results["call_0"].content
    assert "the answer is 42" in results["call_1"].content
    assert "SKILL.md" in results["call_2"].content
    assert "result.txt" in results["call_3"].content


@pytest.mark.parametrize("mode", ["ask", "agent"])
async def test_system_prompt(ws_path, mode):
    model = fake_model(AIMessage(content=text("Hello.")))
    agent = assistant.create_agent(ws_path, mode, model)
    await agent.ainvoke({"messages": [{"role": "user", "content": "Hi"}]})
    [prompt] = model.prompts
    system = "\n".join(m.text for m in prompt if isinstance(m, SystemMessage))
    assert ("You are in Ask mode" in system) == (mode == "ask")
    assert ("## Editing the workflow" in system) == (mode == "agent")
    assert "The other files listed above are internal." in system


async def _stream(ws_path: str, mode: assistant.AssistantMode, monkeypatch) -> str:
    model = fake_model(AIMessage(content=text("Hello there.")))
    monkeypatch.setattr(assistant, "_get_model", lambda: model)
    req = assistant.AssistantCompletionRequest(
        workspace=ws_path,
        mode=mode,
        messages=[assistant.AssistantMessage(role="user", content="Hi")],
    )
    response = await assistant.assistant_stream(req)
    streamed = ""
    async for chunk in response.body_iterator:
        assert isinstance(chunk, str)
        streamed += chunk
    return streamed


async def test_ask_mode_does_not_save_history(ws_path, monkeypatch):
    workspace_json = Path(ws_path) / "workspace.json"
    before = workspace_json.read_text()
    assert await _stream(ws_path, "ask", monkeypatch) == "Hello there."
    assert workspace_json.read_text() == before


async def test_agent_mode_saves_history(ws_path, monkeypatch):
    assert await _stream(ws_path, "agent", monkeypatch) == "Hello there."
    ws = workspace.Workspace.load(ws_path)
    assert ws.assistant_messages is not None
    assert [(m.role, m.content) for m in ws.assistant_messages] == [
        ("user", "Hi"),
        ("assistant", "Hello there."),
    ]


def test_mode_defaults_to_agent():
    req = assistant.AssistantCompletionRequest.model_validate(
        {"workspace": "x", "messages": []}
    )
    assert req.mode == "agent"
