"""A Deep Agents backend that represents a workspace as a file."""

import pathlib
from typing import Any
from deepagents.backends.state import StateBackend
from lynxkite_core import workspace
from . import python_workspace_conversion

BOXES_PLACEHOLDER = '''
"""Custom box definitions for the workspace."""
from lynxkite_core import ops
ENV = "Pillow"
op = ops.op_decorator(ENV)

@op("Flip horizontally")
def flip_horizontally(image):
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
'''.strip()


class WorkspaceBackend(StateBackend):
    def __init__(self, workspace: str) -> None:
        super().__init__()
        self._workspace = workspace

    def _read_files(self) -> dict[str, Any]:
        return {
            "/workspace.py": {
                "content": get_workspace_file_content(self._workspace),
                "modified_at": "",
            },
            "/boxes.py": {"content": get_boxes_file_content(self._workspace), "modified_at": ""},
        }

    def _send_files_update(self, update: dict[str, Any]) -> None:
        for k, v in update.items():
            if k == "/workspace.py":
                set_workspace_file_content(self._workspace, v["content"])
            if k == "/boxes.py":
                set_boxes_file_content(v["content"])


def get_workspace_file_content(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    return python_workspace_conversion.workspace_to_python(ws)


def set_workspace_file_content(ws_path: str, content: str) -> None:
    ws = python_workspace_conversion.python_to_workspace(content)
    ws.save(ws_path)


def get_boxes_file_content(ws_path: str) -> str:
    p = pathlib.Path(__file__).parent / "boxes.py"
    if not p.exists():
        return BOXES_PLACEHOLDER
    with open(p) as f:
        return f.read()


def set_boxes_file_content(content: str) -> None:
    p = pathlib.Path(__file__).parent / "boxes.py"
    with open(p, "w") as f:
        f.write(content)
