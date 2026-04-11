"""A Deep Agents backend that represents a workspace as a file."""

import pathlib
from typing import Any
from deepagents.backends.state import StateBackend
from lynxkite_core import workspace

WORKSPACE_TEMPLATE = '''
"""The Python representation of the workspace."""
def main():
'''.strip()

BOXES_TEMPLATE = '''
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
            "/workspace.py": {"content": get_workspace_file_content(self._workspace)},
            "/boxes.py": {"content": get_boxes_file_content(self._workspace)},
        }

    def _send_files_update(self, update: dict[str, Any]) -> None:
        print("update", update)


def get_workspace_file_content(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    code = [WORKSPACE_TEMPLATE]
    for node in ws.nodes:
        code.append(f"\n    # node: {node.id} ({node.type})\n")
    return "\n".join(code)


def get_boxes_file_content(ws_path: str) -> str:
    p = pathlib.Path(__file__).parent / "boxes.py"
    if not p.exists():
        return BOXES_TEMPLATE
    with open(p) as f:
        return f.read()
