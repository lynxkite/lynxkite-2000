from mcp.server.fastmcp import FastMCP
import json

from lynxkite_core import ops
from lynxkite_assistant import workspace_backend, instructions

mcp = FastMCP("LynxKite Assistant", instructions=instructions.SYSTEM_PROMPT)


@mcp.tool()
def get_lynxkite_workspace_layout(ws_path) -> str:
    """Get the layout of a workspace"""
    return workspace_backend.get_workspace_layout(ws_path)


@mcp.tool()
def get_lynxkite_workspace_file_content(ws_path) -> str:
    """Get the content of a workspace.py file"""
    return workspace_backend.get_workspace_file_content(ws_path)


@mcp.tool()
def get_lynxkite_custom_lynxkite_boxes(ws_path) -> str:
    """Get the source code of the custom LynxKite boxes"""
    return workspace_backend.get_boxes_file_content(ws_path)


@mcp.tool()
def get_lynxkite_workspace_errors(ws_path) -> str:
    """Get the workspace errors"""
    return workspace_backend.get_errors_file_content(ws_path)


@mcp.tool(description=instructions.LAYOUT_INFO)
def edit_layout(ws_path, content: str) -> None:
    """Edit the layout of the workspace"""
    workspace_backend.set_layout_file_content(ws_path, json.loads(content))


@mcp.tool(description=instructions.WORKSPACE_INFO)
async def edit_workspace(ws_path, content: str) -> None:
    """Edit the content of the workspace.py file"""
    await workspace_backend.set_workspace_file_content(ws_path, content)


@mcp.tool()
def edit_boxes(ws_path, content: str) -> None:
    """Edit the content of the boxes.py file to define custom boxes"""
    workspace_backend.set_boxes_file_content(ws_path, content)


def main():
    ops.detect_plugins()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
