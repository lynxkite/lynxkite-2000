#!/usr/bin/env python3
"""Pre-commit hook to check the demo workspaces."""

from lynxkite_core import workspace, ops, workspace_paths
from pathlib import Path
import os
import asyncio
import sys

demo_dir = "examples"


def check_demo_ws(ws_path):
    ws = workspace.Workspace.load(ws_path)
    changed_ws = False
    if not ws.paused:
        ws.paused = True
        changed_ws = True
    workspace_paths.workspace_data_dir(ws_path).mkdir(parents=True, exist_ok=True)
    if ws.assistant_messages:
        ws.assistant_messages = []
        changed_ws = True
    missing_ws_files = False
    for node in ws.nodes:
        if (
            node.type
            in [
                "visualization",
                "graph_visualization",
                "table_view",
                "image",
                "molecule",
            ]
            and not workspace_paths.display_path(ws_path, node.id).exists()
        ):
            missing_ws_files = True
    if missing_ws_files:
        try:
            with open(os.devnull, "w") as f:
                old_stderr = sys.stderr
                sys.stderr = f
                asyncio.run(ops.EXECUTORS[ws.env](ws, ops.CATALOGS[ws.env]))
                sys.stderr = old_stderr
        except Exception as e:
            return f"Error executing workspace {ws_path}: {e}"
        changed_ws = True
    if changed_ws:
        ws.save(ws_path)
    for node in ws.nodes:
        if node.data.error and node.type != "comment" and node.type != "node_group":
            # groups and comments always have "Unknown operation" error, those can be ignored
            return f"{ws_path}: Node '{node.id}' has error: {node.data.error}"


if __name__ == "__main__":
    if not os.getcwd().endswith("lynxkite-2000"):
        print("Please run this script from the lynxkite-2000 directory.")
        sys.exit(1)
    os.chdir(os.path.join(os.getcwd(), demo_dir))
    ops.detect_plugins()
    errors = []
    for ws_file in sys.argv[1:]:
        ws_path = Path(ws_file)
        if (
            ws_path.is_relative_to(demo_dir)
            and (
                ws_path.suffix == ".lynxkite"
                or ws_path.suffixes[-2:] == [".lynxkite", ".json"]
                or (ws_path.name == "workspace.json" and ws_path.parent.suffix == ".lynxkite")
            )
            and ".workspace_files" not in ws_file
            and "generated_samples" not in ws_file
        ):
            if ws_path.name == "workspace.json" and ws_path.parent.suffix == ".lynxkite":
                ws_path = ws_path.parent
            e = check_demo_ws(ws_path.relative_to(demo_dir))
            if e:
                errors.append(e)
    if errors:
        print("Errors found in demo workspaces:")
        for e in errors:
            print(f"\t - {e}")
        sys.exit(1)
