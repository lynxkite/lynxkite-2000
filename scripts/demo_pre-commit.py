#!/usr/bin/env python3
"""Pre-commit hook to check the demo workspaces."""

from lynxkite_core import workspace, ops
from pathlib import Path
import os
import asyncio
import sys

demo_dir = "examples"


def check_demo_ws(ws_path):
    parent_dir = ws_path.parent
    ws_name = ws_path.name
    ws = workspace.Workspace.load(ws_path)
    changed_ws = False
    if not ws.paused:
        ws.paused = True
        changed_ws = True
    if not (parent_dir / ".workspace_files" / ws_name).exists():
        os.makedirs(parent_dir / ".workspace_files" / ws_name, exist_ok=True)
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
            and not (parent_dir / ".workspace_files" / ws_name / f"{node.id}.json").exists()
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
            print(f"Error executing workspace {ws_path}: {e}")
        changed_ws = True
    if changed_ws:
        ws.save(ws_path)
    for node in ws.nodes:
        if node.data.error and node.type != "comment" and node.type != "node_group":
            # groups and comments always have "Unknown operation" error, those can be ignored
            return f"{ws_path}: Node '{node.id}' has error: {node.data.error}"


if __name__ == "__main__":
    os.chdir(os.path.join(os.getcwd(), demo_dir))
    ops.detect_plugins()
    errors = []
    for ws_file in sys.argv[1:]:
        ws_path = Path(ws_file)
        if (
            ws_path.is_relative_to(demo_dir)
            and ws_file[-14:] == ".lynxkite.json"
            and ".workspace_files" not in ws_file
            and "generated_samples" not in ws_file
        ):
            e = check_demo_ws(ws_path.relative_to(demo_dir))
            if e:
                errors.append(e)
    if errors:
        print("Errors found in demo workspaces:")
        for e in errors:
            print(f"\t - {e}")
        sys.exit(1)
