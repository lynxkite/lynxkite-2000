#!/usr/bin/env python3
"""Pre-commit hook to check the demo workspaces."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from lynxkite_core import ops, workspace, workspace_paths

ops.detect_plugins()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--rerun",
    action="store_true",
    help="Allow re-running workspaces",
)
parser.add_argument(
    "--root-dir",
    default="examples",
    help="Run workspaces in this directory",
)
parser.add_argument(
    "filenames",
    nargs="*",
    help="Files to check",
)
args = parser.parse_args()


def check_demo_ws(ws_path: Path, rerun: bool):
    try:
        ws = workspace.Workspace.load(ws_path)
    except Exception as e:
        return f"Could not read {ws_path}: {e}"
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
        if not rerun:
            return f"Missing workspace files for {ws_path}"
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
    os.chdir(args.root_dir)
    ops.detect_plugins()
    errors = []
    for ws_file in args.filenames:
        # From the "workspace.json" file go to the "MyWorkspace.lynxkite" directory.
        ws_path = Path(ws_file).parent
        e = check_demo_ws(ws_path.relative_to(args.root_dir), args.rerun)
        if e:
            errors.append(e)
    if errors:
        print("Errors found in demo workspaces:")
        for e in errors:
            print(f"\t - {e}")
        sys.exit(1)
