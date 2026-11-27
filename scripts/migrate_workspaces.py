#!/usr/bin/env python3
"""Batch-migrate legacy LynxKite workspaces without requiring the app."""

import argparse
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

LEGACY_SUFFIX = ".lynxkite.json"
BUNDLE_SUFFIX = ".lynxkite"
SIDECAR_DIR = ".workspace_files"
WORKSPACE_FILE = "workspace.json"
NODE_DATA_DIR = "node_data"
DISPLAY_FILE = "display.json"
IGNORED_DIRS = {
    ".crdt",
    ".git",
    ".test-workspaces",
    ".workspace_files",
    ".venv",
    "__pycache__",
    "node_modules",
}


def find_legacy_workspaces(root: Path) -> Iterator[Path]:
    """Find legacy workspace files below a data root, excluding bundle internals."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root.name.endswith(BUNDLE_SUFFIX):
        return

    def raise_walk_error(error: OSError) -> None:
        raise error

    for parent, directories, filenames in os.walk(root, onerror=raise_walk_error):
        directories[:] = sorted(
            name
            for name in directories
            if name not in IGNORED_DIRS and not name.endswith(BUNDLE_SUFFIX)
        )
        for filename in sorted(filenames):
            if filename.endswith(LEGACY_SUFFIX):
                yield Path(parent) / filename


def legacy_data_dir(workspace_path: Path) -> Path:
    return workspace_path.parent / SIDECAR_DIR / workspace_path.name


def bundle_data_dir(bundle_path: Path) -> Path:
    return bundle_path / NODE_DATA_DIR


def validate_migration(old_path: Path, new_path: Path) -> list[Path]:
    """Validate a migration and return the legacy display files to convert."""
    if not old_path.is_file():
        raise FileNotFoundError(old_path)
    if not new_path.name.endswith(BUNDLE_SUFFIX):
        raise ValueError(f"Destination is not a workspace bundle: {new_path}")
    if new_path.exists() or new_path.is_symlink():
        raise FileExistsError(new_path)
    if not new_path.parent.is_dir():
        raise NotADirectoryError(new_path.parent)

    old_data_path = legacy_data_dir(old_path)
    if old_data_path.is_symlink():
        raise ValueError(f"Workspace data directory cannot be a symlink: {old_data_path}")
    if old_data_path.exists() and not old_data_path.is_dir():
        raise NotADirectoryError(old_data_path)

    display_files = sorted(old_data_path.glob("*.json")) if old_data_path.exists() else []
    for display_file in display_files:
        if not display_file.is_file():
            raise ValueError(f"Workspace display data is not a file: {display_file}")
        node_data_dir = old_data_path / display_file.stem
        if node_data_dir.exists() or node_data_dir.is_symlink():
            raise FileExistsError(node_data_dir)
    return display_files


def migrate_workspace(old_path: Path, new_path: Path) -> Path:
    """Move a legacy workspace and its sidecar data into a bundle."""
    display_files = validate_migration(old_path, new_path)
    old_data_path = legacy_data_dir(old_path)
    workspace_file = new_path / WORKSPACE_FILE
    new_data_path = bundle_data_dir(new_path)

    new_path.mkdir()
    workspace_moved = False
    data_moved = False
    try:
        old_path.rename(workspace_file)
        workspace_moved = True
        if old_data_path.exists():
            old_data_path.rename(new_data_path)
            data_moved = True
            for display_file in display_files:
                node_data_dir = new_data_path / display_file.stem
                node_data_dir.mkdir()
                (new_data_path / display_file.name).rename(node_data_dir / DISPLAY_FILE)
    except Exception as migration_error:
        rollback_errors: list[str] = []

        def rollback(action: Callable[[], object]) -> None:
            try:
                action()
            except OSError as error:
                rollback_errors.append(str(error))

        if data_moved:
            for display_file in display_files:
                node_data_dir = new_data_path / display_file.stem
                migrated_display = node_data_dir / DISPLAY_FILE
                if migrated_display.exists():
                    rollback(lambda: migrated_display.rename(new_data_path / display_file.name))
                if node_data_dir.exists():
                    rollback(node_data_dir.rmdir)
            rollback(lambda: new_data_path.rename(old_data_path))
        if workspace_moved:
            rollback(lambda: workspace_file.rename(old_path))
        if not rollback_errors:
            rollback(new_path.rmdir)
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Migration failed and rollback was incomplete; inspect {new_path}: {details}"
            ) from migration_error
        raise

    return new_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy .lynxkite.json workspaces under DATA_DIR to .lynxkite bundles. "
            "Stop the LynxKite server before running this command."
        )
    )
    parser.add_argument("data_dir", type=Path, help="LynxKite data directory to migrate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List migrations without changing any files",
    )
    args = parser.parse_args(argv)

    try:
        workspaces = list(find_legacy_workspaces(args.data_dir))
    except OSError as error:
        print(f"Cannot scan data directory: {error}", file=sys.stderr)
        return 2

    if not workspaces:
        print("No legacy workspaces found.")
        return 0

    root = args.data_dir.resolve()
    errors = 0
    migrated = 0
    for old_path in workspaces:
        new_path = old_path.with_suffix("")
        relative_old = old_path.relative_to(root)
        relative_new = new_path.relative_to(root)
        if args.dry_run:
            try:
                validate_migration(old_path, new_path)
            except (OSError, ValueError) as error:
                print(f"Cannot migrate {relative_old}: {error}", file=sys.stderr)
                errors += 1
                continue
            print(f"Would migrate {relative_old} -> {relative_new}")
            continue

        try:
            migrate_workspace(old_path, new_path)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Cannot migrate {relative_old}: {error}", file=sys.stderr)
            errors += 1
            continue
        print(f"Migrated {relative_old} -> {relative_new}")
        migrated += 1

    if errors:
        print(f"Migration finished with {errors} error(s); {migrated} workspace(s) migrated.")
        return 1
    if args.dry_run:
        print(f"Dry run complete: {len(workspaces)} workspace(s) would be migrated.")
    else:
        print(f"Migration complete: {migrated} workspace(s) migrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
