"""Build hook for bundling the frontend into the Python wheel."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return

        root = Path(self.root)
        frontend_dir = root / "web"
        frontend_dist = frontend_dir / "dist"
        package_dir = root / "src" / "lynxkite_app" / "web_assets"

        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm not found on PATH; cannot build LynxKite frontend assets")

        install_command = (
            [npm, "ci"] if (frontend_dir / "package-lock.json").exists() else [npm, "install"]
        )
        self._run(install_command, frontend_dir)
        self._run([npm, "run", "build"], frontend_dir)

        self._replace_generated_assets(frontend_dist, package_dir)
        self._verify_assets(package_dir)

        self._force_include_generated_assets(package_dir, build_data)

    def _run(self, command: list[str], cwd: Path) -> None:
        print(f"Running {shlex.join(command)} in {cwd}", flush=True)
        result = subprocess.run(command, cwd=cwd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {shlex.join(command)}"
            )

    def _replace_generated_assets(self, frontend_dist: Path, package_dir: Path) -> None:
        if not frontend_dist.is_dir():
            raise RuntimeError(f"Frontend build did not create expected directory: {frontend_dist}")

        package_dir.mkdir(parents=True, exist_ok=True)
        self._delete_generated_contents(package_dir)
        shutil.copytree(frontend_dist, package_dir, dirs_exist_ok=True)

    def _delete_generated_contents(self, directory: Path) -> None:
        for item in directory.iterdir():
            if item.name == "__init__.py":
                continue
            if item.is_dir():
                self._delete_generated_contents(item)
                if not any(item.iterdir()):
                    item.rmdir()
            else:
                item.unlink()

    def _verify_assets(self, package_dir: Path) -> None:
        index_html = package_dir / "index.html"
        assets_dir = package_dir / "assets"
        if not index_html.is_file():
            raise RuntimeError(f"Frontend build did not produce {index_html}")
        if not assets_dir.is_dir():
            raise RuntimeError(f"Frontend build did not produce {assets_dir}")
        if not any(path.is_file() and path.name != "__init__.py" for path in assets_dir.rglob("*")):
            raise RuntimeError(f"Frontend build did not produce any files in {assets_dir}")

    def _force_include_generated_assets(
        self, package_dir: Path, build_data: dict[str, Any]
    ) -> None:
        force_include = build_data.setdefault("force_include", {})
        source_root = package_dir.parent.parent
        for path in package_dir.rglob("*"):
            if path.is_file() and path.name != "__init__.py":
                force_include[str(path)] = path.relative_to(source_root).as_posix()
