"""Customized build process for setuptools."""

import subprocess
from setuptools.command.build_py import build_py as _build_py
from pathlib import Path
import shutil


class build_py(_build_py):
    def run(self):
        print("\n\nBuilding frontend...", __file__)
        here = Path(__file__).parent.parent
        frontend_dir = here / "web"
        package_dir = here / "src" / "lynxkite_app" / "web_assets"
        subprocess.check_call(["npm", "install"], cwd=frontend_dir)
        subprocess.check_call(["npm", "run", "build"], cwd=frontend_dir)
        print("files in", frontend_dir / "dist")
        for file in (frontend_dir / "dist").iterdir():
            print(file)
        # shutil.rmtree(package_dir)
        shutil.copytree(frontend_dir / "dist", package_dir, dirs_exist_ok=True)
        # (frontend_dir / "dist").rename(package_dir)
        print("files in", package_dir)
        for file in package_dir.iterdir():
            print(file)
        super().run()
