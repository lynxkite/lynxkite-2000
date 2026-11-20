"""Build static workspace export ZIP files."""

import html
import json
import pathlib
import zipfile

from lynxkite_core import workspace, workspace_paths


def _zip_path(path: pathlib.Path) -> str:
    return path.as_posix()


def _add_directory(
    zf: zipfile.ZipFile,
    source: pathlib.Path,
    target: pathlib.Path,
    skip_root_files: set[str] | None = None,
):
    skip_root_files = skip_root_files or set()
    for path in source.rglob("*"):
        if path.is_file():
            rel = path.relative_to(source)
            if len(rel.parts) == 1 and rel.name in skip_root_files:
                continue
            zf.write(path, _zip_path(target / rel))


def _static_index_html(web_assets_path: pathlib.Path) -> str:
    index_html = (web_assets_path / "index.html").read_text(encoding="utf-8")
    # Use relative paths for the static export.
    index_html = index_html.replace('<base href="/" />', "")
    static_config = {
        "workspace": "data/workspace.lynxkite",
        "filesBase": "data/workspace.lynxkite/node_data/",
    }
    script = (
        "<script>window.LYNXKITE_STATIC_WORKSPACE = "
        + html.escape(json.dumps(static_config), quote=False)
        + ";</script>"
    )
    return index_html.replace("</head>", f"    {script}\n  </head>")


def build_workspace_zip(
    zip_path: pathlib.Path,
    data_path: pathlib.Path,
    workspace_path: pathlib.Path,
    web_assets_path: pathlib.Path,
):
    workspace_bundle = data_path / workspace_path
    workspace_files_path = workspace_paths.workspace_data_dir(workspace_bundle)
    bundle_zip_path = pathlib.Path("data/workspace.lynxkite")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _add_directory(zf, web_assets_path, pathlib.Path(), skip_root_files={"index.html"})
        zf.writestr("index.html", _static_index_html(web_assets_path))
        # Load the workspace to make sure the metadata is updated.
        ws = workspace.Workspace.load(workspace_bundle)
        zf.writestr(_zip_path(bundle_zip_path / "workspace.json"), ws.model_dump_json_sorted())
        if workspace_files_path.exists():
            _add_directory(
                zf,
                workspace_files_path,
                bundle_zip_path / workspace_paths.NODE_DATA_DIRNAME,
            )
