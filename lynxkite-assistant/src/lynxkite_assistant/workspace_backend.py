"""A Deep Agents backend that represents a workspace as a file."""

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileInfo,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    grep_matches_from_files,
)


class WorkspaceBackend(BackendProtocol):
    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    def ls(self, path: str) -> LsResult:
        if path != "/":
            return LsResult(entries=[])
        entries: list[FileInfo] = [
            FileInfo(
                path="/workspace.py",
                is_dir=False,
            ),
            FileInfo(
                path="/boxes.py",
                is_dir=False,
            ),
        ]
        return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        match file_path:
            case "/workspace.py":
                return ReadResult(
                    file_data=FileData(
                        encoding="utf-8", content="# Python representation of the workspace.\n"
                    )
                )
            case "/boxes.py":
                return ReadResult(
                    file_data=FileData(
                        encoding="utf-8", content="# Custom box definitions for the workspace.\n"
                    )
                )
            case _:
                return ReadResult(error=f"File '{file_path}' not found")

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        match file_path:
            case "/workspace.py":
                return WriteResult(path=file_path)
            case "/boxes.py":
                return WriteResult(path=file_path)
            case _:
                return WriteResult(error=f"File '{file_path}' not found")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        match file_path:
            case "/workspace.py":
                return EditResult(path=file_path, occurrences=0)
            case "/boxes.py":
                return EditResult(path=file_path, occurrences=0)
            case _:
                return EditResult(error=f"File '{file_path}' not found")

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search state files for a literal text pattern."""
        return grep_matches_from_files({}, pattern, path if path is not None else "/", glob)

    # def glob(self, pattern: str, path: str = "/") -> GlobResult: ...

    # def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...

    # def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
