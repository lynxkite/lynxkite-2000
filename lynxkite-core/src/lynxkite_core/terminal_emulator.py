"""Terminal emulation utilities for streaming stdout/stderr to LynxKite UI."""

from __future__ import annotations

import contextlib
import fcntl
import io
import os
import shutil
import struct
import sys
import termios
from typing import Protocol, TextIO
import pyte
from pyte.modes import LNM


class TerminalMessageContext(Protocol):
    def set_message(self, message: str) -> None: ...


class _UiTerminal:
    """VT100 screen emulator that feeds rendered content to a TerminalMessageContext."""

    def __init__(
        self,
        op_ctx: TerminalMessageContext,
        columns: int = 120,
        lines: int = 40,
        history: int = 100,
    ):
        self.op_ctx = op_ctx
        self._pyte_screen = pyte.HistoryScreen(columns, lines, history=history, ratio=1.0)
        self._pyte_screen.set_mode(LNM)
        self._pyte_stream = pyte.Stream(self._pyte_screen)

    def _render_pyte(self) -> str:
        """Renders the current content of the pyte screen, including history, into a single string."""
        lines: list[str] = []
        terminal_history = self._pyte_screen.history
        if terminal_history is not None:
            lines.extend([self._render_pyte_line(line) for line in terminal_history.top])
        lines.extend([self._render_pyte_line(line) for line in self._pyte_screen.display])
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def _render_pyte_line(self, line: str | dict[int, pyte.screens.Char]) -> str:
        """Renders a line from pyte, which can be either a simple string or a dict of character cells."""
        if isinstance(line, str):
            return line.rstrip()
        width = self._pyte_screen.columns
        chars = [" "] * width
        for idx, cell in line.items():
            if 0 <= idx < width:
                chars[idx] = getattr(cell, "data", str(cell))
        return "".join(chars).rstrip()

    def feed(self, s: str) -> None:
        if not s:
            return
        self._feed_pyte(s)
        self.op_ctx.set_message(self._render_pyte())

    def _feed_pyte(self, s: str | bytes) -> None:
        if isinstance(s, (bytes, bytearray)):
            byte_stream = pyte.ByteStream(self._pyte_screen)
            byte_stream.feed(bytes(s))
            return
        self._pyte_stream.feed(s)


class _RealtimeStream(io.TextIOBase):
    """A TextIOBase wrapper that feeds written data to a _UiTerminal for
    real-time content extraction.
    """

    def __init__(
        self,
        original: TextIO,
        ui_terminal: _UiTerminal,
    ):
        self._original = original
        self._ui_terminal = ui_terminal

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._original.write(s)
        self._ui_terminal.feed(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "replace")

    @property
    def buffer(self):
        return getattr(self._original, "buffer", None)

    def fileno(self) -> int:
        return self._original.fileno()


@contextlib.contextmanager
def _patched_terminal_size(columns: int, lines: int):
    old_get_terminal_size = os.get_terminal_size
    old_shutil_get_terminal_size = shutil.get_terminal_size
    old_columns_env = os.environ.get("COLUMNS")
    old_lines_env = os.environ.get("LINES")
    old_fcntl_ioctl = fcntl.ioctl
    old_termios_tcgetwinsize = termios.tcgetwinsize

    def _get_terminal_size(_fd: int = 0, /) -> os.terminal_size:
        return os.terminal_size((columns, lines))

    def _shutil_terminal_size(_fallback: tuple[int, int] = (columns, lines)) -> os.terminal_size:
        return os.terminal_size((columns, lines))

    def _termios_tcgetwinsize(_fd) -> tuple[int, int]:
        return (lines, columns)

    # Rewrite lower-level calls that query the terminal size using ioctl
    def _fcntl_ioctl(fd, request, arg=0, mutate_flag=True):
        if request == getattr(termios, "TIOCGWINSZ", None):
            packed = struct.pack("HHHH", lines, columns, 0, 0)
            if isinstance(arg, (bytearray, memoryview)):
                n = min(len(arg), len(packed))
                arg[:n] = packed[:n]
                return 0
            return packed
        return old_fcntl_ioctl(fd, request, arg, mutate_flag)

    try:
        os.environ["COLUMNS"] = str(columns)
        os.environ["LINES"] = str(lines)
        os.get_terminal_size = _get_terminal_size  # type: ignore # same function sig
        shutil.get_terminal_size = _shutil_terminal_size  # type: ignore # same function sig
        termios.tcgetwinsize = _termios_tcgetwinsize  # type: ignore # same function sig
        fcntl.ioctl = _fcntl_ioctl  # type: ignore # same function sig
        yield
    finally:
        os.get_terminal_size = old_get_terminal_size
        shutil.get_terminal_size = old_shutil_get_terminal_size
        termios.tcgetwinsize = old_termios_tcgetwinsize
        fcntl.ioctl = old_fcntl_ioctl
        if old_columns_env is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = old_columns_env
        if old_lines_env is None:
            os.environ.pop("LINES", None)
        else:
            os.environ["LINES"] = old_lines_env


@contextlib.contextmanager
def _capture_output(op_ctx: TerminalMessageContext, columns: int, lines: int, history: int):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    ui_terminal = _UiTerminal(op_ctx, columns=columns, lines=lines, history=history)

    with _patched_terminal_size(columns, lines):
        try:
            # With global patching, multiple threads writing to stdout/stderr will be captured
            # and may be interleaved in the same terminal. When workspaces become their own
            # processes, this solution will be sufficient, as each workspace will have its own stdout/stderr.

            sys.stdout = _RealtimeStream(old_stdout, ui_terminal)
            sys.stderr = _RealtimeStream(old_stderr, ui_terminal)
            yield None
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class StdoutBinder:
    """Context manager that binds stdout/stderr to a terminal
    emulator with the ability to extract emulator content.
    """

    def __init__(self, op_ctx: TerminalMessageContext):
        self.op_ctx = op_ctx
        self._default_cm = None

    def __call__(self, *, columns: int = 75, lines: int = 10, history: int = 100):
        return _capture_output(self.op_ctx, columns=columns, lines=lines, history=history)

    def __enter__(self):
        self._default_cm = self()
        return self._default_cm.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._default_cm is not None
        return self._default_cm.__exit__(exc_type, exc_value, traceback)
