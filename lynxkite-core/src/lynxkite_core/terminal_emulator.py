"""Terminal emulation utilities for streaming stdout/stderr to LynxKite UI."""

# TODO: Fails in multi-threaded contexts
from __future__ import annotations

import contextlib
import io
import os
import shutil
import struct
import sys
from typing import Protocol
import fcntl
import termios
import pyte


class TerminalMessageContext(Protocol):
    def set_message(self, message: str) -> None: ...


class _UiTerminal:
    """VT100 screen emulator for UI updates."""

    def __init__(
        self,
        op_ctx: TerminalMessageContext,
        columns: int = 120,
        lines: int = 40,
        history: int = 100,
    ):
        self.op_ctx = op_ctx
        self._pyte_screen = pyte.HistoryScreen(columns, lines, history=history, ratio=1.0)
        self._pyte_screen.set_mode(pyte.modes.LNM)
        self._pyte_stream = pyte.Stream(self._pyte_screen)

    def _render_pyte(self) -> str:
        lines: list[str] = []
        history = getattr(self._pyte_screen, "history", None)
        if history is not None:
            lines.extend([self._render_pyte_line(line) for line in history.top])
        lines.extend([self._render_pyte_line(line) for line in self._pyte_screen.display])
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def _render_pyte_line(self, line: object) -> str:
        if isinstance(line, str):
            return line.rstrip()
        if isinstance(line, dict):
            width = self._pyte_screen.columns
            chars = [" "] * width
            for idx, cell in line.items():
                if 0 <= idx < width:
                    chars[idx] = getattr(cell, "data", str(cell))
            return "".join(chars).rstrip()
        return str(line).rstrip()

    def feed(self, s: str) -> None:
        if not s:
            return
        self._feed_pyte(s)
        self.op_ctx.set_message(self._render_pyte())

    def _feed_pyte(self, s: str | bytes) -> None:
        if isinstance(s, (bytes, bytearray)):
            stream = pyte.ByteStream(self._pyte_screen)
            stream.feed(bytes(s))
            return
        self._pyte_stream.feed(s)


class _RealtimeStream(io.TextIOBase):
    def __init__(
        self,
        original: io.TextIOBase,
        ui_terminal: _UiTerminal,
    ):
        self.original = original
        self.ui_terminal = ui_terminal

    def write(self, s: str) -> int:
        if not s:
            return 0
        self.original.write(s)
        self.ui_terminal.feed(s)
        return len(s)

    def flush(self) -> None:
        self.original.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return self.original.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.original, "errors", "replace")

    def fileno(self) -> int:
        return self.original.fileno()


@contextlib.contextmanager
def _patched_terminal_size(columns: int, lines: int):
    old_get_terminal_size = os.get_terminal_size
    old_shutil_get_terminal_size = shutil.get_terminal_size
    old_columns_env = os.environ.get("COLUMNS")
    old_lines_env = os.environ.get("LINES")
    old_fcntl_ioctl = getattr(fcntl, "ioctl", None)
    old_termios_tcgetwinsize = getattr(termios, "tcgetwinsize", None)

    def _get_terminal_size(_fd=0):
        return os.terminal_size((columns, lines))

    def _shutil_terminal_size(_fallback=(columns, lines)):
        return os.terminal_size((columns, lines))

    def _termios_tcgetwinsize(_fd):
        return (lines, columns)

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
        os.get_terminal_size = _get_terminal_size
        shutil.get_terminal_size = _shutil_terminal_size
        termios.tcgetwinsize = _termios_tcgetwinsize
        fcntl.ioctl = _fcntl_ioctl
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
def _captured_output(op_ctx: TerminalMessageContext, columns: int, lines: int, history: int):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    ui_terminal = _UiTerminal(op_ctx, columns=columns, lines=lines, history=history)

    with _patched_terminal_size(columns, lines):
        try:
            sys.stdout = _RealtimeStream(old_stdout, ui_terminal)
            sys.stderr = _RealtimeStream(old_stderr, ui_terminal)
            yield None
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class StdoutBinder:
    def __init__(self, op_ctx: TerminalMessageContext):
        self.op_ctx = op_ctx
        self._default_cm = None

    def __call__(self, *, columns: int = 120, lines: int = 40, history: int = 100):
        return _captured_output(self.op_ctx, columns=columns, lines=lines, history=history)

    def __enter__(self):
        self._default_cm = self()
        return self._default_cm.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._default_cm is not None
        return self._default_cm.__exit__(exc_type, exc_value, traceback)
