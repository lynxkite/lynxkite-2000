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
import threading
from typing import Protocol, TextIO
import pyte
from pyte.modes import LNM


class TerminalMessageContext(Protocol):
    def set_message(self, message: str) -> None: ...


DEFAULT_COLUMNS = 70
DEFAULT_LINES = 30
DEFAULT_HISTORY = 500

_terminals: dict[int, "_UiTerminal"] = {}
_sizes: dict[int, tuple[int, int]] = {}
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
_orig___stdout__ = sys.__stdout__
_orig___stderr__ = sys.__stderr__
_orig_get_terminal_size = os.get_terminal_size
_orig_shutil_get_terminal_size = shutil.get_terminal_size
_orig_termios_tcgetwinsize = termios.tcgetwinsize
_orig_fcntl_ioctl = fcntl.ioctl


class _UiTerminal:
    """pyte-based VT100 emulator that pushes rendered output to the UI."""

    def __init__(
        self,
        ctx: TerminalMessageContext,
        columns: int,
        lines: int,
        history: int,
        passthrough: bool = True,
        *,
        owner_tid: int = 0,
    ):
        self.ctx = ctx
        self._owner_tid = owner_tid
        self._passthrough = passthrough
        self._screen = pyte.HistoryScreen(columns, lines, history=history, ratio=1.0)
        self._screen.set_mode(LNM)
        self._stream = pyte.Stream(self._screen)

    def feed(self, data: str) -> None:
        if not data:
            return
        self._stream.feed(data)
        self.ctx.set_message(self._render())

    def _render(self) -> str:
        lines: list[str] = []
        if self._screen.history is not None:
            lines.extend(self._render_line(ln) for ln in self._screen.history.top)
        lines.extend(self._render_line(ln) for ln in self._screen.display)
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def _render_line(self, line: str | dict[int, pyte.screens.Char]) -> str:
        if isinstance(line, str):
            return line.rstrip()
        w = self._screen.columns
        chars = [" "] * w
        for i, cell in line.items():
            if 0 <= i < w:
                chars[i] = getattr(cell, "data", str(cell))
        return "".join(chars).rstrip()


class _Proxy(io.TextIOWrapper):
    """Replacement for sys.stdout/stderr. Routes writes to the active UI terminal (if any)."""

    def __init__(self, original: TextIO | None):
        self._original = original
        if original and isinstance(original, io.TextIOWrapper):
            super().__init__(
                original.buffer,
                encoding=original.encoding,
                errors=original.errors,
                newline=None,
                line_buffering=original.line_buffering,
                write_through=original.write_through,
            )
        else:
            super().__init__(io.BytesIO(), encoding="utf-8")

    def __getattr__(self, name):
        return getattr(self._original, name)

    def write(self, s: str) -> int:
        msg_len = len(s)
        if not s:
            return 0
        t = _terminals.get(threading.get_ident())
        if t is not None:
            t.feed(s)
        if self._original is not None and ((t is not None and t._passthrough) or t is None):
            self._original.write(s)
        return msg_len

    def flush(self) -> None:
        if self._original is not None:
            self._original.flush()

    def isatty(self) -> bool:
        if (t := _terminals.get(threading.get_ident())) is not None and not t._passthrough:
            return True
        try:
            return self._original is not None and self._original.isatty()
        except Exception:
            return False


def _active_size() -> tuple[int, int] | None:
    return _sizes.get(threading.get_ident())


def _get_terminal_size(fd: int = 0, /) -> os.terminal_size:
    s = _active_size()
    return os.terminal_size(s) if s else _orig_get_terminal_size(fd)


def _shutil_terminal_size(fallback: tuple[int, int] = (80, 24)) -> os.terminal_size:
    s = _active_size()
    return os.terminal_size(s) if s else _orig_shutil_get_terminal_size(fallback)


def _termios_tcgetwinsize(fd) -> tuple[int, int]:
    s = _active_size()
    return (s[1], s[0]) if s else _orig_termios_tcgetwinsize(fd)


# Rewrite lower-level calls that query the terminal size using ioctl
def _fcntl_ioctl(fd, request, arg=0, mutate_flag=True):
    s = _active_size()
    if request == getattr(termios, "TIOCGWINSZ", None) and s is not None:
        packed = struct.pack("HHHH", s[1], s[0], 0, 0)
        if isinstance(arg, (bytearray, memoryview)):
            n = min(len(arg), len(packed))
            arg[:n] = packed[:n]
            return 0
        return packed
    return _orig_fcntl_ioctl(fd, request, arg, mutate_flag)


def enable_thread_proxies() -> None:
    """Install stdout/stderr proxies and terminal-size shims."""
    assert not isinstance(sys.stdout, _Proxy), "Proxies are already enabled"
    sys.stdout = _Proxy(_orig_stdout)
    sys.stderr = _Proxy(_orig_stderr)
    sys.__stdout__ = _Proxy(_orig___stdout__)
    sys.__stderr__ = _Proxy(_orig___stderr__)
    os.get_terminal_size = _get_terminal_size  # type: ignore[assignment]
    shutil.get_terminal_size = _shutil_terminal_size  # type: ignore[assignment]
    termios.tcgetwinsize = _termios_tcgetwinsize  # type: ignore[assignment]
    fcntl.ioctl = _fcntl_ioctl  # type: ignore[assignment]


@contextlib.contextmanager
def capture_output(
    op_ctx: TerminalMessageContext,
    columns: int = DEFAULT_COLUMNS,
    lines: int = DEFAULT_LINES,
    history: int = DEFAULT_HISTORY,
    passthrough: bool = True,
):
    """Route stdout/stderr on the current thread to a pyte terminal for *op_ctx*."""
    tid = threading.get_ident()
    terminal = _UiTerminal(op_ctx, columns, lines, history, owner_tid=tid, passthrough=passthrough)
    _sizes[tid] = (columns, lines)
    _terminals[tid] = terminal
    try:
        yield
    finally:
        _terminals.pop(tid, None)
        _sizes.pop(tid, None)
