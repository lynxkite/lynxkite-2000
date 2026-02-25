"""Terminal emulation utilities for streaming stdout/stderr to LynxKite UI."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import struct
import sys
import typing

try:
    import fcntl
except Exception:
    fcntl = None

try:
    import termios
except Exception:
    termios = None


class TerminalMessageContext(typing.Protocol):
    def set_message(self, message: str) -> None: ...


class Data:
    result: str = ""


class UiTerminal:
    """Small VT100 screen emulator for UI updates."""

    def __init__(self, op_ctx: TerminalMessageContext, columns: int = 120, lines: int = 40):
        self.op_ctx = op_ctx
        self._fallback_lines: list[list[str]] = [[]]
        self._fallback_row = 0
        self._fallback_col = 0
        self._has_pyte = False
        try:
            import pyte  # type: ignore

            self._screen = pyte.Screen(columns, lines)
            self._stream = pyte.Stream(self._screen)
            self._has_pyte = True
        except Exception:
            self._screen = None
            self._stream = None

    def _ensure_cursor_in_bounds(self) -> None:
        if self._fallback_row < 0:
            self._fallback_row = 0
        while len(self._fallback_lines) <= self._fallback_row:
            self._fallback_lines.append([])
        if self._fallback_col < 0:
            self._fallback_col = 0

    def _clear_screen_from_cursor(self) -> None:
        self._ensure_cursor_in_bounds()
        line = self._fallback_lines[self._fallback_row]
        if self._fallback_col < len(line):
            del line[self._fallback_col :]
        del self._fallback_lines[self._fallback_row + 1 :]

    def _clear_screen_to_cursor(self) -> None:
        self._ensure_cursor_in_bounds()
        for row in range(self._fallback_row):
            self._fallback_lines[row] = []
        line = self._fallback_lines[self._fallback_row]
        keep_from = min(self._fallback_col + 1, len(line))
        self._fallback_lines[self._fallback_row] = line[keep_from:]

    def _clear_entire_screen(self) -> None:
        self._fallback_lines = [[]]
        self._fallback_row = 0
        self._fallback_col = 0

    def _clear_line_from_cursor(self) -> None:
        self._ensure_cursor_in_bounds()
        line = self._fallback_lines[self._fallback_row]
        if self._fallback_col < len(line):
            del line[self._fallback_col :]

    def _clear_line_to_cursor(self) -> None:
        self._ensure_cursor_in_bounds()
        line = self._fallback_lines[self._fallback_row]
        keep_from = min(self._fallback_col + 1, len(line))
        self._fallback_lines[self._fallback_row] = line[keep_from:]
        self._fallback_col = 0

    def _clear_entire_line(self) -> None:
        self._ensure_cursor_in_bounds()
        self._fallback_lines[self._fallback_row] = []
        self._fallback_col = 0

    def _apply_csi(self, cmd: str, params: str) -> None:
        values = [int(p) if p else 0 for p in params.split(";")] if params else []

        def first_or(default: int) -> int:
            if not values:
                return default
            return values[0] if values[0] != 0 else default

        if cmd == "A":
            self._fallback_row -= first_or(1)
        elif cmd == "B":
            self._fallback_row += first_or(1)
        elif cmd == "C":
            self._fallback_col += first_or(1)
        elif cmd == "D":
            self._fallback_col -= first_or(1)
        elif cmd == "E":
            self._fallback_row += first_or(1)
            self._fallback_col = 0
        elif cmd == "F":
            self._fallback_row -= first_or(1)
            self._fallback_col = 0
        elif cmd == "G":
            self._fallback_col = max(first_or(1) - 1, 0)
        elif cmd in ("H", "f"):
            row = values[0] if len(values) > 0 and values[0] else 1
            col = values[1] if len(values) > 1 and values[1] else 1
            self._fallback_row = max(row - 1, 0)
            self._fallback_col = max(col - 1, 0)
        elif cmd == "J":
            mode = values[0] if values else 0
            if mode == 0:
                self._clear_screen_from_cursor()
            elif mode == 1:
                self._clear_screen_to_cursor()
            elif mode == 2:
                self._clear_entire_screen()
        elif cmd == "K":
            mode = values[0] if values else 0
            if mode == 0:
                self._clear_line_from_cursor()
            elif mode == 1:
                self._clear_line_to_cursor()
            elif mode == 2:
                self._clear_entire_line()

        self._ensure_cursor_in_bounds()

    def _render_fallback(self) -> str:
        return "\n".join("".join(line).rstrip() for line in self._fallback_lines).rstrip()

    def _feed_fallback(self, s: str) -> None:
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == "\x1b":
                if i + 1 < n and s[i + 1] == "[":
                    j = i + 2
                    while j < n and not ("@" <= s[j] <= "~"):
                        j += 1
                    if j < n:
                        self._apply_csi(s[j], s[i + 2 : j])
                        i = j + 1
                        continue
                    break
                i += 1
                continue
            if ch == "\r":
                self._fallback_col = 0
            elif ch == "\n":
                self._fallback_row += 1
                self._fallback_col = 0
                self._ensure_cursor_in_bounds()
            elif ch == "\b":
                self._fallback_col = max(0, self._fallback_col - 1)
            elif ch == "\t":
                self._fallback_col += 8 - (self._fallback_col % 8)
            else:
                self._ensure_cursor_in_bounds()
                line = self._fallback_lines[self._fallback_row]
                if self._fallback_col < len(line):
                    line[self._fallback_col] = ch
                else:
                    if self._fallback_col > len(line):
                        line.extend([" "] * (self._fallback_col - len(line)))
                    line.append(ch)
                self._fallback_col += 1
            i += 1

    def feed(self, s: str) -> None:
        if not s:
            return
        if self._has_pyte:
            self._stream.feed(s)
            text = "\n".join(self._screen.display).rstrip()
            self.op_ctx.set_message(text)
        else:
            self._feed_fallback(s)
            self.op_ctx.set_message(self._render_fallback())


class RealtimeStream(io.TextIOBase):
    def __init__(
        self,
        op_ctx: TerminalMessageContext,
        capturer: io.StringIO,
        original: io.TextIOBase,
        ui_terminal: UiTerminal,
    ):
        self.op_ctx = op_ctx
        self.capturer = capturer
        self.original = original
        self.ui_terminal = ui_terminal

    def write(self, s: str) -> int:
        if not s:
            return 0
        self.capturer.write(s)
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


class StdoutBinder:
    def __init__(self, op_ctx: TerminalMessageContext):
        self.op_ctx = op_ctx
        self._default_cm = None

    def __call__(self, *, columns: int = 120, lines: int = 40):
        @contextlib.contextmanager
        def _stdout():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            old_get_terminal_size = os.get_terminal_size
            old_shutil_get_terminal_size = shutil.get_terminal_size
            old_columns_env = os.environ.get("COLUMNS")
            old_lines_env = os.environ.get("LINES")
            old_fcntl_ioctl = getattr(fcntl, "ioctl", None) if fcntl else None
            old_termios_tcgetwinsize = getattr(termios, "tcgetwinsize", None) if termios else None
            capturer = io.StringIO()
            data = Data()
            ui_terminal = UiTerminal(self.op_ctx, columns=columns, lines=lines)

            def _get_terminal_size(fd=0):
                return os.terminal_size((columns, lines))

            def _shutil_terminal_size(fallback=(columns, lines)):
                return os.terminal_size((columns, lines))

            def _termios_tcgetwinsize(fd):
                return (lines, columns)

            def _fcntl_ioctl(fd, request, arg=0, mutate_flag=True):
                if termios is not None and request == getattr(termios, "TIOCGWINSZ", None):
                    packed = struct.pack("HHHH", lines, columns, 0, 0)
                    if isinstance(arg, (bytearray, memoryview)):
                        n = min(len(arg), len(packed))
                        arg[:n] = packed[:n]
                        return 0
                    return packed
                if old_fcntl_ioctl is None:
                    raise OSError("fcntl.ioctl is unavailable")
                return old_fcntl_ioctl(fd, request, arg, mutate_flag)

            try:
                os.environ["COLUMNS"] = str(columns)
                os.environ["LINES"] = str(lines)
                os.get_terminal_size = _get_terminal_size
                shutil.get_terminal_size = _shutil_terminal_size
                if termios is not None and old_termios_tcgetwinsize is not None:
                    termios.tcgetwinsize = _termios_tcgetwinsize
                if fcntl is not None and old_fcntl_ioctl is not None:
                    fcntl.ioctl = _fcntl_ioctl
                sys.stdout = RealtimeStream(self.op_ctx, capturer, old_stdout, ui_terminal)
                sys.stderr = RealtimeStream(self.op_ctx, capturer, old_stderr, ui_terminal)
                yield data
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                os.get_terminal_size = old_get_terminal_size
                shutil.get_terminal_size = old_shutil_get_terminal_size
                if termios is not None and old_termios_tcgetwinsize is not None:
                    termios.tcgetwinsize = old_termios_tcgetwinsize
                if fcntl is not None and old_fcntl_ioctl is not None:
                    fcntl.ioctl = old_fcntl_ioctl
                if old_columns_env is None:
                    os.environ.pop("COLUMNS", None)
                else:
                    os.environ["COLUMNS"] = old_columns_env
                if old_lines_env is None:
                    os.environ.pop("LINES", None)
                else:
                    os.environ["LINES"] = old_lines_env
                data.result = capturer.getvalue()

        return _stdout()

    def __enter__(self):
        self._default_cm = self()
        return self._default_cm.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._default_cm is not None
        return self._default_cm.__exit__(exc_type, exc_value, traceback)
