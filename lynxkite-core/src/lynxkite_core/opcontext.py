"""Defines the OpContext class, which is passed to operations when they are executed.
This context can be used to send messages to the frontend, capture stdout/stderr, and more."""

import typing
import contextlib
import asyncio
import io
from dataclasses import dataclass, field

if typing.TYPE_CHECKING:
    from .ops import Op
    from . import workspace


class FunctionTerminalEmulator(typing.Protocol):
    def __call__(
        self,
        op_ctx: "OpContext",
        columns: int = 80,
        lines: int = 10,
        history: int = 100,
        passthrough: bool = True,
    ) -> typing.ContextManager: ...


@contextlib.contextmanager
def dummy_terminal_emulator(
    op_ctx: "OpContext",
    columns: int = 80,
    lines: int = 10,
    history: int = 100,
    passthrough: bool = True,
) -> typing.Iterator[None]:
    """
    Default terminal emulator that does nothing. Set TERMINAL_EMULATOR to a function that
    returns a context manager to enable this feature.
    """
    yield


ProgressReporterFactory: typing.TypeAlias = typing.Callable[..., typing.Any]
"""A callable `(op_ctx, iterable=None, *args, **kwargs) -> progress bar`.
Can be a class (like `ProgressReporter`) or a function.
See `lynxkite_app.tqdm_emulator.ProgressReporter` for an example."""


class DummyTqdm:
    def __init__(self, _op_ctx: "OpContext", iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is not None:
            yield from self.iterable

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        pass


# Common name for the context parameter in operations that need access to the OpContext.
CONTEXT_PARAM_NAME = "self"
PROGRESS_REPORTER: ProgressReporterFactory = DummyTqdm
TQDM_CAPTURER: typing.Optional[typing.Callable[["OpContext"], typing.Callable[..., typing.Any]]] = (
    None
)
# Overwrite this to configure a terminal emulator for streaming stdout/stderr to the frontend.
TERMINAL_EMULATOR: FunctionTerminalEmulator = dummy_terminal_emulator

try:
    import tqdm

    def op_ctx_tqdm(_op_ctx: "OpContext", iterable=None, *args, **kwargs):
        return tqdm.tqdm(iterable, *args, **kwargs)

    # If tqdm is available, use it as the default progress reporter. Otherwise, developers can set
    # PROGRESS_REPORTER to a custom implementation.
    PROGRESS_REPORTER = op_ctx_tqdm
except ImportError:
    pass


@dataclass
class OpContext:
    """The context passed to operations when they are executed.

    This context can only be used when the first parameter of the operation function is
    named "self" (by default). The context is currently useful for sending messages to the frontend.

    Example usage:
    ```python
    @op("Example op")
    def example_op(self, input, *, params):
        self.print(f"Received input: {input} and params: {params}")
        result = do_something(input, params)
        self.print(f"Produced result: {result}")
        return result
    ```
    """

    op: "Op | None" = None
    message: str | None = None
    telemetry: dict[str, typing.Any] = field(default_factory=dict)
    node: "workspace.WorkspaceNode | None" = None
    ws: "workspace.Workspace | None" = None
    loop: asyncio.AbstractEventLoop | None = None

    def __init__(
        self,
        op: "Op | None" = None,
        node: "workspace.WorkspaceNode | None" = None,
        ws: "workspace.Workspace | None" = None,
    ):
        self.op = op
        self.node = node
        self.ws = ws
        self.telemetry = {}
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def set_message(self, message: str):
        """Sets the message, and sends it to the frontend if possible."""
        self.message = message
        if self.node is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.node.publish_message, self.message)

    def update_telemetry(self, telemetry: dict[str, typing.Any]):
        """Updates the telemetry data for the current node."""
        self.telemetry.update(telemetry)
        if self.node is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.node.publish_telemetry, telemetry)

    def __getattr__(self, name: str):
        if self.op is not None:
            return getattr(self.op, name)
        raise AttributeError(name)

    def print(
        self,
        *args,
        append: bool = True,
        **kwargs,
    ) -> None:
        """Uses python's print function to send a message to the frontend.

        Args:
            append: If True, the printed message will be appended to the existing message. If False, it will replace the existing message.
        """
        buf = io.StringIO()
        kwargs.pop("file", None)
        print(*args, file=buf, **kwargs)
        message = buf.getvalue()
        if append:
            message = (self.message or "") + message
        self.set_message(message)

    def stdout(
        self, columns: int = 80, lines: int = 10, history: int = 25, passthrough: bool = True
    ) -> typing.ContextManager:
        """A context manager that captures stdout/stderr and sends it to the frontend.
        Example usage:
        ```python
        @op("Example op")
        def example_op(self):
            with self.stdout(columns=60, lines=5):
                print("Starting calculation...")
                for i in tqdm.tqdm(range(4), "Calculating..."):
                    # some calculation here
                print("Done")
        ```

        Args:
            columns: The width of the terminal in characters.
            lines: The number of lines to emulate in the terminal.
            history: The number of lines to keep in the history (for scrolling).
            passthrough: If True, the captured output will also be printed to the original stdout/stderr.
        """
        return TERMINAL_EMULATOR(
            self, columns=columns, lines=lines, history=history, passthrough=passthrough
        )

    def tqdm(self, iterable=None, *args, **kwargs):
        """A wrapper around tqdm.tqdm that sends a tqdm progress bar to the frontend.
        Example usage:
        ```python
        @op("Example op")
        def example_op(self):
            for i in self.tqdm(range(100), "Processing..."):
                # some processing here

            with self.tqdm(total=100) as pbar:
                pbar.update(10)
        ```
        """
        return PROGRESS_REPORTER(self, iterable, *args, **kwargs)

    def tqdm_ctx(self):
        """A wrapper that sends every tqdm progress bar to the frontend inside the captured context.
        Example usage:
        ```python
        @op("Example op")
        def example_op(self):
            with self.tqdm_ctx():
                # any calls to tqdm.tqdm inside this block will be captured and sent to the frontend
        ```
        """
        if TQDM_CAPTURER is not None:
            return TQDM_CAPTURER(self)

        @contextlib.contextmanager
        def dummy_ctx():
            yield

        return dummy_ctx()
