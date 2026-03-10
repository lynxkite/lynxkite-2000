"""Defines the OpContext class, which is passed to operations when they are executed.
This context can be used to send messages to the frontend, capture stdout/stderr, and more."""

import typing
import contextlib
import asyncio
import io
from dataclasses import dataclass

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


# Overwrite this to configure a terminal emulator for streaming stdout/stderr to the frontend.
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


TERMINAL_EMULATOR: FunctionTerminalEmulator = dummy_terminal_emulator


class FunctionTqdm(typing.Protocol):
    def __call__(
        self,
        op_ctx: "OpContext",
        iterable=None,
        *args,
        **kwargs,
    ) -> typing.Iterator[typing.Any]: ...


def dummy_tqdm_func(
    op_ctx: "OpContext",
    iterable=None,
    *args,
    **kwargs,
) -> typing.Iterator[typing.Any]:
    """
    Default tqdm function that just returns the original iterable. Set TQDM_FUNCTION to a
    function that wraps tqdm to enable this feature.
    """
    if iterable is not None:
        yield from iterable
    return


TQDM_FUNCTION: FunctionTqdm = dummy_tqdm_func

# Common name for the context parameter in operations that need access to the OpContext.
CONTEXT_PARAM_NAME = "self"


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
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def set_message(self, message: str):
        """Sets the message, and sends it to the frontend if possible."""
        self.message = message
        if self.node is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.node.publish_message, self.message)

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
        """A wrapper around tqdm.tqdm that sends tqdm progress bars to the frontend.
        Example usage:
        ```python
        @op("Example op")
        def example_op(self):
            for i in self.tqdm(range(100), "Processing..."):
                # some processing here
        ```
        """
        yield from TQDM_FUNCTION(self, iterable, *args, **kwargs)
