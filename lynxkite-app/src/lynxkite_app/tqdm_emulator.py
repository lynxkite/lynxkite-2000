"""Provides a wrapper around tqdm to capture progress bar updates and send them to the frontend via
the OpContext's print method."""

import importlib
import typing
import contextlib
import tqdm

if typing.TYPE_CHECKING:
    from lynxkite_core.opcontext import OpContext


def _tqdm_classes() -> list[type]:
    """Returns a list of tqdm classes to patch. This is necessary because some libraries use
    different classes internally, and we want to patch them all."""
    classes: list[type] = []
    seen: set[type] = set()
    for module_name in ("tqdm", "tqdm.std", "tqdm.auto", "tqdm.autonotebook"):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        tqdm_cls = getattr(module, "tqdm", None)
        if tqdm_cls is not None and tqdm_cls not in seen:
            seen.add(tqdm_cls)
            classes.append(tqdm_cls)
    return classes


def capture_tqdm(op_ctx: "OpContext"):
    """
    Creates a global context manager to capture any and all progress bars.

    Usage:
    ```
    with self.tqdm_ctx():
        # any tqdm bars inside will be printed to frontend
    ```
    """

    @contextlib.contextmanager
    def bind_ctx():
        patched_methods: list[tuple[type, str, typing.Callable[..., typing.Any]]] = []
        for tqdm_cls in _tqdm_classes():
            if not hasattr(tqdm_cls, "display"):
                continue
            original_display = getattr(tqdm_cls, "display")

            # Using _orig=original_display forces early binding so each loop iteration captures its own reference
            def patched_display(
                tqdm_self,
                msg=None,
                pos=None,
                _orig: typing.Callable[..., typing.Any] = original_display,
            ):
                _orig(tqdm_self, msg=msg, pos=pos)
                if getattr(tqdm_self, "disable", False):
                    return
                text = msg if msg is not None else str(tqdm_self)
                if text:
                    if hasattr(tqdm_self, "format_dict"):
                        op_ctx.update_telemetry(tqdm_self.format_dict)

            setattr(tqdm_cls, "display", patched_display)
            patched_methods.append((tqdm_cls, "display", original_display))

            original_close = getattr(tqdm_cls, "close", None)
            if original_close:

                def patched_close(tqdm_self, *args, _orig=original_close, **kwargs):
                    if not getattr(tqdm_self, "disable", False):
                        if hasattr(tqdm_self, "format_dict"):
                            op_ctx.update_telemetry(tqdm_self.format_dict)
                    return _orig(tqdm_self, *args, **kwargs)

                setattr(tqdm_cls, "close", patched_close)
                patched_methods.append((tqdm_cls, "close", original_close))

        try:
            yield
        finally:
            for tqdm_cls, attr_name, original_func in reversed(patched_methods):
                setattr(tqdm_cls, attr_name, original_func)

    return bind_ctx()


class ProgressReporter:
    """A wrapper for a single tqdm progress bar that acts as an iterable and context manager."""

    def __init__(self, op_ctx: "OpContext", iterable=None, *args, **kwargs):
        self.op_ctx = op_ctx
        self.pbar = tqdm.tqdm(iterable, *args, **kwargs)
        self._original_display = getattr(self.pbar, "display", None)
        if not self._original_display:
            return

        def patched_display(msg=None, pos=None):
            self._original_display(msg=msg, pos=pos)  # type: ignore[call-non-callable]
            if getattr(self.pbar, "disable", False):
                return
            text = msg if msg is not None else str(self.pbar)
            if text:
                self.op_ctx.update_telemetry(self.pbar.format_dict)

        self.pbar.display = patched_display  # type: ignore[assignment]

        # Trigger initial print
        if hasattr(self.pbar, "n"):
            self.pbar.display()

        # Patch instance's close method to guarantee final update
        self._original_close = getattr(self.pbar, "close", None)
        if not self._original_close:
            return

        def patched_close(*args, **kwargs):
            if not getattr(self.pbar, "disable", False):
                if hasattr(self.pbar, "format_dict"):
                    self.op_ctx.update_telemetry(self.pbar.format_dict)
            return self._original_close(*args, **kwargs)  # type: ignore[misc]

        self.pbar.close = patched_close  # type: ignore[assignment]

    def __iter__(self):
        try:
            yield from self.pbar
        finally:
            self.pbar.close()

    def __enter__(self):
        self.pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        return getattr(self.pbar, name)


def progress_reporter(op_ctx: "OpContext", iterable=None, *args, **kwargs):
    """
    Creates a wrapper for a single tqdm progress bar.

    Usage:
    ```
    for i in self.tqdm(iterable): ...
    with self.tqdm() as pbar: ...
    ```
    """
    return ProgressReporter(op_ctx, iterable, *args, **kwargs)
