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


def capture_tqdm(op_ctx: "OpContext", iterable=None, *args, **kwargs):
    """Modifies tqdm's display method to also print to the OpContext, allowing tqdm progress bars
    to be captured and sent to the frontend.

    With this we get the unintended side effect that inside a self.tqdm() loop, all tqdm progress
    bars will be captured and sent to the frontend, even if they are in a different library.
    It also allows us to capture progress bars from any library that uses tqdm internally.
    """

    @contextlib.contextmanager
    def _bind():
        patched: list[tuple[type, typing.Callable[..., typing.Any]]] = []
        for tqdm_cls in _tqdm_classes():
            if not hasattr(tqdm_cls, "display"):
                continue
            original_display = getattr(tqdm_cls, "display")

            # Using _orig=original_display forces early binding so each loop iteration captures its own reference
            def patched_display(
                tqdm_self,
                msg: typing.Optional[str] = None,
                pos: typing.Optional[int] = None,
                _orig: typing.Callable[..., typing.Any] = original_display,
            ):
                _orig(tqdm_self, msg=msg, pos=pos)
                if getattr(tqdm_self, "disable", False):
                    return
                text = msg if msg is not None else str(tqdm_self)
                if text:
                    op_ctx.print(text, end="", append=False)

            setattr(tqdm_cls, "display", patched_display)
            patched.append((tqdm_cls, original_display))
        try:
            yield
        finally:
            for tqdm_cls, original_display in reversed(patched):
                setattr(tqdm_cls, "display", original_display)

    if "ncols" not in kwargs:
        desc = kwargs.get("desc", args[0] if len(args) > 0 else "")
        cols = len(str(desc)) + 40
        if op_ctx.node is not None and op_ctx.node.width:
            cols = max(cols, int((op_ctx.node.width - 32) / 8))
        kwargs["ncols"] = cols

    with _bind():
        yield from tqdm.tqdm(iterable, *args, **kwargs)
