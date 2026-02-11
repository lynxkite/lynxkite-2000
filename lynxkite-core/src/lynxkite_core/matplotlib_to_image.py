"""A wrapper for functions that use Matplotlib to instead return an image as a data: URL."""

from __future__ import annotations

import functools
import base64
import io


def matplotlib_to_image(func):
    """Decorator for converting a Matplotlib figure to an image."""
    # Lazy import so the module can be imported even if Matplotlib isn't installed.
    import matplotlib.pyplot as plt
    import matplotlib

    # Make sure we use the non-interactive backend.
    matplotlib.use("agg")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    return wrapper
