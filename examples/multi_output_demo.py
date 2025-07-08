from lynxkite.core.ops import op
import pandas as pd


@op("LynxKite Graph Analytics", "examples", "Multi-output example", outputs=["one", "two"])
def multi_output(*, a_limit=4, b_limit=10):
    """
    Returns two outputs. Also demonstrates Numpy-style docstrings.

    Parameters
    ----------
    a_limit : int
        Number of elements in output "one".
    b_limit : int
        Number of elements in output "two".

    Returns
    -------
    A dict with two DataFrames in it.
    """
    return {
        "one": pd.DataFrame({"a": range(a_limit)}),
        "two": pd.DataFrame({"b": range(b_limit)}),
    }
