"""Graph analytics environment for LynxKite. The core types and functions are imported here for easy access."""

import os
import pandas as pd

if os.environ.get("NX_CUGRAPH_AUTOCONFIG", "").strip().lower() == "true":
    import cudf.pandas  # ty: ignore[unresolved-import]

    cudf.pandas.install()

pd.options.mode.copy_on_write = True  # Prepare for Pandas 3.0.

from .core import *  # noqa (easier access for core classes)
from . import lynxkite_ops  # noqa (imported to trigger registration)
from . import networkx_ops  # noqa (imported to trigger registration)
from . import pytorch  # noqa (imported to trigger registration)
from . import ml_ops  # noqa (imported to trigger registration)
from . import pykeen_ops  # noqa (imported to trigger registration)
