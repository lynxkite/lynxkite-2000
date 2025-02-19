"""Some common configuration."""

import os
import pathlib


DATA_PATH = pathlib.Path(os.environ.get("LYNXKITE_DATA", "lynxkite_data"))
CRDT_PATH = pathlib.Path(os.environ.get("LYNXKITE_CRDT_DATA", "lynxkite_crdt_data"))
