"""
This module provides the "PyTorch model" LynxKite environment. This is a passive
environment: you can build PyTorch models here from neural network layers,
but the workspace can't be executed. Instead, it can be loaded as a model
definition in a "LynxKite Graph Analytics" workspace.
"""

from . import pytorch_core  # noqa
from . import pytorch_ops  # noqa
