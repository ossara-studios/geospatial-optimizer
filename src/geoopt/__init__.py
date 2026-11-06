"""Geospatial Optimizer: adaptive exploration drillhole planning."""

import warnings

__version__ = "0.1.0"

# gpytorch still calls torch.jit.script on import; the warning is noise for users.
warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*", category=FutureWarning)
