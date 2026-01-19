"""
MGMQ Traffic Signal Control Package.

This package contains:
- environment: Traffic signal control environment using SUMO
- models: MGMQ neural network models (GAT, GraphSAGE, Bi-GRU)
- sim: Simulation interfaces
- preprocessing: GPI/FRAP network standardization
"""

from . import environment
from . import sim
from . import preprocessing

# Models require PyTorch - import conditionally
try:
    from . import models
    __all__ = ["environment", "models", "sim", "preprocessing"]
except ImportError:
    __all__ = ["environment", "sim", "preprocessing"]
    print("Warning: MGMQ models not available. Install PyTorch: pip install torch")

