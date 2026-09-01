from importlib.metadata import version

from mouse_core.models import Model, load_model
from mouse_core.models.heads import BaseHead
from mouse_core.objectives import Objective
from mouse_core.optim import AdamW, AdamWFp32
from mouse_core.polyak import PolyakAverager

__version__ = version("mouse-core")

__all__ = [
    "__version__",
    "Model",
    "load_model",
    "BaseHead",
    "Objective",
    "AdamW",
    "AdamWFp32",
    "PolyakAverager",
]
