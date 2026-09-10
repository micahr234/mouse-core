from importlib.metadata import version

from mouse_core.models import Model, load_model
from mouse_core.models.heads import BaseHead
from mouse_core.objectives import Objective
from mouse_core.optim import AdamW
from mouse_core.polyak import Polyak
from mouse_core.schedule import ExponentialDecay, Piecewise

__version__ = version("mouse-core")

__all__ = [
    "__version__",
    "Model",
    "load_model",
    "BaseHead",
    "Objective",
    "AdamW",
    "Polyak",
    "ExponentialDecay",
    "Piecewise",
]
