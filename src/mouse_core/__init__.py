from importlib.metadata import version

from mouse_core.models import Model, load_model
from mouse_core.models.heads import BaseHead, BaseHeadWithTarget
from mouse_core.objectives import Objective

__version__ = version("mouse-core")

__all__ = [
    "__version__",
    "Model",
    "load_model",
    "BaseHead",
    "BaseHeadWithTarget",
    "Objective",
]
