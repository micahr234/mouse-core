from importlib.metadata import version

from mouse_core.models import Model, load_model
from mouse_core.models.heads import BaseHead, BaseHeadWithTarget
from mouse_core.objectives import Objective
from mouse_core.task_eval import (
    DEFAULT_EVAL_SEED_OFFSET,
    make_procedural_frozenlake_group,
    run_task_eval,
)

__version__ = version("mouse-core")

__all__ = [
    "__version__",
    "Model",
    "load_model",
    "BaseHead",
    "BaseHeadWithTarget",
    "Objective",
    "DEFAULT_EVAL_SEED_OFFSET",
    "make_procedural_frozenlake_group",
    "run_task_eval",
]
