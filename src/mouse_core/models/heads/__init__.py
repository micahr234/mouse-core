from mouse_core.models.heads.base import BaseHead, prediction_key
from mouse_core.models.heads.classification import ClassificationHead
from mouse_core.models.heads.discretized_action import DiscretizedActionHead
from mouse_core.models.heads.regression import RegressionHead

__all__ = [
    "BaseHead",
    "prediction_key",
    "ClassificationHead",
    "DiscretizedActionHead",
    "RegressionHead",
]
