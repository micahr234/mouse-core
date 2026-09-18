from mouse_core.models.heads.base import BaseHead, HeadSpec, prediction_key
from mouse_core.models.heads.classification import ClassificationHead
from mouse_core.models.heads.layerwise_regression import LayerwiseRegressionHead
from mouse_core.models.heads.regression import RegressionHead

__all__ = [
    "BaseHead",
    "HeadSpec",
    "prediction_key",
    "ClassificationHead",
    "LayerwiseRegressionHead",
    "RegressionHead",
]
