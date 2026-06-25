from mouse_core.models.embedding.embedding import (
    Encoder,
    StepEmbedder,
    ModalitySpec,
    action_modality,
    observation_modalities,
    # Technique-oriented content embedders (selected by ModalitySpec.type + method)
    DiscreteEmbedder,
    ScalarRFFEmbedder,
    VectorRFFEmbedder,
    VectorLinearEmbedder,
    ImageEmbedder,
    LearnableEmbedder,
)
from mouse_core.models.embedding.encoding import RandomFourierFeatures, NormalizedPixel
from mouse_core.models.embedding.linear import ScaledEmbedding, ScaledLinear, PosLinear, ScaledPosLinear

__all__ = [
    "Encoder",
    "StepEmbedder",
    "ModalitySpec",
    "action_modality",
    "observation_modalities",
    # Technique-based embedders (dispatch is by modality type + method, not semantic role)
    "DiscreteEmbedder",
    "ScalarRFFEmbedder",
    "VectorRFFEmbedder",
    "VectorLinearEmbedder",
    "ImageEmbedder",
    "LearnableEmbedder",
    "RandomFourierFeatures",
    "NormalizedPixel",
    "ScaledEmbedding",
    "ScaledLinear",
    "PosLinear",
    "ScaledPosLinear",
]
