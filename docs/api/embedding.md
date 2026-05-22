# Embedding

Step and token embeddings are built in [`mouse.models.embedding`](../../src/models/embedding/).

## Step embedder

| Name | Source |
|------|--------|
| [`StepEmbedder`](../../src/models/embedding/embedding.py) | Converts `TensorDict[B, S]` steps into a token sequence for the backbone |
| [`TokenType`](../../src/models/embedding/embedding.py) | Enum of modality / compute token ids |

## Per-modality embedders

| Name | Source |
|------|--------|
| [`ActionEmbedder`](../../src/models/embedding/embedding.py) | Discrete actions |
| [`RewardEmbedder`](../../src/models/embedding/embedding.py) | Scalar rewards (RFF) |
| [`DoneEmbedder`](../../src/models/embedding/embedding.py) | Done flags |
| [`TimeEmbedder`](../../src/models/embedding/embedding.py) | Episode step index |
| [`ObsContinuousEmbedder`](../../src/models/embedding/embedding.py) | Continuous observations (RFF) |
| [`ObsContinuousLinearEmbedder`](../../src/models/embedding/embedding.py) | Continuous observations (linear) |
| [`ObsDiscreteEmbedder`](../../src/models/embedding/embedding.py) | Discrete state indices |
| [`ObsImageEmbedder`](../../src/models/embedding/embedding.py) | Image pixels |
| [`TypeEmbedder`](../../src/models/embedding/embedding.py) | Shared token-type table |

## Encoding and linear layers

| Name | Source |
|------|--------|
| [`RandomFourierFeatures`](../../src/models/embedding/encoding.py) | Fourier features for continuous scalars |
| [`NormalizedPixel`](../../src/models/embedding/encoding.py) | Normalised pixel projection |
| [`ScaledEmbedding`](../../src/models/embedding/linear.py) | Scaled embedding table |
| [`ScaledLinear`](../../src/models/embedding/linear.py) | Scaled linear layer |
| [`PosLinear`](../../src/models/embedding/linear.py) | Position-specific linear maps |
| [`ScaledPosLinear`](../../src/models/embedding/linear.py) | Scaled position-specific linear maps |
