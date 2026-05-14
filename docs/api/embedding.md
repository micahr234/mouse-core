# Embedding

The embedding layer converts a `TensorDict[B, S]` of step records into the flat token sequence `[B, S*T, D]` consumed by the backbone.

## StepEmbedder

::: mouse.models.embedding.embedding.StepEmbedder

---

## TokenType

::: mouse.models.embedding.embedding.TokenType

---

## Per-modality embedders

These are constructed and owned by `StepEmbedder`. They are documented here for reference.

::: mouse.models.embedding.embedding.ActionEmbedder

::: mouse.models.embedding.embedding.RewardEmbedder

::: mouse.models.embedding.embedding.DoneEmbedder

::: mouse.models.embedding.embedding.TimeEmbedder

::: mouse.models.embedding.embedding.ObsContinuousEmbedder

::: mouse.models.embedding.embedding.ObsContinuousLinearEmbedder

::: mouse.models.embedding.embedding.ObsDiscreteEmbedder

::: mouse.models.embedding.embedding.ObsImageEmbedder

::: mouse.models.embedding.embedding.TypeEmbedder

---

## Encoding

::: mouse.models.embedding.encoding.RandomFourierFeatures

::: mouse.models.embedding.encoding.NormalizedPixel

---

## Linear layers

::: mouse.models.embedding.linear.ScaledEmbedding

::: mouse.models.embedding.linear.ScaledLinear

::: mouse.models.embedding.linear.PosLinear

::: mouse.models.embedding.linear.ScaledPosLinear
