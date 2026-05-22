# Data

Offline RL data loading, batching, and augmentation.

| Name | Source |
|------|--------|
| [`DatasetStore`](../../src/data/dataset_store.py) | Hugging Face `Dataset`-backed step buffer |
| [`PrefetchBatchifier`](../../src/data/batch.py) | Background batch prefetch → `TensorDict[B, S]` |
| [`TokenAugmenter`](../../src/data/augment.py) | Online token-level augmentation |
| [`AugmentTokensConfig`](../../src/data/augment.py) | Augmentation schedule config |
| [`AugmentScalarSpec`](../../src/data/augment.py) | Scalar augmentation spec |
| [`AugmentMaskProbConfig`](../../src/data/augment.py) | Field mask probabilities |

Hub upload: [`push_to_hub`](../../src/data/hub.py), [`push_stores_to_hub`](../../src/data/hub.py).

Package exports: [`mouse.data`](../../src/data/__init__.py).
