# Built-in validation and local sampling policies — 2026-11-07

The current loader always validates finite JSON-object payloads. There is no
validation callback. Sampling ranges and eligibility are engineer-owned functions
defined immediately above loader construction in the training notebooks. Policies
sit outside repeated training loops so reconstructed loaders can reuse indexes.
Core does not export episode/step sampling policies. Separate environment-protocol
readers remain available for collection/decoding, but are not loader validators.

The measurement uses the same cached FrozenLake shard (358,795 events, 847 complete
episodes), token budget 10,000, seed 147, prefetch 4, Python 3.15 and CPU-only setup
as the historical eager-index baseline. Three trials per worker count, with 12
cold and warm batches each. Medians:

| Workers | Measurement | Eager baseline | Current implementation |
|---:|---|---:|---:|
| 0 | Loader setup | 19.010706 s | 0.456708 s |
| 0 | Unchanged refresh | 18.982641 s | 0.000048 s |
| 0 | Refresh after append | 19.120242 s | 0.464741 s |
| 0 | First batch ready from construction | 19.323789 s | 0.757103 s |
| 0 | Mean cold batch | 0.319321 s | 0.306589 s |
| 0 | Mean warm batch | 0.320897 s | 0.308900 s |
| 4 | Loader setup | 19.139949 s | 0.612515 s |
| 4 | Unchanged refresh | 19.390817 s | 0.235672 s |
| 4 | Refresh after append | 19.646058 s | 0.695594 s |
| 4 | First batch ready from construction | 23.626761 s | 5.000043 s |
| 4 | Mean cold batch | 0.424480 s | 0.423046 s |
| 4 | Mean warm batch | 0.425280 s | 0.420760 s |

Four-worker batch timings include process startup and prefetch, so use the
zero-worker rows to isolate batch preparation. Warmup validates remaining
candidates outside the timed sampling interval solely for this benchmark;
production does not eagerly visit every candidate. Import timing is separate.
The baseline performed environment-protocol checks during indexing; the current
loader guarantees generic JSON-object structure, with episode eligibility chosen
by the engineer. These policies select identical data on this valid shard.

All episode ranges, 12 cold batches, 12 warm batches, and the batch after refresh
match the baseline exactly across worker counts and trials. Appending an episode
produces the same 848 ranges and retains all 847 unchanged eligibility outcomes.
The status buffer remains one byte per candidate, excluding OS page rounding and
Python/range metadata. No GPU performance claim is made.

Validation: full CPU suite **1,611 passed, 116 skipped**. The final notebook column
slice change passed 13 targeted tests; changed core and test files pass Pyright.
Tests cover malformed/nonfinite/nonobject payloads, validation before rejection,
no-callback validation, incomplete tails, worker parity, and cache invalidation.

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
.venv/bin/python bench/bench_sample_index.py SHARD.parquet --trials 3 --repeats 1 \
--workers 0 4 --batches 12 --output RESULT.json
```

[Current raw measurements](sample_index_builtin_2026-11-07.json) ·
[Historical eager baseline](sample_index_before_2026-11-07.json)
