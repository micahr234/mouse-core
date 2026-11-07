# Lazy sample indexing benchmark — 2026-11-07

These are historical measurements from the validation-callback implementation.
The current loader uses automatic structural validation and caller-owned sampling
policies; see [the updated measurements](sample_index_builtin_2026-11-07.md).

One cached FrozenLake Parquet shard: **358,795 events / 847 complete outer episodes**.
Python 3.15, CPU only, token budget 10,000, seed 147, prefetch 4, zero and four
process workers. Three independent trials per worker count. Every trial uses a
fresh store and 12 batches, refreshes, rebuilds the loader, warms validation, and
appends one episode. Medians are reported below. Import time is excluded from
loader timings and is separately recorded in the raw reports.

| Workers | Measurement | Before | Header index + lazy checks |
|---:|---|---:|---:|
| 0 | Initial header/range index | 19.010706 s | 0.598905 s |
| 0 | Unchanged refresh | 18.982641 s | 0.000059 s |
| 0 | Rebuild loader on same store | 19.110669 s | 0.000381 s |
| 0 | Refresh after appending an episode | 19.120242 s | 0.611217 s |
| 0 | Mean cold batch preparation | 0.319321 s | 0.374912 s |
| 0 | Mean warm batch preparation | 0.320897 s | 0.290454 s |
| 4 | Time from construction to first batch | 23.626761 s | 5.204383 s |
| 4 | Refresh call (worker restart included) | 19.390817 s | 0.232520 s |
| 4 | First batch after refresh | 4.424026 s | 4.481426 s |
| 4 | First 12 batches after construction | 5.093759 s | 5.307078 s |

The baseline already includes the Arrow-backed Datastore change. Its episode
selector scans payloads eagerly. The new path indexes header columns and checks
payloads when candidates are first sampled. It also validates full action/result
consistency, so the cold callback does more checking than the old range selector.
Warm measurements explicitly validate all remaining episodes outside the timed
sampling interval; this warmup is a benchmark device, not production behavior.
Four-worker batch times include worker startup and bounded prefetch; they are not
pure per-batch compute times. Compare the zero-worker rows to isolate preparation.

All dataset content, 847 candidate ranges, 12 cold batches, 12 warm batches, and
the next batch after refresh match exactly across revisions, worker counts, and
trials. Appending one complete episode produces the same 848 ranges and retains
all 847 previous validation results. Synthetic tests cover skipped/incomplete
candidates, malformed ineligible data, all-rejected stores, sparse headers, step
windows, worker determinism, cache invalidation and shared-memory lifetime.

Unchanged indexes and check results persist with the store, including across
worker restarts and loader reconstruction. Appends still rescan the changed
store's header columns; they do not rescan unchanged offline stores. A changed
tail is rechecked. The shared check-status array occupies 847 logical bytes for
this shard, plus OS page rounding; range arrays and Python metadata are additional.
No GPU speed or total-process memory claim is made.

Run against the pre-change source snapshot without `--lazy`, and the updated source with it.
The baseline used `/tmp/mouse-core-before-lazy-index-src`; source hashes are in the raw report:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
.venv/bin/python bench/bench_sample_index.py SHARD.parquet --trials 3 --repeats 1 \
--workers 0 4 --batches 12 --lazy --output RESULT.json
```

The raw reports include the shard checksum, source hashes, individual timings,
and exact batch hashes:
[before](sample_index_before_2026-11-07.json),
[after](sample_index_after_2026-11-07.json),
[comparison](sample_index_comparison_2026-11-07.json).

Validation: the full CPU suite passed with 1,594 passed and 116 skipped. After the
final cache-key and atomic-ingestion fixes, 108 focused storage/loader tests passed.
Changed implementation, tests, and benchmark files passed Pyright with no errors.
