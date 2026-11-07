"""Measure HF scheduled full-episode paged-cache RL inference on uniform and ragged inputs.

Times include CPU scheduling, direct paged attention and device execution, after warmup.
Environment/tokenizer time is excluded. Warmup samples are separate; new shapes
can still compile later, so inspect raw samples before claiming steady state.
Example: PYTHONPATH=src .venv/bin/python bench/bench_continuous_decode.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
import platform
import statistics
import time

import torch
from transformers.generation import ContinuousBatchingConfig

from mouse_core.models import Backbone


_WORKLOADS = (
    ('uniform', [512] * 4, [1] * 4),
    ('ragged', [2048, 64, 8, 1], [32, 1, 4, 0]),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--max-batch-tokens', type=int, default=2048)
    parser.add_argument('--cache-capacity-tokens', type=int)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if args.hidden % 8 or min(args.hidden, args.layers, args.max_batch_tokens, args.iterations) <= 0:
        parser.error('hidden must be positive and divisible by 8; layers, budget, iterations positive')
    if args.device == 'cuda' and args.hidden < 128:
        parser.error('CUDA FlexAttention requires head_dim >= 16 (hidden >= 128 for 8 heads)')
    page_size = ContinuousBatchingConfig().page_size
    capacities = {name: sum(((n + step * (10 + args.iterations) + page_size - 1) // page_size) * page_size
                            for n, step in zip(lengths, steps, strict=True))
                  for name, lengths, steps in _WORKLOADS}
    if args.cache_capacity_tokens is not None:
        if args.cache_capacity_tokens % page_size or args.cache_capacity_tokens < max(capacities.values()):
            parser.error(f'cache capacity must be a multiple of {page_size} and at least {max(capacities.values())}')
    torch.manual_seed(78)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    model = Backbone(architecture='qwen3', dtype=torch.float32, hidden_dim=args.hidden,
                     num_layers=args.layers, num_heads=8, num_key_value_heads=4).to(args.device).eval()

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def measured(fn):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        return (time.perf_counter() - start) * 1000

    with torch.no_grad():
        for name, lengths, step_lengths in _WORKLOADS:
            inputs = torch.randn(4, max(lengths), args.hidden, device=args.device)
            step_inputs = torch.randn(4, max(step_lengths), args.hidden, device=args.device)
            before_session = torch.cuda.memory_allocated() if args.device == 'cuda' else None
            session = model.decode_session(4, max_batch_tokens=args.max_batch_tokens,
                                           cache_capacity_tokens=args.cache_capacity_tokens or capacities[name])
            session_bytes = torch.cuda.memory_allocated() - before_session if before_session is not None else None
            def prefill():
                session.reset_rows()
                return session.forward(embeds=inputs, lengths=lengths)
            def decode():
                return session.forward(embeds=step_inputs, lengths=step_lengths)
            try:
                first_ms = measured(prefill)
                first_decode_ms = measured(decode)
                warmup_prefill_ms = [measured(prefill) for _ in range(3)]
                prefill_samples = [measured(prefill) for _ in range(args.iterations)]
                prefill_ms = statistics.median(prefill_samples)
                prefill()
                warmup_decode_ms = [measured(decode) for _ in range(10)]
                decode_samples = [measured(decode) for _ in range(args.iterations)]
                decode_ms = statistics.median(decode_samples)
                print(json.dumps(dict(workload=name, backend="hugging_face", device=args.device,
                                      layers=args.layers, hidden=args.hidden, max_batch_tokens=args.max_batch_tokens,
                                      cache_capacity_tokens=session.cache_capacity_tokens,
                                      session_allocated_bytes=session_bytes,
                                      context_lengths=lengths, query_lengths=step_lengths,
                                      python=platform.python_version(), torch=torch.__version__,
                                      transformers=version('transformers'),
                                      gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                                      first_prefill_ms=first_ms, first_decode_ms=first_decode_ms,
                                      warmup_prefill_ms=warmup_prefill_ms, warmup_decode_ms=warmup_decode_ms,
                                      prefill_samples_ms=prefill_samples, decode_samples_ms=decode_samples,
                                      prefill_ms=round(prefill_ms, 3), decode_ms=round(decode_ms, 3))), flush=True)
            finally:
                session.close()


if __name__ == '__main__':
    main()
