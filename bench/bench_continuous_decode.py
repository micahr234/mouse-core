"""Measure HF scheduled circular-cache RL inference on uniform and ragged inputs.

Times include CPU scheduling, KV gathering and device execution, after warmup.
No environment/tokenizer time or compile warmup is included in steady-state times.
Example: PYTHONPATH=src .venv/bin/python bench/bench_continuous_decode.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from mouse_core.models import Backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--max-batch-tokens', type=int, default=2048)
    parser.add_argument('--max-attention-distance-tokens', type=int, default=16383)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if args.hidden % 8 or min(args.layers, args.max_batch_tokens, args.iterations) <= 0:
        parser.error('hidden must be divisible by 8; layers, token budget, iterations must be positive')
    torch.manual_seed(78)
    torch.set_float32_matmul_precision('high')
    model = Backbone(architecture='qwen3', dtype=torch.float32, hidden_dim=args.hidden,
                     num_layers=args.layers, num_heads=8, num_key_value_heads=4,
                     max_attention_distance_tokens=args.max_attention_distance_tokens).to(args.device).eval()

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
        for name, lengths, step_lengths in (
            ('uniform', [512] * 4, [1] * 4),
            ('ragged', [2048, 64, 8, 1], [32, 1, 4, 0]),
        ):
            inputs = torch.randn(4, max(lengths), args.hidden, device=args.device)
            step_inputs = torch.randn(4, max(step_lengths), args.hidden, device=args.device)
            session = model.decode_session(4, max_batch_tokens=args.max_batch_tokens)
            def prefill():
                session.reset_rows()
                return session.forward(embeds=inputs, lengths=lengths)
            def decode():
                return session.forward(embeds=step_inputs, lengths=step_lengths)
            try:
                first_ms = measured(prefill)
                decode()
                for _ in range(3):
                    prefill()
                prefill_ms = statistics.median(measured(prefill) for _ in range(args.iterations))
                prefill()
                for _ in range(10):
                    decode()
                decode_ms = statistics.median(measured(decode) for _ in range(args.iterations))
                print(json.dumps(dict(workload=name, backend="hugging_face", device=args.device,
                                      layers=args.layers, hidden=args.hidden, max_batch_tokens=args.max_batch_tokens,
                                      first_prefill_ms=round(first_ms, 3),
                                      prefill_ms=round(prefill_ms, 3), decode_ms=round(decode_ms, 3))), flush=True)
            finally:
                session.close()


if __name__ == '__main__':
    main()
