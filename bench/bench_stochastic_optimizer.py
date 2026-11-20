"""Compare fused RMSprop/Polyak with the native stochastic reference path.

Run only when the selected device is idle. Compilation is excluded from timings.
Example: python bench/bench_stochastic_optimizer.py --device cuda
"""
import argparse
import statistics
import time

import torch

from mouse_core.rounding import copy_update_, polyak_parameter_
from mouse_core.precision_optim import PrecisionOptimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--tensors', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if args.tensors < 1 or args.iterations < 1:
        parser.error('tensors and iterations must be positive')
    device = torch.device(args.device)
    torch.set_num_threads(1)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    for legacy in (True, False):
        torch.manual_seed(0)
        parameters = [torch.nn.Parameter(torch.randn(64, 1024, device=device, dtype=torch.bfloat16))
                      for _ in range(args.tensors)]
        targets = [p.detach().clone() for p in parameters]
        optimizer = PrecisionOptimizer(parameters, name='rmsprop', options=dict(alpha=.95, eps=1e-8))
        optimizer.param_groups[0]['lr'] = .0001
        if legacy:
            optimizer.update_transform = lambda parameter, update: None
        for p in parameters:
            p.grad = torch.randn_like(p)

        @torch.no_grad()
        def step():
            optimizer.step()
            for source, target in zip(parameters, targets):
                if legacy:
                    copy_update_(target, target.float().lerp(source.float(), .001))
                else:
                    polyak_parameter_(target, source, .001, stochastic_rounding=True)

        for _ in range(3):
            step()
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(args.iterations):
            sync()
            start = time.perf_counter()
            step()
            sync()
            timings.append(1000 * (time.perf_counter() - start))
        label = 'native reference' if legacy else 'optimized'
        print(f'{label}: {statistics.median(timings):.3f} ms/update (optimizer + targets)')
        if device.type == 'cuda':
            print(f'peak allocated: {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB')
        del optimizer, parameters, targets


if __name__ == '__main__':
    main()
