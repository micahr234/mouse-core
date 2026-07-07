#!/usr/bin/env python
"""Summarize experiment result JSONs into a comparison table.

Usage:
    .venv/bin/python experiments/summarize.py [name ...]

With no arguments, summarizes every JSON in experiments/results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x, width=7):
    if x is None:
        return " " * (width - 2) + "--"
    return f"{x:{width}.3f}"


def summarize(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    cfg = data["config"]
    lines = [
        f"== {cfg['name']} ==  rotate={cfg['rotate_envs']} envs={cfg['num_envs']}"
        f" expl={cfg['exploration_mode']}(floor={cfg['epsilon_floor']})"
        f" gamma_step={cfg['gamma_step']} gamma_ep={cfg['gamma_episode']}"
        f" penalty={cfg['step_penalty']} seq={cfg['sequence_length']}"
        f" bs={cfg['batch_size']} lr={cfg['lr']} grad={cfg['gradient_steps']}"
    ]
    timing = data.get("timing") or {}
    if timing:
        lines.append(
            f"   time: total={timing['total_s']:.0f}s"
            f" (rollout={timing['rollout_s']:.0f} train={timing['train_s']:.0f}"
            f" eval={timing['eval_s']:.0f})"
        )
    cycles = data.get("cycles", [])
    if cycles:
        last = cycles[-1]
        lines.append(
            f"   train rollout (last cycle): mean_reward={last['mean_reward']}"
            f" distinct_envs={last['distinct_envs_visited']}"
        )
    header = (
        "   grad_steps  where      policy temp | success 1st-half 2nd-half"
        "  ep_len  bump  stuck"
    )
    lines.append(header)
    for ev in data.get("evals", []):
        where = "holdout" if ev["holdout"] else "train"
        lines.append(
            f"   {ev.get('grad_steps', 0):>10}  {where:<9}  {ev['policy']:<6}"
            f" {ev['temperature']:<4} |"
            f" {fmt(ev['success_rate'])} {fmt(ev['first_half_success'], 8)}"
            f" {fmt(ev['second_half_success'], 8)}"
            f" {fmt(ev['mean_episode_length'])}"
            f" {fmt(ev['bump_frac'], 5)} {fmt(ev['stuck_frac'], 6)}"
        )
    return lines


def main() -> None:
    results = Path(__file__).parent / "results"
    names = sys.argv[1:]
    paths = (
        [results / f"{n}.json" for n in names]
        if names
        else sorted(results.glob("*.json"))
    )
    for path in paths:
        if not path.exists():
            print(f"== {path.stem} ==  (missing)")
            continue
        print("\n".join(summarize(path)))
        print()


if __name__ == "__main__":
    main()
