#!/usr/bin/env python3
"""Inference loop skeleton using a Hub model id.

Set MODEL_ID to your model on the Hugging Face Hub, for example:

    export MODEL_ID=your-org/your-mouse-model
    python examples/03_inference.py
"""

from __future__ import annotations

import os
import sys

import torch
from tensordict import TensorDict

from mouse import load_model

MODEL_ID = os.environ.get("MODEL_ID", "")


def main() -> None:
    if not MODEL_ID:
        print("Set MODEL_ID to a MOUSE model on the Hub (e.g. your-org/your-model).")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_ID).eval().to(device)

    # Single-step forward with cache (expand for a full env loop).
    step_stream = TensorDict(
        {
            "action": torch.zeros(1, 1, dtype=torch.long),
            "reward": torch.zeros(1, 1, dtype=torch.float32),
            "done": torch.zeros(1, 1, dtype=torch.long),
            "time": torch.zeros(1, 1, dtype=torch.long),
            "obs_discrete": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1, 1),
    )

    with torch.no_grad():
        out, cache = model(step_stream.to(device), cache=None, use_cache=True)

    action = model.get_action(out, temperature=0.0)
    print(f"action={action.tolist()} cache_keys={list(cache.keys()) if cache else None}")


if __name__ == "__main__":
    main()
