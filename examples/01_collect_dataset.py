#!/usr/bin/env python3
"""Collect rollouts from a Gymnasium environment into a DatasetStore.

Requires the examples extra:

    pip install -e ".[examples]"

Optionally push to the Hub by setting REPO_ID (e.g. your-org/my-mouse-dataset).
"""

from __future__ import annotations

import os

try:
    import gymnasium as gym
except ImportError as e:
    raise SystemExit(
        "gymnasium is required for this example. Install with:\n"
        '  pip install -e ".[examples]"'
    ) from e

from mouse.data import DatasetStore, push_stores_to_hub

ENV_ID = os.environ.get("ENV_ID", "FrozenLake-v1")
REPO_ID = os.environ.get("REPO_ID", "")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "50"))


def main() -> None:
    env = gym.make(ENV_ID, is_slippery=True)
    store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)

    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        action = 0
        reward = 0.0
        done_flag = 0

        for step_idx in range(200):
            store.append({
                "observation_discrete": [obs],
                "action": action,
                "reward": reward,
                "done": done_flag,
                "episode_step": step_idx,
            })

            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done_flag = 1 if terminated else (2 if truncated else 0)

            if terminated or truncated:
                store.append({
                    "observation_discrete": [obs],
                    "action": action,
                    "reward": reward,
                    "done": done_flag,
                    "episode_step": step_idx + 1,
                })
                break

    print(store)

    if REPO_ID:
        push_stores_to_hub([store], repo_id=REPO_ID, split="train", private=True)
        print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")
    else:
        print("Set REPO_ID to upload this dataset to the Hugging Face Hub.")


if __name__ == "__main__":
    main()
