#!/usr/bin/env python3
"""Minimal offline training on synthetic in-memory data (no Hub required)."""

from __future__ import annotations

import torch
from mouse.data import DatasetStore, PrefetchBatchifier
from mouse.losses import DqnLossConfig, dqn_loss
from mouse.models.backbone.none import ModelNone

HIDDEN_DIM = 32
MAX_ACTIONS = 4
NUM_STEPS = 200
TRAIN_STEPS = 50


def _build_toy_store() -> DatasetStore:
    store = DatasetStore(max_action_dim=MAX_ACTIONS)
    for episode in range(NUM_STEPS // 20):
        for step_idx in range(20):
            store.append({
                "action": step_idx % MAX_ACTIONS,
                "reward": float(step_idx % 3),
                "done": 1 if step_idx == 19 else 0,
                "episode_step": step_idx,
            })
    loaded = DatasetStore(max_action_dim=MAX_ACTIONS)
    loaded.from_dataset(store.to_dataset())
    return loaded


def _build_model() -> ModelNone:
    embedding_kwargs = {
        "max_num_actions": MAX_ACTIONS,
        "max_num_obs_discrete": 0,
        "max_num_obs_continuous": 0,
        "max_num_obs_image": 0,
        "max_num_time_steps": 256,
        "include_action_token": True,
        "include_reward_token": True,
        "include_done_token": True,
        "include_obs_discrete": False,
        "include_time_token": True,
        "include_obs_continuous": False,
        "include_obs_image": False,
        "include_type_token": True,
        "num_compute_tokens": 0,
        "token_data_len": 1,
        "concat_modalities": False,
        "std": 0.02,
        "fourier_in_min": 0.01,
        "fourier_in_max": 10.0,
    }
    head_kwargs = {"hidden_dim": HIDDEN_DIM, "num_layers": 1, "scale": 0.01}
    return ModelNone(
        hidden_dim=HIDDEN_DIM,
        backbone_kwargs={},
        embedding_kwargs=embedding_kwargs,
        sp_head_kwargs={"num_layers": 0},
        dqn_head_kwargs=head_kwargs,
        sv_head_kwargs={"num_layers": 0},
        vec_dqn_head_kwargs={"num_layers": 0},
        action_head="dqn",
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model().train().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dqn_cfg = DqnLossConfig(weight=1.0, gamma=0.99, tau=0.005)
    optimizer.register_step_post_hook(
        lambda opt, args, kwargs: model.polyak_update(dqn_tau=dqn_cfg.tau)
    )

    store = _build_toy_store()
    bf = PrefetchBatchifier(
        store,
        sequence_length=8,
        batch_size=4,
        sampling="random",
        prefetch=2,
        num_workers=0,
    )

    for step in range(TRAIN_STEPS):
        step_stream = bf.next_batch().to(device)
        out, _ = model(step_stream)
        loss, metrics = dqn_loss(step_stream, out, dqn_cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            print(f"step={step} loss={metrics['dqn_loss']:.4f}")

    bf.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
