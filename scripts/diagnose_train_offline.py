"""Offline training diagnostic script.

Runs the real offline training pipeline at tiny scale (B=4, S=12) using an
IdentityBackbone so no Qwen3 download is needed. Each stage prints detailed
diagnostics and asserts invariants. Run with:

    uv run python scripts/diagnose_train_offline.py
"""

from __future__ import annotations

import math
import sys

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Constants matching the notebook
# ──────────────────────────────────────────────────────────────────────────────

DATASET_ID = "mouse-example-dataset"
B = 4          # batch size
S = 12         # sequence length (small for readability)
D = 32         # hidden dim (tiny — no backbone download)
MAX_ACTIONS = 4
MAX_OBS_DISCRETE = 64

MODALITIES = [
    {"field": "action",      "type": "discrete", "vocab_size": MAX_ACTIONS},
    {"field": "observation", "type": "discrete", "vocab_size": MAX_OBS_DISCRETE},
    {"field": "reward",      "type": "rff",      "in_min": 0.01, "in_max": 100.0},
    {"field": "done",        "type": "discrete", "vocab_size": 5},
    {"type": "learnable"},
]

REQUIRED_FIELDS = {"action", "observation", "reward", "done"}
VALID_DONE_CODES = {0, 1, 2, 3, 4}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
HEAD = "\033[1m\033[34m"
RESET = "\033[0m"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 70
    print(f"\n{HEAD}{bar}\n  {title}\n{bar}{RESET}")


def check(label: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    suffix = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if not ok:
        raise AssertionError(f"FAILED: {label}  {detail}")


def warn(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  [{WARN}] {label}{suffix}")


def info(label: str, detail: str) -> None:
    print(f"  [info] {label}: {detail}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def stage1_dataloader() -> list[list[dict]]:
    section("STAGE 1 — DataLoader")
    from mouse_core.data import DataLoader, load_stores_from_hub

    print("  Loading stores from Hub …")
    stores = load_stores_from_hub(DATASET_ID, split="train")
    info("stores loaded", str(len(stores)))

    loader = DataLoader(stores, sequence_length=S, batch_size=B, num_workers=0)
    batch = loader.next_batch()

    # Shape
    check("outer length == B", len(batch) == B, f"got {len(batch)}")
    check("inner length == S", len(batch[0]) == S, f"got {len(batch[0])}")

    # Fields & values in every row
    all_fields_ok = True
    all_done_ok = True
    for b_idx, seq in enumerate(batch):
        for s_idx, row in enumerate(seq):
            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                all_fields_ok = False
                print(f"    batch[{b_idx}][{s_idx}] missing: {missing}")
            done_val = row.get("done")
            if done_val not in VALID_DONE_CODES:
                all_done_ok = False
                print(f"    batch[{b_idx}][{s_idx}] invalid done={done_val!r}")

    check("all rows contain required fields", all_fields_ok)
    check("all done codes in {0,1,2,3,4}", all_done_ok)

    # Print first row for reference
    row0 = batch[0][0]
    print(f"\n  batch[0][0] (first row):")
    for k, v in sorted(row0.items()):
        print(f"    {k:30s} = {v!r}  (type: {type(v).__name__})")

    # Summary of done code distribution
    done_dist: dict[int, int] = {}
    for seq in batch:
        for row in seq:
            d = row.get("done", -1)
            done_dist[d] = done_dist.get(d, 0) + 1
    info("done code distribution", str(dict(sorted(done_dist.items()))))

    loader.close()
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — StepEmbedder
# ──────────────────────────────────────────────────────────────────────────────

def stage2_embedder(batch: list[list[dict]]) -> tuple:
    section("STAGE 2 — StepEmbedder (sum mode, concat_modalities=False)")
    from mouse_core.models.embedding import StepEmbedder

    encoder = StepEmbedder(
        hidden_dim=D,
        modalities=MODALITIES,
        concat_modalities=False,
        include_type_token=False,
        std=0.02,
    )
    encoder.eval()

    with torch.no_grad():
        embeds, col_values, step_token_indices = encoder(batch)

    # ── Shape checks ──────────────────────────────────────────────────────────
    T = encoder.tokens_per_step
    info("tokens_per_step", str(T))
    check("tokens_per_step == 1 (all modalities Tc=1, sum mode)", T == 1)
    check(
        f"embeds.shape == ({B}, {S*T}, {D})",
        embeds.shape == (B, S * T, D),
        f"got {tuple(embeds.shape)}",
    )

    # ── col_values ────────────────────────────────────────────────────────────
    expected_keys = {"action", "observation", "reward", "done"}
    check(
        "col_values keys == {action, observation, reward, done}",
        set(col_values.keys()) == expected_keys,
        f"got {set(col_values.keys())}",
    )
    for key, expected_dtype in [
        ("action", torch.int64),
        ("observation", torch.int64),
        ("done", torch.int64),
    ]:
        t = col_values[key]
        check(f"col_values[{key!r}].shape == ({B},{S})", t.shape == (B, S), f"got {tuple(t.shape)}")
        check(f"col_values[{key!r}].dtype == int64", t.dtype == expected_dtype, f"got {t.dtype}")

    rw = col_values["reward"]
    check(f"col_values['reward'].shape == ({B},{S})", rw.shape == (B, S), f"got {tuple(rw.shape)}")
    check("col_values['reward'] is float", rw.is_floating_point(), f"got {rw.dtype}")

    # ── step_token_indices ────────────────────────────────────────────────────
    expected_sti = torch.arange(S).unsqueeze(0).expand(B, -1) * T + (T - 1)
    check(
        f"step_token_indices.shape == ({B},{S})",
        step_token_indices.shape == (B, S),
        f"got {tuple(step_token_indices.shape)}",
    )
    check(
        "step_token_indices values == [0,1,...,S-1] (stride 1, T=1)",
        torch.equal(step_token_indices, expected_sti),
        f"\n    expected: {expected_sti[0].tolist()}\n    got:      {step_token_indices[0].tolist()}",
    )

    # ── Std balance (content + type, per modality) ───────────────────────────
    print("\n  Embedding std balance:")
    header = f"  {'modality':<15} {'content_std':>12} {'type_std':>10} {'ratio(t/c)':>12} {'flag':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    content_stds: dict[str, float] = {}
    type_stds: dict[str, float] = {}

    device = embeds.device

    for spec in encoder.modalities:
        assert isinstance(spec.field, str)
        field = spec.field
        mod = encoder.modality_embedders[field]
        include_type = encoder._modality_include_type[field]
        type_id = encoder._modality_token_types[field]

        # Content contribution
        with torch.no_grad():
            if field in encoder._learnable_modalities:
                # LearnableEmbedder: read the parameter directly
                content_vec = mod.embed.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
                # shape [B, S, Tc, D] → [B*S*Tc, D]
                c_std = content_vec.reshape(-1, D).float().std().item()
            else:
                raw = mod(col_values[field])   # [B*S, Tc*D]
                c_std = raw.float().reshape(-1).std().item()

        content_stds[field] = c_std

        # Type embedding contribution (if enabled for this modality)
        if include_type:
            with torch.no_grad():
                type_vec = encoder.type_embedder(type_id, (B, S), device)  # [B, S, D]
            t_std = type_vec.float().reshape(-1).std().item()
            type_stds[field] = t_std
            ratio = t_std / (c_std + 1e-9)
            flag = "FLAG" if ratio > 3.0 else ""
            print(
                f"  {field:<15} {c_std:>12.5f} {t_std:>10.5f} {ratio:>12.2f}x {flag:>6}"
            )
        else:
            print(
                f"  {field:<15} {c_std:>12.5f} {'(no type)':>10} {'—':>12} {'':>6}"
            )

    # Check cross-modality content std spread
    all_c = list(content_stds.values())
    if all_c:
        max_c = max(all_c)
        min_c = min(all_c)
        spread = max_c / (min_c + 1e-9)
        print(f"\n  content std spread (max/min): {spread:.2f}x")
        if spread > 3.0:
            warn(
                f"content std spread is {spread:.1f}x — one modality dominates the summed token",
                f"max={max_c:.5f} min={min_c:.5f}",
            )
        else:
            print(f"  [{PASS}] content stds are balanced (spread {spread:.2f}x < 3x)")

    # ── Verify type embedding application matches include_type_token setting ─
    # Reconstruct the expected summed embedding manually and compare to `embeds`.
    # If include_type_token=False, type contributions must be absent; if True,
    # they must be present. Mismatch means the forward ignored the flag.
    type_status = "ON" if encoder.include_type_token else "OFF"
    print(f"\n  Verifying type embeddings (include_type_token={type_status}) in embeds …")
    T = encoder.tokens_per_step
    with torch.no_grad():
        expected = torch.zeros(B, S, T, D, device=device, dtype=embeds.dtype)
        for spec in encoder.modalities:
            assert isinstance(spec.field, str)
            field = spec.field
            Tc = encoder._modality_tokens[field]
            mod = encoder.modality_embedders[field]
            include_type = encoder._modality_include_type[field]
            type_id = encoder._modality_token_types[field]

            if field in encoder._learnable_modalities:
                from mouse_core.models.embedding.embedding import LearnableEmbedder
                assert isinstance(mod, LearnableEmbedder)
                contrib = mod.embed.to(dtype=embeds.dtype).view(1, 1, Tc, D).expand(B, S, Tc, D)
            else:
                flat = mod(col_values[field]).to(dtype=embeds.dtype)  # [B*S, Tc*D]
                contrib = flat.view(B, S, Tc, D)

            if include_type:
                typ = encoder.type_embedder(type_id, (B, S), device).to(dtype=embeds.dtype)  # [B, S, D]
                contrib = contrib + typ.unsqueeze(2)   # broadcast over Tc dim

            if Tc < T:
                pad = torch.zeros(B, S, T - Tc, D, device=device, dtype=embeds.dtype)
                contrib = torch.cat([contrib, pad], dim=2)

            expected.add_(contrib)

        expected_flat = expected.reshape(B, S * T, D)

    match = torch.allclose(expected_flat, embeds, atol=1e-5)
    check(
        f"manually reconstructed embeds (type {type_status}) match encoder output",
        match,
        "(mismatch means include_type_token flag is not being respected, or formula is wrong)",
    )

    if match and encoder.include_type_token:
        # Show the contribution of type embedding to the total variance
        print("\n  Per-modality type embedding share of total token variance:")
        with torch.no_grad():
            total_var = embeds.float().var().item()
            for spec in encoder.modalities:
                assert isinstance(spec.field, str)
                field = spec.field
                if not encoder._modality_include_type[field]:
                    continue
                type_id = encoder._modality_token_types[field]
                typ = encoder.type_embedder(type_id, (B, S), device).float()
                type_var = typ.var().item()
                share = 100.0 * type_var / (total_var + 1e-9)
                flag = "  ← high" if share > 30.0 else ""
                print(f"    {field:<15}  type_var={type_var:.6f}  total_var={total_var:.6f}  share={share:.1f}%{flag}")
    elif match and not encoder.include_type_token:
        print("  [info] type embeddings disabled — token is purely content-driven")

    return encoder, embeds, col_values, step_token_indices


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Value Head
# ──────────────────────────────────────────────────────────────────────────────

def stage3_value_head(
    encoder: object,
    embeds: torch.Tensor,
    step_token_indices: torch.Tensor,
) -> tuple:
    section("STAGE 3 — DiscreteActionValueHead")
    from mouse_core.models.embedding import StepEmbedder
    from mouse_core.models.heads import DiscreteActionValueHead

    assert isinstance(encoder, StepEmbedder)

    head = DiscreteActionValueHead(
        in_features=D,
        out_features=MAX_ACTIONS,
        hidden_dim=D,
        num_layers=1,
        scale=0.1,
    )
    head.eval()

    # pool_step_reprs: gather the step-representative token from embeds
    # (backbone is identity, so embeds == h)
    with torch.no_grad():
        h_step = encoder.pool_step_reprs(embeds, step_token_indices)  # [B, S, D]

    check(
        f"pool_step_reprs output shape == ({B},{S},{D})",
        h_step.shape == (B, S, D),
        f"got {tuple(h_step.shape)}",
    )

    # Verify gather correctness: h_step[b, s] should equal embeds[b, step_token_indices[b,s]]
    gather_ok = True
    for b in range(B):
        for s in range(S):
            idx = step_token_indices[b, s].item()
            if not torch.allclose(h_step[b, s], embeds[b, idx]):
                gather_ok = False
                break
    check("pool_step_reprs gather is correct (h_step[b,s] == embeds[b, idx[b,s]])", gather_ok)

    print(f"\n  h_step[0, 0] (first step repr, first 5 dims): {h_step[0,0,:5].tolist()}")
    print(f"  embeds[0, 0] (same position, first 5 dims):   {embeds[0,0,:5].tolist()}")

    # Online forward
    with torch.no_grad():
        action_value = head.forward(h_step)

    check(
        f"action_value.shape == ({B},{S},{MAX_ACTIONS})",
        action_value.shape == (B, S, MAX_ACTIONS),
        f"got {tuple(action_value.shape)}",
    )

    # Target forward
    with torch.no_grad():
        action_value_target = head.target_forward(h_step)

    check(
        f"action_value_target.shape == ({B},{S},{MAX_ACTIONS})",
        action_value_target.shape == (B, S, MAX_ACTIONS),
        f"got {tuple(action_value_target.shape)}",
    )

    # Confirm target has no grad tracking
    req_grad_params = [p for p in head.target.parameters() if p.requires_grad]
    check("target network params all require_grad=False", len(req_grad_params) == 0,
          f"{len(req_grad_params)} params still have requires_grad=True")

    info("action_value[0,0]  (first step Q-values)", str(action_value[0, 0].tolist()))
    info("action_value_target[0,0]", str(action_value_target[0, 0].tolist()))

    return head, h_step, action_value, action_value_target


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Full Model.forward + Alignment
# ──────────────────────────────────────────────────────────────────────────────

def stage4_model_forward(batch: list[list[dict]]) -> tuple:
    section("STAGE 4 — Model.forward (IdentityBackbone) & Alignment")
    from mouse_core.models import Model
    from mouse_core.models.backbone import IdentityBackbone
    from mouse_core.models.embedding import StepEmbedder
    from mouse_core.models.heads import DiscreteActionValueHead

    encoder = StepEmbedder(
        hidden_dim=D,
        modalities=MODALITIES,
        concat_modalities=False,
        include_type_token=False,
        std=0.02,
    )
    backbone = IdentityBackbone(hidden_dim=D)
    head = DiscreteActionValueHead(
        in_features=D,
        out_features=MAX_ACTIONS,
        hidden_dim=D,
        num_layers=1,
        scale=0.1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=head)
    model.eval()

    with torch.no_grad():
        predictions, objective_data, _ = model(batch)

    # Shape checks
    av = predictions["action_value"]
    avt = predictions["action_value_target"]
    act = objective_data["action"]

    check(
        f"predictions['action_value'].shape == ({B},{S},{MAX_ACTIONS})",
        av.shape == (B, S, MAX_ACTIONS),
        f"got {tuple(av.shape)}",
    )
    check(
        f"predictions['action_value_target'].shape == ({B},{S},{MAX_ACTIONS})",
        avt.shape == (B, S, MAX_ACTIONS),
        f"got {tuple(avt.shape)}",
    )
    check(
        f"objective_data['action'].shape == ({B},{S})",
        act.shape == (B, S),
        f"got {tuple(act.shape)}",
    )
    check(
        f"objective_data['reward'].shape == ({B},{S})",
        objective_data["reward"].shape == (B, S),
    )
    check(
        f"objective_data['done'].shape == ({B},{S})",
        objective_data["done"].shape == (B, S),
    )
    check(
        f"objective_data['observation'].shape == ({B},{S})",
        objective_data["observation"].shape == (B, S),
    )

    # Alignment print: for sequence 0, show first 5 steps with
    # action stored at that position and done code
    print("\n  Alignment: row layout for batch[0] (first 5 steps)")
    print(f"  {'step':>4}  {'action[t]':>10}  {'obs[t]':>8}  {'reward[t]':>10}  {'done[t]':>8}")
    print("  " + "─" * 50)
    for s in range(min(5, S)):
        a = objective_data["action"][0, s].item()
        o = objective_data["observation"][0, s].item()
        r = objective_data["reward"][0, s].item()
        d = objective_data["done"][0, s].item()
        print(f"  {s:>4}  {a:>10}  {o:>8}  {r:>10.3f}  {d:>8}")

    print()
    print("  DQN objective uses action[:,t+1] to index Q[:,t].")
    print("  action[t] is the action that *produced* obs[t], so:")
    print("  Q(obs[t]) is trained against r[t+1] + γ·max Q(obs[t+1]).")
    print("  This +1 shift is intentional — obs and its producing (action,reward,done)")
    print("  are co-located in the same row, not at adjacent positions.")

    return model, predictions, objective_data


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — DqnObjective
# ──────────────────────────────────────────────────────────────────────────────

def stage5_objective(
    model: object,
    predictions: object,
    objective_data: object,
) -> tuple:
    section("STAGE 5 — DqnObjective loss & alignment")
    from tensordict import TensorDict
    from mouse_core.objectives import DqnObjective

    assert isinstance(predictions, TensorDict)
    assert isinstance(objective_data, TensorDict)

    objective = DqnObjective(
        gamma_step=1.0,
        gamma_episode_terminal=0.99,
        gamma_episode_truncated=0.99,
        gamma_task_terminal=0.99,
        gamma_task_truncated=0.99,
        tau=0.0005,
        cql_weight=0.0,
    )

    loss, metrics = objective(objective_data, predictions)

    check("loss is scalar (0-dim)", loss.ndim == 0, f"ndim={loss.ndim}")
    check("loss is finite", torch.isfinite(loss).item(), f"loss={loss.item()}")
    check("loss >= 0", loss.item() >= 0.0, f"loss={loss.item()}")

    expected_metric_keys = {
        "q_values_mean", "q_values_std", "q_values_min",
        "q_values_max", "q_values_target", "action_value",
    }
    missing_metrics = expected_metric_keys - set(metrics.keys())
    check("all expected metric keys present", not missing_metrics, f"missing: {missing_metrics}")

    print("\n  Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"    {k:<30s} = {v:.6f}")

    # Manual alignment verification: gather should not OOB
    q = predictions["action_value"]        # [B, S, A]
    q_target = predictions["action_value_target"]
    action = objective_data["action"].long()

    curr_q = q[:, :-1, :]             # [B, S-1, A]
    next_actions = action[:, 1:]       # [B, S-1]

    # Verify all action indices are in [0, A)
    A = MAX_ACTIONS
    actions_in_range = (next_actions >= 0).all() and (next_actions < A).all()
    check(
        f"next_actions all in [0, {A}) — gather won't OOB",
        actions_in_range.item(),
        f"min={next_actions.min().item()} max={next_actions.max().item()}",
    )

    gathered = curr_q.gather(-1, next_actions.unsqueeze(-1))
    check(
        f"curr_q.gather shape == ({B},{S-1},1)",
        gathered.shape == (B, S - 1, 1),
        f"got {tuple(gathered.shape)}",
    )

    # All S-1 consecutive pairs within the sampled sequence are trained;
    # the done code at t+1 selects the per-transition discount.
    done = objective_data["done"]
    n_total = B * (S - 1)
    done_counts: dict[int, int] = {}
    for v in done[:, 1:].flatten().tolist():
        done_counts[int(v)] = done_counts.get(int(v), 0) + 1
    info(f"transitions trained: {n_total}", f"done code breakdown: {dict(sorted(done_counts.items()))}")

    return objective, loss


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — Optimizer + Polyak update
# ──────────────────────────────────────────────────────────────────────────────

def stage6_optimizer(model: object, batch: list[list[dict]]) -> None:
    section("STAGE 6 — Optimizer step & Polyak update")
    from mouse_core.models import Model
    from mouse_core.objectives import DqnObjective
    from tensordict import TensorDict

    assert isinstance(model, Model)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    objective = DqnObjective(
        gamma_step=1.0,
        gamma_episode_terminal=0.99,
        gamma_episode_truncated=0.99,
        gamma_task_terminal=0.99,
        gamma_task_truncated=0.99,
        tau=0.005,
        cql_weight=0.0,
    )

    head = model._heads["action_value"]

    # Snapshot weights before the step
    online_params_before = [p.clone().detach() for p in head.online.parameters()]
    target_params_before = [p.clone().detach() for p in head.target.parameters()]

    # Forward + backward + step
    predictions, objective_data, _ = model(batch)
    loss, _ = objective(objective_data, predictions)
    info("loss before step", f"{loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()

    # Check that online params have gradients
    params_with_grad = [p for p in head.online.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    check(
        "online head params received non-zero gradients",
        len(params_with_grad) > 0,
        f"{len(params_with_grad)} of {len(list(head.online.parameters()))} params have grad",
    )

    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    info("gradient norm (all params)", f"{grad_norm:.6f}")

    optimizer.step()

    # Check online weights changed
    online_params_after = list(head.online.parameters())
    any_changed = any(
        not torch.equal(b, a.detach())
        for b, a in zip(online_params_before, online_params_after)
    )
    check("optimizer step changed online head weights", any_changed)

    # Target unchanged before polyak
    target_params_mid = [p.clone().detach() for p in head.target.parameters()]
    target_unchanged = all(
        torch.equal(b, m)
        for b, m in zip(target_params_before, target_params_mid)
    )
    check("target weights unchanged before polyak_update", target_unchanged)

    # Polyak update
    model.polyak_update(action_value_tau=objective.tau)

    target_params_after = [p.clone().detach() for p in head.target.parameters()]
    online_params_final = [p.clone().detach() for p in head.online.parameters()]

    # Target must have moved away from its old value …
    target_moved = any(
        not torch.equal(b, a)
        for b, a in zip(target_params_before, target_params_after)
    )
    check("polyak_update changed target weights", target_moved)

    # … but must not equal the online weights (it's an interpolation, tau < 1)
    target_not_equal_online = any(
        not torch.equal(o, t)
        for o, t in zip(online_params_final, target_params_after)
    )
    check("target weights != online weights after polyak (interpolation, not copy)", target_not_equal_online)

    # Verify interpolation formula: θ_t ← τ·θ_o + (1-τ)·θ_t_old
    tau = objective.tau
    interp_ok = True
    for old_t, old_o, new_t in zip(target_params_before, online_params_final, target_params_after):
        expected = tau * old_o + (1 - tau) * old_t
        if not torch.allclose(expected, new_t, atol=1e-6):
            interp_ok = False
            break
    check("polyak satisfies θ_target ← τ·θ_online + (1-τ)·θ_target_old", interp_ok)

    # Target params must still be frozen
    frozen = all(not p.requires_grad for p in head.target.parameters())
    check("target params still require_grad=False after polyak", frozen)

    # Show a sample weight delta
    delta = (target_params_after[0] - target_params_before[0]).abs().mean().item()
    info("mean |Δtarget| for first param tensor", f"{delta:.8f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═'*70}")
    print("  MOUSE offline training diagnostic")
    print(f"  B={B}  S={S}  D={D}  MAX_ACTIONS={MAX_ACTIONS}")
    print(f"{'═'*70}")

    failed_stages: list[str] = []

    def run(name: str, fn, *args):
        try:
            return fn(*args)
        except AssertionError as exc:
            failed_stages.append(name)
            print(f"\n  {FAIL} {exc}")
            return None
        except Exception as exc:
            failed_stages.append(name)
            print(f"\n  {FAIL} Unexpected error in {name}: {exc}")
            import traceback
            traceback.print_exc()
            return None

    batch = run("Stage 1", stage1_dataloader)
    if batch is None:
        print("\nCannot proceed without data. Aborting.")
        sys.exit(1)

    result2 = run("Stage 2", stage2_embedder, batch)
    encoder = embeds = col_values = step_token_indices = None
    if result2 is not None:
        encoder, embeds, col_values, step_token_indices = result2

    if encoder is not None and embeds is not None and step_token_indices is not None:
        run("Stage 3", stage3_value_head, encoder, embeds, step_token_indices)

    result4 = run("Stage 4", stage4_model_forward, batch)
    model = predictions = objective_data = None
    if result4 is not None:
        model, predictions, objective_data = result4

    if model is not None and predictions is not None and objective_data is not None:
        run("Stage 5", stage5_objective, model, predictions, objective_data)

    if model is not None:
        run("Stage 6", stage6_optimizer, model, batch)

    # Final summary
    bar = "═" * 70
    print(f"\n{HEAD}{bar}{RESET}")
    if failed_stages:
        print(f"  {FAIL} {len(failed_stages)} stage(s) failed: {', '.join(failed_stages)}")
    else:
        print(f"  {PASS} All stages passed.")
    print(f"{HEAD}{bar}{RESET}\n")

    if failed_stages:
        sys.exit(1)


if __name__ == "__main__":
    main()
