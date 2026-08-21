from __future__ import annotations

import copy

import numpy as np
import pytest

from mouse_core.data import Augmenter, Selector, SequenceAugmentFieldSpec, compose
from mouse_core.data.augmenter import _stable_hash


def _rng_for_key(
    *,
    seed: int | None,
    seed_field: str,
    key: object,
    generation: int = 0,
) -> np.random.Generator:
    base = 0 if seed is None else int(seed)
    return np.random.default_rng(
        np.random.SeedSequence(
            [base, generation, _stable_hash(seed_field, key)]
        )
    )


def test_disabled_augmenter_returns_input_step() -> None:
    step = {"action": 1, "reward": 2.0, "task_index": 0}
    augment = Augmenter(
        enabled=False,
        seed_field="task_index",
        fields=[
            {"type": 'linear', "input_field": 'reward', "output_field": 'reward', "scale_in_low": 0.0, "scale_out_low": 0.0, "scale_in_high": 1.0, "scale_out_high": 2.0}
        ],
    )
    assert augment(step) is step


def test_linear_scale_endpoints_copy_without_mutating_input() -> None:
    step = {"reward": 2.0, "obs_continuous": [1.0, 2.0], "task_index": 0}
    original = copy.deepcopy(step)
    augment = Augmenter(
        seed_field="task_index",
        fields=[
            {"type": 'linear', "input_field": 'reward', "output_field": 'reward', "scale_in_low": 0.0, "scale_out_low": 1.0, "scale_in_high": 1.0, "scale_out_high": 3.0},
            {"type": 'linear', "input_field": 'obs_continuous', "output_field": 'obs_continuous', "scale_in_low": 0.0, "scale_out_low": -1.0, "scale_in_high": 1.0, "scale_out_high": 2.0},
        ]
    )
    out = augment(step)
    assert step == original
    assert out is not step
    assert out["reward"] == 5.0
    assert out["obs_continuous"] == [2.0, 5.0]


def test_mask_probabilities_apply_to_configured_fields() -> None:
    step = {
        "action": 3,
        "reward": 1.5,
        "episode_done": 2,
        "observation": 7,
        "pixels": [10, 20],
        "step_index": 12,
        "task_index": 0,
    }
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": 'discrete', "input_field": 'action', "output_field": 'action', "mask_prob": 1.0},
            {"type": 'linear', "input_field": 'reward', "output_field": 'reward', "mask_prob": 1.0},
            {"type": 'discrete', "input_field": 'episode_done', "output_field": 'episode_done', "mask_prob": 1.0},
            {"type": 'discrete', "input_field": 'observation', "output_field": 'observation', "mask_prob": 1.0},
            {"type": 'image', "input_field": 'pixels', "output_field": 'pixels', "mask_prob": 1.0},
            {"type": 'discrete', "input_field": 'step_index', "output_field": 'step_index', "mask_prob": 1.0},
        ],
    )
    out = augment(step)
    assert out == {
        "action": 0,
        "reward": 0.0,
        "episode_done": 0,
        "observation": 0,
        "pixels": [0, 0],
        "step_index": 0,
        "task_index": 0,
    }


def test_shared_action_permutation_also_permutates_action_value_targets() -> None:
    step = {"action": 0, "prev_action": 1, "info_q_star": [10.0, 20.0, 30.0], "task_index": 0}
    expected_perm = _rng_for_key(seed=0, seed_field="task_index", key=0).permutation(3)
    expected_inverse = np.empty_like(expected_perm)
    expected_inverse[expected_perm] = np.arange(len(expected_perm))
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": ("action", "prev_action"),
                "output_field": ("action", "prev_action"),
                "input_vector_field": "info_q_star",
                "output_vector_field": "info_q_star",
                "vocab_size": 3,
                "permute": True,
            }
        ],
    )
    out = augment(step)
    assert out["action"] == int(expected_perm[0])
    assert out["prev_action"] == int(expected_perm[1])
    assert out["info_q_star"] == np.take([10.0, 20.0, 30.0], expected_inverse).tolist()


def test_input_vector_field_is_skipped_when_absent() -> None:
    expected_perm = _rng_for_key(seed=0, seed_field="task_index", key=0).permutation(3)
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "input_vector_field": "info_q_star",
                "output_vector_field": "info_q_star",
                "vocab_size": 3,
                "permute": True,
            }
        ],
    )
    out = augment({"action": 0, "task_index": 0})
    assert out["action"] == int(expected_perm[0])
    assert "info_q_star" not in out


def test_mask_does_not_zero_vector_field() -> None:
    expected_perm = _rng_for_key(seed=0, seed_field="task_index", key=0).permutation(3)
    expected_inverse = np.empty_like(expected_perm)
    expected_inverse[expected_perm] = np.arange(len(expected_perm))
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "input_vector_field": "info_q_star",
                "output_vector_field": "info_q_star",
                "vocab_size": 3,
                "permute": True,
                "mask_prob": 1.0,
            }
        ],
    )
    out = augment({"action": 0, "info_q_star": [10.0, 20.0, 30.0], "task_index": 0})
    assert out["action"] == 0
    assert out["info_q_star"] == np.take([10.0, 20.0, 30.0], expected_inverse).tolist()


def test_vector_rename_writes_output_keeps_input() -> None:
    expected_perm = _rng_for_key(seed=0, seed_field="task_index", key=0).permutation(3)
    expected_inverse = np.empty_like(expected_perm)
    expected_inverse[expected_perm] = np.arange(len(expected_perm))
    q = [10.0, 20.0, 30.0]
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "input_vector_field": "info_q_star",
                "output_vector_field": "q_star_aug",
                "vocab_size": 3,
                "permute": True,
            }
        ],
    )
    out = augment({"action": 0, "info_q_star": q, "task_index": 0})
    assert out["info_q_star"] == q
    assert out["q_star_aug"] == np.take(q, expected_inverse).tolist()
    assert out["action"] == int(expected_perm[0])


def test_vector_on_input_field_raises() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "info_q_star",
                "output_field": "info_q_star",
                "vocab_size": 3,
                "permute": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="input_vector_field"):
        augment({"info_q_star": [10.0, 20.0, 30.0], "task_index": 0})


def test_mask_decisions_vary_per_step_within_one_generation() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[{"type": 'discrete', "input_field": 'action', "output_field": 'action', "mask_prob": 0.5}],
    )
    # Same seed key, same generation: masking must still be a per-step draw.
    outs = [augment({"action": 3, "task_index": 0})["action"] for _ in range(64)]
    assert 0 in outs and 3 in outs


def test_reseed_replaces_draw_cache() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": 'discrete', "input_field": 'action', "output_field": 'action', "vocab_size": 10, "permute": True}
        ],
    )
    for key in range(5):
        augment({"action": 0, "task_index": key})
    assert len(augment._draw_cache) == 5
    augment.reseed()
    assert len(augment._draw_cache) == 0
    augment({"action": 0, "task_index": 0})
    assert len(augment._draw_cache) == 1


def test_multi_field_mask_uses_one_decision_per_step() -> None:
    step = {"action": 3, "episode_done": 2, "task_index": 0}
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[{"type": 'discrete', "input_field": ('action', 'episode_done'), "output_field": ('action', 'episode_done'), "mask_prob": 0.5}],
    )
    out = augment(step)
    # One mask decision for the multi-field modality; either both kept or both zeroed.
    assert (out["action"], out["episode_done"]) in {(3, 2), (0, 0)}


def test_seed_field_shares_draws_across_steps() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": 'discrete', "input_field": 'action', "output_field": 'action', "vocab_size": 10, "permute": True}
        ],
    )
    a = augment({"action": 0, "task_index": 7})
    b = augment({"action": 0, "task_index": 7})
    c = augment({"action": 0, "task_index": 8})
    assert a["action"] == b["action"]
    assert a["action"] == augment({"action": 0, "task_index": 7})["action"]
    # Different seed keys use independent cached draws (may collide by chance).
    assert isinstance(c["action"], int)


def test_reseed_changes_draws_for_same_key() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action", "vocab_size": 10, "permute": True}
        ],
    )
    before = augment({"action": 0, "task_index": 7})
    assert before["action"] == int(
        _rng_for_key(seed=0, seed_field="task_index", key=7, generation=0).permutation(10)[0]
    )
    augment.reseed()
    after = augment({"action": 0, "task_index": 7})
    assert after["action"] == int(
        _rng_for_key(seed=0, seed_field="task_index", key=7, generation=1).permutation(10)[0]
    )
    assert after["action"] == augment({"action": 0, "task_index": 7})["action"]


def test_eval_compose_omits_augmenter() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action", "vocab_size": 10, "permute": True}
        ],
    )
    selector = Selector(
        fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "task_index", "output_field": "task_index"},
        ]
    )
    train_transform = compose(augment, selector)
    eval_transform = compose(selector)
    step = {"action": 0, "task_index": 7}
    assert eval_transform(step) == {"action": 0, "task_index": 7}
    assert train_transform(step) == selector(augment(step))
    assert train_transform is not eval_transform


def test_compose_reseed_forwards_to_augmenter() -> None:
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action", "vocab_size": 10, "permute": True}
        ],
    )
    transform = compose(
        augment,
        Selector(
            fields=[
                {"input_field": "action", "output_field": "action"},
                {"input_field": "task_index", "output_field": "task_index"},
            ]
        ),
    )
    assert augment._generation == 0
    transform.reseed()
    assert augment._generation == 1
    out = transform({"action": 0, "task_index": 7, "extra": 1})
    assert "extra" not in out
    assert out["action"] == int(
        _rng_for_key(seed=0, seed_field="task_index", key=7, generation=1).permutation(10)[0]
    )


def test_missing_seed_field_raises() -> None:
    augment = Augmenter(
        seed_field="task_index",
        fields=[
            {"type": 'discrete', "input_field": 'action', "output_field": 'action', "vocab_size": 10, "permute": True}
        ],
    )
    with pytest.raises(KeyError, match="task_index"):
        augment({"action": 0})


def test_discrete_and_done_permutations_use_configured_vocab_sizes() -> None:
    step = {"episode_done": 2, "observation": 1, "task_index": 0}
    rng = _rng_for_key(seed=0, seed_field="task_index", key=0)
    done_perm = rng.permutation(3)
    obs_perm = rng.permutation(4)
    augment = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {"type": 'discrete', "input_field": 'episode_done', "output_field": 'episode_done', "vocab_size": 3, "permute": True},
            {"type": 'discrete', "input_field": 'observation', "output_field": 'observation', "vocab_size": 4, "permute": True},
        ],
    )
    out = augment(step)
    assert out["episode_done"] == int(done_perm[2])
    assert out["observation"] == int(obs_perm[1])


def test_image_scale_shift_clamps_to_pixel_range() -> None:
    step = {"obs_image": [-10, 10, 300], "task_index": 0}
    augment = Augmenter(
        seed_field="task_index",
        fields=[
            {"type": 'image', "input_field": 'obs_image', "output_field": 'obs_image', "scale_mean": 2.0, "shift_mean": 10.0}
        ]
    )
    assert augment(step)["obs_image"] == [0, 30, 255]


def test_augmenter_rename_writes_output_keeps_input() -> None:
    step = {"reward": 1.0, "task_index": 0}
    augment = Augmenter(
        seed_field="task_index",
        fields=[
            {
                "type": "linear",
                "input_field": "reward",
                "output_field": "reward_aug",
                "scale_in_low": 0.0,
                "scale_out_low": 0.0,
                "scale_in_high": 1.0,
                "scale_out_high": 2.0,
            }
        ]
    )
    out = augment(step)
    assert out["reward"] == 1.0
    assert out["reward_aug"] == 2.0


def test_augmenter_rejects_legacy_field_key() -> None:
    with pytest.raises(TypeError, match="input_field"):
        Augmenter(
            seed_field="task_index",
            fields=[{"field": "reward", "type": "linear", "mask_prob": 0.0}],
        )


def test_invalid_mask_probability_raises() -> None:
    with pytest.raises(ValueError, match="reward"):
        SequenceAugmentFieldSpec(input_field="reward",
            output_field="reward", type="linear", mask_prob=1.1)


def test_linear_scale_endpoints_must_be_complete() -> None:
    with pytest.raises(ValueError, match="scale_in_low"):
        SequenceAugmentFieldSpec(
            input_field="reward",
            output_field="reward", type="linear", scale_in_low=0.0, scale_out_low=0.0
        )


def test_discrete_scale_shift_raises() -> None:
    with pytest.raises(ValueError, match="only apply to type='image'"):
        SequenceAugmentFieldSpec(
            input_field="observation",
            output_field="observation",
            type="discrete",
            vocab_size=10,
            scale_mean=2.0,
            shift_mean=5.0,
        )


def test_linear_scale_input_endpoints_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        SequenceAugmentFieldSpec(
            input_field="reward",
            output_field="reward",
            type="linear",
            scale_in_low=1.0,
            scale_out_low=0.0,
            scale_in_high=1.0,
            scale_out_high=2.0,
        )


def test_input_vector_field_requires_permute() -> None:
    with pytest.raises(ValueError, match="input_vector_field requires permute"):
        SequenceAugmentFieldSpec(
            input_field="action",
            output_field="action",
            type="discrete",
            vocab_size=3,
            input_vector_field="info_q_star",
            output_vector_field="info_q_star",
        )


def test_input_vector_field_must_not_overlap_input_field() -> None:
    with pytest.raises(ValueError, match="also appear on input_field"):
        SequenceAugmentFieldSpec(
            input_field=("action", "info_q_star"),
            output_field=("action", "info_q_star"),
            type="discrete",
            vocab_size=3,
            permute=True,
            input_vector_field="info_q_star",
            output_vector_field="info_q_star",
        )


def test_output_vector_field_defaults_to_input() -> None:
    spec = SequenceAugmentFieldSpec(
        input_field="action",
        type="discrete",
        vocab_size=3,
        permute=True,
        input_vector_field="info_q_star",
    )
    assert spec.output_field == "action"
    assert spec.output_vector_field == "info_q_star"


def test_output_vector_field_requires_input() -> None:
    with pytest.raises(ValueError, match="output_vector_field requires"):
        SequenceAugmentFieldSpec(
            input_field="action",
            type="discrete",
            vocab_size=3,
            permute=True,
            output_vector_field="info_q_star",
        )


def test_vector_field_arity_mismatch() -> None:
    with pytest.raises(ValueError, match="arity mismatch"):
        SequenceAugmentFieldSpec(
            input_field="action",
            output_field="action",
            type="discrete",
            vocab_size=3,
            permute=True,
            input_vector_field=("info_q_star", "info_q_star_2"),
            output_vector_field="info_q_star",
        )
