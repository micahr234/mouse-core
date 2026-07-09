from __future__ import annotations

import copy

import numpy as np
import pytest

from mouse_core.data import (
    SequenceAugmentModalitySpec,
    Augmenter,
)


def test_disabled_augmenter_returns_input_batch() -> None:
    batch = [[{"action": 1, "reward": 2.0}]]
    augment = Augmenter(
        [
            {
                "field": "reward",
                "type": "linear",
                "scale_in_low": 0.0,
                "scale_out_low": 0.0,
                "scale_in_high": 1.0,
                "scale_out_high": 2.0,
            }
        ],
        enabled=False,
    )

    assert augment(batch) is batch


def test_linear_scale_endpoints_copy_without_mutating_input() -> None:
    batch = [[{"reward": 2.0, "obs_continuous": [1.0, 2.0]}]]
    original = copy.deepcopy(batch)
    augment = Augmenter(
        [
            {
                "field": "reward",
                "type": "linear",
                "scale_in_low": 0.0,
                "scale_out_low": 1.0,
                "scale_in_high": 1.0,
                "scale_out_high": 3.0,
            },
            {
                "field": "obs_continuous",
                "type": "linear",
                "scale_in_low": 0.0,
                "scale_out_low": -1.0,
                "scale_in_high": 1.0,
                "scale_out_high": 2.0,
            },
        ]
    )

    out = augment(batch)

    assert batch == original
    assert out is not batch
    assert out[0][0] is not batch[0][0]
    assert out[0][0]["reward"] == 5.0
    assert out[0][0]["obs_continuous"] == [2.0, 5.0]


def test_mask_probabilities_apply_to_configured_fields() -> None:
    batch = [
        [
            {
                "action": 3,
                "reward": 1.5,
                "done": 4,
                "observation": 7,
                "pixels": [10, 20],
                "time": 12,
            }
        ]
    ]
    augment = Augmenter(
        [
            {"field": "action", "type": "discrete", "mask_prob": 1.0},
            {"field": "reward", "type": "linear", "mask_prob": 1.0},
            {"field": "done", "type": "discrete", "mask_prob": 1.0},
            {"field": "observation", "type": "discrete", "mask_prob": 1.0},
            {"field": "pixels", "type": "image", "mask_prob": 1.0},
            {"field": "time", "type": "discrete", "mask_prob": 1.0},
        ],
        seed=0,
    )

    out = augment(batch)

    assert out == [[{"action": 0, "reward": 0.0, "done": 0, "observation": 0, "pixels": [0, 0], "time": -1}]]


def test_shared_action_permutation_also_permutates_action_value_targets() -> None:
    batch = [[{"action": 0, "prev_action": 1, "info_q_star": [10.0, 20.0, 30.0]}]]
    expected_perm = np.random.default_rng(0).permutation(3)
    expected_inverse = np.empty_like(expected_perm)
    expected_inverse[expected_perm] = np.arange(len(expected_perm))

    augment = Augmenter(
        [
            {
                "field": ("action", "prev_action", "info_q_star"),
                "type": "discrete",
                "vocab_size": 3,
                "permute": True,
            }
        ],
        seed=0,
    )

    out = augment(batch)

    assert out[0][0]["action"] == int(expected_perm[0])
    assert out[0][0]["prev_action"] == int(expected_perm[1])
    assert out[0][0]["info_q_star"] == np.take([10.0, 20.0, 30.0], expected_inverse).tolist()


def test_multi_field_mask_uses_one_decision_per_step() -> None:
    batch = [[{"action": 3, "done": 4}]]
    augment = Augmenter(
        [{"field": ("action", "done"), "type": "discrete", "mask_prob": 0.5}],
        seed=0,
    )

    # default_rng(0).random() is > 0.5, so the shared mask decision keeps both fields.
    assert augment(batch) == [[{"action": 3, "done": 4}]]


def test_permutation_is_sampled_independently_per_sequence() -> None:
    batch = [[{"action": 0}], [{"action": 0}]]
    rng = np.random.default_rng(0)
    first_perm = rng.permutation(10)
    second_perm = rng.permutation(10)

    augment = Augmenter(
        [{"field": "action", "type": "discrete", "vocab_size": 10, "permute": True}],
        seed=0,
    )

    out = augment(batch)

    assert out[0][0]["action"] == int(first_perm[0])
    assert out[1][0]["action"] == int(second_perm[0])


def test_discrete_and_done_permutations_use_configured_vocab_sizes() -> None:
    batch = [[{"done": 2, "observation": 1}]]
    rng = np.random.default_rng(0)
    done_perm = rng.permutation(5)
    obs_perm = rng.permutation(4)

    augment = Augmenter(
        [
            {"field": "done", "type": "discrete", "vocab_size": 5, "permute": True},
            {"field": "observation", "type": "discrete", "vocab_size": 4, "permute": True},
        ],
        seed=0,
    )

    out = augment(batch)

    assert out[0][0]["done"] == int(done_perm[2])
    assert out[0][0]["observation"] == int(obs_perm[1])


def test_image_scale_shift_clamps_to_pixel_range() -> None:
    batch = [[{"obs_image": [-10, 10, 300]}]]
    augment = Augmenter(
        [
            {
                "field": "obs_image",
                "type": "image",
                "scale_mean": 2.0,
                "shift_mean": 10.0,
            }
        ]
    )

    assert augment(batch)[0][0]["obs_image"] == [0, 30, 255]


def test_invalid_mask_probability_raises() -> None:
    with pytest.raises(ValueError, match="reward"):
        SequenceAugmentModalitySpec(field="reward", type="linear", mask_prob=1.1)


def test_linear_scale_endpoints_must_be_complete() -> None:
    with pytest.raises(ValueError, match="scale_in_low"):
        SequenceAugmentModalitySpec(
            field="reward",
            type="linear",
            scale_in_low=0.0,
            scale_out_low=0.0,
        )


def test_discrete_scale_shift_raises() -> None:
    with pytest.raises(ValueError, match="only apply to type='image'"):
        SequenceAugmentModalitySpec(
            field="observation",
            type="discrete",
            vocab_size=10,
            scale_mean=2.0,
            shift_mean=5.0,
        )


def test_linear_scale_input_endpoints_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        SequenceAugmentModalitySpec(
            field="reward",
            type="linear",
            scale_in_low=1.0,
            scale_out_low=0.0,
            scale_in_high=1.0,
            scale_out_high=2.0,
        )


def test_keep_fields_removes_unlisted_keys() -> None:
    batch = [[{"obs": 0, "action": 1, "reward": 2.0, "done": False}]]
    augment = Augmenter([], keep_fields=["obs", "action"])
    result = augment(batch)
    assert result == [[{"obs": 0, "action": 1}]]


def test_keep_fields_does_not_mutate_original() -> None:
    step = {"obs": 0, "action": 1, "reward": 2.0}
    batch = [[step]]
    augment = Augmenter([], keep_fields=["obs"])
    augment(batch)
    assert "reward" in step


def test_keep_fields_with_augmentation() -> None:
    batch = [[{"obs": 0, "action": 1, "reward": 1.0}]]
    augment = Augmenter(
        [{"field": "reward", "type": "linear",
          "scale_in_low": 0.0, "scale_out_low": 0.0,
          "scale_in_high": 1.0, "scale_out_high": 2.0}],
        keep_fields=["obs", "action"],
    )
    result = augment(batch)
    assert list(result[0][0].keys()) == ["obs", "action"]


def test_keep_fields_propagated_by_fork() -> None:
    augment = Augmenter([], keep_fields=["obs"])
    forked = augment.fork(seed=0)
    assert forked.keep_fields == ("obs",)
