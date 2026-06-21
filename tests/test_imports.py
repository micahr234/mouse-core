"""Smoke tests for public package exports."""

from __future__ import annotations


def test_mouse_root_exports() -> None:
    import mouse_core

    assert hasattr(mouse_core, "load_model")
    assert hasattr(mouse_core, "Model")
    assert hasattr(mouse_core, "ObjectiveConfig")


def test_mouse_data_exports() -> None:
    from mouse_core.data import (
        ACTION_KEY_DISCRETE,
        DataLoader,
        DatasetStore,
        OBS_KEY_IMAGE,
        push_to_hub,
    )

    assert DatasetStore is not None
    assert DataLoader is not None
    assert callable(push_to_hub)
    assert ACTION_KEY_DISCRETE == "discrete"
    assert OBS_KEY_IMAGE == "image"
