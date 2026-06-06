"""Smoke tests for public package exports."""

from __future__ import annotations


def test_mouse_root_exports() -> None:
    import mouse_core

    assert hasattr(mouse_core, "load_model")
    assert hasattr(mouse_core, "Model")
    assert hasattr(mouse_core, "LossConfig")


def test_mouse_data_exports() -> None:
    from mouse_core.data import DatasetStore, PrefetchBatchifier, push_to_hub

    assert DatasetStore is not None
    assert PrefetchBatchifier is not None
    assert callable(push_to_hub)
