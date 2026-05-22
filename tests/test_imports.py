"""Smoke tests for public package exports."""

from __future__ import annotations


def test_mouse_root_exports() -> None:
    import mouse

    assert hasattr(mouse, "load_model")
    assert hasattr(mouse, "Model")
    assert hasattr(mouse, "LossConfig")


def test_mouse_data_exports() -> None:
    from mouse.data import DatasetStore, PrefetchBatchifier, push_to_hub

    assert DatasetStore is not None
    assert PrefetchBatchifier is not None
    assert callable(push_to_hub)
