"""Tests for Datastore append and raw-row retrieval."""

from __future__ import annotations

from mouse_core.data import Datastore


def test_name_is_optional() -> None:
    store = Datastore()
    assert store.name is None
    assert repr(store) == "Datastore(steps=0)"


def test_named_store_repr() -> None:
    store = Datastore(name="cartpole")
    assert store.name == "cartpole"
    assert repr(store) == "Datastore(name='cartpole', steps=0)"


def test_name_is_not_saved_as_a_dataset_column() -> None:
    store = Datastore(name="cartpole")
    store.append({"action": 1, "reward": 0.5, "done": 0})
    assert "name" not in store.to_dataset().column_names


def test_append_and_len() -> None:
    store = Datastore()
    store.append({"action": 1, "reward": 0.5, "done": 0})
    store.append({"action": 2, "reward": 1.0, "done": 0})
    assert len(store) == 2


def test_append_store() -> None:
    left = Datastore(name="left")
    left.append({"action": 1, "reward": 0.0, "done": 0})
    right = Datastore(name="right")
    right.append({"action": 2, "reward": 1.0, "done": 0})

    left.append(right)

    assert len(left) == 2
    rows = left.__getitem__([0, 1])
    assert [r["action"] for r in rows] == [1, 2]


def test_append_store_list() -> None:
    target = Datastore()
    stores = []
    for action in (1, 2):
        store = Datastore()
        store.append({"action": action, "reward": float(action), "done": 0})
        stores.append(store)

    target.append(stores)

    assert len(target) == 2
    rows = target.__getitem__([0, 1])
    assert [r["action"] for r in rows] == [1, 2]


def test_getitem_returns_list_of_dicts() -> None:
    store = Datastore()
    store.append({"action": 3, "reward": -1.0, "done": 1, "time": 4})
    rows = store.__getitem__(0)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["action"] == 3
    assert rows[0]["reward"] == -1.0
    assert rows[0]["done"] == 1
    assert rows[0]["time"] == 4


def test_getitem_slice_preserves_order() -> None:
    store = Datastore()
    for i in range(5):
        store.append({"action": i, "reward": float(i), "done": 0})
    rows = store.__getitem__([2, 4, 0])
    assert [r["action"] for r in rows] == [2, 4, 0]


def test_dataset_roundtrip() -> None:
    """Append rows, export to a HuggingFace Dataset, reload, and re-retrieve."""
    store = Datastore()
    for t in range(3):
        store.append({
            "action": t % 2,
            "reward": float(t),
            "done": 0,
            "time": t,
            "group_id": "CartPole-v1#0",
        })

    ds = store.to_dataset()
    assert len(ds) == 3

    reloaded = Datastore()
    reloaded.from_dataset(ds)
    rows = reloaded.__getitem__([0, 1, 2])
    assert [r["action"] for r in rows] == [0, 1, 0]
    assert rows[2]["reward"] == 2.0


def test_getitem_mixed_source_and_buffer() -> None:
    """Rows from the HF source and from the append buffer interleave correctly."""
    store = Datastore()
    store.append({"action": 10, "reward": 0.0, "done": 0})
    store.append({"action": 20, "reward": 0.0, "done": 0})
    # Flush buffer to source
    ds = store.to_dataset()
    store2 = Datastore()
    store2.from_dataset(ds)
    # Now add more rows to the buffer
    store2.append({"action": 30, "reward": 0.0, "done": 0})

    rows = store2.__getitem__([0, 1, 2])
    assert [r["action"] for r in rows] == [10, 20, 30]
