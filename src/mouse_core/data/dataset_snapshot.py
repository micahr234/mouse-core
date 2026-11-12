"""Scoped, memory-mapped source snapshots for spawned data workers."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import weakref

from datasets import Dataset, load_from_disk
from datasets.table import ConcatenationTable, MemoryMappedTable


def _memory_mapped(table) -> bool:
    if table is None or isinstance(table, MemoryMappedTable):
        return True
    if isinstance(table, ConcatenationTable):
        return all(_memory_mapped(block) for row in table.blocks for block in row)
    return False


class DatasetSnapshot:
    """Keep an immutable Dataset cheaply serializable for a worker generation.

    Fully memory-mapped data and selection indices borrow their existing files.
    In-memory or mixed storage is written once, preserving selected row order,
    duplicates, sparse columns and dataset features. Only filenames and small
    metadata then travel to spawned workers, rather than the Arrow payload.

    Keep this owner alive until all readers/workers stop. ``close()`` removes
    only files owned by this snapshot; borrowed source files are never removed.
    Returned *batches* own their arrays and can outlive the snapshot. A returned
    Dataset must not be reopened or serialized after its owner is closed.
    ``directory`` selects the parent for the owned temporary directory.
    """

    def __init__(self, dataset: Dataset, *, directory: str | Path | None = None):
        if not isinstance(dataset, Dataset):
            raise TypeError('DatasetSnapshot requires a datasets.Dataset')
        self._source = weakref.ref(dataset)
        self._directory: TemporaryDirectory | None = None
        self._dataset: Dataset | None = None
        self.path: Path | None = None
        self.bytes_written = 0
        if not len(dataset):
            # Empty selections can retain large backing buffers. Reconstruct
            # the empty schema instead of serializing those buffers or asking
            # HF to reopen an on-disk dataset with zero Arrow shards.
            self._dataset = Dataset.from_dict({name: [] for name in dataset.column_names},
                                              features=dataset.features)
            return
        if _memory_mapped(dataset.data) and _memory_mapped(dataset._indices):
            self._dataset = dataset.with_format(None)
            return
        self._directory = TemporaryDirectory(prefix='mouse-core-snapshot-', dir=directory)
        self.path = Path(self._directory.name)
        try:
            dataset.with_format(None).save_to_disk(str(self.path))
            loaded = load_from_disk(str(self.path), keep_in_memory=False)
            if not isinstance(loaded, Dataset):
                raise TypeError('A source snapshot must contain one Dataset')
            self._dataset = loaded
            self.bytes_written = sum(path.stat().st_size for path in self.path.rglob('*') if path.is_file())
        except BaseException:
            self.close()
            raise

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError('DatasetSnapshot is closed')
        return self._dataset

    def matches(self, dataset: Dataset) -> bool:
        """Reuse an unchanged object without trusting caller-supplied fingerprints."""
        return self._dataset is not None and self._source() is dataset

    def close(self) -> None:
        self._dataset = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def __enter__(self) -> DatasetSnapshot:
        if self._dataset is None:
            raise RuntimeError('DatasetSnapshot is closed')
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError('Serialize snapshot.dataset, and keep DatasetSnapshot ownership in the parent')

    def __del__(self):
        if hasattr(self, '_directory'):
            self.close()
