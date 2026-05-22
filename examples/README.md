# Examples

Run from the repository root after `source scripts/install.sh` and `source .venv/bin/activate`.

| Script | Description |
|--------|-------------|
| `01_collect_dataset.py` | Gymnasium rollouts → `DatasetStore` (`pip install -e ".[examples]"`) |
| `02_train_offline.py` | Train tiny model on synthetic data (no Hub) |
| `03_inference.py` | Inference skeleton — set `MODEL_ID` |

See [docs/examples.md](../docs/examples.md) for copy-paste training loops.
