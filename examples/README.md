# Examples

Runnable scripts for common MOUSE workflows. Run from the repository root after installing:

```bash
source scripts/install.sh
source .venv/bin/activate
```

| Script | Description | Extra dependencies |
|--------|-------------|-------------------|
| `01_collect_dataset.py` | Collect rollouts and optionally push to the Hub | `pip install -e ".[examples]"` |
| `02_train_offline.py` | Train a tiny model on synthetic in-memory data | (core only) |
| `03_inference.py` | Inference loop skeleton | Set `MODEL_ID` env var |
