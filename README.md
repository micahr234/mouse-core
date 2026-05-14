# Meta-Optimization Using Sequential Experiences

<p align="center"><img src="docs/mouse.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for use. APIs will change without notice.

**MOUSE** is a modular PyTorch library for in-context reinforcement learning. It provides the building blocks — embeddings, transformer backbones, output heads, losses, and data utilities — for training and deploying agents that adapt their behaviour by attending over their own transition history, with no weight updates at inference time.

## Install

```bash
pip install mouse-core
```

For the latest development version:

```bash
pip install "git+https://github.com/micahr234/mouse-core.git"
```

## Documentation

📖 **[micahr234.github.io/mouse-core](https://micahr234.github.io/mouse-core/)**

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
