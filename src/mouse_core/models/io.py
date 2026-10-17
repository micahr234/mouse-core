"""Online model weights and Hub uploads; architecture is supplied by the caller."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from mouse_core.data.tokenizer import Tokenizer


def save_model(*, model: nn.Module, path: str | Path) -> None:
    """Save online weights and identifying metadata, without training state.

    Accepts an individual module or a ModuleDict of separate encoders and heads.

    The caller owns the model implementation and construction settings.
    Optimizers, delayed models, and Python forward code are not serialized.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "mouse-core-model-v3",
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(model.state_dict(), path / "pytorch_model.bin")


def load_model[T: nn.Module](
    *, model: T, repo_id_or_path: str | Path, force_download: bool = True,
    local_dir: str | Path | None = None, map_location: str | torch.device = "cpu",
    **kwargs: Any,
) -> T:
    """Load weights into the required caller-constructed model and return it.

    Configure architecture, kernels, precision, and device in your implementation
    before loading. Loading preserves that instance's class and trainability.
    No default model or architecture reconstruction is provided. Extra keyword
    arguments (for example ``revision`` or ``token``) go to Hub downloads.
    """
    obsolete = set(kwargs) & {"train_kernel", "decode_kernel", "dtype", "train_autocast_dtype", "decode_autocast_dtype"}
    if obsolete:
        raise TypeError(f"configure {sorted(obsolete)} on the supplied model before loading.")
    local = Path(repo_id_or_path)
    if local.exists():
        config_path, weights_path = local / "config.json", local / "pytorch_model.bin"
    else:
        from huggingface_hub import hf_hub_download

        repo_id = _hub_repo_id_for_user(str(repo_id_or_path), token=kwargs.get("token"))
        download_kwargs = {"force_download": force_download, **kwargs}
        if local_dir is not None:
            download_kwargs["local_dir"] = str(local_dir)
        config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json", **download_kwargs))
        weights_path = Path(hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **download_kwargs))
    config = json.loads(config_path.read_text())
    if config.get("format") != "mouse-core-model-v3":
        raise ValueError("Unsupported model config format; expected mouse-core-model-v3.")
    model.load_state_dict(torch.load(weights_path, map_location=map_location, weights_only=True))
    return model


def push_model_to_hub(
    *,
    model: nn.Module,
    tokenizer: "Tokenizer",
    repo_id: str,
    commit_message: str = "Upload MOUSE model and tokenizer",
    private: bool = False,
    clear: bool = False,
    **kwargs: Any,
) -> str:
    """Upload model weights and tokenizer.json together to one repository."""
    from huggingface_hub import HfApi
    from mouse_core.data.tokenizer import Tokenizer, save_tokenizer

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(f"tokenizer must be a Tokenizer, got {type(tokenizer).__name__}")
    model_url, hub_model_id = _create_hub_repo(
        repo_id=repo_id, private=private, clear=clear, token=kwargs.get("token")
    )
    with tempfile.TemporaryDirectory() as directory:
        save_model(model=model, path=directory)
        save_tokenizer(tokenizer=tokenizer, path=directory)
        _write_model_card(
            model=model, repo_id=hub_model_id, path=Path(directory) / "README.md"
        )
        HfApi().upload_folder(
            repo_id=hub_model_id, folder_path=directory,
            commit_message=commit_message, **kwargs,
        )
    return model_url


def _hub_repo_id_for_user(repo_id: str, token: str | bool | None = None) -> str:
    """Resolve an unscoped Hub repo name under the authenticated user."""
    if "/" in repo_id:
        return repo_id

    from huggingface_hub import HfApi

    user = HfApi().whoami(token=token)["name"]
    return f"{user}/{repo_id}"


def _create_hub_repo(
    *,
    repo_id: str,
    private: bool,
    clear: bool,
    token: str | bool | None,
) -> tuple[str, str]:
    """Create or open a Hub repo, optionally clearing it. Returns ``(url, hub_repo_id)``."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
        token=token,
    )
    hub_repo_id = repo_url.repo_id
    if clear:
        # huggingface_hub never deletes .gitattributes, so exclude it from the
        # list to avoid a spurious "no files modified" warning when it is the
        # only file present (e.g. a freshly created repo).
        existing = [f for f in api.list_repo_files(hub_repo_id) if f != ".gitattributes"]
        if existing:
            api.delete_files(
                repo_id=hub_repo_id,
                delete_patterns=existing,
                commit_message="Clear repository before upload",
            )
    return str(repo_url), hub_repo_id


def _write_model_card(*, model: nn.Module, repo_id: str, path: Path) -> None:
    path.write_text(
        f"---\nlibrary_name: mouse-core\ntags:\n- mouse-core\n- reinforcement-learning\n---\n\n"
        f"# {repo_id}\n\n"
        "This checkpoint contains online model weights only. Optimizer state and delayed models "
        "are not included. The model implementation and its construction settings must be "
        "provided by the application. No default architecture is inferred.\n\n"
        f"## Architecture\n\nClass: `{type(model).__module__}.{type(model).__qualname__}`. "
        f"\n\n```text\n{model}\n```\n\n"
        "## Load weights\n\nConstruct your model, then load into it:\n\n"
        "```python\nfrom mouse_core.models import load_model\n\n"
        f"load_model(model=model, repo_id_or_path={repo_id!r})\nmodel.eval()\n```\n\n"
        f"The tokenizer is stored in this repository (`{repo_id}`). Load it with "
        "`mouse_core.data.load_tokenizer`. Kernels, dtype, and all modules are specified "
        "when constructing your model, before loading weights.\n",
        encoding="utf-8",
    )


def _write_tokenizer_card(*, path: Path, repo_id: str) -> None:
    path.write_text(
        f"""---
library_name: mouse-core
tags:
- mouse-core
- tokenizer
---

# {repo_id}

MOUSE tokenizer packing spec (`tokenizer.json`). Load with:

```python
from mouse_core.data import load_tokenizer

tokenizer = load_tokenizer(repo_id_or_path="{repo_id}")
```
""",
        encoding="utf-8",
    )
