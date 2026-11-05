"""Restore a checkpoint's vocabulary readout without loading another transformer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


class VocabularyHead(nn.Module):
    """Pretrained output RMSNorm + vocabulary projection on Mouse's residual stream.

    Backbone deliberately removes the original output norm; restore it only on
    the vocabulary branch. The existing value head keeps its own normalization.
    """
    def __init__(self, *, hidden_dim: int, vocab_size: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.default_end_token_ids: tuple[int, ...] = ()
        self.norm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps, dtype=torch.float32)
        self.projection = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden.float()))

    @classmethod
    def from_pretrained(cls, pretrained: str | Path, *, hub_kwargs: dict[str, Any] | None = None) -> VocabularyHead:
        """Read only norm/readout tensors from a Llama/Qwen3 safetensors checkpoint."""
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.utils.hub import cached_file

        kwargs = hub_kwargs or {}
        config = AutoConfig.from_pretrained(pretrained, **kwargs)
        if config.model_type not in ("qwen3", "llama"):
            raise ValueError("Pretrained vocabulary readout currently supports Mouse's Qwen3 and Llama stacks")
        index = cached_file(pretrained, "model.safetensors.index.json",
                            _raise_exceptions_for_missing_entries=False, **kwargs)
        if index is not None:
            weight_map = json.loads(Path(index).read_text())["weight_map"]
        else:
            path = cached_file(pretrained, "model.safetensors", **kwargs)
            if path is None:
                raise FileNotFoundError("Vocabulary readout requires a safetensors checkpoint")
            with safe_open(path, framework="pt", device="cpu") as handle:
                weight_map = {key: "model.safetensors" for key in handle.keys()}
        choices = [("model.norm.weight", "norm.weight"), ("lm_head.weight",)]
        if config.tie_word_embeddings:
            choices[1] += ("model.embed_tokens.weight", "embed_tokens.weight")
        weights = []
        for candidates in choices:
            key = next((candidate for candidate in candidates if candidate in weight_map), None)
            if key is None:
                raise ValueError(f"Checkpoint lacks vocabulary readout weight: {candidates}")
            path = cached_file(pretrained, weight_map[key], **kwargs)
            assert path is not None
            with safe_open(path, framework="pt", device="cpu") as handle:
                weights.append(handle.get_tensor(key).float())
        head = cls(hidden_dim=config.hidden_size, vocab_size=config.vocab_size, rms_norm_eps=config.rms_norm_eps)
        generation_path = cached_file(pretrained, "generation_config.json",
                                      _raise_exceptions_for_missing_entries=False, **kwargs)
        generation = json.loads(Path(generation_path).read_text()) if generation_path else {}
        end_ids = generation.get("eos_token_id", getattr(config, "eos_token_id", None))
        head.default_end_token_ids = (() if end_ids is None else
                                     (end_ids,) if isinstance(end_ids, int) else tuple(end_ids))
        head.load_state_dict({"norm.weight": weights[0], "projection.weight": weights[1]}, strict=True)
        return head
