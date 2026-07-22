"""TextEmbedder — format/tokenize in prepare → typed embed_tokens on TokenBatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from string import Formatter
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from mouse_core.models.embedding.embedding import (
    Encoder,
    _field_names,
    _unwrap_scalar,
    _values_equal,
)
from mouse_core.models.embedding.token_batch import TokenBatch, empty_token_batch

# Single embedding-table type for all text / token-modality / image-token ids.
_TYPE_TEXT = 0


@dataclass
class TextModalitySpec:
    """Modality for :class:`TextEmbedder`."""

    type: str
    field: str | Sequence[str] | None = None
    format: str | None = None
    skip: Any = None
    required: bool = True

    _VALID_TYPES: ClassVar[tuple[str, ...]] = ("text", "token", "image")

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown TextEmbedder modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES} "
                "(learnable scratch tokens are NumericEmbedder-only)"
            )
        object.__setattr__(self, "type", k)
        if k == "text":
            if not self.format:
                raise ValueError(f"text modality {self.field!r} requires format=")
        elif k == "token" and self.format is not None:
            raise ValueError(
                f"token modality {self.field!r} must not set format= "
                "(the integer value selects embed_tokens[id] directly)"
            )


def _expand_text_modality_spec(spec: TextModalitySpec) -> list[TextModalitySpec]:
    fields = _field_names(spec.field)
    if not fields:
        raise ValueError("text/token/image modalities must set a field")
    return [replace(spec, field=field) for field in fields]


def _infer_col_dtype(raw_values: list[Any]) -> torch.dtype:
    for value in raw_values:
        if value is None:
            continue
        v = _unwrap_scalar(value)
        if isinstance(v, (float, np.floating)):
            return torch.float32
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                return torch.float32
            if v.dtype == torch.uint8:
                return torch.uint8
            return torch.int64
        if isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                return torch.float32
            if v.dtype == np.uint8:
                return torch.uint8
            return torch.int64
        if isinstance(v, (list, tuple)) and v:
            sample = _unwrap_scalar(v[0])
            if isinstance(sample, (float, np.floating)):
                return torch.float32
    return torch.int64


class TextEmbedder(Encoder):
    """Pretrained token embeddings over a flat :class:`TokenBatch`.

    :meth:`prepare` formats and tokenizes on CPU; :meth:`forward` only runs
    ``embed_tokens`` (and optional vision) on the GPU.
    """

    def __init__(
        self,
        hidden_dim: int,
        modalities: list[dict | TextModalitySpec] | None = None,
        *,
        format: str | None = None,
        pretrained: str | Path | None = None,
        tokenizer=None,
        embed_tokens: nn.Embedding | None = None,
        image_processor=None,
        hub_kwargs: dict | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self._hub_kwargs = dict(hub_kwargs or {})
        self.format = format

        raw = modalities or []
        self.modalities: list[TextModalitySpec] = []
        for m in raw:
            spec = m if isinstance(m, TextModalitySpec) else TextModalitySpec(**m)
            self.modalities.extend(_expand_text_modality_spec(spec))

        has_text = any(s.type == "text" for s in self.modalities)
        has_token = any(s.type == "token" for s in self.modalities)
        has_image = any(s.type == "image" for s in self.modalities)
        needs_format = has_text or has_image or has_token
        if needs_format and format is None:
            raise TypeError(
                "TextEmbedder requires format= when text, token, or image modalities "
                "are declared"
            )
        if format is not None and not (has_text or has_token or has_image):
            raise TypeError("format= requires at least one text, token, or image modality")

        self._text_by_field = {
            s.field: s for s in self.modalities if s.type == "text" and isinstance(s.field, str)
        }
        self._token_by_field = {
            s.field: s for s in self.modalities if s.type == "token" and isinstance(s.field, str)
        }
        self._image_by_field = {
            s.field: s for s in self.modalities if s.type == "image" and isinstance(s.field, str)
        }

        if format is not None:
            for _, name, _, _ in Formatter().parse(format):
                if name is None or name == "":
                    continue
                if name not in self._text_by_field and name not in self._token_by_field and name not in self._image_by_field:
                    raise ValueError(
                        f"format placeholder {{{name}}} has no matching text/token/image modality"
                    )

        if embed_tokens is not None:
            if embed_tokens.embedding_dim != hidden_dim:
                raise ValueError(
                    f"embed_tokens dim {embed_tokens.embedding_dim} != hidden_dim {hidden_dim}"
                )
            self.embed_tokens = embed_tokens
        elif pretrained is not None:
            self.embed_tokens = self._load_embed_tokens(pretrained, hidden_dim)
        else:
            raise TypeError("TextEmbedder requires pretrained= or embed_tokens=")

        needs_tokenizer = format is not None and has_text
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif pretrained is not None and needs_tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, **self._hub_kwargs)
        elif needs_tokenizer:
            raise TypeError("TextEmbedder with format= requires tokenizer= or pretrained=")
        else:
            self.tokenizer = None

        self.image_processor = image_processor
        self._vision: nn.Module | None = None
        self._pretrained = pretrained
        if has_image:
            if image_processor is not None and callable(image_processor) and not hasattr(
                image_processor, "image_processor"
            ):
                # Custom callable that returns token ids or [n,D] — prefer ids for TokenBatch.
                pass
            elif pretrained is not None:
                self._vision, self.image_processor = self._load_vision(pretrained)
            else:
                raise TypeError(
                    "TextEmbedder with image modalities requires image_processor= or "
                    "a vision-capable pretrained="
                )

        if freeze_embeddings:
            self.embed_tokens.weight.requires_grad_(False)
            if self._vision is not None:
                for p in self._vision.parameters():
                    p.requires_grad_(False)

    def _load_embed_tokens(self, pretrained: str | Path, hidden_dim: int) -> nn.Embedding:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(pretrained, **self._hub_kwargs)
        src = model.get_input_embeddings()
        if src.embedding_dim != hidden_dim:
            raise ValueError(
                f"pretrained embedding dim {src.embedding_dim} != hidden_dim {hidden_dim}"
            )
        emb = nn.Embedding(src.num_embeddings, hidden_dim, padding_idx=getattr(src, "padding_idx", None))
        with torch.no_grad():
            emb.weight.copy_(src.weight)
        del model
        return emb

    def _load_vision(self, pretrained: str | Path) -> tuple[nn.Module, Any]:
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError("transformers AutoProcessor required for image modalities") from exc

        processor = AutoProcessor.from_pretrained(pretrained, **self._hub_kwargs)
        if not hasattr(processor, "image_processor") and type(processor).__name__ == "PreTrainedTokenizerFast":
            raise ValueError(
                f"pretrained {pretrained!r} has no image processor; "
                "cannot declare type='image' modalities"
            )
        from transformers import AutoModel

        model = AutoModel.from_pretrained(pretrained, **self._hub_kwargs)
        vision = getattr(model, "vision_model", None) or getattr(model, "vision_tower", None)
        if vision is None:
            raise ValueError(
                f"pretrained {pretrained!r} exposes no vision_model/vision_tower; "
                "cannot declare type='image' modalities"
            )
        projector = getattr(model, "multi_modal_projector", None) or getattr(
            model, "mm_projector", None
        )
        wrapper = _VisionEmbedder(vision, projector, self._hidden_dim)
        wrapper.load_from(model)
        del model
        return wrapper, processor

        self._pretrained = str(pretrained) if pretrained is not None else None

    @property
    def pretrained(self) -> str | Path | None:
        return self._pretrained

    @pretrained.setter
    def pretrained(self, value: str | Path | None) -> None:
        self._pretrained = str(value) if value is not None else None

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        return 0

    def make_preparer(self) -> Callable[..., TokenBatch]:
        format_str = self.format
        text_by_field = dict(self._text_by_field)
        token_by_field = dict(self._token_by_field)
        image_by_field = dict(self._image_by_field)
        modalities = list(self.modalities)
        tokenizer = self.tokenizer
        image_processor = self.image_processor
        # Vision path in prepare: only support tokenizer-like callables that return ids.
        # Full vision tower runs in forward when prepare stores a sentinel — for now
        # image prepare requires a callable returning Sequence[int] token ids.
        has_vision_module = self._vision is not None

        def _prepare(
            batch: list[list[dict]],
            segment_ids: list[list[int]] | None = None,
        ) -> TokenBatch:
            return _prepare_text(
                batch,
                segment_ids,
                format_str=format_str,
                text_by_field=text_by_field,
                token_by_field=token_by_field,
                image_by_field=image_by_field,
                modalities=modalities,
                tokenizer=tokenizer,
                image_processor=image_processor,
                has_vision_module=has_vision_module,
            )

        return _prepare

    def prepare(
        self,
        batch: list[list[dict]],
        segment_ids: list[list[int]] | None = None,
    ) -> TokenBatch:
        return self.make_preparer()(batch, segment_ids)

    def forward(
        self, token_batch: TokenBatch | list[list[dict]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if not isinstance(token_batch, TokenBatch):
            token_batch = self.prepare(token_batch)
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        t = token_batch.to_tensors(device)
        ids = t["token_ids"]
        types = t["token_types"]
        L = ids.shape[0]
        D = self._hidden_dim
        embeds = torch.zeros(L, D, device=device, dtype=dtype)
        if L > 0:
            text_mask = types == _TYPE_TEXT
            embeds[text_mask] = self.embed_tokens(ids[text_mask]).to(dtype=dtype)
            vision_mask = types == 1
            embeds[vision_mask] = self.embed_tokens(ids[vision_mask]).to(dtype=dtype)

        col_values = t["col_values"]
        sti = t["step_token_indices"].view(token_batch.B, token_batch.S)
        return embeds, col_values, sti

    def pool_step_reprs(self, h: torch.Tensor, step_token_indices: torch.Tensor) -> torch.Tensor:
        D = self._hidden_dim
        if h.ndim == 2:
            idx = step_token_indices.reshape(-1)
            return h[idx].view(*step_token_indices.shape, D)
        B, S = step_token_indices.shape
        idx = step_token_indices.unsqueeze(-1).expand(B, S, D)
        return h.gather(1, idx)

    def _coerce_col_value(self, value: Any, dt: torch.dtype) -> Any:
        if value is None:
            if dt in (torch.int64, torch.int32, torch.uint8):
                return 0
            return 0.0
        v = _unwrap_scalar(value)
        if dt in (torch.float32, torch.float64):
            if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
                return np.asarray(
                    v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                    dtype=np.float32,
                )
            return float(v)
        if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
            np_dtype = np.uint8 if dt == torch.uint8 else np.int64
            return np.asarray(
                v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                dtype=np_dtype,
            )
        return int(v)


def _field_text_value(spec: TextModalitySpec, row: dict[str, Any]) -> str | None:
    assert isinstance(spec.field, str)
    assert spec.format is not None
    value = row.get(spec.field)
    if value is None:
        if spec.required:
            raise KeyError(f"Required modality {spec.field!r} is missing")
        return None
    if spec.skip is not None and _values_equal(value, spec.skip):
        return None
    return spec.format.format_map({spec.field: _unwrap_scalar(value)})


def _tokenize_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).tolist()
    elif ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    out: list[int] = []
    for i in ids:
        if isinstance(i, torch.Tensor):
            out.append(int(i.item()))
        else:
            out.append(int(i))
    return out


def _prepare_text(
    batch: list[list[dict]],
    segment_ids: list[list[int]] | None,
    *,
    format_str: str | None,
    text_by_field: dict[str, TextModalitySpec],
    token_by_field: dict[str, TextModalitySpec],
    image_by_field: dict[str, TextModalitySpec],
    modalities: list[TextModalitySpec],
    tokenizer: Any,
    image_processor: Any,
    has_vision_module: bool,
) -> TokenBatch:
    B = len(batch)
    S = len(batch[0]) if B > 0 else 0
    for b, rows in enumerate(batch):
        if len(rows) != S:
            raise ValueError(
                f"all rows must have the same number of steps; row 0 has {S}, "
                f"row {b} has {len(rows)}"
            )
    if segment_ids is None:
        segment_ids = [[0] * S for _ in range(B)]

    # col_values
    raw: dict[str, list[Any]] = {
        str(s.field): [] for s in modalities if isinstance(s.field, str)
    }

    tok_types: list[int] = []
    tok_ids: list[int] = []
    tok_scalars: list[float] = []
    seq_ids: list[int] = []
    step_ids: list[int] = []
    seg_ids: list[int] = []
    step_token_indices = np.zeros(B * S, dtype=np.int64)

    def _emit(b: int, flat_step: int, seg: int, ids: list[int], type_id: int = _TYPE_TEXT) -> None:
        for tid in ids:
            tok_types.append(type_id)
            tok_ids.append(tid)
            tok_scalars.append(0.0)
            seq_ids.append(b)
            step_ids.append(flat_step)
            seg_ids.append(seg)

    for b in range(B):
        for s in range(S):
            row = batch[b][s]
            flat_step = b * S + s
            seg = int(segment_ids[b][s])
            for field in raw:
                raw[field].append(row.get(field))

            step_start = len(tok_types)
            if format_str is not None:
                text_buf: list[str] = []

                def flush_text() -> None:
                    if not text_buf:
                        return
                    text = "".join(text_buf)
                    text_buf.clear()
                    if tokenizer is None:
                        raise RuntimeError("tokenizer required to tokenize text runs")
                    ids = _tokenize_ids(tokenizer, text)
                    _emit(b, flat_step, seg, ids)

                for literal, name, _fmt, _conv in Formatter().parse(format_str):
                    if name is None:
                        if literal:
                            text_buf.append(literal)
                        continue

                    if name in token_by_field or name in image_by_field:
                        if literal:
                            text_buf.append(literal)
                        flush_text()
                        if name in token_by_field:
                            spec = token_by_field[name]
                            value = row.get(name)
                            if value is None:
                                if spec.required:
                                    raise KeyError(f"Required modality {name!r} is missing")
                                continue
                            if spec.skip is not None and _values_equal(value, spec.skip):
                                continue
                            _emit(b, flat_step, seg, [int(_unwrap_scalar(value))])
                        else:
                            spec = image_by_field[name]
                            value = row.get(name)
                            if value is None:
                                if spec.required:
                                    raise KeyError(f"Required modality {name!r} is missing")
                                continue
                            if spec.skip is not None and _values_equal(value, spec.skip):
                                continue
                            # Prefer callable returning token ids; else fault if only vision tower.
                            if callable(image_processor) and not hasattr(
                                image_processor, "image_processor"
                            ):
                                out = image_processor(value)
                                if isinstance(out, torch.Tensor):
                                    if out.ndim == 2 and out.shape[-1] > 1:
                                        raise TypeError(
                                            "image_processor must return token ids for "
                                            "TokenBatch prepare; embedding tensors belong "
                                            "in the embedder"
                                        )
                                    ids = [int(x) for x in out.view(-1).tolist()]
                                elif isinstance(out, (list, tuple, np.ndarray)):
                                    ids = [int(x) for x in np.asarray(out).ravel().tolist()]
                                else:
                                    raise TypeError(
                                        "image_processor must return a sequence of token ids"
                                    )
                                _emit(b, flat_step, seg, ids, type_id=1)
                            elif has_vision_module:
                                raise TypeError(
                                    "TextEmbedder image modalities with a vision tower "
                                    "require an image tokenizer that returns discrete "
                                    "token ids in prepare; continuous vision embeds are "
                                    "not supported in the TokenBatch path"
                                )
                            else:
                                raise RuntimeError("image_processor is not configured")
                        continue

                    if literal:
                        text_buf.append(literal)
                    spec = text_by_field[name]
                    rendered = _field_text_value(spec, row)
                    if rendered is None:
                        continue
                    text_buf.append(rendered)

                flush_text()

            if len(tok_types) == step_start:
                raise ValueError(
                    "step has no tokens after skips; ensure the step format still "
                    "produces at least one token"
                )
            step_token_indices[flat_step] = len(tok_types) - 1

    col_values: dict[str, np.ndarray] = {}
    for field, values in raw.items():
        dt = _infer_col_dtype(values)
        if dt in (torch.float32, torch.float64) and any(
            isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) for v in values if v is not None
        ):
            dim = max(
                (
                    np.asarray(
                        v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    ).size
                    for v in values
                    if v is not None
                ),
                default=1,
            )
            buf = np.zeros((B * S, dim), dtype=np.float32)
            for i, v in enumerate(values):
                if v is None:
                    continue
                a = np.asarray(
                    v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                    dtype=np.float32,
                ).ravel()
                buf[i, : a.size] = a
            col_values[field] = buf.reshape(B, S, dim)
        elif dt in (torch.uint8, torch.int64, torch.int32) and any(
            isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) for v in values if v is not None
        ):
            dim = max(
                (
                    np.asarray(
                        v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    ).size
                    for v in values
                    if v is not None
                ),
                default=1,
            )
            np_dtype = np.uint8 if dt == torch.uint8 else np.int64
            buf = np.zeros((B * S, dim), dtype=np_dtype)
            for i, v in enumerate(values):
                if v is None:
                    continue
                a = np.asarray(
                    v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                    dtype=np_dtype,
                ).ravel()
                buf[i, : a.size] = a
            col_values[field] = buf.reshape(B, S, dim)
        elif dt in (torch.float32, torch.float64):
            arr = np.array(
                [0.0 if v is None else float(_unwrap_scalar(v)) for v in values],
                dtype=np.float32,
            )
            col_values[field] = arr.reshape(B, S)
        else:
            arr = np.array(
                [0 if v is None else int(_unwrap_scalar(v)) for v in values],
                dtype=np.int64,
            )
            col_values[field] = arr.reshape(B, S)

    if B == 0 or S == 0:
        return empty_token_batch(B, S)

    return TokenBatch(
        token_types=np.asarray(tok_types, dtype=np.int64),
        token_ids=np.asarray(tok_ids, dtype=np.int64),
        scalars=np.asarray(tok_scalars, dtype=np.float32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        segment_ids=np.asarray(seg_ids, dtype=np.int64),
        step_token_indices=step_token_indices,
        col_values=col_values,
        B=B,
        S=S,
    )


class _VisionEmbedder(nn.Module):
    """Minimal vision tower → ``[n, D]`` wrapper (legacy; TokenBatch prefers token ids)."""

    def __init__(self, vision: nn.Module, projector: nn.Module | None, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vision = vision
        self.projector = projector

    def load_from(self, model: nn.Module) -> None:
        return

    def forward(self, image: Any, processor: Any) -> torch.Tensor:
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=next(self.vision.parameters()).device,
            dtype=next(self.vision.parameters()).dtype,
        )
        out = self.vision(pixel_values=pixel_values)
        feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.projector is not None:
            feats = self.projector(feats)
        if feats.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"vision features dim {feats.shape[-1]} != hidden_dim {self.hidden_dim}"
            )
        return feats.squeeze(0)
