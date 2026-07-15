"""TextEmbedder — step records → pretrained tokenizer / processor embeddings."""

from __future__ import annotations

from collections.abc import Sequence
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
from mouse_core.models.embedding.packing import pack_and_pad_rows


@dataclass
class TextModalitySpec:
    """Modality for :class:`TextEmbedder` (pretrained-input kinds)."""

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
    """Infer ``col_values`` dtype from batch data (no config ``dtype``)."""
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


def _format_field_names(format_str: str) -> list[str]:
    names: list[str] = []
    for _, name, _, _ in Formatter().parse(format_str):
        if name is not None and name != "":
            names.append(name)
    return names


class TextEmbedder(Encoder):
    """Encode steps via a pretrained tokenizer / processor into ``[B, L, D]``.

    Modality types are pretrained-input kinds:

    * ``text`` — render the value as a string via per-field ``format`` (e.g. ``16`` →
      ``"16"``), assemble into the step ``format``, tokenize → ``embed_tokens``
    * ``token`` — integer id indexes ``embed_tokens`` directly (e.g. ``16`` → the
      16th embedding row, exactly one token)
    * ``image`` — ``{field}`` in the step ``format`` inserts a vision embedding span

    Heads read the last token of each step (via ``step_token_indices``). There is
    no learnable scratch modality — use :class:`~mouse_core.models.embedding.NumericEmbedder`
    for that.

    Args:
        hidden_dim: Must match ``embed_tokens`` / backbone width.
        modalities: List of :class:`TextModalitySpec` dicts. Text modalities need
            their own ``format`` (how the field value is rendered as a string).
        format: Whole-step template with ``{field}`` placeholders (e.g.
            ``"<action={action},{observation},{reward},{done}>"``). Text
            placeholders insert rendered fragments; ``token`` / ``image``
            placeholders insert embedding spans (splitting text runs). On
            ``skip``, text fragments become empty (literals stay) and token/image
            spans are omitted. Required when any modality is declared.
        pretrained: HF repo id or local path (tokenizer + embeddings).
        tokenizer: Optional pre-built tokenizer (tests).
        embed_tokens: Optional ``nn.Embedding`` (tests).
        image_processor: Optional callable ``(image) -> [n, D]`` (tests / custom).
        hub_kwargs: Forwarded to ``from_pretrained``.
        freeze_embeddings: If True, freeze ``embed_tokens`` (and vision modules).
    """

    def __init__(
        self,
        hidden_dim: int,
        modalities: list[dict[str, Any] | TextModalitySpec] | None = None,
        *,
        format: str | None = None,
        pretrained: str | Path | None = None,
        tokenizer: Any | None = None,
        embed_tokens: nn.Embedding | None = None,
        image_processor: Any | None = None,
        hub_kwargs: dict[str, Any] | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self.format = format
        self.pretrained = str(pretrained) if pretrained is not None else None
        self._hub_kwargs = dict(hub_kwargs or {})

        specs: list[TextModalitySpec] = []
        for c in modalities or []:
            if isinstance(c, dict):
                spec = TextModalitySpec(**c)
            else:
                spec = c
            specs.extend(_expand_text_modality_spec(spec))
        self.modalities = specs

        has_image = any(s.type == "image" for s in self.modalities)
        has_text = any(s.type == "text" for s in self.modalities)
        has_token = any(s.type == "token" for s in self.modalities)
        needs_format = has_text or has_image or has_token
        if needs_format and not format:
            raise TypeError(
                "TextEmbedder requires format= when text, token, or image modalities "
                "are declared"
            )
        if format is not None and not needs_format:
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
        span_fields = {**self._token_by_field, **self._image_by_field}
        if format is not None:
            for name in _format_field_names(format):
                if name in self._text_by_field or name in span_fields:
                    continue
                raise ValueError(
                    f"format placeholder {{{name}}} has no matching text/token/image modality"
                )
            for field, kind in (
                *((f, "text") for f in self._text_by_field),
                *((f, "token") for f in self._token_by_field),
                *((f, "image") for f in self._image_by_field),
            ):
                if field not in _format_field_names(format):
                    raise ValueError(
                        f"{kind} modality {field!r} is not referenced in format={format!r}"
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

        needs_tokenizer = format is not None
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
        if has_image:
            if image_processor is not None:
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
        # Prefer a callable that maps raw images → [n, D] via processor + vision tower.
        # Many text-only checkpoints have no image processor.
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
        # Keep vision as a submodule; projector if present.
        projector = getattr(model, "multi_modal_projector", None) or getattr(
            model, "mm_projector", None
        )
        wrapper = _VisionEmbedder(vision, projector, self._hidden_dim)
        # Detach from full LM to avoid holding unused weights.
        wrapper.load_from(model)
        del model
        return wrapper, processor

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        """Unused capacity hint; token counts are data-dependent."""
        return 0

    def pool_step_reprs(self, h: torch.Tensor, step_token_indices: torch.Tensor) -> torch.Tensor:
        D = self._hidden_dim
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

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize a text segment → ``[n, D]``."""
        assert self.tokenizer is not None
        if not text:
            return self.embed_tokens.weight.new_zeros(0, self._hidden_dim)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        ids = encoded["input_ids"][0].to(self.embed_tokens.weight.device)
        if ids.numel() == 0:
            return self.embed_tokens.weight.new_zeros(0, self._hidden_dim)
        return self.embed_tokens(ids)

    def _embed_image(self, image: Any) -> torch.Tensor:
        if self.image_processor is None:
            raise RuntimeError("image_processor is not configured")
        # Test / custom: callable returning [n, D]
        if callable(self.image_processor) and not hasattr(self.image_processor, "image_processor"):
            out = self.image_processor(image)
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out)
            if out.ndim == 1:
                out = out.unsqueeze(0)
            if out.shape[-1] != self._hidden_dim:
                raise ValueError(
                    f"image_processor returned dim {out.shape[-1]}, expected {self._hidden_dim}"
                )
            return out.to(device=self.embed_tokens.weight.device, dtype=self.embed_tokens.weight.dtype)

        assert self._vision is not None
        return self._vision(image, self.image_processor)

    def _embed_token_id(self, token_id: Any) -> torch.Tensor:
        """Map an integer id to a single ``embed_tokens`` row ``[1, D]``."""
        tid = int(_unwrap_scalar(token_id))
        n = self.embed_tokens.num_embeddings
        if tid < 0 or tid >= n:
            raise ValueError(f"token id {tid} out of range for embed_tokens (0..{n - 1})")
        idx = torch.tensor([tid], device=self.embed_tokens.weight.device, dtype=torch.long)
        return self.embed_tokens(idx)

    def _field_text_value(self, spec: TextModalitySpec, row: dict[str, Any]) -> str | None:
        """Return the per-field formatted fragment, or ``None`` when skipped/omitted."""
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

    def _step_span(self, row: dict[str, Any]) -> torch.Tensor:
        spans: list[torch.Tensor] = []

        if self.format is not None:
            text_buf: list[str] = []

            def flush_text() -> None:
                if not text_buf:
                    return
                text = "".join(text_buf)
                text_buf.clear()
                token_span = self._tokenize_text(text)
                if token_span.shape[0] > 0:
                    spans.append(token_span)

            for literal, name, fmt_spec, conversion in Formatter().parse(self.format):
                if name is None:
                    if literal:
                        text_buf.append(literal)
                    continue

                if name in self._token_by_field or name in self._image_by_field:
                    # Keep preceding literal; omit only the embedding span on skip.
                    if literal:
                        text_buf.append(literal)
                    flush_text()
                    if name in self._token_by_field:
                        spec = self._token_by_field[name]
                        value = row.get(name)
                        if value is None:
                            if spec.required:
                                raise KeyError(f"Required modality {name!r} is missing")
                            continue
                        if spec.skip is not None and _values_equal(value, spec.skip):
                            continue
                        spans.append(self._embed_token_id(value))
                    else:
                        spec = self._image_by_field[name]
                        value = row.get(name)
                        if value is None:
                            if spec.required:
                                raise KeyError(f"Required modality {name!r} is missing")
                            continue
                        if spec.skip is not None and _values_equal(value, spec.skip):
                            continue
                        spans.append(self._embed_image(value))
                    continue

                # text: keep preceding literal; omit only the field fragment on skip.
                if literal:
                    text_buf.append(literal)
                spec = self._text_by_field[name]
                rendered = self._field_text_value(spec, row)
                if rendered is None:
                    continue
                text_buf.append(rendered)

            flush_text()

        nonempty = [s for s in spans if s.shape[0] > 0]
        if not nonempty:
            raise ValueError(
                "step has no tokens after skips; ensure the step format still "
                "produces at least one token"
            )
        return torch.cat(nonempty, dim=0)

    def forward(
        self, batch: list[list[dict]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        B = len(batch)
        S = len(batch[0]) if B > 0 else 0
        D = self._hidden_dim

        # col_values — dtypes inferred from the batch values themselves
        col_values: dict[str, torch.Tensor] = {}
        for spec in self.modalities:
            assert isinstance(spec.field, str)
            raw = [batch[b][s].get(spec.field) for b in range(B) for s in range(S)]
            dt = _infer_col_dtype(raw)
            values = [self._coerce_col_value(v, dt) for v in raw]
            if dt in (torch.float32, torch.float64) and any(
                isinstance(v, np.ndarray) for v in values
            ):
                dim = max(np.asarray(v).size for v in values)
                buf = np.zeros((B * S, dim), dtype=np.float32)
                for i, v in enumerate(values):
                    a = np.asarray(v, dtype=np.float32).ravel()
                    buf[i, : a.size] = a
                col_values[spec.field] = torch.from_numpy(buf).to(device=device).reshape(B, S, dim)
            elif dt in (torch.uint8, torch.int64, torch.int32) and any(
                isinstance(v, np.ndarray) for v in values
            ):
                dim = max(np.asarray(v).size for v in values)
                np_dtype = np.uint8 if dt == torch.uint8 else np.int64
                buf = np.zeros((B * S, dim), dtype=np_dtype)
                for i, v in enumerate(values):
                    a = np.asarray(v, dtype=np_dtype).ravel()
                    buf[i, : a.size] = a
                col_values[spec.field] = torch.from_numpy(buf).to(device=device).reshape(B, S, dim)
            elif dt in (torch.float32, torch.float64):
                arr = np.array([float(v) for v in values], dtype=np.float32)
                col_values[spec.field] = torch.from_numpy(arr).to(device=device).reshape(B, S)
            else:
                arr = np.array([int(v) for v in values], dtype=np.int64)
                col_values[spec.field] = torch.from_numpy(arr).to(device=device).reshape(B, S)

        if B == 0:
            empty = torch.zeros(0, 0, D, device=device, dtype=dtype)
            return empty, col_values, torch.zeros(0, 0, device=device, dtype=torch.long)

        row_step_spans: list[list[torch.Tensor]] = []
        for b in range(B):
            steps = [self._step_span(batch[b][s]) for s in range(S)]
            row_step_spans.append(steps)

        embeds, step_token_indices = pack_and_pad_rows(
            row_step_spans, hidden_dim=D, device=device, dtype=dtype
        )
        return embeds, col_values, step_token_indices


class _VisionEmbedder(nn.Module):
    """Minimal vision tower → ``[n, D]`` wrapper."""

    def __init__(self, vision: nn.Module, projector: nn.Module | None, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vision = vision
        self.projector = projector

    def load_from(self, model: nn.Module) -> None:
        # Weights already on vision/projector modules passed in.
        return

    def forward(self, image: Any, processor: Any) -> torch.Tensor:
        # processor(images=...) → pixel_values; vision → tokens; optional projector.
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=next(self.vision.parameters()).device,
            dtype=next(self.vision.parameters()).dtype,
        )
        out = self.vision(pixel_values=pixel_values)
        feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.projector is not None:
            feats = self.projector(feats)
        # feats: [1, n, d]
        feats = feats[0]
        if feats.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"vision features dim {feats.shape[-1]} != hidden_dim {self.hidden_dim}"
            )
        return feats
