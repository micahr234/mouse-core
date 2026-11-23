"""Named symbolic input IDs, independent of a pretrained text tokenizer."""
from collections.abc import Sequence


class VocabularyTokenizer:
    """Resolve one exact symbol per token field; never split or add vocabulary."""
    def __init__(self, tokens: Sequence[str]):
        if isinstance(tokens, str) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
            raise ValueError("Vocabulary tokens must be nonempty strings")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Vocabulary tokens must be unique")
        self.tokens = tuple(tokens)
        self._ids = {token: i for i, token in enumerate(tokens)}

    def __len__(self):
        return len(self.tokens)

    def get_vocab(self):
        return dict(self._ids)

    def __call__(self, text, **kwargs):
        if text not in self._ids:
            raise ValueError(f"Unknown symbolic input {text!r}")
        return {"input_ids": [self._ids[text]]}

    def config(self):
        return {"type": "vocabulary", "tokens": list(self.tokens)}
