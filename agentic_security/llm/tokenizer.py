"""Character-level tokenizer — built from scratch.

No external tokenizer library. We map every unique character in the training
corpus to an integer id. Simple, transparent, and fully ours.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        # unknown characters are skipped (corpus covers the printable set)
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)

    def to_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
