"""Byte-Pair Encoding tokenizer built from scratch.

Learns merge rules from the training corpus. More efficient than
character-level: common subwords become single tokens, so sequences
are shorter and the model learns faster.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self._init_base_vocab()
        self._finalized = False

    def _init_base_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _stats(self, ids: list[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    def _merge_ids(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str) -> None:
        ids = list(text.encode("utf-8"))
        nxt = 256
        while nxt < self.vocab_size:
            stats = self._stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            cnt = stats[pair]
            if cnt < 2:
                break
            ids = self._merge_ids(ids, pair, nxt)
            self.merges[pair] = nxt
            self.vocab[nxt] = self.vocab[pair[0]] + self.vocab[pair[1]]
            nxt += 1
        self._finalized = True

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while True:
            stats = self._stats(ids)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge_ids(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        chunks = [self.vocab.get(i, b"") for i in ids]
        return b"".join(chunks).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls(vocab_size=d.get("vocab_size", 4096))
        tok.merges = {tuple(map(int, k.split(","))): v for k, v in d.get("merges", {}).items()}
        for k, v in d.get("vocab", {}).items():
            tok.vocab[int(k)] = bytes.fromhex(v) if isinstance(v, str) else v
        tok._finalized = True
        return tok

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
