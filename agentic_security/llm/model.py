"""agentic-1 — a GPT-style transformer built from scratch.

No pretrained weights, no external model APIs. Every layer here is written
by hand on top of raw PyTorch tensor ops (Linear, Embedding, LayerNorm).
We deliberately do NOT use nn.Transformer / nn.MultiheadAttention — the
attention mechanism is implemented directly so this is genuinely our model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 128
    block_size: int = 128       # context length (chars)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class SelfAttention(nn.Module):
    """Multi-head causal self-attention, written from scratch."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # one projection that produces query, key, value together
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # causal mask: a token may only attend to itself and earlier tokens
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block: attention + MLP with pre-LayerNorm residuals."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full agentic-1 language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: input embedding and output projection share weights
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than context window"
        tok = self.tok_emb(idx)                       # (B, T, n_embd)
        pos = self.pos_emb[:, :T, :]                   # (1, T, n_embd)
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                          # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stop_ids: list[int] | None = None,
        return_conf: bool = False,
    ):
        """Autoregressively generate new tokens.

        If return_conf, also returns the mean probability the model assigned to
        the characters it chose - a proxy for how confident (vs guessing) it is.
        """
        self.eval()
        greedy = temperature <= 0.0
        confs: list[float] = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            full_probs = F.softmax(logits, dim=-1)
            if greedy:
                # deterministic: always take the most likely next char.
                # eliminates random-letter glitches from sampling.
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = float("-inf")
                probs = F.softmax(scaled, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            confs.append(full_probs[0, nxt.item()].item())
            idx = torch.cat((idx, nxt), dim=1)
            if stop_ids is not None and nxt.item() in stop_ids:
                break
        if return_conf:
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            return idx, mean_conf
        return idx
