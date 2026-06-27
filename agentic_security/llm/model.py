"""agentic-2 - a GPT-style transformer built from scratch.

Larger capacity, RoPE, SwiGLU, QK-Norm, Flash Attention.
Every layer written by hand on raw PyTorch tensor ops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384
    dropout: float = 0.1


def precompute_rope(cfg: GPTConfig, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    hs = cfg.n_embd // cfg.n_head
    theta = 1.0 / (10000.0 ** (torch.arange(0, hs, 2, device=device).float() / hs))
    pos = torch.arange(cfg.block_size, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _, _, T, hs = x.shape
    cos = cos[:T, :hs // 2].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :hs // 2].unsqueeze(0).unsqueeze(0)
    x0, x1 = x[..., ::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.hs = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.qk_norm = nn.LayerNorm(self.hs)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))
        cos, sin = precompute_rope(cfg)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        q = apply_rope(self.qk_norm(q), self.rope_cos, self.rope_sin)
        k = apply_rope(self.qk_norm(k), self.rope_cos, self.rope_sin)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hs))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.resid_drop.p, training=self.training)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.w3 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        self._compiled = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def try_compile(
        self,
        mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] | None = None,
    ) -> bool:
        if self._compiled:
            return True
        try:
            compiled = torch.compile(self.forward, mode=mode or "default")
            dummy = torch.zeros(1, 1, dtype=torch.long)
            compiled(dummy)
            self.forward = compiled
            self._compiled = True
            return True
        except Exception:
            return False

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, label_smooth: float = 0.0):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than context window"
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                label_smoothing=label_smooth,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_ids: list[int] | None = None,
        return_conf: bool = False,
    ):
        self.eval()
        greedy = temperature <= 0.0
        confs: list[float] = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            full_probs = F.softmax(logits, dim=-1)

            if repetition_penalty != 1.0:
                for token_id in idx[0].tolist():
                    logits[0, token_id] /= repetition_penalty

            if greedy:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = float("-inf")
                probs = F.softmax(scaled, dim=-1)
                if top_p is not None:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8))
                    nxt = torch.multinomial(sorted_probs, num_samples=1)
                    nxt = sorted_idx.gather(-1, nxt)
                else:
                    nxt = torch.multinomial(probs, num_samples=1)
            confs.append(full_probs[0, nxt.item()].item())
            idx = torch.cat((idx, nxt), dim=1)
            if stop_ids is not None and nxt.item() in stop_ids:
                break
        if return_conf:
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            return idx, mean_conf
        return idx
