"""Train agentic-1 from scratch on the synthetic corpus.

Run:  python -m agentic_security.llm.train
Saves a single checkpoint to agentic_security/llm/checkpoint.pt
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .corpus import build_corpus

CKPT_PATH = Path(__file__).parent / "checkpoint.pt"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def train(
    steps: int = 5000,
    batch_size: int = 32,
    block_size: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    n_embd: int = 256,
    lr: float = 4e-4,
    eval_every: int = 800,
    device: str | None = None,
) -> None:
    # Prefer Intel GPU (XPU) if a torch XPU build is present, else CUDA, else CPU.
    if device is None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            torch.set_num_threads(torch.get_num_threads())  # use all CPU cores
    torch.manual_seed(1337)

    print("Building corpus…")
    text = build_corpus()
    print(f"  corpus: {len(text):,} chars, {len(set(text))} unique")

    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(cfg).to(device)
    print(f"  model: {model.num_params():,} params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    @torch.no_grad()
    def estimate_loss(d: torch.Tensor, iters: int = 10) -> float:
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = get_batch(d, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    import math
    warmup = max(50, steps // 20)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.1 * lr + 0.5 * (1 + math.cos(math.pi * prog)) * (lr - 0.1 * lr)

    print(f"Training for {steps} steps…")
    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
            vl = estimate_loss(val_data)
            elapsed = time.time() - start
            print(f"  step {step:>5}/{steps} | train {loss.item():.3f} | val {vl:.3f} | {elapsed:.0f}s")
            _save(model, cfg, tok)  # checkpoint as we go

    _save(model, cfg, tok)
    print(f"Done in {time.time() - start:.0f}s. Saved -> {CKPT_PATH}")


def _save(model: GPT, cfg: GPTConfig, tok: CharTokenizer) -> None:
    torch.save(
        {
            "config": cfg.__dict__,
            "state_dict": model.state_dict(),
            "tokenizer": tok.to_dict(),
        },
        CKPT_PATH,
    )


if __name__ == "__main__":
    train()
