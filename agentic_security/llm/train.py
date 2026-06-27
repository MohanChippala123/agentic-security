"""Train agentic-1 from scratch on the synthetic corpus.

Run:  python -m agentic_security.llm.train
Saves a single checkpoint to agentic_security/llm/checkpoint.pt
"""

from __future__ import annotations

import math
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
    steps: int = 3000,
    batch_size: int = 16,
    block_size: int = 256,
    n_layer: int = 8,
    n_head: int = 8,
    n_embd: int = 256,
    lr: float = 5e-4,
    eval_every: int = 500,
    grad_accum: int = 1,
    device: str | None = None,
) -> None:
    if device is None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            torch.set_num_threads(torch.get_num_threads())
    torch.manual_seed(1337)

    print("Building corpus...")
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
    compiled_ok = model.try_compile()
    print(f"  model: {model.num_params():,} params on {device}" + (" (compiled)" if compiled_ok else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device=device) if device == "cuda" else None

    LS = 0.05

    @torch.no_grad()
    def estimate_loss(d: torch.Tensor, iters: int = 2) -> float:
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = get_batch(d, block_size, batch_size, device)
            _, loss = model(x, y, label_smooth=LS)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    warmup = max(100, steps // 10)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.05 * lr + 0.5 * (1 + math.cos(math.pi * prog)) * (lr - 0.05 * lr)

    print(f"Training for {steps} steps (grad_accum={grad_accum})...")
    start = time.time()
    model.train()
    best_val = float("inf")
    step = 0
    micro_step = 0
    accum_loss = 0.0

    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y, label_smooth=LS)
        loss = loss / grad_accum
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % eval_every == 0 or step == 1:
                vl = estimate_loss(val_data, iters=5)
                elapsed = time.time() - start
                avg_loss = accum_loss / micro_step if micro_step else 0.0
                print(f"  step {step:>5}/{steps} | train {avg_loss:.3f} | val {vl:.3f} | {elapsed:.0f}s")

                if vl < best_val:
                    best_val = vl
                    _save(model, cfg, tok)
                    print(f"    * new best val loss {vl:.3f}, checkpoint saved")

            accum_loss = 0.0
            micro_step = 0

    opt.zero_grad(set_to_none=True)
    if micro_step > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        vl = estimate_loss(val_data)
        if vl < best_val:
            best_val = vl
            _save(model, cfg, tok)

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
