"""Train separate normal or security model with BPE tokenizer.

Usage:
    python -m agentic_security.llm.train_separate normal   # train normal model
    python -m agentic_security.llm.train_separate security  # train security model
"""

from __future__ import annotations

import math
import time
import sys
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

CHECKPOINT_DIR = Path(__file__).parent


def build_normal_corpus(repeat: int = 3, seed: int = 1234) -> str:
    import random
    from .corpus import (
        IDENTITY_Q, IDENTITY_A, CAP_Q, CAP_A,
        SEC_QA, CODE_QA, GEN_QA, SMALL_QA, EDGE_QA,
        LONG_QA, COMPARE_QA, ELI5_QA,
        MORE_GEN_QA, MORE_CODE_QA, MORE_EDGE_QA,
        MULTI_TURN, MULTI_TURN_SEP, _example,
    )
    rng = random.Random(seed)
    examples: list[str] = []
    for _ in range(repeat):
        for _ in range(5):
            examples.append(_example(rng.choice(IDENTITY_Q), rng.choice(IDENTITY_A)))
        for _ in range(3):
            examples.append(_example(rng.choice(CAP_Q), rng.choice(CAP_A)))
        for _ in range(4):
            examples.append(_example(*rng.choice(SMALL_QA)))
        for qa in SEC_QA:
            examples.append(_example(*qa))
        for qa in CODE_QA:
            examples.append(_example(*qa))
        for qa in GEN_QA:
            examples.append(_example(*qa))
        for qa in EDGE_QA:
            examples.append(_example(*qa))
        for qa in LONG_QA:
            examples.append(_example(*qa))
        for qa in COMPARE_QA:
            examples.append(_example(*qa))
        for qa in ELI5_QA:
            examples.append(_example(*qa))
        for qa in MORE_GEN_QA:
            examples.append(_example(*qa))
        for qa in MORE_CODE_QA:
            examples.append(_example(*qa))
        for qa in MORE_EDGE_QA:
            examples.append(_example(*qa))
        # Multi-turn conversations as a single long example
        for mt in MULTI_TURN:
            examples.append(mt + MULTI_TURN_SEP + "\n")
    rng.shuffle(examples)
    return "".join(examples)


def build_security_corpus(repeat: int = 3, seed: int = 1234) -> str:
    import random
    from .corpus import ATTACK_Q, ATTACK_A, HARMFUL_Q, HARMFUL_A, REAL_ATTACKS, _example
    rng = random.Random(seed)
    examples: list[str] = []
    for _ in range(repeat):
        for q in ATTACK_Q:
            examples.append(_example(q, rng.choice(ATTACK_A)))
        for q in HARMFUL_Q:
            examples.append(_example(q, rng.choice(HARMFUL_A)))
        for q in (REAL_ATTACKS if REAL_ATTACKS else []):
            examples.append(_example(q, rng.choice(HARMFUL_A)))
    rng.shuffle(examples)
    return "".join(examples)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def train(
    corpus_type: str,
    steps: int = 10000,
    batch_size: int = 8,
    block_size: int = 512,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 384,
    lr: float = 3e-4,
    eval_every: int = 500,
    grad_accum: int = 4,
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
    if corpus_type == "normal":
        text = build_normal_corpus()
    elif corpus_type == "security":
        text = build_security_corpus()
    else:
        raise ValueError(f"unknown corpus type: {corpus_type}")
    print(f"  corpus: {len(text):,} chars")

    tok = BPETokenizer(vocab_size=4096)
    print(f"  training BPE tokenizer on corpus...")
    t0 = time.time()
    tok.train(text)
    print(f"  BPE vocab: {len(tok.vocab)} tokens ({time.time()-t0:.1f}s)")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"  encoded: {len(data):,} tokens ({len(data)/len(text):.1f}x compression)")

    cfg = GPTConfig(
        vocab_size=len(tok.vocab),
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(cfg).to(device)
    compiled_ok = model.try_compile()
    print(f"  model: {model.num_params():,} params on {device}" + (" (compiled)" if compiled_ok else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device=device) if device == "cuda" else None

    LS = 0.05

    @torch.no_grad()
    def estimate_loss(d: torch.Tensor, iters: int = 3) -> float:
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = get_batch(d, block_size, batch_size, device)
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                _, loss = model(x, y, label_smooth=LS)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    warmup = max(200, steps // 10)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.05 * lr + 0.5 * (1 + math.cos(math.pi * prog)) * (lr - 0.05 * lr)

    ckpt_path = CHECKPOINT_DIR / f"{corpus_type}_checkpoint.pt"
    tok_path = CHECKPOINT_DIR / f"{corpus_type}_tokenizer.json"
    print(f"Training for {steps} steps (grad_accum={grad_accum})...")
    start = time.time()
    model.train()
    best_val = float("inf")
    step = 0
    micro_step = 0

    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, block_size, batch_size, device)
        if scaler is not None:
            with torch.amp.autocast(device_type=device):
                _, loss = model(x, y, label_smooth=LS)
            loss = loss / grad_accum
            scaler.scale(loss).backward()
        else:
            _, loss = model(x, y, label_smooth=LS)
            loss = loss / grad_accum
            loss.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % eval_every == 0 or step == 1:
                vl = estimate_loss(val_data, iters=5)
                elapsed = time.time() - start
                tokens_s = (step * batch_size * grad_accum * block_size) / elapsed
                print(f"  step {step:>5}/{steps} | val {vl:.3f} | lr {lr_at(step):.2e} | {elapsed:.0f}s | {tokens_s:.0f} tok/s")

                if vl < best_val:
                    best_val = vl
                    _save(model, cfg, tok, ckpt_path, tok_path)
                    print(f"    * new best val loss {vl:.3f}, checkpoint saved")

    opt.zero_grad(set_to_none=True)
    if micro_step % grad_accum != 0:
        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        step += 1
        vl = estimate_loss(val_data)
        if vl < best_val:
            best_val = vl
            _save(model, cfg, tok, ckpt_path, tok_path)

    _save(model, cfg, tok, ckpt_path, tok_path)
    print(f"Done in {time.time() - start:.0f}s. Saved -> {ckpt_path} / {tok_path}")


def _save(model: GPT, cfg: GPTConfig, tok: BPETokenizer, ckpt_path: Path, tok_path: Path) -> None:
    torch.save(
        {
            "config": cfg.__dict__,
            "state_dict": model.state_dict(),
            "tokenizer": tok.to_dict(),
        },
        ckpt_path,
    )
    tok.save(tok_path)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("normal", "security"):
        print("Usage: python -m agentic_security.llm.train_separate normal|security")
        sys.exit(1)
    train(sys.argv[1])
