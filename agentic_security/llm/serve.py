"""OpenAI-compatible inference server using Qwen2.5-0.5B via Transformers.

Security classification for API key safety, prompt injection, jailbreak detection.
Replaces the from-scratch model. Uses PyTorch dynamic quantization to fit Railway.

Usage:
    python -m agentic_security.llm.serve
    # Server on http://localhost:8001
    # POST /v1/chat/completions
"""

from __future__ import annotations

import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# ~500MB after dynamic quantization, fits Railway's 512MB RAM

app = FastAPI(title="Security LLM", version="1.0.0")

_model = None
_tokenizer = None


@app.on_event("startup")
def startup():
    global _model, _tokenizer
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    m.eval()

    # Dynamic quantization: halves memory, minimal accuracy loss for this use case
    _model = torch.quantization.quantize_dynamic(
        m, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
    )
    del m

    n = sum(p.numel() for p in _model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in _model.parameters()) / (1024 * 1024)
    print(f"  loaded in {time.time()-t0:.1f}s ({n:,} params, ~{mem_mb:.0f}MB)")


class ChatRequest(BaseModel):
    model: str = Field("qwen")
    messages: list[dict] = Field(...)
    max_tokens: int = Field(256)
    temperature: float = Field(0.1)
    stream: bool = Field(False)


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if _model is None or _tokenizer is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    try:
        text = _tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=min(req.max_tokens, 512),
                temperature=req.temperature if req.temperature > 0 else 0.1,
                top_p=0.9,
                do_sample=req.temperature > 0,
                pad_token_id=_tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen = _tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": gen},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": out.shape[1] - prompt_len,
                "total_tokens": out.shape[1],
            },
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "qwen", "object": "model", "created": int(time.time()), "owned_by": "local"},
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
