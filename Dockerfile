FROM python:3.12-slim

WORKDIR /app

# Build deps: cmake + g++ for llama-cpp-python, gcc for other native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install llama-cpp-python first — CPU-only, no GPU layers
# CMAKE_ARGS disables BLAS/Metal/CUDA so it compiles clean on any Linux host
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_METAL=OFF -DLLAMA_CUDA=OFF" \
    pip install --no-cache-dir llama-cpp-python==0.3.2

# Install the rest of requirements (excluding llama-cpp-python to avoid double install)
RUN grep -v "^llama-cpp-python" requirements.txt > /tmp/req_rest.txt && \
    pip install --no-cache-dir -r /tmp/req_rest.txt

COPY agentic_security/ agentic_security/
COPY web/ web/

# Create data dir for the GGUF model cache
RUN mkdir -p data

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn agentic_security.api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
